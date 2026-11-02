// The machines drawn as see-through shells through the site's camera (three.js), and the lines drawn over them.
// Plain script text, inlined inside the page's module script after the three bundle; it holds no page state.

/** The site frame (x forward, y left, z up) as three's world (y up): (x, y, z) -> (x, z, -y) */
function toThree(x, y, z) { return new globalThis.THREE.Vector3(x, z, -y); }

/** The page camera where the real one is (as PS's video view does it): the picture's frustum from K, at -R^T t.
 *  The camera is given in the site frame (p_cam = R p + t, x right, y down, z forward): its position and axes are
 *  carried into three's world; three looks down -z with y up. */
function placeCamera(camera, { intr, extr }) {
  const THREE = globalThis.THREE;
  const r = extr.R;
  const t = extr.t;
  const fullWidth = 2 * Math.max(intr.cx, intr.width - intr.cx);
  const fullHeight = 2 * Math.max(intr.cy, intr.height - intr.cy);
  camera.fov = (2 * Math.atan(fullHeight / 2 / intr.fy)) * 180 / Math.PI;
  camera.setViewOffset(fullWidth, fullHeight, fullWidth / 2 - intr.cx, fullHeight / 2 - intr.cy, intr.width, intr.height);
  camera.aspect = (fullWidth / fullHeight) * (intr.fy / intr.fx);
  camera.position.copy(toThree(-(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]), -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]), -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])));
  camera.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(
    toThree(r[0], r[1], r[2]), toThree(-r[3], -r[4], -r[5]), toThree(-r[6], -r[7], -r[8])));
  camera.updateProjectionMatrix();
}

/** Base64 text to an ArrayBuffer. */
function bytesOf(b64) {
  const s = atob(b64);
  const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i += 1) u[i] = s.charCodeAt(i);
  return u.buffer;
}

/** The scene for a machine's shells: `model.links` read from `buffer` (positions, then triangle indices, each link
 *  in its own frame), one group of meshes per colour. Returns { renderer, scene, camera: null, site, records:
 *  [{ group, meshes }], geometries, materials }; the caller makes the camera, calls placeCamera and stores it in
 *  three.camera. */
function buildShells(glCanvas, model, buffer, colors) {
  const THREE = globalThis.THREE;
  const geometries = model.links.map((l) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buffer, l.offset, l.vCount * 3), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, l.offset + l.vCount * 12, l.iCount), 1));
    g.computeVertexNormals();
    return g;
  });
  const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, alpha: true, antialias: true });
  renderer.setPixelRatio(1);
  renderer.setSize(glCanvas.width, glCanvas.height, false);
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x707070, 2.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(5, 10, 3);
  scene.add(sun);
  // The site frame (x forward, y left, z up) inside, three's world (y up) outside: (x, y, z) -> (x, z, -y)
  const site = new THREE.Group();
  site.matrixAutoUpdate = false;
  site.matrix.makeBasis(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 1, 0));
  scene.add(site);
  // Each record in its own colour only (none of the machine's own), see-through. A machine first writes its
  // depth without colour, then its colour over only its front faces: it reads as one tinted shell rather
  // than a tangle of its own insides
  const depthOnly = new THREE.MeshBasicMaterial({ colorWrite: false });
  const materials = colors.map((c) => new THREE.MeshLambertMaterial({ color: c, transparent: true, opacity: 0.5, depthWrite: false, depthFunc: THREE.LessEqualDepth }));
  const records = colors.map((_, r) => {
    const group = new THREE.Group();
    const meshes = [];
    model.links.forEach((l, i) => {
      for (const material of [depthOnly, materials[r]]) {
        const mesh = new THREE.Mesh(geometries[i], material);
        mesh.matrixAutoUpdate = false;   // the pose comes from the server as it is
        mesh.frustumCulled = false;
        group.add(mesh);
        meshes.push({ mesh, link: i });
      }
    });
    site.add(group);
    return { group, meshes };
  });
  return { renderer, scene, camera: null, site, records, geometries, materials: [depthOnly, ...materials] };
}

/** Draws the shells. posesPerRecord[r] is record r's link poses for the frame (16 numbers per link, row by row)
 *  or null; base is the machine's base transform (16 numbers, row by row) or null for identity; shown[r] says
 *  whether record r is drawn at all. */
function drawShells(three, posesPerRecord, base, shown) {
  const THREE = globalThis.THREE;
  const { renderer, scene, camera, records } = three;
  const draws = [];
  const local = new THREE.Matrix4();
  // Each link's pose is in the machine's own frame; the machine's base transform (where it stands) comes apart
  const baseM = base ? new THREE.Matrix4().set(...base) : new THREE.Matrix4();
  records.forEach((rec, r) => {
    const poses = posesPerRecord[r];
    rec.group.visible = false;
    if (!(shown[r] && poses)) return;
    for (const { mesh, link } of rec.meshes) {
      mesh.matrix.copy(baseM).multiply(local.set(...poses[link]));   // base x pose
      mesh.matrixWorldNeedsUpdate = true;
    }
    draws.push(rec);
  });
  // One machine at a time with the depth cleared between them: no machine hides another, they all show through
  renderer.autoClear = false;
  renderer.clear();
  for (const rec of draws) {
    rec.group.visible = true;
    renderer.clearDepth();
    renderer.render(scene, camera);
    rec.group.visible = false;
  }
}

/** Frees the shells' GPU memory (shapes, materials, the renderer's own buffers). */
function releaseShells(three) {
  if (!three) return;
  three.geometries.forEach((g) => g.dispose());
  three.materials.forEach((m) => m.dispose());
  three.renderer.dispose();
}

/** Draws the points as lines on ctx; a null (a point behind the camera) breaks the line. */
function strokePath(ctx, points, close) {
  let open = false;
  ctx.beginPath();
  for (const p of points) {
    if (!p) { open = false; continue; }
    if (open) ctx.lineTo(p[0], p[1]); else { ctx.moveTo(p[0], p[1]); open = true; }
  }
  if (close) ctx.closePath();
}
