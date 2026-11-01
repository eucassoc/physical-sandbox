"""The display bundle the browser draws (bundle.json and RBM1 meshes), made from a URDF."""
import json
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from physical_sandbox.bundle import MAX_FACES, decimate, make_bundle, read_mesh, read_stl

MINI = Path(__file__).parent / "fixtures" / "mini-robot"
PROTO = Path(os.environ.get("PS_PROTO_DIR", "~/work/physical-sandbox-proto")).expanduser()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rbm1(path):
    blob = Path(path).read_bytes()
    flags, nv, ni = struct.unpack_from("<III", blob, 4)
    pos = np.frombuffer(blob, "<f4", nv * 3, 16).reshape(-1, 3)
    off = 16 + nv * 12 + (nv * 12 if flags & 2 else 0) + (nv * 8 if flags & 1 else 0)
    idx = np.frombuffer(blob, "<u4", ni, off)
    return blob[:4], flags, pos, idx


def _mini(tmp_path):
    return make_bundle(MINI / "mini.urdf", product=_json(MINI / "product.json"),
                       profile=_json(MINI / "profile.json"), name="mini", out_dir=tmp_path / "out")


def sphere_stl(path, slices=150, stacks=101, r=0.1):
    """A binary STL UV sphere with 2 * slices * (stacks - 1) triangles (30,000 by default)."""
    tris = []
    for i in range(stacks):
        t0, t1 = math.pi * i / stacks, math.pi * (i + 1) / stacks
        for j in range(slices):
            p0, p1 = 2 * math.pi * j / slices, 2 * math.pi * (j + 1) / slices
            a, b, c, d = ([r * math.sin(t) * math.cos(p), r * math.sin(t) * math.sin(p), r * math.cos(t)]
                          for t, p in ((t0, p0), (t1, p0), (t1, p1), (t0, p1)))
            if i > 0:
                tris.append((a, b, d))
            if i < stacks - 1:
                tris.append((b, c, d))
    rec = np.zeros(len(tris), [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = np.array(tris, "<f4")
    path.write_bytes(bytes(80) + len(tris).to_bytes(4, "little") + rec.tobytes())
    return len(tris)


def one_link_urdf(folder: Path, mesh_ref: str) -> Path:
    path = folder / "one.urdf"
    path.write_text(f"""<?xml version="1.0"?>
<robot name="one"><link name="body"><visual><geometry><mesh filename="{mesh_ref}"/></geometry></visual></link></robot>
""")
    return path


def test_bundle_json_has_the_root_the_joint_and_both_links(tmp_path):
    result = _mini(tmp_path)
    data = _json(result.bundle_path)
    assert result.bundle_path == tmp_path / "out" / "bundle.json"
    assert data["id"] == "mini"
    assert data["rootLink"] == "base_link"
    assert data["joints"][0]["name"] == "pan"
    assert len(data["links"]) == 2 and result.links == 2
    assert result.joints == ["pan"]
    for key in ("version", "displayName", "vendor", "licenseId", "urdfSourceUrl", "totalMassKg", "locomotion",
                "hoverHeightM", "motion", "rootLink", "links", "joints", "chains", "dimensions"):
        assert key in data, key
    assert data["displayName"] == "Mini arm"
    assert data["licenseId"] == "Apache-2.0"
    assert data["chains"] == [{"name": "arm", "baseLink": "base_link", "tipLink": "arm_link", "joints": ["pan"]}]
    assert data["motion"] == {"maxJointVel": [2.0], "maxJointAcc": [5.0]}  # the '_' notes stay behind
    j = data["joints"][0]
    assert (j["type"], j["parent"], j["child"], j["lower"], j["upper"]) == ("revolute", "base_link", "arm_link",
                                                                             -1.5, 1.5)
    assert j["origin"] == {"xyz": [0.0, 0.0, 0.04], "rpy": [0.0, 0.0, 0.0]}


def test_meshes_are_rbm1_and_a_scaled_use_is_its_own_baked_file(tmp_path):
    result = _mini(tmp_path)
    data = _json(result.bundle_path)
    base = data["links"]["base_link"]["visuals"][0]["mesh"]
    arm = data["links"]["arm_link"]["visuals"][0]["mesh"]
    assert base == "assets/box.bin"
    assert arm != base and arm.startswith("assets/box") and arm.endswith(".bin")
    assert sorted(result.mesh_paths) == sorted([tmp_path / "out" / base, tmp_path / "out" / arm])
    magic, _, pos, idx = _rbm1(tmp_path / "out" / base)
    assert magic == b"RBM1"
    assert len(idx) == 36 and np.allclose(pos.max(0), [0.04, 0.04, 0.04])
    _, _, pos2, _ = _rbm1(tmp_path / "out" / arm)
    assert np.allclose(pos2.max(0), [0.08, 0.02, 0.02])  # scale 2 0.5 0.5 baked in


def test_dimensions_are_measured_from_the_meshes_at_rest(tmp_path):
    data = _json(_mini(tmp_path).bundle_path)
    # base cube 4 cm centred on x, y; the arm (8 x 2 x 2 cm) on top of it from x = 0
    assert data["dimensions"] == {"lengthM": 0.1, "widthM": 0.04, "heightM": 0.06}


def test_a_30000_face_mesh_is_thinned_to_at_most_20000(tmp_path):
    n = sphere_stl(tmp_path / "ball.stl")
    assert n == 30000
    result = make_bundle(one_link_urdf(tmp_path, "ball.stl"), product={}, profile=None, name="ball",
                         out_dir=tmp_path / "out")
    _, _, pos, idx = _rbm1(result.mesh_paths[0])
    assert 0 < len(idx) // 3 <= MAX_FACES
    assert np.allclose(np.linalg.norm(pos, axis=1).max(), 0.1, atol=0.01)  # still the same ball
    assert any("thinned" in w for w in result.warnings)


def test_decimate_drops_faces_that_collapse():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1e-6, 0, 0]], float)
    faces = np.array([[0, 1, 2], [0, 3, 1]])
    p, f, _ = decimate(pos, faces, max_faces=1)
    assert len(f) == 1 and len(p) == 3


def test_a_missing_mesh_names_the_mesh(tmp_path):
    shutil.copytree(MINI, tmp_path / "mini")
    (tmp_path / "mini" / "meshes" / "box.stl").unlink()
    with pytest.raises(FileNotFoundError, match="box.stl"):
        make_bundle(tmp_path / "mini" / "mini.urdf", product={}, profile=None, name="mini",
                    out_dir=tmp_path / "out")
    assert not (tmp_path / "out" / "bundle.json").exists()  # no data file pointing at meshes that are not there


def test_ascii_and_binary_stl_read_the_same(tmp_path):
    ascii_v, ascii_f = read_stl(MINI / "meshes" / "box.stl")
    tris = ascii_v[ascii_f]
    rec = np.zeros(len(tris), [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = tris
    (tmp_path / "b.stl").write_bytes(b"solid but binary".ljust(80, b" ") + len(tris).to_bytes(4, "little")
                                     + rec.tobytes())
    bin_v, bin_f = read_stl(tmp_path / "b.stl")
    assert np.allclose(bin_v[bin_f], tris)
    assert len(ascii_v) == 8 and len(ascii_f) == 12  # shared corners


def test_obj_keeps_uvs_and_dae_applies_its_node_transform(tmp_path):
    (tmp_path / "t.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
    mesh = read_mesh(tmp_path / "t.obj")
    assert mesh.uvs is not None and len(mesh.faces) == 1
    (tmp_path / "t.dae").write_text("""<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array id="pa" count="9">0 0 0 1 0 0 0 1 0</float_array></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
 </mesh></geometry></library_geometries>
 <library_visual_scenes><visual_scene id="s"><node><translate>0 0 5</translate>
  <instance_geometry url="#g"/></node></visual_scene></library_visual_scenes>
</COLLADA>
""")
    dae = read_mesh(tmp_path / "t.dae")
    assert np.allclose(dae.positions[:, 2], 5) and len(dae.faces) == 1


def test_a_glb_without_trimesh_is_skipped_with_a_warning(tmp_path, monkeypatch):
    import physical_sandbox.bundle as bundle

    monkeypatch.setattr(bundle, "_trimesh", lambda: None)
    (tmp_path / "part.glb").write_bytes(b"glTF")
    result = make_bundle(one_link_urdf(tmp_path, "part.glb"), product={}, profile=None, name="one",
                         out_dir=tmp_path / "out")
    assert result.mesh_paths == []
    assert any("trimesh" in w for w in result.warnings)
    assert _json(result.bundle_path)["links"]["body"]["visuals"] == []


@pytest.mark.skipif(not (PROTO / "robots/so101/so101.urdf").exists(), reason="the old open build is not here")
def test_so101_matches_the_old_export(tmp_path):
    product = _json(PROTO / "robots/so101/product.json")
    profile = _json(PROTO / "core/data/profiles/so101.json")
    result = make_bundle(PROTO / "robots/so101/so101.urdf", product=product, profile=profile, name="so101",
                         out_dir=tmp_path)
    ours, theirs = _json(result.bundle_path), _json(PROTO / "app/data/so101.json")
    assert ours["links"] == theirs["links"]
    assert ours["joints"] == theirs["joints"]
    assert ours["rootLink"] == theirs["rootLink"] and ours["chains"] == theirs["chains"]
