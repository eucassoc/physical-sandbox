# Machines and objects for the catalog

A machine you register in the PS.com catalog is a folder: its URDF, the meshes the URDF points at,
`product.json` and, optionally, `profile.json`. A mesh-only thing (a statue, a shelf, a fixture) is an
**object**: one mesh file and `object.json`.

```sh
psbx machine init my-arm/      # writes product.json and profile.json templates (never overwrites)
psbx machine check my-arm/     # one line per check; exit 1 while any line says NG
psbx machine init statue.stl   # writes object.json beside the mesh
psbx machine check statue.stl
```

`init` sets every field a person has to decide to `"<fill>"`, and `check` stays NG until none is left. A
field you do not need (say `maker` on a generic product) can be deleted instead.

The browser does not read URDF. When a machine is pushed, psbx turns it into a **display bundle**:
`bundle.json` (links and their visual meshes, joints, chains, the profile's figures) and one binary mesh per
visual under `assets/`. In Python:

```python
from pathlib import Path
from physical_sandbox.bundle import make_bundle

result = make_bundle(Path("my-arm/my-arm.urdf"), product=product, profile=profile, name="my-arm",
                     out_dir=Path("bundle"))
print(result.links, result.joints, result.warnings)
```

What the bundle does with the meshes:

- STL (binary or ASCII), OBJ (with its UVs, normals and `map_Kd` texture) and Collada DAE (with the scene's
  node transforms) are read with numpy alone. GLB, glTF and 3MF need `trimesh` (`pip install trimesh`);
  without it those visuals are skipped with a warning, and `machine check` says so.
- A `<mesh scale="...">` is baked into the vertices (a mirror reverses the winding). One mesh used at two
  scales becomes two files.
- A mesh with more than 20,000 triangles is thinned by vertex clustering (64 cells along each side of its
  bounding box, coarser if that is still too many), and a warning says how far. Keep display meshes under
  20,000 triangles if the detail matters to you.
- Only visual meshes are carried; `<box>`, `<cylinder>` and `<sphere>` visuals are not drawn (a warning
  names them).
- `dimensions` is measured from the meshes with every joint at 0. A footprint in `profile.json`'s
  `motion.footprint` wins for length and width.

Mesh references are found as written relative to the URDF, then with `.obj` for `.stl`, then under
`meshes/`. `package://<pkg>/` is dropped, since the package folder is not known.

## product.json

What the catalog shows. The fields are those of the old open build's `schema/product.schema.json`
(vocabulary 0.4).

| field | required | what |
|---|---|---|
| `id` | yes | `a-z 0-9 _ -`, starting with a letter or digit |
| `displayName` | yes | `{"ja": "...", "en": "..."}`, both |
| `licenseId` | yes | an SPDX identifier or expression (`Apache-2.0`, `MIT OR Apache-2.0`, `LicenseRef-<name>` for your own terms). It is the licence of the URDF and meshes you push, so it must be one you may grant |
| `maker` | no | who makes it. Left out for a generic product |
| `generic` | yes | `true` when it is not a particular maker's model |
| `class` | yes | `equipment`, `sensor`, `mover`, `substance` or `item` |
| `category` | for push | the catalog category (`arm`, `amr`, `figure`, ...) when not given with `--category` |
| `<class>` | per class | the class's own block, under its name (`"mover": {"robotId": "...", "speedMps": 0.5}`) |
| `arm` | no | `null` for a robot without an arm; otherwise `{"mount": ..., "hand": ...}` (below) |

`arm.mount` is what a hand bolts onto, `{"link": "tool0", "standard": "ISO 9409-1-50-4-M6"}`, or `null`
when the hand is part of the robot. `arm.hand` is the hand it comes with: `name`, `grasp` (or `null` with
`graspMissing` saying why) and `actuation` (`{"kind": "gripper", "joints": {"<chain>": ["<joint>"]},
"open": .., "closed": ..}`).

## profile.json

How it moves. Every field is optional; keys starting with `_` are notes for the person editing the file
(where a figure came from) and are not carried into the bundle.

| field | what |
|---|---|
| `vendor` | shown with the machine (falls back to `product.json`'s `maker`) |
| `urdfSourceUrl` | where the URDF came from |
| `totalMassKg` | `init` sums the URDF's link masses |
| `locomotion` | `Static` (default), or how it moves over the ground |
| `hoverHeightM` | how high above the surface it rests (0 for anything standing on the ground) |
| `chains` | `[{"name", "baseLink", "tipLink", "jointNames"}]`: the joints that get handles, and the root and tip the IK works between. `init` guesses one from the root to the deepest link |
| `motion` | `maxJointVel`, `maxJointAcc` (one per chain joint), `maxLinearVel`, `maxLinearAcc`, `maxYawRate`, `maxYawAcc`, `footprint` `{lengthM, widthM}` |
| `environment` | `fordingDepthM` with `fordingRated` (whether the maker publishes that depth); both or neither |
| `wheels` | below |

`check` makes sure every chain's links and joints are in the URDF.

### wheels

For a machine that drives, which joints turn as it goes:

```json
"wheels": {
  "radius": 0.30,
  "halfTrack": 0.45,
  "joints": [
    {"name": "front_left_wheel", "spin": 1, "turn": -1, "radius": 0.25},
    {"name": "rear_left_wheel", "spin": 1}
  ]
}
```

`radius` is the drive wheels' radius and `halfTrack` half the distance between the left and right wheels, in
metres. Each joint names a URDF joint, which `check` requires to be `continuous`; `spin` is +1 or -1 (which
way the joint turns when the machine goes forward), `turn` the same for a steered wheel, and a joint's own
`radius` overrides the common one (a smaller front wheel).

## Measuring grasp

`arm.hand.grasp` tells a simulation how the hand holds something. The SO-101's values come from a trial on
2026-10-14 (S0, in MuJoCo 3.10, the cube lifted and held), and they are measured, not guessed:

1. Put the arm in the pose it grasps from, with the jaws closed on the object (for the SO-101, a 2.5 cm cube
   at gripper value 0.10).
2. From each jaw's mesh vertices at that pose, find the face that touches the object. Put a thin box there,
   1.5 mm thick, as a **pad**: `{"link", "pos", "quatWxyz", "half"}` in that jaw link's own frame, its z axis
   the face normal pointing at the object. A jaw's convex-hull collision shape is fat and tilted, and a
   clamped object squirts out; a flat pad holds it.
3. `tcp` `{"link", "pos"}` is the point between the pads; `approachLocal` the direction the hand comes in
   along, in that link's frame; `aimShiftM` the small offset from the object's centre to where the tool point
   should aim.
4. `gripperJoint`, `open`, `closed`: the joint and its values; `padFriction`; and `gains` `{kp, kd,
   effortNm}` per joint, tuned until the object lifts and stays.

Write where each figure came from in `grasp._source`. When you have not measured it, set `grasp` to `null`
and say why in `graspMissing`.

## object.json

```json
{"name": "statue", "display": {"ja": "像", "en": "Statue"}, "license": "CC-BY-4.0",
 "category": "figure", "unit": "mm"}
```

| field | what |
|---|---|
| `name` | the catalog name: `a-z 0-9 _ -`, two characters or more |
| `display` | `{ja, en}` |
| `license` | SPDX, as for machines |
| `category` | `figure` by default |
| `unit` | the mesh file's unit, `m` or `mm`. A mesh carries no unit, and a model exported from CAD or a 3D scan in millimetres is drawn 1000 times too big if this is wrong. Required; there is no default |

The mesh can be GLB, STL, OBJ or 3MF. When it is pushed, psbx wraps it in a one-link URDF
(`physical_sandbox.machine_tools.wrap_object(mesh, unit="mm", out_dir=...)`, `scale="0.001 0.001 0.001"`
for millimetres) and makes the same display bundle a machine gets.
