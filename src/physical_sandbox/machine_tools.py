"""``psbx machine check`` / ``psbx machine init``, and a mesh-only object wrapped as a one-link URDF.

A machine folder holds the URDF (or MJCF), its meshes, ``product.json`` (what the catalog shows: names, licence,
class) and optionally ``profile.json`` (vendor, mass, locomotion, motion, chains, wheels). An object folder holds
one mesh file and ``object.json``. ``check`` says, line by line, whether such a folder can be pushed; ``init``
writes the templates, with every field a person has to fill in set to ``"<fill>"``. See docs/machines.md.
"""
from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .errors import InputError

FILL = "<fill>"
MESH_SUFFIXES = (".glb", ".gltf", ".stl", ".obj", ".3mf", ".dae")
OBJECT_SUFFIXES = (".glb", ".stl", ".obj", ".3mf")
CLASSES = ("equipment", "sensor", "mover", "substance", "item")
UNITS = {"m": 1.0, "mm": 0.001}

# the SPDX identifiers machines and models are usually published under; LicenseRef-* is accepted too
SPDX = {
    "0BSD", "AFL-3.0", "AGPL-3.0-only", "AGPL-3.0-or-later", "Apache-1.1", "Apache-2.0", "Artistic-2.0",
    "BSD-1-Clause", "BSD-2-Clause", "BSD-3-Clause", "BSD-3-Clause-Clear", "BSD-4-Clause", "BSL-1.0",
    "CC0-1.0", "CC-BY-3.0", "CC-BY-4.0", "CC-BY-NC-4.0", "CC-BY-NC-ND-4.0", "CC-BY-NC-SA-4.0", "CC-BY-ND-4.0",
    "CC-BY-SA-3.0", "CC-BY-SA-4.0", "CDDL-1.0", "CERN-OHL-P-2.0", "CERN-OHL-S-2.0", "CERN-OHL-W-2.0",
    "EPL-1.0", "EPL-2.0", "EUPL-1.2", "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
    "ISC", "LGPL-2.1-only", "LGPL-2.1-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later", "MIT", "MIT-0",
    "MPL-2.0", "MS-PL", "NCSA", "ODbL-1.0", "OFL-1.1", "PDDL-1.0", "Python-2.0", "TAPR-OHL-1.0", "Unlicense",
    "UPL-1.0", "W3C", "WTFPL", "Zlib", "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0", "AGPL-3.0",
}
_SPDX_LOWER = {s.lower() for s in SPDX}


@dataclass
class CheckResult:
    items: list[tuple[bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.items) and all(ok for ok, _ in self.items)

    def add(self, ok: bool, text: str) -> None:
        self.items.append((bool(ok), text))


def is_spdx(expression) -> bool:
    """An SPDX identifier or expression (``MIT OR Apache-2.0``, ``GPL-2.0-only WITH Classpath-exception-2.0``)."""
    if not isinstance(expression, str) or not expression.strip():
        return False
    tokens = re.split(r"\s+(?:OR|AND)\s+", expression.strip().strip("()"))
    for token in tokens:
        token = token.strip("() ").split(" WITH ")[0].strip().rstrip("+")
        if not (token.lower() in _SPDX_LOWER or re.fullmatch(r"LicenseRef-[A-Za-z0-9.\-]+", token)):
            return False
    return True


def _is_mjcf(path: Path) -> bool:
    try:
        return ET.parse(path).getroot().tag == "mujoco"
    except (ET.ParseError, OSError):
        return False


def _find_model(folder: Path) -> Path | None:
    """The folder's URDF (the one named after the folder when there are several), else its MJCF."""
    urdfs = sorted(folder.glob("*.urdf"))
    if urdfs:
        named = [u for u in urdfs if u.stem == folder.name]
        return (named or urdfs)[0]
    return next((x for x in sorted(folder.glob("*.xml")) if _is_mjcf(x)), None)


def _meshes_in(folder: Path, suffixes=OBJECT_SUFFIXES) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def _read_json(path: Path, result: CheckResult):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        result.add(False, f"{path.name}: does not read as JSON ({e})")
        return None
    if not isinstance(data, dict):
        result.add(False, f"{path.name}: is not a JSON object")
        return None
    return data


def _fills(data, prefix="") -> list[str]:
    """The keys (dotted) still set to "<fill>"."""
    out = []
    if isinstance(data, dict):
        for k, v in data.items():
            out += _fills(v, f"{prefix}{k}.")
    elif isinstance(data, list):
        for i, v in enumerate(data):
            out += _fills(v, f"{prefix}{i}.")
    elif data == FILL:
        out.append(prefix[:-1])
    return out


def _given(value) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value != FILL


# ---- check ----

def check(path: Path) -> CheckResult:
    """Whether a machine folder (or an object folder / mesh file) can be pushed, as (ok, text) lines."""
    path = Path(path).expanduser()
    result = CheckResult()
    if path.is_file() and path.suffix.lower() in MESH_SUFFIXES:
        return _check_object(path.parent, path, result)
    if path.is_file():
        folder, model = path.parent, path
    elif path.is_dir():
        folder, model = path, _find_model(path)
    else:
        result.add(False, f"{path}: no such file or folder")
        return result
    if model is None:
        if (folder / "object.json").is_file() or len(_meshes_in(folder)) == 1:
            return _check_object(folder, None, result)
        result.add(False, f"{folder}: no URDF (*.urdf) or MJCF here, and no single mesh to be an object")
        return result
    if model.suffix.lower() == ".urdf":
        _check_urdf(model, result)
    else:
        _check_mjcf(model, result)
    _check_product(folder, result)
    _check_profile(folder, model, result)
    return result


def _check_urdf(urdf: Path, result: CheckResult) -> None:
    from .bundle import _trimesh
    from .errors import OutsideFolder
    from .machine import _find_mesh, load_machine

    result.add(True, f"URDF: {urdf.name}")
    try:
        root = ET.parse(urdf).getroot()
    except ET.ParseError as e:
        result.add(False, f"{urdf.name}: not XML ({e})")
        return
    refs = list(dict.fromkeys(m.get("filename") for m in root.iter("mesh") if m.get("filename")))
    missing = 0
    for ref in refs:
        try:
            found = _find_mesh(urdf.parent, ref)
        except OutsideFolder as e:
            missing += 1
            result.add(False, str(e))
            continue
        except FileNotFoundError:
            missing += 1
            result.add(False, f"mesh {ref!r} not found next to {urdf.name} (nor under meshes/)")
            continue
        if found.suffix.lower() in (".glb", ".gltf", ".3mf") and _trimesh() is None:
            result.add(False, f"mesh {ref!r}: {found.suffix} needs trimesh to be drawn (pip install trimesh)")
    if not missing:
        result.add(True, f"meshes: all {len(refs)} found")
        try:
            m = load_machine(urdf)
            result.add(True, f"reads in MuJoCo: {len(m.joint_names)} moving joints ({', '.join(m.joint_names)})")
        except Exception as e:  # noqa: BLE001 - whatever MuJoCo says is the reason it does not read
            result.add(False, f"does not read in MuJoCo: {' '.join(str(e).split())}")


def _check_mjcf(mjcf: Path, result: CheckResult) -> None:
    import mujoco

    result.add(True, f"MJCF: {mjcf.name}")
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf))
        result.add(True, f"reads in MuJoCo: {model.njnt} joints, {model.nmesh} meshes")
    except Exception as e:  # noqa: BLE001
        result.add(False, f"does not read in MuJoCo: {' '.join(str(e).split())}")
    result.add(False, "the display bundle is made from a URDF; an MJCF is not converted yet (export a URDF)")


def _check_product(folder: Path, result: CheckResult) -> None:
    path = folder / "product.json"
    if not path.is_file():
        result.add(False, "no product.json (psbx machine init writes a template)")
        return
    product = _read_json(path, result)
    if product is None:
        return
    names = product.get("displayName") if isinstance(product.get("displayName"), dict) else {}
    wanted = {"displayName.ja": names.get("ja"), "displayName.en": names.get("en"),
              "licenseId": product.get("licenseId"), "class": product.get("class")}
    unfilled = [k for k, v in wanted.items() if not _given(v)]
    unfilled += [k for k in _fills(product) if k not in unfilled]
    if unfilled:
        result.add(False, f"product.json: fill in {', '.join(unfilled)}")
    else:
        result.add(True, f"product.json: {names['en']} / {names['ja']}, class {product['class']}")
    if _given(product.get("class")) and product["class"] not in CLASSES:
        result.add(False, f"product.json: class {product['class']!r} is not one of {', '.join(CLASSES)}")
    if _given(product.get("licenseId")):
        if is_spdx(product["licenseId"]):
            result.add(True, f"licence: {product['licenseId']} (SPDX)")
        else:
            result.add(False, f"product.json: licenseId {product['licenseId']!r} is not an SPDX identifier "
                              f"(https://spdx.org/licenses/; LicenseRef-<name> for your own)")


def _check_profile(folder: Path, model: Path, result: CheckResult) -> None:
    path = folder / "profile.json"
    if not path.is_file():
        result.add(True, "no profile.json (optional: vendor, mass, motion, chains, wheels)")
        return
    profile = _read_json(path, result)
    if profile is None:
        return
    joints, links = {}, set()
    if model.suffix.lower() == ".urdf":
        try:
            root = ET.parse(model).getroot()
            joints = {j.get("name"): j.get("type") for j in root.findall("joint")}
            links = {link.get("name") for link in root.findall("link")}
        except ET.ParseError:
            pass
    bad = [f"fill in {k}" for k in _fills(profile)]
    wheels = profile.get("wheels")
    if isinstance(wheels, dict):
        for j in wheels.get("joints") or []:
            jname = j.get("name") if isinstance(j, dict) else None
            if jname not in joints:
                bad.append(f"wheel joint {jname!r} is not a joint of {model.name}")
            elif joints[jname] != "continuous":
                bad.append(f"wheel joint {jname!r} is {joints[jname]} in {model.name}, not continuous")
    for chain in profile.get("chains") or []:
        if not isinstance(chain, dict):
            continue
        for key in ("baseLink", "tipLink"):
            if links and chain.get(key) not in links:
                bad.append(f"chain {chain.get('name')!r}: {key} {chain.get(key)!r} is not a link")
        for jname in chain.get("jointNames") or []:
            if joints and jname not in joints:
                bad.append(f"chain {chain.get('name')!r}: joint {jname!r} is not a joint")
    for b in bad:
        result.add(False, f"profile.json: {b}")
    if not bad:
        result.add(True, "profile.json: " + ("wheels are continuous joints" if isinstance(wheels, dict)
                                             else "reads, its joints and links are in the model"))


def _check_object(folder: Path, mesh: Path | None, result: CheckResult) -> CheckResult:
    meshes = [mesh] if mesh is not None else _meshes_in(folder)
    if len(meshes) != 1:
        result.add(False, f"an object folder holds one mesh ({', '.join(OBJECT_SUFFIXES)}), found {len(meshes)}")
    else:
        result.add(True, f"object mesh: {meshes[0].name}")
    path = folder / "object.json"
    if not path.is_file():
        result.add(False, "no object.json (psbx machine init writes a template)")
        return result
    obj = _read_json(path, result)
    if obj is None:
        return result
    display = obj.get("display") if isinstance(obj.get("display"), dict) else {}
    wanted = {"name": obj.get("name"), "display.ja": display.get("ja"), "display.en": display.get("en"),
              "license": obj.get("license"), "category": obj.get("category")}
    unfilled = [k for k, v in wanted.items() if not _given(v)]
    if unfilled:
        result.add(False, f"object.json: fill in {', '.join(unfilled)}")
    if obj.get("unit") not in UNITS:
        result.add(False, f"object.json: unit must be \"m\" or \"mm\" (the mesh's unit), not {obj.get('unit')!r}")
    if _given(obj.get("license")) and not is_spdx(obj["license"]):
        result.add(False, f"object.json: license {obj['license']!r} is not an SPDX identifier")
    if not unfilled and obj.get("unit") in UNITS and is_spdx(obj.get("license")):
        result.add(True, f"object.json: {display['en']}, {obj['license']}, in {obj['unit']}")
    return result


# ---- init ----

def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-_")
    return s if len(s) >= 2 else (s + "-object" if s else "object")


def _write_new(path: Path, data: dict, written: list[Path]) -> None:
    if path.exists():
        return  # never overwrite what a person filled in
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written.append(path)


def _urdf_guess(urdf: Path) -> tuple[float, list]:
    """The URDF's mass summed, and a chain from the root to the deepest link through its moving joints."""
    root = ET.parse(urdf).getroot()
    mass = 0.0
    for m in root.iter("mass"):
        try:
            mass += float(m.get("value") or 0)
        except ValueError:
            pass
    joints = root.findall("joint")
    child_of = {}
    for j in joints:
        p, c = j.find("parent"), j.find("child")
        if p is not None and c is not None:
            child_of[c.get("link")] = (p.get("link"), j)
    links = [link.get("name") for link in root.findall("link")]
    parents = {p for p, _ in child_of.values()}
    roots = [n for n in links if n not in child_of and n in parents]
    if len(roots) != 1:
        return round(mass, 6), []

    def path_to(link):
        out = []
        while link in child_of:
            parent, j = child_of[link]
            out.append(j)
            link = parent
        return out[::-1]

    tip = max(links, key=lambda n: (len(path_to(n)), -links.index(n)))
    moving = [j.get("name") for j in path_to(tip) if j.get("type") in ("revolute", "prismatic", "continuous")]
    chains = [{"name": "arm", "baseLink": roots[0], "tipLink": tip, "jointNames": moving}] if moving else []
    return round(mass, 6), chains


def init(path: Path) -> list[Path]:
    """Write the templates a folder lacks; returns the files written (none are overwritten).

    A URDF/MJCF folder gets ``product.json`` and ``profile.json``; a GLB/STL/OBJ/3MF file (or a folder holding
    just one) gets ``object.json`` beside it.
    """
    path = Path(path).expanduser()
    written: list[Path] = []
    if path.is_file() and path.suffix.lower() in OBJECT_SUFFIXES:
        _write_new(path.parent / "object.json", _object_template(path), written)
        return written
    if path.is_file():
        folder, model = path.parent, path
    elif path.is_dir():
        folder, model = path, _find_model(path)
    else:
        raise InputError(f"{path}: no such file or folder")
    if model is None:
        meshes = _meshes_in(folder)
        if len(meshes) == 1:
            _write_new(folder / "object.json", _object_template(meshes[0]), written)
            return written
        raise InputError(f"{folder}: no URDF or MJCF here, and not one single mesh ({', '.join(OBJECT_SUFFIXES)}) "
                         f"to make an object of")
    mass, chains = _urdf_guess(model) if model.suffix.lower() == ".urdf" else (0.0, [])
    _write_new(folder / "product.json", {
        "id": _slug(folder.name),
        "displayName": {"ja": FILL, "en": FILL},
        "licenseId": FILL,
        "maker": FILL,
        "generic": False,
        "class": FILL,
        "category": FILL,
    }, written)
    _write_new(folder / "profile.json", {
        "vendor": "",
        "urdfSourceUrl": "",
        "totalMassKg": mass,
        "locomotion": "Static",
        "chains": chains,
        "motion": {},
    }, written)
    return written


def _object_template(mesh: Path) -> dict:
    return {"name": _slug(mesh.stem), "display": {"ja": FILL, "en": FILL}, "license": FILL,
            "category": "figure", "unit": "<m|mm>"}


# ---- objects as one link ----

def _copy_obj_material(src: Path, out_dir: Path) -> None:
    """An OBJ's mtllib and the images its map_Kd lines name, copied beside the copy (same relative names)."""
    for line in src.read_text(errors="replace").splitlines():
        if line.startswith("mtllib "):
            mtl = src.parent / line.split(None, 1)[1].strip()
            if mtl.is_file():
                (out_dir / mtl.name).write_bytes(mtl.read_bytes())
                for ml in mtl.read_text(errors="replace").splitlines():
                    if ml.strip().startswith("map_Kd "):
                        image = mtl.parent / ml.strip().split(None, 1)[1].strip()
                        if image.is_file():
                            dest = out_dir / image.relative_to(mtl.parent)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(image, dest)
            break


def wrap_object(mesh_path: Path, *, unit: str, out_dir: Path) -> Path:
    """Write a one-link URDF drawing ``mesh_path`` (copied into ``out_dir``); ``unit`` "mm" scales it by 0.001."""
    if unit not in UNITS:
        raise InputError(f"unit must be \"m\" or \"mm\" (the mesh's unit), not {unit!r}")
    mesh_path = Path(mesh_path).expanduser().resolve()
    if not mesh_path.is_file():
        raise InputError(f"{mesh_path}: no such mesh file")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / mesh_path.name
    if dest.resolve() != mesh_path:
        shutil.copyfile(mesh_path, dest)
        if mesh_path.suffix.lower() == ".obj":
            _copy_obj_material(mesh_path, out_dir)
    name = _slug(mesh_path.stem)
    mesh = ET.Element("mesh", filename=mesh_path.name)
    if unit == "mm":
        mesh.set("scale", "0.001 0.001 0.001")
    robot = ET.Element("robot", name=name)
    link = ET.SubElement(robot, "link", name="base_link")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="1")
    ET.SubElement(inertial, "inertia", ixx="0.01", iyy="0.01", izz="0.01", ixy="0", ixz="0", iyz="0")
    visual = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", mesh.attrib)
    ET.indent(robot)
    urdf = out_dir / f"{name}.urdf"
    urdf.write_text('<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n", encoding="utf-8")
    return urdf
