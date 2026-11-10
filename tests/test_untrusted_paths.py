"""A machine folder from someone else may name files anywhere: every file it leads to is read only from inside it.

A URDF names meshes, an OBJ names a material library, the library names an image, a glTF names buffers and images.
Each is followed only while it stays inside the machine's folder (after links are resolved), so a crafted folder
cannot make ``psbx`` read, cache or upload a file the person never put there (``psbx push`` uploads the bundle).
"""
import json
import struct
from pathlib import Path

import pytest

from physical_sandbox.bundle import make_bundle, read_mesh
from physical_sandbox.errors import InputError, OutsideFolder
from physical_sandbox.machine import load_machine
from physical_sandbox.machine_tools import check

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 24
SECRET = '{"token": "not to be uploaded"}'
TRI = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n"


def urdf(folder: Path, mesh_ref: str) -> Path:
    path = folder / "one.urdf"
    path.write_text(f"""<?xml version="1.0"?>
<robot name="one"><link name="body"><visual><geometry><mesh filename="{mesh_ref}"/></geometry></visual></link></robot>
""")
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A machine folder ``robot/`` and, beside it, a file it must never reach (``secret.json``)."""
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    robot = tmp_path / "robot"
    robot.mkdir()
    (tmp_path / "secret.json").write_text(SECRET)
    return robot


def textured_obj(robot: Path, mtllib: str, map_kd: str, obj="part.obj") -> None:
    (robot / obj).parent.mkdir(parents=True, exist_ok=True)
    (robot / obj).write_text(f"mtllib {mtllib}\n{TRI}usemtl m\nf 1/1 2/2 3/3\n")
    (robot / obj).parent.joinpath("part.mtl").write_text(f"newmtl m\nmap_Kd {map_kd}\n")


def bundle(robot: Path, mesh_ref="part.obj", tmp=None):
    return make_bundle(urdf(robot, mesh_ref), product={}, profile=None, name="one", out_dir=robot.parent / "out")


def nothing_secret_written(robot: Path) -> bool:
    out = robot.parent / "out"
    return not out.exists() or all(SECRET not in p.read_text(errors="replace") for p in out.rglob("*") if p.is_file())


# ---- OBJ -> MTL -> image ----

@pytest.mark.parametrize("map_kd", ["{abs}", "../secret.json", "../robot/../secret.json"])
def test_a_material_image_outside_the_folder_is_refused_and_never_copied(world, map_kd):
    textured_obj(world, "part.mtl", map_kd.format(abs=world.parent / "secret.json"))
    with pytest.raises(OutsideFolder, match="outside"):
        bundle(world)
    assert nothing_secret_written(world)


def test_a_material_library_outside_the_folder_is_refused(world):
    (world.parent / "evil.mtl").write_text("newmtl m\nmap_Kd secret.json\n")
    (world / "part.obj").write_text(f"mtllib ../evil.mtl\n{TRI}f 1/1 2/2 3/3\n")
    with pytest.raises(OutsideFolder, match="evil.mtl"):
        bundle(world)
    assert nothing_secret_written(world)


def test_a_link_inside_the_folder_that_leads_outside_is_refused(world):
    (world / "skin.png").symlink_to(world.parent / "secret.json")
    textured_obj(world, "part.mtl", "skin.png")
    with pytest.raises(OutsideFolder):
        bundle(world)
    assert nothing_secret_written(world)


def test_a_material_image_that_is_not_a_png_or_jpeg_is_refused(world):
    (world / "skin.png").write_text(SECRET)
    textured_obj(world, "part.mtl", "skin.png")
    with pytest.raises(InputError, match="not a PNG or JPEG"):
        bundle(world)
    assert nothing_secret_written(world)


def test_images_in_a_subfolder_or_beside_the_meshes_folder_still_come_along(world):
    (world / "textures").mkdir()
    (world / "textures" / "skin.png").write_bytes(PNG)
    textured_obj(world, "part.mtl", "../textures/skin.png", obj="meshes/part.obj")
    result = bundle(world, "meshes/part.obj")
    assert [p.name for p in result.mesh_paths if p.suffix == ".png"] == ["skin.png"]


def test_reading_an_obj_alone_keeps_its_references_to_its_own_folder(world):
    textured_obj(world, "part.mtl", "../secret.json")
    with pytest.raises(OutsideFolder):
        read_mesh(world / "part.obj")


# ---- glTF -> buffers and images ----

@pytest.fixture
def loader(monkeypatch):
    """trimesh as the bundle sees it: loading anything fails the test, so a refusal proves it came first."""
    import physical_sandbox.bundle as bundle_module

    class Never:
        @staticmethod
        def load(*a, **k):
            raise AssertionError("the glTF was loaded before its references were checked")
    monkeypatch.setattr(bundle_module, "_trimesh", lambda: Never)

def gltf(uri: str) -> dict:
    return {"asset": {"version": "2.0"}, "buffers": [{"uri": uri, "byteLength": 36}], "images": [], "meshes": []}


def test_a_gltf_buffer_outside_the_folder_is_refused_before_it_is_read(world, loader):
    (world / "part.gltf").write_text(json.dumps(gltf("../secret.json")))
    with pytest.raises(OutsideFolder, match="secret.json"):
        bundle(world, "part.gltf")


def test_a_glb_image_outside_the_folder_is_refused(world, loader):
    doc = json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": str(world.parent / "secret.json")}]}).encode()
    doc += b" " * (-len(doc) % 4)
    blob = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(doc)) + struct.pack("<I", len(doc)) + b"JSON" + doc
    (world / "part.glb").write_bytes(blob)
    with pytest.raises(OutsideFolder):
        bundle(world, "part.glb")


def test_embedded_and_inside_gltf_references_are_left_alone(world):
    from physical_sandbox.bundle import _gltf_references_inside

    (world / "part.bin").write_bytes(b"\0" * 36)
    doc = gltf("part.bin")
    doc["images"] = [{"uri": "data:image/png;base64,iVBORw0KGgo="}, {"bufferView": 0}]
    (world / "part.gltf").write_text(json.dumps(doc))
    _gltf_references_inside(world / "part.gltf", world)  # no error


# ---- URDF -> mesh ----

@pytest.mark.parametrize("ref", ["{abs}", "../outside.stl", "package://pkg/../outside.stl"])
def test_a_urdf_mesh_outside_the_folder_is_not_loaded_bundled_or_passed_by_the_check(world, ref):
    (world.parent / "outside.stl").write_text("solid x\nendsolid x\n")
    path = urdf(world, ref.format(abs=world.parent / "outside.stl"))
    with pytest.raises(OutsideFolder, match="outside"):
        load_machine(path)
    with pytest.raises(OutsideFolder):
        make_bundle(path, product={}, profile=None, name="one", out_dir=world.parent / "out")
    result = check(world)
    assert not result.ok
    assert any(not ok and "outside" in line for ok, line in result.items)


def test_a_urdf_mesh_link_that_leads_outside_is_refused(world):
    (world.parent / "outside.obj").write_text(f"{TRI}f 1 2 3\n")
    (world / "part.obj").symlink_to(world.parent / "outside.obj")
    with pytest.raises(OutsideFolder):
        load_machine(urdf(world, "part.obj"))
