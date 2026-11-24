"""Build the visual assets of the P4 "real-looking" pick-place scene (run with isaacsim-env: pxr + trimesh + PIL).

    ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\make_real_scene_assets.py --wood_photo <photo of the real table>

Writes, under assets/props/:
  - camera_mount_v2.usd : the printed wrist camera mount (isaac/Pièces/CameraMount_v2.stl, mm -> m, its own frame),
    off-white PETG, VISUAL ONLY (no collision, no rigid body: the arm's physics is unchanged), plus a simplified
    camera (black board 25 x 24 x 1.6 mm, Waveshare outline; black lens barrel d 14 mm) so the front camera sees it;
  - table_wood.usd + table_wood.png : a 0.80 x 0.70 m textured quad laid on the table top (visual only), texture =
    crop of the user's photo of the real wooden table, mirrored so it tiles without seams.
Colours measured on the user's photos (2026-11-24) are printed for the scene config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--stl", default=str(REPO.parent / "Pièces" / "CameraMount_v2.stl"))
ap.add_argument("--wood_photo", required=True)
ap.add_argument("--wood_crop", type=int, nargs=4, default=[1010, 40, 1490, 1240], metavar=("X0", "Y0", "X1", "Y1"))
ap.add_argument("--wood_tile_m", type=float, nargs=2, default=[0.09, 0.22],
                help="physical size of one mirrored tile (u, v); ESTIMATE from the photo (~10 px/mm)")
ap.add_argument("--out", default=str(REPO / "assets" / "props"))
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

LENS_TOP_Z = 0.011  # just behind the wrist camera's nominal optical centre (wrist_mount.LENS_CENTRE_Z = 0.012)


def srgb_to_linear(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def preview_material(stage, path, rgb, roughness=0.5, texture=None):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if texture is None:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
    else:
        st = UsdShade.Shader.Define(stage, f"{path}/uv")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        # brightness / tint, randomised per episode by cta_learn.sim.mdp.randomize_texture_scale
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 1, 1, 1))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def add_mesh(stage, path, tm: trimesh.Trimesh, material):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(tm.vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tm.faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tm.faces.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


# ---------------------------------------------------------------- camera mount (+ simplified camera)
stage = Usd.Stage.CreateNew(str(out / "camera_mount_v2.usd"))
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
root = UsdGeom.Xform.Define(stage, "/CameraMount")
stage.SetDefaultPrim(root.GetPrim())
petg = preview_material(stage, "/CameraMount/Looks/PETG", srgb_to_linear([235, 235, 228]), roughness=0.45)
black = preview_material(stage, "/CameraMount/Looks/Black", srgb_to_linear([20, 20, 22]), roughness=0.35)
m = trimesh.load(args.stl)
m.apply_scale(0.001)
add_mesh(stage, "/CameraMount/plate", m, petg)
board = trimesh.creation.box(extents=(0.025, 0.024, 0.0016))
board.apply_translation((0.0, 0.00375, -0.0008))  # behind the flat face, centred on the lens opening
add_mesh(stage, "/CameraMount/camera_board", board, black)
lens = trimesh.creation.cylinder(radius=0.007, height=LENS_TOP_Z, sections=48)
lens.apply_translation((0.0, 0.00375, LENS_TOP_Z / 2))
add_mesh(stage, "/CameraMount/camera_lens", lens, black)
stage.GetRootLayer().Save()
print("wrote", out / "camera_mount_v2.usd", "plate faces", len(m.faces))

# ---------------------------------------------------------------- wooden table top
img = Image.open(args.wood_photo).convert("RGB")
crop = np.asarray(img.crop(tuple(args.wood_crop)))
tile = np.concatenate([crop, crop[:, ::-1]], axis=1)
tile = np.concatenate([tile, tile[::-1]], axis=0)  # mirrored 2 x 2: seamless when repeated
tile_img = Image.fromarray(tile).resize((512, int(512 * tile.shape[0] / tile.shape[1])), Image.LANCZOS)
tile_img.save(out / "table_wood.png")
sx, sy = 0.80, 0.70
stage = Usd.Stage.CreateNew(str(out / "table_wood.usd"))
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
root = UsdGeom.Xform.Define(stage, "/TableWood")
stage.SetDefaultPrim(root.GetPrim())
wood = preview_material(stage, "/TableWood/Looks/Wood", None, roughness=0.6, texture="./table_wood.png")
quad = UsdGeom.Mesh.Define(stage, "/TableWood/top")
quad.CreatePointsAttr([(-sx / 2, -sy / 2, 0), (sx / 2, -sy / 2, 0), (sx / 2, sy / 2, 0), (-sx / 2, sy / 2, 0)])
quad.CreateFaceVertexCountsAttr([4])
quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
quad.CreateNormalsAttr([(0, 0, 1)] * 4)
quad.CreateSubdivisionSchemeAttr("none")
# grain of the photo runs along the image width (u) -> laid along the robot's Y (left-right in the front camera)
tu, tv = args.wood_tile_m
st = [(-sy / 2 / tu, -sx / 2 / tv), (-sy / 2 / tu, sx / 2 / tv), (sy / 2 / tu, sx / 2 / tv), (sy / 2 / tu, -sx / 2 / tv)]
pv = UsdGeom.PrimvarsAPI(quad).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
pv.Set(st)
UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(wood)
stage.GetRootLayer().Save()
print("wrote", out / "table_wood.usd", "texture", tile_img.size)

w = crop.reshape(-1, 3)
print("wood median sRGB", np.median(w, 0), "-> linear", srgb_to_linear(np.median(w, 0)).round(3))
