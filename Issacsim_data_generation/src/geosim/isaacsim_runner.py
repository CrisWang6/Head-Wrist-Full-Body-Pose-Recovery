from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a geosim motion cache in Isaac Sim.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--motion-label", default="motion")
    parser.add_argument("--renderer", default="RayTracedLighting", choices=("RayTracedLighting", "PathTracing"))
    parser.add_argument("--rt-subframes", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=12)
    parser.add_argument("--video-format", default="mp4", choices=("mp4", "avi"))
    parser.add_argument("--hide-wrist-tags", action="store_true", help="Do not render wrist tag meshes into RGB outputs.")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug override. 0 renders every cached frame.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(cache_path, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int32)
    camera_names = [str(name) for name in data["camera_names"]]
    camera_positions = np.asarray(data["camera_positions"], dtype=np.float32)
    camera_rotations = np.asarray(data["camera_rotations"], dtype=np.float32)
    tag_names = [str(name) for name in data["tag_names"]]
    tag_corners = np.asarray(data["tag_corners"], dtype=np.float32)
    body_face_groups = np.asarray(data["body_face_groups"], dtype=np.int32) if "body_face_groups" in data.files else np.zeros(len(faces), dtype=np.int32)
    body_group_colors = np.asarray(data["body_group_colors"], dtype=np.float32) if "body_group_colors" in data.files else np.asarray([[0.72, 0.58, 0.49]], dtype=np.float32)
    scene_config = _scene_config_from_cache(data)
    sensor_width, sensor_height = [int(value) for value in data["image_size"]]
    video_width, video_height = [int(value) for value in data["video_size"]] if "video_size" in data.files else (sensor_width, sensor_height)
    fisheye_fov_deg = float(np.asarray(data["fisheye_fov_deg"]).reshape(-1)[0]) if "fisheye_fov_deg" in data.files else 220.0
    fps = float(np.asarray(data["output_fps"]).reshape(-1)[0])

    if args.camera_name not in camera_names:
        raise ValueError(f"Unknown camera {args.camera_name!r}. Available cameras: {', '.join(camera_names)}")
    camera_idx = camera_names.index(args.camera_name)
    frame_count = int(vertices.shape[0])
    if args.max_frames > 0:
        frame_count = min(frame_count, int(args.max_frames))

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": int(sensor_width),
            "height": int(sensor_height),
            "renderer": args.renderer,
        }
    )

    try:
        import carb
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

        settings = carb.settings.get_settings()
        settings.set("/omni/replicator/captureOnPlay", False)
        settings.set("/omni/replicator/asyncRendering", False)
        settings.set("/app/asyncRendering", False)
        settings.set("/rtx/post/dlss/execMode", 2)
        if args.renderer == "PathTracing":
            settings.set("/rtx/pathtracing/spp", max(1, int(args.rt_subframes)))

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        _create_materials(stage, body_group_colors, scene_config)
        body_meshes = _create_body(stage, vertices[0], faces, body_face_groups, body_group_colors, Vt)
        tag_meshes = [] if args.hide_wrist_tags else _create_tags(stage, tag_names, tag_corners[:, 0], Vt)
        camera = _create_camera(stage, sensor_width, sensor_height, fisheye_fov_deg)
        _create_scene(stage, vertices, scene_config)

        render_product = rep.create.render_product(camera.GetPath(), (int(sensor_width), int(sensor_height)))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach(render_product)

        for _ in range(max(1, int(args.warmup_frames))):
            app.update()

        output_path = output_dir / f"{args.motion_label}_{args.camera_name}.{args.video_format}"
        writer = _open_video_writer(output_path, fps, video_width, video_height, args.video_format)
        timings: list[float] = []
        t_all = time.perf_counter()
        try:
            for frame_idx in range(frame_count):
                t0 = time.perf_counter()
                for mesh in body_meshes:
                    _set_mesh_points(mesh, vertices[frame_idx], Vt)
                for tag_idx, tag_mesh in enumerate(tag_meshes):
                    _set_mesh_points(tag_mesh, tag_corners[tag_idx, frame_idx], Vt)
                _set_xform_matrix(
                    camera,
                    _geosim_camera_to_usd_matrix(
                        camera_positions[camera_idx, frame_idx],
                        camera_rotations[camera_idx, frame_idx],
                        Gf,
                    ),
                )
                rep.orchestrator.step(rt_subframes=max(1, int(args.rt_subframes)))
                rgb = rgb_annotator.get_data()
                frame = _rgb_to_bgr_frame(rgb)
                frame = _center_crop(frame, video_width, video_height)
                writer.write(frame)
                timings.append(time.perf_counter() - t0)
                if (frame_idx + 1) % 30 == 0 or frame_idx + 1 == frame_count:
                    print(f"[IsaacSim] {args.camera_name} {frame_idx + 1}/{frame_count}", flush=True)
        finally:
            writer.release()

        total = time.perf_counter() - t_all
        stats = {
            "camera": args.camera_name,
            "output": str(output_path),
            "frames": int(frame_count),
            "fps": float(fps),
            "sensor_resolution": [int(sensor_width), int(sensor_height)],
            "video_resolution": [int(video_width), int(video_height)],
            "camera_model": "opencv_fisheye_equidistant",
            "fisheye_fov_deg": float(fisheye_fov_deg),
            "wrist_tags_rendered": not bool(args.hide_wrist_tags),
            "total_seconds": float(total),
            "seconds_per_frame_mean": float(np.mean(timings)) if timings else 0.0,
            "seconds_per_frame_median": float(np.median(timings)) if timings else 0.0,
        }
        stats_path = output_dir / f"{args.motion_label}_{args.camera_name}_isaacsim_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2), flush=True)
    finally:
        app.close()
    return 0


def _open_video_writer(path: Path, fps: float, width: int, height: int, video_format: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if video_format == "mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _rgb_to_bgr_frame(rgb: object) -> np.ndarray:
    frame = np.asarray(rgb)
    if frame.ndim == 1:
        raise RuntimeError(f"Unexpected flat RGB buffer shape: {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _center_crop(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    width = int(width)
    height = int(height)
    if src_w == width and src_h == height:
        return frame
    if width > src_w or height > src_h:
        raise ValueError(f"Crop {width}x{height} is larger than rendered frame {src_w}x{src_h}.")
    x0 = (src_w - width) // 2
    y0 = (src_h - height) // 2
    return frame[y0 : y0 + height, x0 : x0 + width]


def _scene_config_from_cache(data) -> dict[str, object]:
    def array_or_default(name: str, default):
        return np.asarray(data[name]).reshape(-1).tolist() if name in data.files else default

    floor_style = str(np.asarray(data["scene_floor_style"]).reshape(-1)[0]) if "scene_floor_style" in data.files else "concrete"
    return {
        "floor_style": floor_style,
        "floor_color": array_or_default("scene_floor_color", [0.50, 0.49, 0.45]),
        "floor_accent": array_or_default("scene_floor_accent", [0.22, 0.22, 0.22]),
        "sun_rotation": array_or_default("scene_sun_rotation", [42.0, 0.0, 28.0]),
        "sun_intensity": float(np.asarray(data["scene_sun_intensity"]).reshape(-1)[0]) if "scene_sun_intensity" in data.files else 2600.0,
    }


def _create_materials(stage, body_group_colors: np.ndarray, scene_config: dict[str, object]) -> None:
    names = ("Skin", "Top", "Bottom", "Shoes")
    for idx, name in enumerate(names):
        color_idx = min(idx, len(body_group_colors) - 1)
        _make_material(stage, f"/World/Materials/Body{name}", tuple(float(v) for v in body_group_colors[color_idx, :3]))
    _make_material(stage, "/World/Materials/Floor", tuple(float(v) for v in scene_config["floor_color"]))
    _make_material(stage, "/World/Materials/FloorAccent", tuple(float(v) for v in scene_config["floor_accent"]))
    _make_material(stage, "/World/Materials/Block", (0.18, 0.30, 0.36))
    _make_material(stage, "/World/Materials/TagWhite", (0.95, 0.95, 0.90))


def _make_material(stage, path: str, color: tuple[float, float, float]):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.62)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim, material_path: str) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material.Get(prim.GetStage(), material_path))


def _create_body(
    stage,
    vertices: np.ndarray,
    faces: np.ndarray,
    body_face_groups: np.ndarray,
    body_group_colors: np.ndarray,
    Vt,
) -> list[object]:
    from pxr import UsdGeom

    body_face_groups = np.asarray(body_face_groups, dtype=np.int32).reshape(-1)
    if len(body_face_groups) != len(faces):
        body_face_groups = np.zeros(len(faces), dtype=np.int32)
    material_names = ("Skin", "Top", "Bottom", "Shoes")
    meshes = []
    for group_idx, material_name in enumerate(material_names):
        group_faces = faces[body_face_groups == group_idx]
        if len(group_faces) == 0:
            continue
        mesh = UsdGeom.Mesh.Define(stage, f"/World/SMPLXBody_{material_name}")
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * int(len(group_faces))))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(group_faces, dtype=np.int32).reshape(-1).tolist()))
        _set_mesh_points(mesh, vertices, Vt)
        mesh.CreateSubdivisionSchemeAttr("none")
        _bind_material(mesh.GetPrim(), f"/World/Materials/Body{material_name}")
        meshes.append(mesh)
    if not meshes:
        mesh = UsdGeom.Mesh.Define(stage, "/World/SMPLXBody")
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * int(len(faces))))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(faces, dtype=np.int32).reshape(-1).tolist()))
        _set_mesh_points(mesh, vertices, Vt)
        mesh.CreateSubdivisionSchemeAttr("none")
        _bind_material(mesh.GetPrim(), "/World/Materials/BodySkin")
        meshes.append(mesh)
    return meshes


def _create_tags(stage, tag_names: list[str], first_corners: np.ndarray, Vt) -> list[object]:
    from pxr import UsdGeom

    meshes = []
    for tag_idx, tag_name in enumerate(tag_names):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Tags/{tag_name}")
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
        _set_mesh_points(mesh, first_corners[tag_idx], Vt)
        mesh.CreateSubdivisionSchemeAttr("none")
        _bind_material(mesh.GetPrim(), "/World/Materials/TagWhite")
        meshes.append(mesh)
    return meshes


def _set_mesh_points(mesh, points: np.ndarray, Vt) -> None:
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))


def _create_camera(stage, width: int, height: int, fisheye_fov_deg: float):
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, "/World/RenderCamera")
    camera.CreateClippingRangeAttr((0.01, 80.0))
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955 * float(height) / float(width))
    camera.CreateFocalLengthAttr(5.0)
    prim = camera.GetPrim()
    prim.ApplyAPI("OmniLensDistortionOpenCvFisheyeAPI")
    _set_or_create_attr(prim, "omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "opencvFisheye")
    max_theta = math.radians(float(fisheye_fov_deg)) * 0.5
    focal_px = min(float(width), float(height)) * 0.5 / max_theta
    _set_or_create_attr(prim, "omni:lensdistortion:opencvFisheye:imageSize", Sdf.ValueTypeNames.Int2, Gf.Vec2i(int(width), int(height)))
    _set_or_create_attr(prim, "omni:lensdistortion:opencvFisheye:cx", Sdf.ValueTypeNames.Float, float(width) * 0.5)
    _set_or_create_attr(prim, "omni:lensdistortion:opencvFisheye:cy", Sdf.ValueTypeNames.Float, float(height) * 0.5)
    _set_or_create_attr(prim, "omni:lensdistortion:opencvFisheye:fx", Sdf.ValueTypeNames.Float, float(focal_px))
    _set_or_create_attr(prim, "omni:lensdistortion:opencvFisheye:fy", Sdf.ValueTypeNames.Float, float(focal_px))
    for coeff_name in ("k1", "k2", "k3", "k4"):
        _set_or_create_attr(prim, f"omni:lensdistortion:opencvFisheye:{coeff_name}", Sdf.ValueTypeNames.Float, 0.0)
    return camera


def _set_or_create_attr(prim, name: str, value_type, value) -> None:
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, value_type, False)
    attr.Set(value)


def _create_scene(stage, vertices: np.ndarray, scene_config: dict[str, object]) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    bounds_min = vertices.reshape(-1, 3).min(axis=0)
    bounds_max = vertices.reshape(-1, 3).max(axis=0)
    center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
    floor_z = float(bounds_min[2]) - 0.015

    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), floor_z - 0.025))
    floor.AddScaleOp().Set(Gf.Vec3d(7.0, 7.0, 0.025))
    _bind_material(floor.GetPrim(), "/World/Materials/Floor")
    _create_floor_pattern(stage, center_xy, floor_z, scene_config)

    for idx, offset in enumerate(((-3.2, 2.8, 0.35), (3.0, 3.4, 0.5), (2.8, -3.0, 0.25))):
        block = UsdGeom.Cube.Define(stage, f"/World/DistantBlock_{idx}")
        block.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0] + offset[0]), float(center_xy[1] + offset[1]), floor_z + offset[2]))
        block.AddScaleOp().Set(Gf.Vec3d(0.6 + 0.175 * idx, 0.175, 0.25 + 0.125 * idx))
        _bind_material(block.GetPrim(), "/World/Materials/Block")

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(550.0)
    dome.CreateColorAttr((0.78, 0.82, 0.88))

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(scene_config["sun_intensity"]))
    sun.CreateAngleAttr(0.55)
    sun.AddRotateXYZOp().Set(tuple(float(v) for v in scene_config["sun_rotation"]))


def _create_floor_pattern(stage, center_xy: np.ndarray, floor_z: float, scene_config: dict[str, object]) -> None:
    from pxr import Gf, UsdGeom

    style = str(scene_config.get("floor_style", "concrete"))
    cx, cy = float(center_xy[0]), float(center_xy[1])
    if style == "tile":
        for idx, x in enumerate(np.linspace(-6.0, 6.0, 13)):
            line = UsdGeom.Cube.Define(stage, f"/World/FloorGridX_{idx}")
            line.AddTranslateOp().Set(Gf.Vec3d(cx + float(x), cy, floor_z + 0.002))
            line.AddScaleOp().Set(Gf.Vec3d(0.01, 6.0, 0.003))
            _bind_material(line.GetPrim(), "/World/Materials/FloorAccent")
        for idx, y in enumerate(np.linspace(-6.0, 6.0, 13)):
            line = UsdGeom.Cube.Define(stage, f"/World/FloorGridY_{idx}")
            line.AddTranslateOp().Set(Gf.Vec3d(cx, cy + float(y), floor_z + 0.003))
            line.AddScaleOp().Set(Gf.Vec3d(6.0, 0.01, 0.003))
            _bind_material(line.GetPrim(), "/World/Materials/FloorAccent")
    elif style == "carpet":
        for idx, y in enumerate(np.linspace(-5.5, 5.5, 9)):
            stripe = UsdGeom.Cube.Define(stage, f"/World/CarpetStripe_{idx}")
            stripe.AddTranslateOp().Set(Gf.Vec3d(cx, cy + float(y), floor_z + 0.003))
            stripe.AddScaleOp().Set(Gf.Vec3d(6.0, 0.035, 0.004))
            _bind_material(stripe.GetPrim(), "/World/Materials/FloorAccent")
    elif style == "grass":
        for idx, (x, y) in enumerate(_grass_offsets()):
            blade = UsdGeom.Cube.Define(stage, f"/World/GrassPatch_{idx}")
            blade.AddTranslateOp().Set(Gf.Vec3d(cx + x, cy + y, floor_z + 0.004))
            blade.AddScaleOp().Set(Gf.Vec3d(0.045, 0.012, 0.005))
            _bind_material(blade.GetPrim(), "/World/Materials/FloorAccent")


def _grass_offsets() -> list[tuple[float, float]]:
    rng = np.random.default_rng(17)
    return [(float(rng.uniform(-5.5, 5.5)), float(rng.uniform(-5.5, 5.5))) for _ in range(80)]


def _set_xform_matrix(xformable, matrix) -> None:
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(xformable)
    ops = xform.GetOrderedXformOps()
    if ops:
        ops[0].Set(matrix)
    else:
        xform.AddTransformOp().Set(matrix)


def _geosim_camera_to_usd_matrix(position: np.ndarray, rotation_cam_to_world: np.ndarray, Gf):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation_cam_to_world, dtype=float) @ np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return Gf.Matrix4d(matrix.T.tolist())


if __name__ == "__main__":
    raise SystemExit(main())
