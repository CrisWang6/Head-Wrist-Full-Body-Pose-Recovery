import blenderproc as bproc

import argparse
import math
from pathlib import Path
import shutil
import subprocess

import bpy
from mathutils import Matrix
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render geosim motion cache in BlenderProc.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-name", default="", help="Render only this camera. Empty renders every cached camera sequentially.")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--engine", default="CYCLES", choices=("CYCLES", "BLENDER_EEVEE_NEXT"))
    parser.add_argument("--device", default="GPU", choices=("GPU", "CPU"))
    parser.add_argument("--denoise", action="store_true", help="Enable Cycles denoising. Disabled by default for sharper synthetic camera data.")
    parser.add_argument("--video-format", default="mp4", choices=("mp4", "avi"))
    parser.add_argument("--video-crf", type=int, default=16, help="libx264 CRF for final MP4 crop. Lower is higher quality.")
    parser.add_argument("--motion-label", default="motion")
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
    scene_config = _load_scene_config(data)
    width, height = [int(value) for value in data["image_size"]]
    video_width, video_height = [int(value) for value in data["video_size"]] if "video_size" in data.files else (width, height)
    fps = float(np.asarray(data["output_fps"]).reshape(-1)[0])
    fisheye_fov = math.radians(float(np.asarray(data["fisheye_fov_deg"]).reshape(-1)[0]))
    camera_indices = _select_camera_indices(camera_names, args.camera_name)
    selected_camera_names = [camera_names[idx] for idx in camera_indices]
    selected_camera_positions = camera_positions[camera_indices]
    selected_camera_rotations = camera_rotations[camera_indices]

    bproc.init()
    _configure_render(
        width=width,
        height=height,
        fps=fps,
        engine=args.engine,
        samples=args.samples,
        device=args.device,
        denoise=args.denoise,
        video_format=args.video_format,
    )
    _create_open_scene(vertices, scene_config)
    body_obj = _create_body_mesh(vertices[0], faces, body_face_groups, body_group_colors)
    tag_objs = _create_tag_meshes(tag_names, tag_corners[:, 0], cache_path.parent)
    camera_objs = _create_cameras(selected_camera_names, fisheye_fov)

    def apply_frame(frame_number: int) -> None:
        frame_idx = int(np.clip(frame_number - 1, 0, vertices.shape[0] - 1))
        _update_mesh_vertices(body_obj.data, vertices[frame_idx])
        for tag_idx, tag_obj in enumerate(tag_objs):
            _update_mesh_vertices(tag_obj.data, tag_corners[tag_idx, frame_idx])
        for cam_idx, cam_obj in enumerate(camera_objs):
            cam_obj.matrix_world = Matrix(
                _geosim_camera_to_blender_matrix(
                    selected_camera_positions[cam_idx, frame_idx],
                    selected_camera_rotations[cam_idx, frame_idx],
                )
            )

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(lambda scene: apply_frame(scene.frame_current))
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = int(vertices.shape[0])
    apply_frame(1)

    rendered_paths = []
    for camera_name, camera_obj in zip(selected_camera_names, camera_objs):
        scene.camera = camera_obj
        output_path = output_dir / f"{args.motion_label}_{camera_name}.{args.video_format}"
        render_path = _render_path_for_output(output_path, width, height, video_width, video_height)
        scene.render.filepath = str(render_path)
        print(f"Rendering {camera_name} -> {output_path}", flush=True)
        bpy.ops.render.render(animation=True, write_still=False)
        if render_path != output_path:
            _center_crop_video(render_path, output_path, video_width, video_height, fps, video_crf=args.video_crf)
            render_path.unlink(missing_ok=True)
        rendered_paths.append(str(output_path))
    print("Rendered videos:")
    for path in rendered_paths:
        print(path)
    return 0


def _select_camera_indices(camera_names: list[str], camera_name: str) -> list[int]:
    if not camera_name:
        return list(range(len(camera_names)))
    if camera_name not in camera_names:
        raise ValueError(f"Unknown camera {camera_name!r}. Available cameras: {', '.join(camera_names)}")
    return [camera_names.index(camera_name)]


def _render_path_for_output(output_path: Path, width: int, height: int, video_width: int, video_height: int) -> Path:
    if width == video_width and height == video_height:
        return output_path
    return output_path.with_name(f"{output_path.stem}_sensor{width}x{height}{output_path.suffix}")


def _center_crop_video(input_path: Path, output_path: Path, width: int, height: int, fps: float, video_crf: int) -> None:
    import cv2

    if shutil.which("ffmpeg"):
        crop_expr = f"crop={int(width)}:{int(height)}:(in_w-{int(width)})/2:(in_h-{int(height)})/2"
        if output_path.suffix.lower() == ".mp4":
            codec_args = [
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(int(video_crf)),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        else:
            codec_args = ["-c:v", "mjpeg", "-q:v", "2"]
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            crop_expr,
            "-r",
            f"{float(fps):.6f}",
            *codec_args,
            "-an",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError:
            pass

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open rendered sensor video for cropping: {input_path}")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*("mp4v" if output_path.suffix.lower() == ".mp4" else "MJPG")),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open cropped video writer: {output_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            src_h, src_w = frame.shape[:2]
            if width > src_w or height > src_h:
                raise ValueError(f"Crop {width}x{height} is larger than rendered frame {src_w}x{src_h}.")
            x0 = (src_w - int(width)) // 2
            y0 = (src_h - int(height)) // 2
            cropped = frame[y0 : y0 + int(height), x0 : x0 + int(width)]
            writer.write(cropped)
    finally:
        cap.release()
        writer.release()


def _configure_render(
    *,
    width: int,
    height: int,
    fps: float,
    engine: str,
    samples: int,
    device: str,
    denoise: bool,
    video_format: str,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.fps = int(round(fps))
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "FFMPEG"
    if video_format == "mp4":
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
    else:
        scene.render.ffmpeg.format = "AVI"
        scene.render.ffmpeg.codec = "MPEG4"
    scene.render.ffmpeg.constant_rate_factor = "PERC_LOSSLESS"
    scene.render.ffmpeg.video_bitrate = 90000
    scene.render.ffmpeg.maxrate = 120000
    scene.render.ffmpeg.buffersize = 120000
    scene.render.ffmpeg.gopsize = 1
    if engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = bool(denoise)
        scene.cycles.max_bounces = 4
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 2
        scene.cycles.transparent_max_bounces = 4
        scene.cycles.device = device
        if device == "GPU":
            _enable_gpu_cycles()
    else:
        scene.eevee.taa_render_samples = max(8, int(samples))
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def _enable_gpu_cycles() -> None:
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return
    cycles_prefs = prefs.preferences
    for compute_type in ("OPTIX", "CUDA", "HIP", "METAL"):
        try:
            cycles_prefs.compute_device_type = compute_type
            cycles_prefs.get_devices()
            break
        except Exception:
            continue
    for dev in cycles_prefs.devices:
        dev_type = str(getattr(dev, "type", "")).upper()
        dev_name = str(getattr(dev, "name", "")).upper()
        dev.use = dev_type != "CPU" and "CPU" not in dev_name


def _load_scene_config(data) -> dict[str, object]:
    def _array_or_default(name: str, default):
        return np.asarray(data[name]).tolist() if name in data.files else default

    return {
        "floor_style": str(np.asarray(data["scene_floor_style"]).reshape(-1)[0]) if "scene_floor_style" in data.files else "concrete",
        "floor_color": _array_or_default("scene_floor_color", [0.50, 0.49, 0.45]),
        "floor_accent": _array_or_default("scene_floor_accent", [0.34, 0.34, 0.32]),
        "sun_rotation": _array_or_default("scene_sun_rotation", [42.0, 0.0, 28.0]),
        "sun_intensity": float(np.asarray(data["scene_sun_intensity"]).reshape(-1)[0]) if "scene_sun_intensity" in data.files else 2600.0,
    }


def _create_open_scene(vertices: np.ndarray, scene_config: dict[str, object]) -> None:
    bounds_min = vertices.reshape(-1, 3).min(axis=0)
    bounds_max = vertices.reshape(-1, 3).max(axis=0)
    center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
    floor_z = float(bounds_min[2]) - 0.015

    floor_color = tuple(float(v) for v in scene_config.get("floor_color", [0.52, 0.50, 0.46])) + (1.0,)
    floor_mat = _make_principled_material("mat_floor", floor_color, roughness=0.82)
    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(float(center_xy[0]), float(center_xy[1]), floor_z))
    floor = bpy.context.object
    floor.name = "open_concrete_floor"
    floor.data.materials.append(floor_mat)

    # Add a few low-contrast distant primitives so the cameras see a real scene
    # without turning this into a collision or physics simulation.
    accent_color = tuple(float(v) for v in scene_config.get("floor_accent", [0.18, 0.30, 0.36])) + (1.0,)
    accent_mat = _make_principled_material("mat_floor_accent", accent_color, roughness=0.7)
    for idx, offset in enumerate(((-3.2, 2.8, 0.35), (3.0, 3.4, 0.5), (2.8, -3.0, 0.25))):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(float(center_xy[0] + offset[0]), float(center_xy[1] + offset[1]), floor_z + offset[2]))
        block = bpy.context.object
        block.name = f"distant_scene_block_{idx}"
        block.dimensions = (1.2 + 0.35 * idx, 0.35, 0.5 + 0.25 * idx)
        block.data.materials.append(accent_mat)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.78, 0.82, 0.88)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 5.0))
    sun = bpy.context.object
    sun.name = "soft_sun"
    sun.data.energy = float(scene_config.get("sun_intensity", 2600.0)) / 1400.0
    sun.rotation_euler = tuple(math.radians(float(v)) for v in scene_config.get("sun_rotation", [42.0, 0.0, 28.0]))

    bpy.ops.object.light_add(type="AREA", location=(float(center_xy[0] - 2.0), float(center_xy[1] - 3.0), floor_z + 4.0))
    area = bpy.context.object
    area.name = "large_softbox"
    area.data.energy = 360.0
    area.data.size = 5.0


def _create_body_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    body_face_groups: np.ndarray,
    body_group_colors: np.ndarray,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("smplx_body_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new("smplx_body", mesh)
    bpy.context.collection.objects.link(obj)
    colors = np.asarray(body_group_colors, dtype=float).reshape(-1, 3)
    for group_idx, color in enumerate(colors):
        obj.data.materials.append(
            _make_principled_material(f"mat_body_group_{group_idx}", tuple(float(v) for v in color) + (1.0,), roughness=0.62)
        )
    if not obj.data.materials:
        obj.data.materials.append(_make_principled_material("mat_smplx_skin", (0.72, 0.58, 0.49, 1.0), roughness=0.62))
    groups = np.asarray(body_face_groups, dtype=np.int32).reshape(-1)
    if len(groups) == len(mesh.polygons):
        for polygon, group_idx in zip(mesh.polygons, groups):
            polygon.material_index = int(np.clip(group_idx, 0, len(obj.data.materials) - 1))
    obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    return obj


def _create_tag_meshes(tag_names: list[str], first_corners: np.ndarray, texture_dir: Path) -> list[bpy.types.Object]:
    objects = []
    for tag_idx, tag_name in enumerate(tag_names):
        mesh = bpy.data.meshes.new(f"{tag_name}_mesh")
        mesh.from_pydata(first_corners[tag_idx].tolist(), [], [(0, 1, 2, 3)])
        mesh.update()
        uv_layer = mesh.uv_layers.new(name="UVMap")
        uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        for loop_idx, uv in enumerate(uvs):
            uv_layer.data[loop_idx].uv = uv
        obj = bpy.data.objects.new(tag_name, mesh)
        bpy.context.collection.objects.link(obj)
        texture_name = "tag0_aruco.png" if tag_name.endswith("tag0") else "tag1_aruco.png"
        obj.data.materials.append(_make_image_material(f"mat_{tag_name}", texture_dir / texture_name))
        objects.append(obj)
    return objects


def _create_cameras(camera_names: list[str], fisheye_fov: float) -> list[bpy.types.Object]:
    objects = []
    for camera_name in camera_names:
        camera_data = bpy.data.cameras.new(camera_name)
        camera_data.clip_start = 0.01
        camera_data.clip_end = 80.0
        camera_data.display_size = 0.08
        camera_data.type = "PANO"
        try:
            camera_data.panorama_type = "FISHEYE_EQUIDISTANT"
            camera_data.fisheye_fov = fisheye_fov
        except Exception:
            camera_data.type = "PERSP"
            camera_data.angle = min(fisheye_fov, math.radians(170.0))
        obj = bpy.data.objects.new(camera_name, camera_data)
        bpy.context.collection.objects.link(obj)
        objects.append(obj)
    return objects


def _update_mesh_vertices(mesh: bpy.types.Mesh, vertices: np.ndarray) -> None:
    mesh.vertices.foreach_set("co", np.asarray(vertices, dtype=np.float32).reshape(-1))
    mesh.update()


def _make_principled_material(name: str, color: tuple[float, float, float, float], roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
    return mat


def _make_image_material(name: str, image_path: Path) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path))
    tex.extension = "CLIP"
    if principled is not None:
        mat.node_tree.links.new(tex.outputs["Color"], principled.inputs["Base Color"])
        principled.inputs["Roughness"].default_value = 0.45
    return mat


def _geosim_camera_to_blender_matrix(position: np.ndarray, rotation_cam_to_world: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation_cam_to_world, dtype=float) @ np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


if __name__ == "__main__":
    raise SystemExit(main())
