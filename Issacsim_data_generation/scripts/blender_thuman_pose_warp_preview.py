#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _import_obj(path: Path, name_prefix: str):
    import bpy

    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    objects = [obj for obj in bpy.data.objects if obj not in before and getattr(obj, "type", "") == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh objects were imported from {path}")
    for obj in objects:
        obj.name = f"{name_prefix}_{obj.name}"
    return objects


def _single_mesh(objects, label: str):
    if len(objects) != 1:
        raise RuntimeError(f"{label} expected one mesh object, got {len(objects)}")
    return objects[0]


def _coords(obj):
    import numpy as np

    return np.asarray([obj.matrix_world @ vertex.co for vertex in obj.data.vertices], dtype=float)


def _warp_scan_by_nearest_smplx(source_scan, source_smplx, target_smplx):
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.kdtree import KDTree

    scan_xyz = _coords(source_scan)
    source_xyz = _coords(source_smplx)
    target_xyz = _coords(target_smplx)
    if source_xyz.shape != target_xyz.shape:
        raise RuntimeError(f"SMPL-X vertex count mismatch: {source_xyz.shape} vs {target_xyz.shape}")

    kd = KDTree(len(source_xyz))
    for idx, xyz in enumerate(source_xyz):
        kd.insert(Vector(xyz), idx)
    kd.balance()

    nearest = np.empty(len(scan_xyz), dtype=np.int32)
    residual = np.empty_like(scan_xyz)
    for idx, xyz in enumerate(scan_xyz):
        co, nearest_idx, _ = kd.find(Vector(xyz))
        nearest[idx] = int(nearest_idx)
        residual[idx] = xyz - source_xyz[nearest_idx]

    warped_xyz = target_xyz[nearest] + residual
    mesh = source_scan.data.copy()
    warped_obj = source_scan.copy()
    warped_obj.data = mesh
    warped_obj.name = "warped_0000_texture_to_0001_pose"
    bpy.context.collection.objects.link(warped_obj)
    for idx, vertex in enumerate(warped_obj.data.vertices):
        vertex.co = warped_obj.matrix_world.inverted() @ Vector(warped_xyz[idx])

    stats = _warp_stats(source_scan, scan_xyz, warped_xyz, residual)
    return warped_obj, stats


def _triangle_basis(vertices):
    import numpy as np

    a, b, c = vertices
    e1 = b - a
    e2 = c - a
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n = n / n_norm
    t = e1
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-12:
        t = np.cross(n, np.array([1.0, 0.0, 0.0], dtype=float))
        t_norm = np.linalg.norm(t)
    t = t / max(t_norm, 1e-12)
    bitangent = np.cross(n, t)
    return t, bitangent, n


def _barycentric_coordinates(point, triangle):
    import numpy as np

    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=float)


def _warp_scan_by_smplx_surface(source_scan, source_smplx, target_smplx, *, keep_tangent_residual: bool):
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    scan_xyz = _coords(source_scan)
    source_xyz = _coords(source_smplx)
    target_xyz = _coords(target_smplx)
    if source_xyz.shape != target_xyz.shape:
        raise RuntimeError(f"SMPL-X vertex count mismatch: {source_xyz.shape} vs {target_xyz.shape}")

    faces = [tuple(int(v) for v in polygon.vertices) for polygon in source_smplx.data.polygons]
    if len(faces) != len(target_smplx.data.polygons):
        raise RuntimeError("Source and target SMPL-X face counts do not match.")
    bvh = BVHTree.FromObject(source_smplx, bpy.context.evaluated_depsgraph_get())

    warped_xyz = np.empty_like(scan_xyz)
    face_indices = np.empty(len(scan_xyz), dtype=np.int32)
    normal_offsets = np.empty(len(scan_xyz), dtype=float)
    for idx, xyz in enumerate(scan_xyz):
        nearest = bvh.find_nearest(Vector(xyz))
        if nearest is None:
            warped_xyz[idx] = xyz
            face_indices[idx] = -1
            normal_offsets[idx] = 0.0
            continue
        nearest_point, _nearest_normal, face_idx, _distance = nearest
        face_indices[idx] = int(face_idx)
        face = faces[int(face_idx)]
        src_tri = source_xyz[list(face)]
        tgt_tri = target_xyz[list(face)]
        nearest_np = np.asarray(nearest_point, dtype=float)
        bary = _barycentric_coordinates(nearest_np, src_tri)
        src_t, src_b, src_n = _triangle_basis(src_tri)
        tgt_t, tgt_b, tgt_n = _triangle_basis(tgt_tri)
        residual = xyz - nearest_np
        normal_offsets[idx] = float(np.dot(residual, src_n))
        target_surface = bary @ tgt_tri
        if keep_tangent_residual:
            local = np.array([np.dot(residual, src_t), np.dot(residual, src_b), np.dot(residual, src_n)])
            warped_xyz[idx] = target_surface + local[0] * tgt_t + local[1] * tgt_b + local[2] * tgt_n
        else:
            warped_xyz[idx] = target_surface + normal_offsets[idx] * tgt_n

    mesh = source_scan.data.copy()
    warped_obj = source_scan.copy()
    warped_obj.data = mesh
    warped_obj.name = "warped_0000_texture_to_0001_pose_surface_bound"
    bpy.context.collection.objects.link(warped_obj)
    for idx, vertex in enumerate(warped_obj.data.vertices):
        vertex.co = warped_obj.matrix_world.inverted() @ Vector(warped_xyz[idx])

    stats = _warp_stats(source_scan, scan_xyz, warped_xyz, scan_xyz - warped_xyz)
    stats["binding"] = {
        "method": "nearest SMPL-X triangle barycentric surface binding",
        "valid_bound_vertices": int(np.sum(face_indices >= 0)),
        "normal_offset_m": _quantiles(np.abs(normal_offsets)),
        "keep_tangent_residual": bool(keep_tangent_residual),
    }
    return warped_obj, stats


def _warp_scan_incremental_surface(
    source_scan,
    source_smplx,
    target_smplx,
    *,
    steps: int,
    keep_tangent_residual: bool,
):
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    original_scan_xyz = _coords(source_scan)
    current_scan_xyz = original_scan_xyz.copy()
    source_xyz = _coords(source_smplx)
    target_xyz = _coords(target_smplx)
    if source_xyz.shape != target_xyz.shape:
        raise RuntimeError(f"SMPL-X vertex count mismatch: {source_xyz.shape} vs {target_xyz.shape}")
    faces = [tuple(int(v) for v in polygon.vertices) for polygon in source_smplx.data.polygons]
    steps = max(1, int(steps))
    normal_offsets_all = []

    for step_idx in range(steps):
        alpha0 = step_idx / steps
        alpha1 = (step_idx + 1) / steps
        current_body = (1.0 - alpha0) * source_xyz + alpha0 * target_xyz
        next_body = (1.0 - alpha1) * source_xyz + alpha1 * target_xyz
        bvh_vertices = [Vector(row) for row in current_body]
        bvh = BVHTree.FromPolygons(bvh_vertices, faces)
        next_scan_xyz = np.empty_like(current_scan_xyz)
        normal_offsets = np.empty(len(current_scan_xyz), dtype=float)

        for idx, xyz in enumerate(current_scan_xyz):
            nearest = bvh.find_nearest(Vector(xyz))
            if nearest is None:
                next_scan_xyz[idx] = xyz
                normal_offsets[idx] = 0.0
                continue
            nearest_point, _nearest_normal, face_idx, _distance = nearest
            face = faces[int(face_idx)]
            cur_tri = current_body[list(face)]
            next_tri = next_body[list(face)]
            nearest_np = np.asarray(nearest_point, dtype=float)
            bary = _barycentric_coordinates(nearest_np, cur_tri)
            cur_t, cur_b, cur_n = _triangle_basis(cur_tri)
            next_t, next_b, next_n = _triangle_basis(next_tri)
            residual = xyz - nearest_np
            normal_offsets[idx] = float(np.dot(residual, cur_n))
            target_surface = bary @ next_tri
            if keep_tangent_residual:
                local = np.array([np.dot(residual, cur_t), np.dot(residual, cur_b), np.dot(residual, cur_n)])
                next_scan_xyz[idx] = target_surface + local[0] * next_t + local[1] * next_b + local[2] * next_n
            else:
                next_scan_xyz[idx] = target_surface + normal_offsets[idx] * next_n

        current_scan_xyz = next_scan_xyz
        normal_offsets_all.append(np.abs(normal_offsets))

    mesh = source_scan.data.copy()
    warped_obj = source_scan.copy()
    warped_obj.data = mesh
    warped_obj.name = "warped_0000_texture_to_0001_pose_incremental_surface"
    bpy.context.collection.objects.link(warped_obj)
    for idx, vertex in enumerate(warped_obj.data.vertices):
        vertex.co = warped_obj.matrix_world.inverted() @ Vector(current_scan_xyz[idx])

    stats = _warp_stats(source_scan, original_scan_xyz, current_scan_xyz, original_scan_xyz - current_scan_xyz)
    stats["binding"] = {
        "method": "incremental nearest SMPL-X triangle barycentric surface tracking",
        "steps": int(steps),
        "normal_offset_m": _quantiles(np.concatenate(normal_offsets_all)),
        "keep_tangent_residual": bool(keep_tangent_residual),
    }
    return warped_obj, stats


def _warp_scan_by_knn_displacement(source_scan, source_smplx, target_smplx, *, k: int, sigma: float):
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.kdtree import KDTree

    original_scan_xyz = _coords(source_scan)
    source_xyz = _coords(source_smplx)
    target_xyz = _coords(target_smplx)
    if source_xyz.shape != target_xyz.shape:
        raise RuntimeError(f"SMPL-X vertex count mismatch: {source_xyz.shape} vs {target_xyz.shape}")

    deltas = target_xyz - source_xyz
    kd = KDTree(len(source_xyz))
    for idx, xyz in enumerate(source_xyz):
        kd.insert(Vector(xyz), idx)
    kd.balance()

    k = max(1, int(k))
    sigma = max(float(sigma), 1e-6)
    warped_xyz = np.empty_like(original_scan_xyz)
    nearest_dist = np.empty(len(original_scan_xyz), dtype=float)
    for idx, xyz in enumerate(original_scan_xyz):
        neighbors = kd.find_n(Vector(xyz), k)
        dists = np.asarray([max(float(item[2]), 1e-8) for item in neighbors], dtype=float)
        indices = np.asarray([int(item[1]) for item in neighbors], dtype=np.int32)
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-12:
            weights = 1.0 / dists
        weights = weights / weights.sum()
        delta = weights @ deltas[indices]
        warped_xyz[idx] = xyz + delta
        nearest_dist[idx] = float(dists[0])

    mesh = source_scan.data.copy()
    warped_obj = source_scan.copy()
    warped_obj.data = mesh
    warped_obj.name = "warped_0000_texture_to_0001_pose_knn_displacement"
    bpy.context.collection.objects.link(warped_obj)
    for idx, vertex in enumerate(warped_obj.data.vertices):
        vertex.co = warped_obj.matrix_world.inverted() @ Vector(warped_xyz[idx])

    stats = _warp_stats(source_scan, original_scan_xyz, warped_xyz, original_scan_xyz - warped_xyz)
    stats["binding"] = {
        "method": "KNN smooth SMPL-X displacement field",
        "k": int(k),
        "sigma_m": float(sigma),
        "nearest_source_smplx_distance_m": _quantiles(nearest_dist),
    }
    return warped_obj, stats


def _build_vertex_adjacency(mesh) -> list[list[int]]:
    adjacency = [set() for _ in mesh.vertices]
    for polygon in mesh.polygons:
        verts = list(polygon.vertices)
        for a, b in zip(verts, verts[1:] + verts[:1]):
            if a != b:
                adjacency[int(a)].add(int(b))
                adjacency[int(b)].add(int(a))
    return [sorted(neighbors) for neighbors in adjacency]


def _laplacian_smooth_copy(source_obj, *, iterations: int, alpha: float):
    import bpy
    import numpy as np
    from mathutils import Vector

    smoothed = source_obj.copy()
    smoothed.data = source_obj.data.copy()
    smoothed.name = f"{source_obj.name}_laplacian_smoothed"
    bpy.context.collection.objects.link(smoothed)

    adjacency = _build_vertex_adjacency(smoothed.data)
    coords = np.asarray([vertex.co[:] for vertex in smoothed.data.vertices], dtype=float)
    alpha = float(alpha)
    for _ in range(max(0, int(iterations))):
        new_coords = coords.copy()
        for idx, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            mean = coords[neighbors].mean(axis=0)
            new_coords[idx] = (1.0 - alpha) * coords[idx] + alpha * mean
        coords = new_coords
    for idx, vertex in enumerate(smoothed.data.vertices):
        vertex.co = Vector(coords[idx])
    return smoothed


def _warp_stats(scan_obj, original_xyz, warped_xyz, residual):
    import numpy as np

    edges = set()
    for polygon in scan_obj.data.polygons:
        verts = list(polygon.vertices)
        for a, b in zip(verts, verts[1:] + verts[:1]):
            if a != b:
                edges.add((min(int(a), int(b)), max(int(a), int(b))))
    edge_pairs = np.asarray(sorted(edges), dtype=np.int32)
    orig_len = np.linalg.norm(original_xyz[edge_pairs[:, 0]] - original_xyz[edge_pairs[:, 1]], axis=1)
    warp_len = np.linalg.norm(warped_xyz[edge_pairs[:, 0]] - warped_xyz[edge_pairs[:, 1]], axis=1)
    valid = orig_len > 1e-9
    ratio = warp_len[valid] / orig_len[valid]
    displacement = np.linalg.norm(warped_xyz - original_xyz, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)

    return {
        "scan_vertices": int(len(original_xyz)),
        "scan_edges": int(len(edge_pairs)),
        "edge_length_ratio_warped_over_original": _quantiles(ratio),
        "vertex_displacement_m": _quantiles(displacement),
        "surface_residual_to_source_smplx_m": _quantiles(residual_norm),
    }


def _quantiles(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _make_material(name: str, color: tuple[float, float, float, float]):
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.55
    return mat


def _assign_material(objects, material) -> None:
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def _bounds(objects):
    from mathutils import Vector

    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
    return mins, maxs


def _robust_bounds(objects, low: float = 0.01, high: float = 0.99):
    import numpy as np
    from mathutils import Vector

    coords = []
    for obj in objects:
        coords.extend([obj.matrix_world @ vertex.co for vertex in obj.data.vertices])
    arr = np.asarray(coords, dtype=float)
    mins = np.quantile(arr, low, axis=0)
    maxs = np.quantile(arr, high, axis=0)
    return Vector(mins), Vector(maxs)


def _normalize_group(objects, target_x: float, target_height: float = 1.8, robust: bool = True) -> None:
    from mathutils import Matrix, Vector

    mins, maxs = _robust_bounds(objects) if robust else _bounds(objects)
    center = (mins + maxs) * 0.5
    height = max(maxs.z - mins.z, 1e-6)
    scale = target_height / height
    transform = Matrix.Translation(Vector((target_x, 0.0, 0.0))) @ Matrix.Scale(scale, 4) @ Matrix.Translation(-center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world


def _add_label(text: str, x: float) -> None:
    import bpy

    bpy.ops.object.text_add(location=(x, -0.05, -1.38), rotation=(math.radians(75), 0.0, 0.0))
    obj = bpy.context.object
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.size = 0.13
    obj.data.materials.append(_make_material(f"mat_label_{text}", (0.02, 0.02, 0.02, 1.0)))


def _look_at(obj, target) -> None:
    from mathutils import Vector

    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warp one THuman textured scan to another fitted SMPL-X pose and render a preview.")
    parser.add_argument("--source-asset-dir", required=True)
    parser.add_argument("--target-asset-dir", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--smooth-iterations", type=int, default=0)
    parser.add_argument("--smooth-alpha", type=float, default=0.25)
    parser.add_argument(
        "--warp-method",
        default="surface",
        choices=("nearest_vertex", "surface", "incremental_surface", "knn_displacement"),
        help="nearest_vertex is the rough diagnostic baseline; surface uses nearest SMPL-X triangle barycentric binding.",
    )
    parser.add_argument("--keep-tangent-residual", action="store_true")
    parser.add_argument("--incremental-steps", type=int, default=12)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--knn-sigma", type=float, default=0.04)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    import bpy

    args = parse_args(argv)
    source_dir = Path(args.source_asset_dir).expanduser().resolve()
    target_dir = Path(args.target_asset_dir).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    source_manifest = _load_json(source_dir / "asset_manifest.json")
    target_manifest = _load_json(target_dir / "asset_manifest.json")
    textured_mesh = source_manifest.get("textured_mesh") or ""
    if not textured_mesh:
        raise RuntimeError(f"Source asset has no textured mesh: {source_dir}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    source_scan = _single_mesh(_import_obj(Path(textured_mesh), "source_scan"), "source textured scan")
    source_smplx = _single_mesh(_import_obj(Path(source_manifest["smplx_mesh"]), "source_smplx"), "source SMPL-X")
    target_smplx = _single_mesh(_import_obj(Path(target_manifest["smplx_mesh"]), "target_smplx"), "target SMPL-X")
    if args.warp_method == "nearest_vertex":
        warped_obj, stats = _warp_scan_by_nearest_smplx(source_scan, source_smplx, target_smplx)
    elif args.warp_method == "surface":
        warped_obj, stats = _warp_scan_by_smplx_surface(
            source_scan,
            source_smplx,
            target_smplx,
            keep_tangent_residual=args.keep_tangent_residual,
        )
    elif args.warp_method == "incremental_surface":
        warped_obj, stats = _warp_scan_incremental_surface(
            source_scan,
            source_smplx,
            target_smplx,
            steps=args.incremental_steps,
            keep_tangent_residual=args.keep_tangent_residual,
        )
    else:
        warped_obj, stats = _warp_scan_by_knn_displacement(
            source_scan,
            source_smplx,
            target_smplx,
            k=args.knn_k,
            sigma=args.knn_sigma,
        )
    smoothed_obj = None
    if args.smooth_iterations > 0:
        smoothed_obj = _laplacian_smooth_copy(
            warped_obj,
            iterations=args.smooth_iterations,
            alpha=args.smooth_alpha,
        )

    source_smplx.hide_render = True
    source_smplx.hide_viewport = True
    _assign_material([target_smplx], _make_material("mat_target_smplx", (0.78, 0.80, 0.84, 1.0)))

    if smoothed_obj is None:
        _normalize_group([source_scan], target_x=-2.15)
        _normalize_group([warped_obj], target_x=0.0)
        _normalize_group([target_smplx], target_x=2.15)
        _add_label("0000 textured scan", -2.15)
        _add_label("0000 warped to 0001 pose", 0.0)
        _add_label("0001 SMPL-X target", 2.15)
    else:
        _normalize_group([source_scan], target_x=-2.55)
        _normalize_group([warped_obj], target_x=-0.85)
        _normalize_group([smoothed_obj], target_x=0.85)
        _normalize_group([target_smplx], target_x=2.55)
        _add_label("0000 scan", -2.55)
        _add_label("raw warp", -0.85)
        _add_label("smoothed warp", 0.85)
        _add_label("0001 target", 2.55)

    bpy.ops.mesh.primitive_plane_add(size=7.6, location=(0.0, 0.0, -1.18))
    floor = bpy.context.object
    floor.data.materials.append(_make_material("mat_floor", (0.82, 0.84, 0.82, 1.0)))

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.4, 4.0))
    light = bpy.context.object
    light.data.energy = 750.0
    light.data.size = 4.8

    bpy.ops.object.camera_add(location=(0.0, -7.2, 0.42))
    camera = bpy.context.object
    _look_at(camera, (0.0, 0.0, 0.0))
    camera.data.lens = 30
    bpy.context.scene.camera = camera

    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    if hasattr(bpy.context.scene, "eevee"):
        bpy.context.scene.eevee.taa_render_samples = 64
    bpy.context.scene.render.resolution_x = int(args.width)
    bpy.context.scene.render.resolution_y = int(args.height)
    bpy.context.scene.world.color = (1.0, 1.0, 1.0)
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    bpy.context.scene.render.filepath = str(output_png)
    bpy.ops.render.render(write_still=True)

    report = {
        "source_asset_dir": str(source_dir),
        "target_asset_dir": str(target_dir),
        "output_png": str(output_png),
        "method": args.warp_method,
        "stats": stats,
        "smoothing": {
            "iterations": int(args.smooth_iterations),
            "alpha": float(args.smooth_alpha),
            "note": "Laplacian smoothing changes visual roughness but does not fix incorrect nearest-vertex correspondences.",
        },
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    raise SystemExit(main(argv))
