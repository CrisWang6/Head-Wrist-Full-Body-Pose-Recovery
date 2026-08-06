#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

try:
    from pupil_apriltags import Detector as PupilAprilTagDetector
except ModuleNotFoundError:
    PupilAprilTagDetector = None


SOCKETS = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
    "CAM_E": dai.CameraBoardSocket.CAM_E,
    "cama": dai.CameraBoardSocket.CAM_A,
    "camb": dai.CameraBoardSocket.CAM_B,
    "camc": dai.CameraBoardSocket.CAM_C,
    "camd": dai.CameraBoardSocket.CAM_D,
    "came": dai.CameraBoardSocket.CAM_E,
    "rgb": dai.CameraBoardSocket.CAM_A,
    "left": dai.CameraBoardSocket.CAM_B,
    "right": dai.CameraBoardSocket.CAM_C,
}

DEFAULT_FISHEYE_SOCKETS = ["CAM_A", "CAM_B", "CAM_C", "CAM_D"]
ARUCO_DICTIONARY_ALIASES = {
    "DICT_4X4": "DICT_4X4_50",
    "4X4": "DICT_4X4_50",
    "TAG36H11": "DICT_APRILTAG_36H11",
    "36H11": "DICT_APRILTAG_36H11",
    "APRILTAG_36H11": "DICT_APRILTAG_36H11",
}
PUPIL_APRILTAG_FAMILY_ALIASES = {
    "DICT_APRILTAG_36H11": "tag36h11",
    "APRILTAG_36H11": "tag36h11",
    "TAG36H11": "tag36h11",
    "36H11": "tag36h11",
    "tag36h11": "tag36h11",
}
_PUPIL_DETECTORS = {}
COLOR_RESOLUTIONS_BY_SIZE = {
    (1280, 720): dai.ColorCameraProperties.SensorResolution.THE_720_P,
    (1280, 800): dai.ColorCameraProperties.SensorResolution.THE_800_P,
    (1920, 1080): dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    (1920, 1200): dai.ColorCameraProperties.SensorResolution.THE_1200_P,
}


def socket_from_name(name):
    if name not in SOCKETS:
        valid = ", ".join(sorted(SOCKETS))
        raise ValueError(f"Unknown socket '{name}'. Valid sockets: {valid}")
    return SOCKETS[name]


def socket_label(socket):
    return socket.name


def normalize_socket_names(names):
    return [socket_label(socket_from_name(name)) for name in names]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def to_list(value):
    return np.asarray(value, dtype=float).tolist()


def make_color_camera_output(pipeline, socket, width, height, fps):
    size = (int(width), int(height))
    if size in COLOR_RESOLUTIONS_BY_SIZE:
        cam = pipeline.createColorCamera()
        cam.setBoardSocket(socket)
        cam.setResolution(COLOR_RESOLUTIONS_BY_SIZE[size])
        cam.setFps(float(fps))
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        return cam.isp

    cam = pipeline.create(dai.node.Camera)
    if hasattr(cam, "setSensorType") and hasattr(dai, "CameraSensorType"):
        cam.setSensorType(dai.CameraSensorType.COLOR)
    cam.setBoardSocket(socket)
    cam.setSize(size)
    cam.setFps(float(fps))
    return cam.video


def make_camera_pipeline(socket_names, width, height, fps):
    pipeline = dai.Pipeline()
    streams = {}

    for name in socket_names:
        socket = socket_from_name(name)
        stream_name = socket_label(socket)

        color_output = make_color_camera_output(pipeline, socket, width, height, fps)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(stream_name)
        color_output.link(xout.input)
        streams[stream_name] = stream_name

    return pipeline, streams


def frame_timestamp_ms(frame_msg):
    ts = frame_msg.getTimestampDevice()
    return ts.total_seconds() * 1000.0


def make_grid(frames, labels, cell_width=640):
    cells = []
    for label in labels:
        frame = frames.get(label)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "waiting", (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        scale = cell_width / frame.shape[1]
        resized = cv2.resize(frame, (cell_width, int(frame.shape[0] * scale)))
        cv2.putText(resized, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cells.append(resized)

    if not cells:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    h = min(cell.shape[0] for cell in cells)
    cells = [cell[:h, :, :] for cell in cells]
    if len(cells) <= 2:
        return np.hstack(cells)

    while len(cells) < 4:
        cells.append(np.zeros_like(cells[0]))

    top = np.hstack(cells[:2])
    bottom = np.hstack(cells[2:4])
    return np.vstack([top, bottom])


def aruco_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing. Install opencv-contrib-python.")
    name = ARUCO_DICTIONARY_ALIASES.get(str(name).upper(), name)
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown aruco dictionary '{name}'")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def aruco_detector_params(corner_refinement=None):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    if corner_refinement is not None and hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = corner_refinement
    if hasattr(params, "aprilTagQuadDecimate"):
        params.aprilTagQuadDecimate = 1.0
    if hasattr(params, "aprilTagMinClusterPixels"):
        params.aprilTagMinClusterPixels = 5
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 53
    if hasattr(params, "adaptiveThreshWinSizeStep"):
        params.adaptiveThreshWinSizeStep = 4
    return params


def detect_markers(gray, dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return marker_corners, marker_ids


def detect_apriltag_ids_in_image(image_path, dictionary_name="DICT_APRILTAG_36H11"):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read AprilTag image: {image_path}")

    _, marker_ids = detect_pupil_apriltags(gray, dictionary_name)
    if marker_ids is not None and len(marker_ids) > 0:
        return [int(value) for value in marker_ids.reshape(-1)]

    dictionary = aruco_dictionary(dictionary_name)
    params = aruco_detector_params()
    _, marker_ids = detect_markers(gray, dictionary, params)
    if marker_ids is None or len(marker_ids) == 0:
        return []
    return [int(value) for value in marker_ids.reshape(-1)]


def pupil_apriltag_family(name):
    return PUPIL_APRILTAG_FAMILY_ALIASES.get(str(name), PUPIL_APRILTAG_FAMILY_ALIASES.get(str(name).upper(), str(name)))


def pupil_detector(family):
    if PupilAprilTagDetector is None:
        return None
    family = pupil_apriltag_family(family)
    if family not in _PUPIL_DETECTORS:
        _PUPIL_DETECTORS[family] = PupilAprilTagDetector(
            families=family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
            debug=False,
        )
    return _PUPIL_DETECTORS[family]


def detect_pupil_apriltags(gray, family):
    detector = pupil_detector(family)
    if detector is None:
        return [], None
    results = detector.detect(gray, estimate_tag_pose=False)
    if not results:
        return [], None

    corners = []
    ids = []
    for result in results:
        # pupil_apriltags returns [top-right, top-left, bottom-left, bottom-right].
        # The rest of this calibration code uses OpenCV's [top-left, top-right, bottom-right, bottom-left].
        corners.append(np.asarray(result.corners, dtype=np.float64)[[1, 0, 3, 2]].reshape(1, 4, 2))
        ids.append([int(result.tag_id)])
    return corners, np.asarray(ids, dtype=np.int32)


def tag_edge_stats(image_points):
    tag_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 4, 2)
    if len(tag_points) == 0:
        return {
            "mean_tag_edge_px": 0.0,
            "min_tag_edge_px": 0.0,
            "max_tag_edge_px": 0.0,
        }

    edges = []
    for corners in tag_points:
        for index in range(4):
            edges.append(float(np.linalg.norm(corners[(index + 1) % 4] - corners[index])))

    return {
        "mean_tag_edge_px": float(np.mean(edges)),
        "min_tag_edge_px": float(np.min(edges)),
        "max_tag_edge_px": float(np.max(edges)),
    }


def aprilgrid_object_corners(tag_id, rows, cols, tag_size_m, tag_spacing_m, start_id=0):
    index = int(tag_id) - int(start_id)
    if index < 0 or index >= int(rows) * int(cols):
        return None

    row = index // int(cols)
    col = index % int(cols)
    pitch = float(tag_size_m) + float(tag_spacing_m)
    x0 = col * pitch
    y0 = row * pitch
    x1 = x0 + float(tag_size_m)
    y1 = y0 + float(tag_size_m)
    return np.array(
        [
            [x0, y0, 0.0],
            [x1, y0, 0.0],
            [x1, y1, 0.0],
            [x0, y1, 0.0],
        ],
        dtype=np.float64,
    )


def single_tag_object_corners(tag_size_m):
    tag_size = float(tag_size_m)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [tag_size, 0.0, 0.0],
            [tag_size, tag_size, 0.0],
            [0.0, tag_size, 0.0],
        ],
        dtype=np.float64,
    )


def detect_single_apriltag(
    image,
    tag_id=0,
    tag_size_m=0.10,
    dictionary_name="DICT_APRILTAG_36H11",
    robust=False,
):
    detection = detect_aprilgrid(
        image,
        rows=1,
        cols=1,
        tag_size_m=tag_size_m,
        tag_spacing_m=0.0,
        start_id=tag_id,
        end_id=tag_id,
        dictionary_name=dictionary_name,
        min_tags=1,
        robust=robust,
    )
    if detection is None:
        return None

    detection["object_points"] = single_tag_object_corners(tag_size_m).reshape(-1, 1, 3)
    detection["ids"] = np.asarray([tag_id * 4 + corner_index for corner_index in range(4)], dtype=np.int32)
    detection["tag_ids"] = [int(tag_id)]
    detection["tag_count"] = 1
    detection["corner_count"] = 4
    return detection


def detect_aprilgrid(
    image,
    rows=6,
    cols=6,
    tag_size_m=0.0352,
    tag_spacing_m=0.01056,
    start_id=0,
    end_id=35,
    dictionary_name="DICT_APRILTAG_36H11",
    min_tags=4,
    robust=False,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    marker_corners, marker_ids = detect_pupil_apriltags(gray, dictionary_name)
    detector_name = "pupil_apriltags" if marker_ids is not None and len(marker_ids) > 0 else ""

    if marker_ids is None or len(marker_ids) == 0:
        dictionary = aruco_dictionary(dictionary_name)
        refinement = cv2.aruco.CORNER_REFINE_APRILTAG if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG") else None
        refinements = [refinement]
        candidates = [gray]
        if robust:
            if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
                refinements.append(cv2.aruco.CORNER_REFINE_SUBPIX)
            refinements.append(None)

            equalized = cv2.equalizeHist(gray)
            if not np.array_equal(equalized, gray):
                candidates.append(equalized)
            candidates.append(255 - gray)

        marker_corners = []
        marker_ids = None
        for candidate in candidates:
            for refinement in refinements:
                params = aruco_detector_params(refinement)
                marker_corners, marker_ids = detect_markers(candidate, dictionary, params)
                if marker_ids is not None and len(marker_ids) > 0:
                    detector_name = "opencv_aruco"
                    break
            if marker_ids is not None and len(marker_ids) > 0:
                break

    if marker_ids is None or len(marker_ids) < int(min_tags):
        return None

    object_points = []
    image_points = []
    ids = []
    tag_ids = []
    for tag_id, corners in zip(marker_ids.flatten(), marker_corners):
        tag_id = int(tag_id)
        if tag_id < int(start_id) or tag_id > int(end_id):
            continue
        obj = aprilgrid_object_corners(tag_id, rows, cols, tag_size_m, tag_spacing_m, start_id)
        if obj is None:
            continue
        object_points.append(obj)
        image_points.append(corners.reshape(4, 2).astype(np.float64))
        ids.extend([tag_id * 4 + corner_index for corner_index in range(4)])
        tag_ids.append(tag_id)

    if len(tag_ids) < int(min_tags):
        return None

    object_points_array = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    image_points_array = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    return {
        "object_points": object_points_array,
        "image_points": image_points_array,
        "ids": np.asarray(ids, dtype=np.int32),
        "tag_ids": tag_ids,
        "tag_count": len(tag_ids),
        "corner_count": len(tag_ids) * 4,
        "detector": detector_name,
        **tag_edge_stats(image_points_array),
    }


def draw_detection(image, detection, board_type, corners_x=None, corners_y=None):
    overlay = image.copy()
    if detection is None:
        cv2.putText(overlay, "no board", (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return overlay

    points = detection["image_points"].astype(np.float32)
    tag_ids = detection.get("tag_ids", [])
    tag_points = points.reshape(-1, 4, 2)
    for index, corners in enumerate(tag_points):
        pts = np.round(corners).astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
        center = tuple(np.round(corners.mean(axis=0)).astype(int))
        label = str(tag_ids[index]) if index < len(tag_ids) else ""
        cv2.putText(overlay, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        for point in pts:
            cv2.circle(overlay, tuple(point), 3, (0, 255, 0), -1)

    tag_count = int(detection.get("tag_count", len(points) // 4))
    corner_count = int(detection.get("corner_count", len(points)))
    edge_px = float(detection.get("mean_tag_edge_px", 0.0))
    text = f"{tag_count} tags / {corner_count} corners"
    if edge_px > 0.0:
        text += f" / edge {edge_px:.0f}px"
    cv2.putText(overlay, text, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    return overlay


