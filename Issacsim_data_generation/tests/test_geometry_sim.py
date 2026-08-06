from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from geosim.camera import make_default_camera_rig, make_default_wrist_camera_rig
from geosim.config import SimulationConfig
from geosim.imu import fuse_wrist_visual_imu, simulate_wrist_imu
from geosim.linalg import frame_from_forearm
from geosim.motion import load_motion_npz, synthetic_motion
from geosim.pose_tracks import PoseEstimate, save_pose_tracks, smooth_pose_sequence
from geosim.runner import run_motion_sequence, summarize_results
from geosim.tag_rig import make_wrist_tag_rig


class GeometrySimulationTest(unittest.TestCase):
    def test_default_rig_camera_positions(self) -> None:
        cameras = make_default_camera_rig(SimulationConfig().camera_rig)
        positions = {camera.name: camera.position_head for camera in cameras}
        np.testing.assert_allclose(positions["CAM_A"], [-0.10, 0.22, 0.0])
        np.testing.assert_allclose(positions["CAM_B"], [0.10, 0.22, 0.0])
        np.testing.assert_allclose(positions["CAM_C"], [-0.10, -0.22, 0.0])
        np.testing.assert_allclose(positions["CAM_D"], [0.10, -0.22, 0.0])

    def test_wrist_camera_axis_lines_pass_through_wrist_and_share_tag_plane(self) -> None:
        config = SimulationConfig().camera_rig
        distance = 0.04
        cameras = make_default_wrist_camera_rig(config)
        self.assertEqual([camera.name for camera in cameras], ["WRIST_PALM_NORMAL", "WRIST_FORWARD"])
        for camera in cameras:
            self.assertEqual(camera.image_width, config.image_width)
            self.assertEqual(camera.image_height, config.image_height)
            self.assertEqual(camera.fov_deg, config.fisheye_fov_deg)
            optical_axis_wrist = camera.rotation_cam_to_wrist @ np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(camera.position_wrist - distance * optical_axis_wrist, np.zeros(3), atol=1e-12)
            self.assertLess(float(np.dot(optical_axis_wrist, -camera.position_wrist)), 0.0)
            self.assertAlmostEqual(float(optical_axis_wrist[0]), 0.0, places=12)

        axes = [camera.rotation_cam_to_wrist @ np.array([0.0, 0.0, 1.0]) for camera in cameras]
        np.testing.assert_allclose(axes[0], [0.0, -1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(axes[1], [0.0, 0.0, -1.0], atol=1e-12)
        self.assertAlmostEqual(float(np.dot(axes[0], axes[1])), 0.0, places=12)

        natural_wrist_rot = frame_from_forearm(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        forward_axis_world = natural_wrist_rot @ axes[1]
        np.testing.assert_allclose(forward_axis_world, [0.0, -1.0, 0.0], atol=1e-12)

    def test_amass_camera_optical_axes_point_down(self) -> None:
        motion_path = ROOT / "test_motion/HumanEva/S1/Static_stageii.npz"
        model_path = ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"
        if not motion_path.exists() or not model_path.exists():
            self.skipTest("AMASS fixture or SMPL-X model is not available.")
        motion = load_motion_npz(motion_path, smplx_model_path=model_path)
        down = np.array([0.0, 0.0, -1.0])
        for camera in make_default_camera_rig(SimulationConfig().camera_rig):
            _, cam_rot = camera.world_pose(motion.head_pos[0], motion.head_rot[0])
            np.testing.assert_allclose(cam_rot @ np.array([0.0, 0.0, 1.0]), down, atol=1e-9)

    def test_wrist_tags_are_coplanar_and_perpendicular_to_forearm(self) -> None:
        tag_rig = make_wrist_tag_rig(SimulationConfig().tag_rig)
        points = tag_rig.model_points
        xs = np.array([point[0] for point in points.values()])
        np.testing.assert_allclose(xs, np.zeros_like(xs), atol=1e-12)

        tag0_edge = points["tag0_c1"] - points["tag0_c0"]
        tag1_edge = points["tag1_c1"] - points["tag1_c0"]
        cos_angle = np.dot(tag0_edge, tag1_edge) / (np.linalg.norm(tag0_edge) * np.linalg.norm(tag1_edge))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        self.assertAlmostEqual(angle, 45.0, places=7)

    def test_noise_free_synthetic_motion_is_reconstructed(self) -> None:
        config = SimulationConfig(max_frames=30)
        results = run_motion_sequence(synthetic_motion(frames=30), config)
        summary = summarize_results(results)
        self.assertEqual(summary["frames"], 30)
        self.assertGreater(summary["success_rate"], 0.9)
        self.assertLess(summary["position_max_m"], 1e-8)
        self.assertLess(summary["rotation_max_deg"], 1e-5)

    def test_synthetic_motion_includes_mirrored_right_wrist(self) -> None:
        motion = synthetic_motion(frames=6)
        self.assertIsNotNone(motion.right_wrist_pos)
        self.assertIsNotNone(motion.right_elbow_pos)
        self.assertIsNotNone(motion.right_wrist_rot)
        assert motion.right_wrist_pos is not None
        assert motion.right_elbow_pos is not None
        np.testing.assert_allclose(motion.right_wrist_pos[:, 0], -motion.left_wrist_pos[:, 0])
        np.testing.assert_allclose(motion.right_wrist_pos[:, 1:], motion.left_wrist_pos[:, 1:])
        np.testing.assert_allclose(motion.right_elbow_pos[:, 0], -motion.left_elbow_pos[:, 0])

    def test_pose_sequence_interpolates_missing_frame(self) -> None:
        sequence = [
            PoseEstimate(np.array([0.0, 0.0, 0.0]), np.eye(3), "stereo"),
            None,
            PoseEstimate(np.array([2.0, 0.0, 0.0]), np.eye(3), "stereo"),
        ]
        smoothed = smooth_pose_sequence(sequence)
        self.assertEqual(len(smoothed), 3)
        self.assertEqual(smoothed[1].source, "interpolated")
        np.testing.assert_allclose(smoothed[1].position_world, [1.0, 0.0, 0.0])

    def test_pose_track_round_trip_shapes(self) -> None:
        path = ROOT / "outputs/test_pose_track_round_trip.npz"
        tag_names = ("tag0",)
        tag_truth = {"tag0": [(np.eye(3), np.array([0.0, 0.0, 0.0]))]}
        tag_estimates = {"tag0": [PoseEstimate(np.array([0.1, 0.0, 0.0]), np.eye(3), "stereo")]}
        wrist_estimates = [PoseEstimate(np.array([0.2, 0.0, 0.0]), np.eye(3), "tag0:stereo")]
        save_pose_tracks(
            path,
            frame_indices=np.array([0]),
            output_fps=30.0,
            tag_names=tag_names,
            tag_estimates=tag_estimates,
            tag_truth=tag_truth,
            wrist_estimates=wrist_estimates,
            wrist_truth_pos=np.array([[0.0, 0.0, 0.0]]),
            wrist_truth_rot=np.eye(3)[None, :, :],
            raw_sources={"tag0": {"stereo": 1}},
        )
        data = np.load(path)
        self.assertEqual(data["wrist_est_pos"].shape, (1, 3))
        self.assertEqual(data["tag_est_rot"].shape, (1, 1, 3, 3))

    def test_simulated_imu_fuses_wrist_pose_sequence(self) -> None:
        motion = synthetic_motion(frames=8, fps=40.0)
        imu = simulate_wrist_imu(motion.left_wrist_pos, motion.left_wrist_rot, motion.fps, seed=2)
        self.assertEqual(imu.gyro_rad_s.shape, (8, 3))
        self.assertEqual(imu.accel_m_s2.shape, (8, 3))
        visual = [
            PoseEstimate(motion.left_wrist_pos[0], motion.left_wrist_rot[0], "visual"),
            None,
            None,
            PoseEstimate(motion.left_wrist_pos[3], motion.left_wrist_rot[3], "visual"),
            None,
            None,
            None,
            PoseEstimate(motion.left_wrist_pos[7], motion.left_wrist_rot[7], "visual"),
        ]
        fused = fuse_wrist_visual_imu(visual, imu, motion.left_wrist_pos[0], motion.left_wrist_rot[0])
        self.assertEqual(len(fused), 8)
        self.assertTrue(np.isfinite(np.stack([estimate.position_world for estimate in fused])).all())
        self.assertTrue(all(estimate.source.startswith("imu") for estimate in fused))


if __name__ == "__main__":
    unittest.main()
