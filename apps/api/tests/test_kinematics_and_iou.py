import math

import numpy as np

from src.modules.perception.kinematics import (
    compute_binary_mask_iou,
    compute_mask_pose_pca,
    estimate_revolute_pivot_2d,
    unwrap_axial_angles,
)


def test_compute_mask_pose_pca_horizontal_bar_and_contour() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 20:80] = 1

    pose = compute_mask_pose_pca(mask)
    assert math.isclose(pose.centroid_xy[0], 49.5, abs_tol=0.1)
    assert math.isclose(pose.centroid_xy[1], 49.5, abs_tol=0.1)
    assert pose.angle_rad is not None
    assert math.isclose(abs(pose.angle_rad), 0.0, abs_tol=0.05) or math.isclose(
        abs(pose.angle_rad), math.pi, abs_tol=0.05
    )
    assert pose.eccentricity > 0.90
    assert pose.area_px == 600
    assert len(pose.contour_polygon_px) >= 4


def test_compute_mask_pose_pca_returns_none_angle_for_square_or_circle() -> None:
    square_mask = np.zeros((100, 100), dtype=np.uint8)
    square_mask[30:70, 30:70] = 1  # Symmetric square -> eccentricity ~ 0.0

    pose = compute_mask_pose_pca(square_mask)
    assert pose.eccentricity < 0.05
    assert pose.angle_rad is None


def test_unwrap_axial_angles_eliminates_180_degree_pca_flips() -> None:
    # Simulate a smooth rotation where PCA flips by -pi on frame 2
    raw = [0.1, 0.2, 0.3 - math.pi, 0.4]
    unwrapped = unwrap_axial_angles(raw)
    assert np.allclose(unwrapped, [0.1, 0.2, 0.3, 0.4], atol=1e-4)


def test_compute_binary_mask_iou_returns_zero_on_empty_masks() -> None:
    empty1 = np.zeros((10, 10), dtype=np.uint8)
    empty2 = np.zeros((10, 10), dtype=np.uint8)

    # Empty masks must return 0.0 so a broken/blank render never passes verification
    assert compute_binary_mask_iou(empty1, empty2) == 0.0

    m1 = np.zeros((10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10), dtype=np.uint8)
    m1[0:4, 0:4] = 1  # 16 px
    m2[0:4, 0:2] = 1  # 8 px
    assert math.isclose(compute_binary_mask_iou(m1, m2), 0.5)
    assert compute_binary_mask_iou(m1, empty2) == 0.0


def test_estimate_revolute_pivot_2d_with_moving_parent() -> None:
    # Parent translates along X while child rotates around a hinge at (20, 0) in parent's local frame
    parent_positions = [(100.0 + i * 5.0, 80.0) for i in range(5)]
    local_hinge = (20.0, 0.0)
    radius = 40.0
    angles = [0.0, 0.25, 0.50, 0.75, 1.00]

    child_positions = [
        (
            px + local_hinge[0] + radius * math.cos(a),
            py + local_hinge[1] + radius * math.sin(a),
        )
        for (px, py), a in zip(parent_positions, angles, strict=False)
    ]

    pivot_world, joint_type = estimate_revolute_pivot_2d(
        parent_centroids_seq=parent_positions,
        child_centroids_seq=child_positions,
        child_angles_seq=angles,
    )
    assert joint_type == "revolute"
    # At frame 0, parent is at (100, 80) + local_hinge (20, 0) -> world pivot (120, 80)
    assert math.isclose(pivot_world[0], 120.0, abs_tol=1e-2)
    assert math.isclose(pivot_world[1], 80.0, abs_tol=1e-2)


def test_estimate_revolute_pivot_2d_falls_back_to_fixed_under_low_motion() -> None:
    centroids = [(50.0, 50.0), (50.1, 50.0), (50.0, 50.1), (50.1, 50.1)]
    angles = [0.01, 0.015, 0.02, 0.018]

    _, joint_type = estimate_revolute_pivot_2d(
        parent_centroids_seq=(30.0, 50.0),
        child_centroids_seq=centroids,
        child_angles_seq=angles,
    )
    assert joint_type == "fixed"


def test_estimate_revolute_pivot_2d_detects_collinear_prismatic_slider() -> None:
    centroids = [(10.0, 20.0), (20.0, 20.0), (30.0, 20.0), (40.0, 20.0)]
    angles = [0.0, 0.2, 0.0, 0.2]

    _, joint_type = estimate_revolute_pivot_2d(
        parent_centroids_seq=(0.0, 20.0),
        child_centroids_seq=centroids,
        child_angles_seq=angles,
    )
    assert joint_type == "prismatic"
