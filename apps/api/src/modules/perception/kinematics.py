from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskPose2D:
    centroid_xy: tuple[float, float]
    angle_rad: float | None  # None when mask is near-circular (low eccentricity)
    eccentricity: float
    area_px: int
    contour_polygon_px: list[tuple[float, float]]


def extract_mask_contour_polygon(
    binary_mask: np.ndarray, epsilon_factor: float = 0.015
) -> list[tuple[float, float]]:
    """
    Extracts a simplified 2D polygon contour from a binary mask so the frontend
    overlay and silhouette verifier render actual part geometry instead of rectangles.
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_factor * peri, True)
    return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]


def compute_mask_pose_pca(binary_mask: np.ndarray, min_eccentricity: float = 0.20) -> MaskPose2D:
    """
    Computes 2D centroid, principal axis orientation angle, eccentricity, and contour
    polygon from a 2D binary mask. If eccentricity < min_eccentricity (near-circular
    or square part), angle_rad is returned as None rather than inventing an unstable angle.
    """
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array")

    ys, xs = np.nonzero(binary_mask > 0)
    area_px = int(len(xs))
    if area_px < 3:
        raise ValueError("Mask contains fewer than 3 foreground pixels; cannot solve PCA")

    mean_x = float(np.mean(xs))
    mean_y = float(np.mean(ys))

    coords = np.vstack((xs - mean_x, ys - mean_y))
    cov = np.cov(coords)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    lambda1, lambda2 = float(eigenvalues[order[0]]), float(eigenvalues[order[1]])
    principal_vec = eigenvectors[:, order[0]]

    if lambda1 <= 1e-9:
        raise ValueError("Degenerate zero-variance mask geometry")

    eccentricity = float(np.clip(1.0 - (max(lambda2, 0.0) / lambda1), 0.0, 1.0))
    angle_rad: float | None = None
    if eccentricity >= min_eccentricity:
        angle_rad = float(math.atan2(principal_vec[1], principal_vec[0]))

    polygon = extract_mask_contour_polygon(binary_mask)

    return MaskPose2D(
        centroid_xy=(mean_x, mean_y),
        angle_rad=angle_rad,
        eccentricity=eccentricity,
        area_px=area_px,
        contour_polygon_px=polygon,
    )


def unwrap_axial_angles(angles_rad: list[float]) -> np.ndarray:
    """
    PCA eigenvectors have a 180-degree (pi) sign ambiguity (v vs -v).
    Unwrapping 2*theta modulo 2*pi and dividing by 2 eliminates 180-degree flips
    across consecutive video frames.
    """
    arr = np.asarray(angles_rad, dtype=np.float64)
    return np.unwrap(2.0 * arr) / 2.0


def compute_binary_mask_iou(mask_sam: np.ndarray, mask_sim: np.ndarray) -> float:
    """
    Computes exact pixel Intersection-over-Union (IoU) between SAM 3.1 ground-truth
    mask and a rendered simulation silhouette.
    Returns 0.0 if either mask is empty so blank renders never score 1.0.
    """
    if mask_sam.shape != mask_sim.shape:
        raise ValueError(f"Mask dimension mismatch: {mask_sam.shape} vs {mask_sim.shape}")

    sam_bool = mask_sam > 0
    sim_bool = mask_sim > 0

    sam_area = int(sam_bool.sum())
    sim_area = int(sim_bool.sum())
    if sam_area == 0 or sim_area == 0:
        return 0.0

    intersection = int(np.logical_and(sam_bool, sim_bool).sum())
    union = int(np.logical_or(sam_bool, sim_bool).sum())
    return float(intersection / union)


def estimate_revolute_pivot_2d(
    parent_centroids_seq: list[tuple[float, float]] | tuple[float, float],
    child_centroids_seq: list[tuple[float, float]],
    child_angles_seq: list[float],
    parent_angles_seq: list[float] | None = None,
    min_rotation_rad: float = 0.08,
    min_translation_px: float = 5.0,
) -> tuple[tuple[float, float], str]:
    """
    Estimates the 2D joint constraint between a parent part and child part by
    transforming the child's trajectory into the parent's reference frame.
    Classifies the joint as:
      - 'fixed': neither rotation nor translation exceeds noise thresholds
      - 'prismatic': collinear translation (SVD minor/major singular value ratio < 0.05)
      - 'revolute': circular arc trajectory around a hinge pivot solved via Kasa circle fit
    """
    n_frames = len(child_centroids_seq)
    if n_frames != len(child_angles_seq) or n_frames < 3:
        raise ValueError("At least 3 synchronized pose frames are required")

    # Normalize parent_centroids_seq if a single static (x, y) tuple was passed
    if (
        isinstance(parent_centroids_seq, tuple)
        and len(parent_centroids_seq) == 2
        and isinstance(parent_centroids_seq[0], (int, float))
    ):
        parent_pts = np.tile(np.array(parent_centroids_seq, dtype=np.float64), (n_frames, 1))
    else:
        parent_pts = np.asarray(parent_centroids_seq, dtype=np.float64)
        if len(parent_pts) != n_frames:
            raise ValueError("parent_centroids_seq length must match child_centroids_seq")

    parent_angles = (
        unwrap_axial_angles(parent_angles_seq)
        if parent_angles_seq is not None
        else np.zeros(n_frames, dtype=np.float64)
    )
    child_angles = unwrap_axial_angles(child_angles_seq)
    rel_angles = child_angles - parent_angles
    angular_span = float(np.max(rel_angles) - np.min(rel_angles))

    child_pts = np.asarray(child_centroids_seq, dtype=np.float64)

    # Express child displacement in the parent's local coordinate frame at each frame
    rel_pts = np.zeros_like(child_pts)
    for idx in range(n_frames):
        dx, dy = child_pts[idx] - parent_pts[idx]
        cos_p = math.cos(-parent_angles[idx])
        sin_p = math.sin(-parent_angles[idx])
        rel_pts[idx, 0] = cos_p * dx - sin_p * dy
        rel_pts[idx, 1] = sin_p * dx + cos_p * dy

    # Measure total translation span and SVD collinearity in parent frame
    centered = rel_pts - np.mean(rel_pts, axis=0)
    pairwise_dists = np.linalg.norm(rel_pts[:, None, :] - rel_pts[None, :, :], axis=-1)
    translation_span = float(np.max(pairwise_dists))

    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    sigma1 = float(singular_values[0]) if len(singular_values) > 0 else 0.0
    sigma2 = float(singular_values[1]) if len(singular_values) > 1 else 0.0
    collinearity_ratio = (sigma2 / sigma1) if sigma1 > 1e-6 else 1.0

    # World-space midpoint at Frame 0 for fixed/prismatic anchor placement
    p0_x, p0_y = float(parent_pts[0, 0]), float(parent_pts[0, 1])
    mean_child_x, mean_child_y = float(np.mean(child_pts[:, 0])), float(np.mean(child_pts[:, 1]))
    midpoint_world = (0.5 * (p0_x + mean_child_x), 0.5 * (p0_y + mean_child_y))

    # 1. Fixed check: neither significant rotation nor translation
    if angular_span < min_rotation_rad and translation_span < min_translation_px:
        return midpoint_world, "fixed"

    # 2. Prismatic check: trajectory is collinear (straight line) in parent frame
    if translation_span >= min_translation_px and collinearity_ratio < 0.05:
        return midpoint_world, "prismatic"

    # 3. Revolute check: requires angular span >= threshold
    if angular_span < min_rotation_rad:
        return midpoint_world, "fixed"

    # Solve Kasa circle fit in parent's relative coordinate frame:
    # 2*cx*x + 2*cy*y + (R^2 - cx^2 - cy^2) = x^2 + y^2
    rx = rel_pts[:, 0]
    ry = rel_pts[:, 1]
    a_mat = np.column_stack((2.0 * rx, 2.0 * ry, np.ones_like(rx)))
    b_vec = rx**2 + ry**2

    solution, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    rel_pivot_x, rel_pivot_y = float(solution[0]), float(solution[1])

    # Guard against near-collinear arcs where fitted radius is huge (> 10x chord span)
    fitted_radius = math.hypot(rel_pts[0, 0] - rel_pivot_x, rel_pts[0, 1] - rel_pivot_y)
    if translation_span > 1e-3 and (fitted_radius / translation_span) > 10.0:
        return midpoint_world, "prismatic"

    # Transform relative pivot back to Frame 0 world coordinates
    cos_0 = math.cos(parent_angles[0])
    sin_0 = math.sin(parent_angles[0])
    world_pivot_x = p0_x + (cos_0 * rel_pivot_x - sin_0 * rel_pivot_y)
    world_pivot_y = p0_y + (sin_0 * rel_pivot_x + cos_0 * rel_pivot_y)

    return (world_pivot_x, world_pivot_y), "revolute"
