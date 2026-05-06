"""
Unified coordinate transforms for single- and multi-arm setups.

Frames (in transform order):
    camera  -> end-effector (EE / tool) -> robot base -> world -> other robot base

Primitive transforms:
    camera_to_ee()   -- hand-eye calibration (T_cam_to_ee)
    ee_to_base()     -- forward kinematics from TCP pose
    base_to_ee()     -- inverse of ee_to_base
    base_to_world()  -- per-arm static transform (+ linear rail offset)
    world_to_base()  -- inverse of base_to_world

Composite transforms:
    base_to_base()          -- from_arm base -> world -> to_arm base
    camera_to_base()        -- camera -> EE -> same arm base
    camera_to_world()       -- camera -> EE -> base -> world
    camera_to_other_base()  -- camera -> EE -> from_arm base -> world -> to_arm base

Config:
    configs/arm_transforms.json -- per-arm T_base_to_world (4x4) and rail_axis.
    Falls back to identity when missing (single-arm backward compat).
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aira.utils.math import pose_to_matrix
from aira.utils.paths import get_project_root

_BASE = get_project_root()

# ---------------------------------------------------------------------------
# Config loading (cached)
# ---------------------------------------------------------------------------

_arm_transforms: Optional[Dict[str, Dict[str, Any]]] = None
_transforms_lock = threading.Lock()

_IDENTITY_4x4 = np.eye(4, dtype=np.float64)


def load_arm_transforms() -> Dict[str, Dict[str, Any]]:
    """Load configs/arm_transforms.json and return {arm_name: {T_base_to_world, rail_axis}}.
    Cached after first call.  Missing file -> empty dict (identity fallback)."""
    global _arm_transforms
    if _arm_transforms is not None:
        return _arm_transforms
    with _transforms_lock:
        if _arm_transforms is not None:
            return _arm_transforms
        path = _BASE / "configs" / "arm_transforms.json"
        result: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                for name, entry in raw.items():
                    T = np.array(entry["T_base_to_world"], dtype=np.float64)
                    rail = entry.get("rail_axis")
                    rail_arr = np.array(rail, dtype=np.float64) if rail is not None else None
                    result[name] = {"T_base_to_world": T, "rail_axis": rail_arr}
            except Exception:
                pass
        _arm_transforms = result
        return _arm_transforms


def reset_arm_transforms() -> None:
    """Clear cached transforms (e.g. after re-calibration)."""
    global _arm_transforms
    with _transforms_lock:
        _arm_transforms = None


def _get_T_base_to_world(arm_name: str, rail_position: Optional[float] = None) -> np.ndarray:
    """Return 4x4 T_base_to_world for arm_name. Identity if unconfigured.
    For rail arms, translates by rail_axis * rail_position."""
    transforms = load_arm_transforms()
    entry = transforms.get(arm_name)
    if entry is None:
        return _IDENTITY_4x4.copy()
    T = entry["T_base_to_world"].copy()
    rail_axis = entry.get("rail_axis")
    if rail_axis is not None and rail_position is not None:
        T[:3, 3] += rail_axis * rail_position
    return T


# ---------------------------------------------------------------------------
# Primitive transforms
# ---------------------------------------------------------------------------

def camera_to_ee(
    p_cam: np.ndarray,
    T_cam_to_ee: np.ndarray,
    tare: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Transform a 3D point from camera frame to end-effector (tool) frame.

    Args:
        p_cam: [x, y, z] in camera frame (mm).
        T_cam_to_ee: 4x4 hand-eye calibration matrix.
        tare: (dx, dy, dz) correction offset applied in EE frame.

    Returns:
        [x, y, z] in end-effector frame (mm).
    """
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(p_cam, dtype=np.float64).ravel()[:3]
    p_ee = (T_cam_to_ee @ p)[:3]
    p_ee += np.array(tare, dtype=np.float64)
    return p_ee


def ee_to_base(
    p_ee: np.ndarray,
    ee_pose: List[float],
) -> np.ndarray:
    """Transform from end-effector frame to robot base frame.

    Args:
        p_ee: [x, y, z] in end-effector frame (mm).
        ee_pose: [x, y, z, roll, pitch, yaw] current TCP pose in base frame (mm/deg).

    Returns:
        [x, y, z] in robot base frame (mm).
    """
    T_base_to_ee = pose_to_matrix(ee_pose)
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(p_ee, dtype=np.float64).ravel()[:3]
    return (T_base_to_ee @ p)[:3]


def base_to_ee(
    p_base: np.ndarray,
    ee_pose: List[float],
) -> np.ndarray:
    """Transform from robot base frame to end-effector frame (inverse of ee_to_base).

    Args:
        p_base: [x, y, z] in base frame (mm).
        ee_pose: [x, y, z, roll, pitch, yaw] current TCP pose in base frame (mm/deg).

    Returns:
        [x, y, z] in end-effector frame (mm).
    """
    T_base_to_ee = pose_to_matrix(ee_pose)
    T_ee_to_base = np.linalg.inv(T_base_to_ee)
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(p_base, dtype=np.float64).ravel()[:3]
    return (T_ee_to_base @ p)[:3]


def base_to_world(
    p_base: np.ndarray,
    arm_name: str,
    rail_position: Optional[float] = None,
) -> np.ndarray:
    """Transform from robot base frame to shared world frame.

    For arms on a linear rail, ``rail_position`` (mm along rail) shifts the
    base-to-world origin by ``rail_axis * rail_position``.

    Falls back to identity when no transform is configured (single-arm compat).
    """
    T = _get_T_base_to_world(arm_name, rail_position)
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(p_base, dtype=np.float64).ravel()[:3]
    return (T @ p)[:3]


def world_to_base(
    p_world: np.ndarray,
    arm_name: str,
    rail_position: Optional[float] = None,
) -> np.ndarray:
    """Transform from shared world frame to robot base frame (inverse of base_to_world)."""
    T = _get_T_base_to_world(arm_name, rail_position)
    T_inv = np.linalg.inv(T)
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(p_world, dtype=np.float64).ravel()[:3]
    return (T_inv @ p)[:3]


# ---------------------------------------------------------------------------
# Composite transforms
# ---------------------------------------------------------------------------

def base_to_base(
    p: np.ndarray,
    from_arm: str,
    to_arm: str,
    from_rail_pos: Optional[float] = None,
    to_rail_pos: Optional[float] = None,
) -> np.ndarray:
    """Convert a point from one arm's base frame to another's.

    Pipeline: from_base -> world -> to_base.
    """
    p_world = base_to_world(p, from_arm, from_rail_pos)
    return world_to_base(p_world, to_arm, to_rail_pos)


def camera_to_base(
    p_cam: np.ndarray,
    arm_name: str,
    ee_pose: List[float],
    T_cam_to_ee: np.ndarray,
    tare: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Full chain: camera -> end-effector -> base frame of the same arm."""
    p_ee = camera_to_ee(p_cam, T_cam_to_ee, tare)
    return ee_to_base(p_ee, ee_pose)


def camera_to_world(
    p_cam: np.ndarray,
    arm_name: str,
    ee_pose: List[float],
    T_cam_to_ee: np.ndarray,
    tare: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rail_position: Optional[float] = None,
) -> np.ndarray:
    """Full chain: camera -> end-effector -> base -> world."""
    p_base = camera_to_base(p_cam, arm_name, ee_pose, T_cam_to_ee, tare)
    return base_to_world(p_base, arm_name, rail_position)


def camera_to_other_base(
    p_cam: np.ndarray,
    from_arm: str,
    to_arm: str,
    ee_pose: List[float],
    T_cam_to_ee: np.ndarray,
    tare: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    from_rail_pos: Optional[float] = None,
    to_rail_pos: Optional[float] = None,
) -> np.ndarray:
    """Cross-arm pipeline: camera -> EE -> from_arm base -> world -> to_arm base.

    This is the key function for scenarios like
    "right arm camera sees object, left arm moves there".
    """
    p_world = camera_to_world(
        p_cam, from_arm, ee_pose, T_cam_to_ee, tare, from_rail_pos,
    )
    return world_to_base(p_world, to_arm, to_rail_pos)
