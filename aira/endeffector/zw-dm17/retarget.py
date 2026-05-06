"""
PyBullet-based retargeting from MediaPipe hand landmarks to ZWHAND DM17
motor positions.

Loads the vendor URDF in headless (DIRECT) mode, then for each finger
solves a bounded least-squares optimisation (scipy L-BFGS-B) that
minimises the distance between FK link positions and the transformed
MediaPipe target positions.  Joint angles are then mapped to the
0-1000 motor command range.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
import pybullet as p

logger = logging.getLogger(__name__)

_DEFAULT_URDF = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "deps" / "ZWHAND-DM17" / "URDF" / "urdf" / "src"
    / "zwhand_17dof_left" / "urdf" / "zwhand_17dof_left.urdf"
)

_DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "configs" / "mediapipe_calibration.json"
)

# ── MediaPipe landmark indices ────────────────────────────────────────
WRIST = 0
TH_CMC, TH_MCP, TH_IP, TH_TIP = 1, 2, 3, 4
IX_MCP, IX_PIP, IX_DIP, IX_TIP = 5, 6, 7, 8
MD_MCP, MD_PIP, MD_DIP, MD_TIP = 9, 10, 11, 12
RG_MCP, RG_PIP, RG_DIP, RG_TIP = 13, 14, 15, 16
PK_MCP, PK_PIP, PK_DIP, PK_TIP = 17, 18, 19, 20

# ── Per-finger chain definitions ──────────────────────────────────────
# Each entry: (pb_joint_indices, [(pb_link, mp_landmark), ...])
# Two targets per finger: fingertip + one intermediate joint.
_FINGER_CHAINS = [
    # Thumb: pb joints [0,1,2,3] → targets tip (link 3) + IP (link 2)
    ([0, 1, 2, 3], [(3, TH_TIP), (2, TH_IP)]),
    # Index: pb joints [4,5,6,7] → targets tip (link 7) + DIP (link 6)
    ([4, 5, 6, 7], [(7, IX_TIP), (6, IX_DIP)]),
    # Middle: pb joints [8,9,10] → targets tip (link 10) + DIP (link 9)
    ([8, 9, 10], [(10, MD_TIP), (9, MD_DIP)]),
    # Ring: pb joints [11,12,13] → targets tip (link 13) + DIP (link 12)
    ([11, 12, 13], [(13, RG_TIP), (12, RG_DIP)]),
    # Little: pb joints [14,15,16] → targets tip (link 16) + DIP (link 15)
    ([14, 15, 16], [(16, PK_TIP), (15, PK_DIP)]),
]


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


class DM17Retargeter:
    """
    Solves MediaPipe-to-DM17 retargeting via per-finger bounded
    least-squares FK optimisation in PyBullet.

    Usage::

        rt = DM17Retargeter()
        motor_positions = rt.retarget(world_landmarks)  # 17 ints, 0-1000
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
    ) -> None:
        self._cid = p.connect(p.DIRECT)

        urdf = Path(urdf_path) if urdf_path else _DEFAULT_URDF
        if not urdf.exists():
            raise FileNotFoundError(f"URDF not found: {urdf}")

        fixed_urdf = self._fix_urdf(urdf)
        try:
            self._hand = p.loadURDF(
                fixed_urdf, useFixedBase=True, physicsClientId=self._cid,
            )
        finally:
            os.unlink(fixed_urdf)

        self._nj = p.getNumJoints(self._hand, physicsClientId=self._cid)
        assert self._nj == 17

        self._lower: list[float] = []
        self._upper: list[float] = []
        self._pb_to_dm17: list[int] = []

        for i in range(self._nj):
            info = p.getJointInfo(self._hand, i, physicsClientId=self._cid)
            name = info[1].decode()
            dm17_idx = int(re.search(r"\d+", name).group())
            self._pb_to_dm17.append(dm17_idx)
            self._lower.append(info[8])
            self._upper.append(info[9])

        self._prev_angles = np.zeros(self._nj)

        self._finger_bounds = []
        for joints, _ in _FINGER_CHAINS:
            self._finger_bounds.append(
                [(self._lower[j], self._upper[j]) for j in joints]
            )

        self._urdf_frame, self._urdf_scale = self._compute_urdf_reference()

        self._cal_min = list(self._lower)
        self._cal_max = list(self._upper)
        self._load_calibration(calibration_path)

        logger.info(
            "DM17Retargeter ready (%d joints, scale=%.4fm)",
            self._nj, self._urdf_scale,
        )

    # ── URDF preprocessing ────────────────────────────────────────────

    @staticmethod
    def _fix_urdf(urdf_path: Path) -> str:
        pkg_dir = urdf_path.parent.parent
        text = urdf_path.read_text()
        text = text.replace(
            "package://zwhand_17dof_left/",
            str(pkg_dir) + "/",
        )
        fd, tmp = tempfile.mkstemp(suffix=".urdf")
        os.write(fd, text.encode())
        os.close(fd)
        return tmp

    # ── Calibration loading ─────────────────────────────────────────────

    def _load_calibration(self, path: Optional[str]) -> None:
        """
        Load per-joint observed min/max from a calibration JSON.

        Falls back to the default path, and silently skips if the file
        does not exist (URDF limits are used instead).
        """
        cal_path = Path(path) if path else _DEFAULT_CALIBRATION
        if not cal_path.exists():
            logger.debug("No calibration file at %s — using URDF limits", cal_path)
            return

        try:
            with open(cal_path) as f:
                data = json.load(f)
            ranges = data["joint_ranges"]
            obs_min = ranges["observed_min"]
            obs_max = ranges["observed_max"]
            if len(obs_min) != self._nj or len(obs_max) != self._nj:
                raise ValueError(
                    f"Expected {self._nj} joints, got "
                    f"min={len(obs_min)}, max={len(obs_max)}"
                )
            for i in range(self._nj):
                if obs_max[i] - obs_min[i] > 1e-4:
                    self._cal_min[i] = obs_min[i]
                    self._cal_max[i] = obs_max[i]
            logger.info("Loaded calibration from %s", cal_path)
        except Exception:
            logger.warning(
                "Failed to load calibration from %s — using URDF limits",
                cal_path, exc_info=True,
            )

    # ── Reference frame from URDF rest pose ───────────────────────────

    def _compute_urdf_reference(self) -> tuple[np.ndarray, float]:
        for i in range(self._nj):
            p.resetJointState(self._hand, i, 0.0, physicsClientId=self._cid)

        def _link_pos(link_idx: int) -> np.ndarray:
            return np.array(
                p.getLinkState(self._hand, link_idx, physicsClientId=self._cid)[0]
            )

        # Index MCP (pb link 4), Middle MCP (pb link 8), Pinky MCP (pb link 14)
        idx_mcp = _link_pos(4)
        mid_mcp = _link_pos(8)
        pky_mcp = _link_pos(14)

        wrist = np.zeros(3)
        across = _normalize(pky_mcp - idx_mcp)
        mid = 0.5 * (idx_mcp + pky_mcp)
        forward = _normalize(mid - wrist)
        normal = _normalize(np.cross(across, forward))
        forward = _normalize(np.cross(normal, across))

        R = np.column_stack([across, forward, normal])
        scale = float(np.linalg.norm(mid_mcp - wrist))
        return R, scale

    # ── Coordinate alignment ──────────────────────────────────────────

    @staticmethod
    def _mp_palm_frame(
        pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        wrist = pts[WRIST]
        idx = pts[IX_MCP]
        pky = pts[PK_MCP]
        mid = pts[MD_MCP]

        across = _normalize(pky - idx)
        center = 0.5 * (idx + pky)
        forward = _normalize(center - wrist)
        normal = _normalize(np.cross(across, forward))
        forward = _normalize(np.cross(normal, across))

        R = np.column_stack([across, forward, normal])
        scale = float(np.linalg.norm(mid - wrist))
        return R, wrist, max(scale, 1e-6)

    def _transform_landmarks(self, pts: np.ndarray) -> np.ndarray:
        R_mp, wrist_mp, scale_mp = self._mp_palm_frame(pts)
        scale = self._urdf_scale / scale_mp
        R = self._urdf_frame @ np.linalg.inv(R_mp)
        centered = pts - wrist_mp
        return (R @ centered.T).T * scale

    # ── Per-finger FK optimisation ────────────────────────────────────

    def _solve_finger(
        self,
        joints: list[int],
        targets: list[tuple[int, np.ndarray]],
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Find joint angles for one finger chain that minimise the sum of
        squared distances between FK link positions and target positions.
        """
        cid = self._cid
        hand = self._hand
        x0 = np.array([self._prev_angles[j] for j in joints])
        target_links = [link for link, _ in targets]
        target_pos = [pos for _, pos in targets]

        def cost(x: np.ndarray) -> float:
            for j_idx, j in enumerate(joints):
                p.resetJointState(hand, j, float(x[j_idx]), physicsClientId=cid)
            err = 0.0
            for link, tgt in zip(target_links, target_pos):
                fk = np.array(p.getLinkState(hand, link, physicsClientId=cid)[0])
                err += np.sum((fk - tgt) ** 2)
            return float(err)

        res = minimize(
            cost, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 80, "ftol": 1e-12},
        )
        return res.x

    # ── Public API ────────────────────────────────────────────────────

    def retarget(self, world_landmarks: Sequence) -> List[int]:
        """
        Retarget MediaPipe hand world landmarks to DM17 motor positions.

        Parameters
        ----------
        world_landmarks
            The 21-element ``hand_world_landmarks`` from MediaPipe
            (each has ``.x .y .z`` in metres), or an (21, 3) array.

        Returns
        -------
        list[int]
            17 motor positions (0-1000).  ``positions[0]`` = DM17 motor 1.
        """
        if hasattr(world_landmarks[0], "x"):
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in world_landmarks],
                dtype=np.float64,
            )
        else:
            pts = np.asarray(world_landmarks, dtype=np.float64)

        transformed = self._transform_landmarks(pts)

        angles = np.array(self._prev_angles, copy=True)

        for chain_idx, (joints, link_mp_pairs) in enumerate(_FINGER_CHAINS):
            targets = [
                (link, transformed[mp_idx]) for link, mp_idx in link_mp_pairs
            ]
            result = self._solve_finger(
                joints, targets, self._finger_bounds[chain_idx],
            )
            for j_idx, j in enumerate(joints):
                angles[j] = result[j_idx]
                p.resetJointState(
                    self._hand, j, float(result[j_idx]),
                    physicsClientId=self._cid,
                )

        self._prev_angles = angles
        return self._rad_to_motor(angles)

    # ── Conversion ────────────────────────────────────────────────────

    def _rad_to_motor(self, pb_angles: np.ndarray) -> list[int]:
        positions = [0] * 17
        for pb_idx in range(self._nj):
            dm17 = self._pb_to_dm17[pb_idx]
            lo = self._cal_min[pb_idx]
            hi = self._cal_max[pb_idx]
            rng = hi - lo
            frac = (pb_angles[pb_idx] - lo) / rng if rng > 1e-8 else 0.0
            positions[dm17 - 1] = max(0, min(1000, int(round(frac * 1000))))
        return positions

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        if self._cid >= 0:
            p.disconnect(self._cid)
            self._cid = -1

    def __del__(self) -> None:
        self.close()
