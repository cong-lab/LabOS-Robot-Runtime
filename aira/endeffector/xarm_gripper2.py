"""
EndEffector implementation for the xArm built-in parallel gripper.

Wraps the xArm SDK gripper API into the standard EndEffector interface so
that gripper position can be recorded, replayed, and used in protocols
just like any other end-effector.

State dict format::

    {"type": "xarm-gripper2", "position": 450.0}

Position is a float in the range [0, 850] where 0 = fully closed and
850 = fully open (exact max depends on the gripper model).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aira.endeffector import EndEffector

logger = logging.getLogger(__name__)

DEFAULT_SPEED = 5000
DEFAULT_MAX_POS = 850.0


class XArmGripper2(EndEffector):
    """EndEffector wrapper for the xArm built-in gripper."""

    def __init__(self, speed: int = DEFAULT_SPEED) -> None:
        self._arm: Any = None
        self._speed = speed

    @property
    def name(self) -> str:
        return "xarm-gripper2"

    @property
    def num_joints(self) -> int:
        return 1

    def connect(self, arm: Any = None, **kwargs: Any) -> bool:  # type: ignore[override]
        if arm is None:
            raise TypeError("connect() requires an xArm API instance")
        self._arm = arm
        try:
            self._arm.set_gripper_mode(0)
            self._arm.set_gripper_enable(True)
            self._arm.set_gripper_speed(self._speed)
            logger.info("xArm gripper enabled (speed=%d)", self._speed)
            return True
        except Exception:
            logger.exception("Failed to enable xArm gripper")
            return False

    def disconnect(self) -> bool:
        self._arm = None
        return True

    def calibrate(self) -> bool:
        if self._arm is None:
            return False
        try:
            self._arm.set_gripper_position(DEFAULT_MAX_POS, wait=True)
            self._arm.set_gripper_position(0, wait=True)
            self._arm.set_gripper_position(DEFAULT_MAX_POS, wait=True)
            logger.info("Gripper calibration cycle complete")
            return True
        except Exception:
            logger.exception("Gripper calibration failed")
            return False

    def stop(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        if self._arm is None:
            raise RuntimeError("Not connected")
        ret = self._arm.get_gripper_position()
        if isinstance(ret, (list, tuple)) and len(ret) >= 2:
            code, pos = int(ret[0]), ret[1]
        else:
            raise RuntimeError(f"Unexpected get_gripper_position return: {ret}")
        if code != 0 or pos is None:
            raise RuntimeError(f"Failed to read gripper position (code={code})")
        return {"type": self.name, "position": round(float(pos), 1)}

    def load_state_dict(self, state: Dict[str, Any]) -> bool:
        if self._arm is None:
            logger.error("Not connected")
            return False
        pos = state.get("position")
        if pos is None:
            logger.error("state_dict missing 'position' key")
            return False
        pos = float(pos)
        try:
            self._arm.set_gripper_position(pos, wait=True, speed=self._speed)
            return True
        except Exception:
            logger.exception("Failed to set gripper position")
            return False

    # ------------------------------------------------------------------
    # Direct access
    # ------------------------------------------------------------------

    def get_position(self) -> float:
        """Current gripper position (0 = closed, ~850 = open)."""
        if self._arm is None:
            raise RuntimeError("Not connected")
        ret = self._arm.get_gripper_position()
        if isinstance(ret, (list, tuple)) and len(ret) >= 2 and int(ret[0]) == 0:
            return float(ret[1])
        raise RuntimeError(f"Failed to read gripper position: {ret}")

    def set_position(self, pos: float, wait: bool = True) -> bool:
        """Set gripper position (0 = closed, ~850 = open)."""
        if self._arm is None:
            return False
        try:
            self._arm.set_gripper_position(pos, wait=wait, speed=self._speed)
            return True
        except Exception:
            logger.exception("Failed to set gripper position")
            return False
