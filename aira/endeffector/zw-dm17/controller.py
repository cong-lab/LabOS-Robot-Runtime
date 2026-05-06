"""
High-level EndEffector implementation for the ZWHAND DM17-V6 hand
connected via the xArm tool-end RS485 bus.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aira.endeffector import EndEffector
from .vendor_api import NUM_JOINTS, _DEFAULT_BAUD
from .xarm_api import ZWDM17XArmHand

logger = logging.getLogger(__name__)


class ZWDM17XArmController(EndEffector):
    """
    EndEffector wrapper around :class:`ZWDM17XArmHand`.

    Intended usage::

        ctrl = ZWDM17XArmController(device_id=1, baudrate=115200)
        ctrl.connect(arm)          # arm is an xArm API instance
        ctrl.calibrate()
        ctrl.set_all_absolute([500] * 17)
        state = ctrl.state_dict()  # snapshot for recording
        ctrl.stop()
        ctrl.disconnect()
    """

    def __init__(
        self,
        device_id: int = 1,
        baudrate: int = _DEFAULT_BAUD,
    ) -> None:
        self._device_id = device_id
        self._baudrate = baudrate
        self._api: Optional[ZWDM17XArmHand] = None

    # ------------------------------------------------------------------
    # EndEffector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "zw-dm17"

    @property
    def num_joints(self) -> int:
        return NUM_JOINTS

    def connect(self, arm: Any = None, **kwargs: Any) -> bool:  # type: ignore[override]
        """
        Initialise the hand via the xArm's tool-end RS485.

        Parameters
        ----------
        arm : XArmAPI
            A connected xArm SDK instance whose tool RS485 is wired to the
            DM17 hand.
        """
        if arm is None:
            raise TypeError("connect() requires an xArm API instance as the first argument")
        try:
            self._api = ZWDM17XArmHand(
                arm,
                device_id=self._device_id,
                baudrate=self._baudrate,
            )
        except Exception:
            logger.exception("Failed to create ZWDM17XArmHand")
            return False

        return self.init()

    def init(self, retries: int = 10, retry_delay: float = 1.0) -> bool:
        """
        Verify the hand is communicating, clear errors, and log diagnostics.

        Called automatically by :meth:`connect` but can be re-run at any
        time to re-check the hand.

        Parameters
        ----------
        retries : int
            Number of attempts to read init_state before giving up.
        retry_delay : float
            Seconds to wait between retries.
        """
        import time

        if self._api is None:
            logger.error("init() called but no API instance — call connect() first")
            return False

        init_state: Any = False
        for attempt in range(1, retries + 1):
            init_state = self._api.get_init_state()
            if init_state:
                break
            if attempt < retries:
                logger.info(
                    "Waiting for hand (attempt %d/%d, init_state=%s) …",
                    attempt, retries, init_state,
                )
                time.sleep(retry_delay)

        if not init_state:
            logger.error(
                "Hand did not report ready after %d attempts (init_state=%s)",
                retries, init_state,
            )
            return False
        logger.info("Hand init_state OK (%s) after %d attempt(s)", init_state, attempt)

        errors = self._api.get_errors()
        if errors is False:
            logger.warning("Failed to read error registers — continuing")
        elif any(e != 0 for e in errors):
            active = {i: v for i, v in enumerate(errors) if v != 0}
            logger.warning(
                "Hand has active errors (reg:code): %s — attempting clear",
                active,
            )
            if self._api.clear_errors():
                errors = self._api.get_errors()
                if errors is not False and any(e != 0 for e in errors):
                    active = {i: v for i, v in enumerate(errors) if v != 0}
                    logger.warning(
                        "Errors persist after clear (reg:code): %s  "
                        "— may resolve after calibration", active,
                    )
                else:
                    logger.info("Errors cleared successfully")
            else:
                logger.warning("clear_errors command failed — continuing")
        else:
            logger.info("No active errors")

        voltage = self._api.get_voltage()
        if voltage is not None:
            logger.info("Supply voltage: %.3f V", voltage)
        else:
            logger.warning("Could not read supply voltage")

        bl = self._api.get_bootloader_version()
        hw = self._api.get_hardware_version()
        sw = self._api.get_software_version()
        logger.info(
            "Firmware — bootloader=%s, hardware=%s, software=%s",
            bl if bl is not False else "?",
            hw if hw is not False else "?",
            sw if sw is not False else "?",
        )

        logger.info(
            "ZWDM17 hand ready (id=%d, baud=%d)",
            self._device_id, self._baudrate,
        )
        return True

    def disconnect(self) -> bool:
        self._api = None
        return True

    def calibrate(self) -> bool:
        """Zero-position calibration for all 17 joints (~13 s)."""
        if self._api is None:
            logger.error("Not connected")
            return False
        return self._api.calibrate_all()

    def stop(self) -> bool:
        """Emergency-stop all joint motors."""
        if self._api is None:
            logger.error("Not connected")
            return False
        return self._api.stop_all()

    def state_dict(self) -> Dict[str, Any]:
        """
        Snapshot current joint angles for recording.

        Returns a dict like::

            {"type": "zw-dm17", "angles": [0, 0, ..., 0]}
        """
        if self._api is None:
            raise RuntimeError("Not connected — cannot read state")
        angles = self._api.get_angles()
        if angles is False:
            raise RuntimeError("Failed to read joint angles from hand")
        return {"type": self.name, "angles": list(angles)}

    def load_state_dict(self, state: Dict[str, Any]) -> bool:
        """
        Restore the hand to a previously recorded state.

        The *state* dict must contain an ``"angles"`` key with a 17-element
        list of integer angle steps (0-1000).
        """
        if self._api is None:
            logger.error("Not connected")
            return False
        angles = state.get("angles")
        if angles is None or len(angles) != NUM_JOINTS:
            logger.error("Invalid state dict: expected 'angles' with %d entries", NUM_JOINTS)
            return False
        return self._api.set_all_absolute([int(a) for a in angles])

    # ------------------------------------------------------------------
    # Visual editing
    # ------------------------------------------------------------------

    def visual_edit(
        self,
        start_angles: Optional[List[int]] = None,
        port: int = 8080,
        urdf_path: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Open a viser-based 3D visual editor for the DM17 hand.

        Uses the connected hand for live read/send when available.
        Returns a ``state_dict``-compatible dict on save, or ``None`` on cancel.
        """
        from .visual import dm17_visual_edit

        angles = dm17_visual_edit(
            ee_ctrl=self if self._api is not None else None,
            start_angles=start_angles,
            port=port,
            urdf_path=urdf_path,
        )
        if angles is None:
            return None
        return {"type": self.name, "angles": angles}

    # ------------------------------------------------------------------
    # Direct access
    # ------------------------------------------------------------------

    @property
    def api(self) -> ZWDM17XArmHand:
        """Direct access to the low-level driver for advanced operations."""
        if self._api is None:
            raise RuntimeError("Not connected")
        return self._api

    # ------------------------------------------------------------------
    # Device configuration pass-throughs
    # ------------------------------------------------------------------

    def set_id(self, new_id: int) -> bool:
        return self.api.set_id(new_id)

    def set_baudrate(self, level: int) -> bool:
        return self.api.set_baudrate(level)

    def clear_errors(self) -> bool:
        return self.api.clear_errors()

    def save_config(self) -> bool:
        return self.api.save_config()

    def save_params(self) -> bool:
        return self.api.save_params()

    def factory_reset(self) -> bool:
        return self.api.factory_reset()

    # ------------------------------------------------------------------
    # Speed & current pass-throughs
    # ------------------------------------------------------------------

    def set_speed(self, joint: int, speed: int) -> bool:
        return self.api.set_speed(joint, speed)

    def set_all_speeds(self, speed: int) -> bool:
        return self.api.set_all_speeds(speed)

    def set_current(self, joint: int, current: int) -> bool:
        return self.api.set_current(joint, current)

    def set_all_currents(self, current: int) -> bool:
        return self.api.set_all_currents(current)

    # ------------------------------------------------------------------
    # Emergency stop pass-throughs
    # ------------------------------------------------------------------

    def stop_joint(self, joint: int) -> bool:
        """Emergency-stop a single joint motor."""
        return self.api.stop(joint)

    def stop_all(self) -> bool:
        """Emergency-stop all 17 joint motors."""
        return self.api.stop_all()

    # ------------------------------------------------------------------
    # Position control pass-throughs
    # ------------------------------------------------------------------

    def set_absolute(self, joint: int, position: int) -> bool:
        return self.api.set_absolute(joint, position)

    def set_all_absolute(self, positions: list[int]) -> bool:
        return self.api.set_all_absolute(positions)

    def set_relative(self, joint: int, offset: int) -> bool:
        return self.api.set_relative(joint, offset)

    def set_all_relative(self, offsets: list[int]) -> bool:
        return self.api.set_all_relative(offsets)

    # ------------------------------------------------------------------
    # Calibration pass-throughs
    # ------------------------------------------------------------------

    def calibrate_joint(self, joint: int) -> bool:
        """Zero-position calibration for a single joint (~1 s)."""
        return self.api.calibrate(joint)

    def calibrate_all_steppers(self) -> bool:
        """Zero-position calibration for stepper joints only (~1 s)."""
        return self.api.calibrate_all_steppers()

    # ------------------------------------------------------------------
    # Telemetry pass-throughs
    # ------------------------------------------------------------------

    def get_init_state(self) -> Union[int, bool]:
        return self.api.get_init_state()

    def get_angles(self) -> Union[List[int], bool]:
        return self.api.get_angles()

    def get_errors(self) -> Union[List[int], bool]:
        return self.api.get_errors()

    def get_voltage(self) -> Optional[float]:
        return self.api.get_voltage()

    def get_stall_states(self) -> Union[List[int], bool]:
        return self.api.get_stall_states()

    def get_fingertip_torques(self) -> Union[List[int], bool]:
        return self.api.get_fingertip_torques()

    def get_moving_range(self) -> Union[List[int], bool]:
        return self.api.get_moving_range()

    def get_bootloader_version(self) -> Union[int, bool]:
        return self.api.get_bootloader_version()

    def get_hardware_version(self) -> Union[int, bool]:
        return self.api.get_hardware_version()

    def get_software_version(self) -> Union[int, bool]:
        return self.api.get_software_version()

    # ------------------------------------------------------------------
    # Vendor-compatible aliases
    # ------------------------------------------------------------------

    set_error_clear = clear_errors
    set_single_motor_speed = set_speed
    set_all_motor_speed = set_all_speeds
    set_single_motor_current = set_current
    set_all_motor_current = set_all_currents
    set_single_motor_stop = stop_joint
    set_all_motor_stop = stop_all
    set_single_motor_absolute = set_absolute
    set_all_motor_absolute = set_all_absolute
    set_single_motor_relative = set_relative
    set_all_motor_relative = set_all_relative
    set_single_motor_calibration = calibrate_joint
    set_all_motor_calibration = calibrate
    set_all_step_motor_calibration = calibrate_all_steppers
    get_initialize_state = get_init_state
    get_device_error = get_errors
    get_device_voltage = get_voltage
    get_motor_locked_state = get_stall_states
    get_motor_real_angle = get_angles
    get_fingertip_skin_moment = get_fingertip_torques
