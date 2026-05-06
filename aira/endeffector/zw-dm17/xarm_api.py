"""
ZWHAND DM17-V6 driver routed through the xArm tool-end RS485 controller.

Same public interface as :class:`vendor_api.ZWDM17Hand` but all Modbus
frames are sent/received via ``arm.getset_tgpio_modbus_data(host_id=9)``.

Uses **transparent transmission mode** so that raw Modbus RTU frames
(including CRC) are passed directly to/from the RS485 bus.  This is
required because the xArm's normal Modbus mode silently drops responses
that use function code 0x04 (Read Input Registers) — the code the
ZWHAND uses for all status reads.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Union

from .vendor_api import (
    NUM_JOINTS,
    BAUD_LEVELS,
    _ADDR_ABSOLUTE,
    _ADDR_ALL_CAL,
    _ADDR_ALL_STEP_CAL,
    _ADDR_ANGLE,
    _ADDR_BOOTLOADER_VER,
    _ADDR_CLEAR_ERROR,
    _ADDR_CURRENT,
    _ADDR_ERROR,
    _ADDR_FACTORY_RESET,
    _ADDR_FINGERTIP,
    _ADDR_HW_VER,
    _ADDR_INIT_STATE,
    _ADDR_MOVING_RANGE,
    _ADDR_POWER_OFF_SAVE,
    _ADDR_RELATIVE,
    _ADDR_SET_BAUD,
    _ADDR_SET_ID,
    _ADDR_SINGLE_CAL,
    _ADDR_SPEED,
    _ADDR_STALL,
    _ADDR_STOP,
    _ADDR_SW_VER,
    _ADDR_VOLTAGE,
    _DEFAULT_BAUD,
)

logger = logging.getLogger(__name__)

TGPIO_HOST_ID = 9


def _modbus_crc(data: list[int]) -> list[int]:
    """Standard Modbus CRC-16 (polynomial 0xA001). Returns [crc_lo, crc_hi]."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    return [crc & 0xFF, (crc >> 8) & 0xFF]


class ZWDM17XArmHand:
    """
    Driver for one ZWHAND DM17-V6 hand wired to the xArm tool-end RS485.

    Instead of opening a serial port directly, all Modbus RTU traffic is
    tunnelled through the xArm controller using
    ``arm.getset_tgpio_modbus_data``.
    """

    def __init__(
        self,
        arm: Any,
        device_id: int = 1,
        baudrate: int = _DEFAULT_BAUD,
        modbus_timeout_ms: int = 500,
    ) -> None:
        if not 1 <= device_id <= 255:
            raise ValueError(f"device_id must be 1-255, got {device_id}")

        self._arm = arm
        self._id = device_id
        self._baudrate = baudrate

        code = self._arm.set_tgpio_modbus_baudrate(baudrate)
        if code != 0:
            raise RuntimeError(
                f"Failed to set tool RS485 baudrate to {baudrate} (code={code})"
            )
        logger.info("Tool RS485 baudrate set to %d", baudrate)

        time.sleep(2)

        code = self._arm.set_tgpio_modbus_timeout(
            modbus_timeout_ms, is_transparent_transmission=True,
        )
        if code != 0:
            logger.warning(
                "set_tgpio_modbus_timeout (transparent) returned code=%d – "
                "continuing", code,
            )

    # ------------------------------------------------------------------
    # Low-level Modbus-over-xArm helpers
    # ------------------------------------------------------------------

    def _send(
        self,
        frame: list[int],
        min_res_len: int,
    ) -> tuple[int, list[int]]:
        """
        Send a Modbus RTU frame through the xArm tool RS485 in
        **transparent transmission** mode.

        Calls ``arm_cmd.tgpio_set_modbus`` directly, bypassing the SDK's
        ``_check_modbus_code`` which can latch ControllerError 19 and
        cascade failures to all subsequent calls.

        CRC is appended before sending and stripped from the response,
        so callers work with plain Modbus PDUs — identical to normal mode.

        Returns ``(xarm_code, response_pdu)`` where *response_pdu* is the
        raw Modbus response bytes (no CRC).
        """
        if not self._arm.connected:
            logger.error("_send: arm is not connected")
            return -1, []

        frame_with_crc = list(frame) + _modbus_crc(frame)

        ret = self._arm.core.tgpio_set_modbus(
            frame_with_crc,
            len(frame_with_crc),
            host_id=TGPIO_HOST_ID,
            is_transparent_transmission=True,
        )

        code = ret[0]
        if code != 0:
            logger.warning("_send: transport error code=%d", code)
            return code, []

        res = list(ret[2:])
        if len(res) >= 2:
            res = res[:-2]
        return 0, res

    @staticmethod
    def _parse_registers(response: list[int], header_len: int) -> Union[int, List[int], bool]:
        """
        Extract 16-bit register values from a Modbus response PDU.

        For a function-0x04 read response the layout is:
            [slave, 0x04, byte_count, data_hi, data_lo, ...]
        *header_len* is the number of bytes before the register data (3).

        For a function-0x10 write echo the layout is:
            [slave, 0x10, addr_hi, addr_lo, count_hi, count_lo]
        which carries no payload — return True.
        """
        payload = response[header_len:]
        if not payload:
            return True

        values: list[int] = []
        for i in range(0, len(payload) - 1, 2):
            values.append((payload[i] << 8) | payload[i + 1])
        return values[0] if len(values) == 1 else values

    def _write_single(self, address: int, value: int) -> Union[int, List[int], bool]:
        """Write one 16-bit register (Modbus function 0x10, 1 register)."""
        hi, lo = value.to_bytes(2, "big", signed=True)
        frame = [self._id, 0x10, 0x00, address, 0x00, 0x01, 0x02, hi, lo]

        code, res = self._send(frame, min_res_len=6)
        if code != 0:
            logger.error(
                "Write single failed (addr=0x%02X, code=%d)", address, code
            )
            return False

        if len(res) < 6:
            logger.error("Write single: response too short (%d bytes)", len(res))
            return False

        for i in range(min(6, len(res))):
            if res[i] != frame[i]:
                logger.error(
                    "Write single: echo mismatch at byte %d "
                    "(expected 0x%02X, got 0x%02X)",
                    i, frame[i], res[i],
                )
                return False

        return self._parse_registers(res, 6)

    def _write_multiple(
        self,
        start_address: int,
        values: list[int],
    ) -> Union[int, List[int], bool]:
        """Write N consecutive 16-bit registers (Modbus function 0x10)."""
        count = len(values)
        frame = [
            self._id, 0x10,
            0x00, start_address,
            0x00, count,
            count * 2,
        ]
        for v in values:
            hi, lo = v.to_bytes(2, "big", signed=True)
            frame.extend([hi, lo])

        code, res = self._send(frame, min_res_len=6)
        if code != 0:
            logger.error(
                "Write multiple failed (addr=0x%02X, code=%d)",
                start_address, code,
            )
            return False

        if len(res) < 6:
            logger.error(
                "Write multiple: response too short (%d bytes)", len(res)
            )
            return False

        for i in range(min(6, len(res))):
            if res[i] != frame[i]:
                logger.error(
                    "Write multiple: echo mismatch at byte %d "
                    "(expected 0x%02X, got 0x%02X)",
                    i, frame[i], res[i],
                )
                return False

        return self._parse_registers(res, 6)

    def _read_registers(
        self,
        start_address: int,
        count: int,
        _retries: int = 3,
    ) -> Union[int, List[int], bool]:
        """Read *count* consecutive input registers (Modbus function 0x04).

        Retries up to *_retries* times on empty/short responses (the hand
        may still be busy after calibration or a large write).
        """
        frame = [self._id, 0x04, 0x00, start_address, 0x00, count]
        expected_data_bytes = count * 2

        for attempt in range(1, _retries + 1):
            code, res = self._send(frame, min_res_len=3 + expected_data_bytes)
            if code != 0:
                logger.warning(
                    "Read registers (addr=0x%02X) attempt %d/%d code=%d",
                    start_address, attempt, _retries, code,
                )
                if attempt < _retries:
                    time.sleep(0.5)
                    continue
                return False

            if len(res) < 3 + expected_data_bytes:
                logger.warning(
                    "Read registers (addr=0x%02X) attempt %d/%d: "
                    "response too short (%d bytes, expected >= %d)",
                    start_address, attempt, _retries,
                    len(res), 3 + expected_data_bytes,
                )
                if attempt < _retries:
                    time.sleep(0.5)
                    continue
                return False

            if res[0] != self._id or res[1] != 0x04 or res[2] != expected_data_bytes:
                logger.error(
                    "Read registers: header mismatch "
                    "(id=0x%02X/0x%02X, func=0x%02X/0x04, len=%d/%d)",
                    res[0], self._id, res[1], res[2], expected_data_bytes,
                )
                return False

            return self._parse_registers(res, 3)

        return False

    # ------------------------------------------------------------------
    # Validation helpers (same as vendor_api)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_joint(joint: int) -> None:
        if not isinstance(joint, int) or not 1 <= joint <= NUM_JOINTS:
            raise ValueError(f"joint must be 1-{NUM_JOINTS}, got {joint}")

    @staticmethod
    def _check_range(name: str, value: int, lo: int, hi: int) -> None:
        if not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f"{name} must be {lo}-{hi}, got {value}")

    # ------------------------------------------------------------------
    # Device configuration
    # ------------------------------------------------------------------

    def set_id(self, new_id: int) -> bool:
        """Change the Modbus slave address (1-255)."""
        self._check_range("new_id", new_id, 1, 255)
        if self._write_single(_ADDR_SET_ID, new_id):
            self._id = new_id
            logger.info("Device ID set to %d", new_id)
            return True
        logger.error("Failed to set device ID to %d", new_id)
        return False

    def set_baudrate(self, level: int) -> bool:
        """
        Change the *hand's* baud rate by level (1-4).

        Also reconfigures the xArm tool RS485 to match and waits for the
        tool MCU reboot.
        """
        if level not in BAUD_LEVELS:
            raise ValueError(f"level must be one of {list(BAUD_LEVELS)}, got {level}")

        if not self._write_single(_ADDR_SET_BAUD, level):
            logger.error("Failed to set hand baud rate level %d", level)
            return False

        new_baud = BAUD_LEVELS[level]
        code = self._arm.set_tgpio_modbus_baudrate(new_baud)
        if code != 0:
            logger.error(
                "Failed to reconfigure xArm tool RS485 to %d (code=%d)",
                new_baud, code,
            )
            return False

        time.sleep(2)
        self._baudrate = new_baud
        logger.info("Baud rate set to %d (hand + xArm tool RS485)", new_baud)
        return True

    def clear_errors(self) -> bool:
        """Clear the device error register."""
        return bool(self._write_single(_ADDR_CLEAR_ERROR, 1))

    def save_config(self) -> bool:
        """Persist configuration (device ID, baud rate) across power cycles."""
        return bool(self._write_single(_ADDR_POWER_OFF_SAVE, 1))

    def save_params(self) -> bool:
        """Persist motion parameters (speed, current) across power cycles."""
        return bool(self._write_single(_ADDR_POWER_OFF_SAVE, 2))

    def factory_reset(self) -> bool:
        """
        Restore factory defaults (ID -> 1, baud -> 115200).

        Reconfigures the xArm tool RS485 back to 115200 automatically.
        """
        if not self._write_single(_ADDR_FACTORY_RESET, 1):
            logger.error("Factory reset command failed")
            return False

        self._id = 1
        if self._baudrate != _DEFAULT_BAUD:
            code = self._arm.set_tgpio_modbus_baudrate(_DEFAULT_BAUD)
            if code != 0:
                logger.error(
                    "Failed to reset xArm tool RS485 baud to %d", _DEFAULT_BAUD
                )
            time.sleep(2)
            self._baudrate = _DEFAULT_BAUD

        logger.info("Factory reset complete")
        return True

    # ------------------------------------------------------------------
    # Speed & current
    # ------------------------------------------------------------------

    def set_speed(self, joint: int, speed: int) -> bool:
        """Set speed for a single joint motor (1-100)."""
        self._check_joint(joint)
        self._check_range("speed", speed, 1, 100)
        return bool(self._write_single(_ADDR_SPEED + joint - 1, speed))

    def set_all_speeds(self, speed: int) -> bool:
        """Set uniform speed for all 17 joint motors (1-100)."""
        self._check_range("speed", speed, 1, 100)
        return bool(self._write_multiple(_ADDR_SPEED, [speed] * NUM_JOINTS))

    def set_current(self, joint: int, current: int) -> bool:
        """Set current limit for a single joint motor (1-100)."""
        self._check_joint(joint)
        self._check_range("current", current, 1, 100)
        return bool(self._write_single(_ADDR_CURRENT + joint - 1, current))

    def set_all_currents(self, current: int) -> bool:
        """Set uniform current limit for all 17 joint motors (1-100)."""
        self._check_range("current", current, 1, 100)
        return bool(self._write_multiple(_ADDR_CURRENT, [current] * NUM_JOINTS))

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def stop(self, joint: int) -> bool:
        """Emergency-stop a single joint motor."""
        self._check_joint(joint)
        return bool(self._write_single(_ADDR_STOP + joint - 1, 1))

    def stop_all(self) -> bool:
        """Emergency-stop all 17 joint motors."""
        return bool(self._write_multiple(_ADDR_STOP, [1] * NUM_JOINTS))

    # ------------------------------------------------------------------
    # Absolute position control (angle steps 0-1000)
    # ------------------------------------------------------------------

    def set_absolute(self, joint: int, position: int) -> bool:
        """Move one joint to an absolute angle-step position (0-1000)."""
        self._check_joint(joint)
        self._check_range("position", position, 0, 1000)
        return bool(self._write_single(_ADDR_ABSOLUTE + joint - 1, position))

    def set_all_absolute(self, positions: list[int]) -> bool:
        """Move all 17 joints to absolute angle-step positions (each 0-1000)."""
        if len(positions) != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} positions, got {len(positions)}")
        for i, p in enumerate(positions):
            self._check_range(f"positions[{i}]", p, 0, 1000)
        return bool(self._write_multiple(_ADDR_ABSOLUTE, positions))

    # ------------------------------------------------------------------
    # Relative position control (angle steps -1000 to 1000)
    # ------------------------------------------------------------------

    def set_relative(self, joint: int, offset: int) -> bool:
        """Move one joint by a relative angle-step offset (-1000 to 1000)."""
        self._check_joint(joint)
        self._check_range("offset", offset, -1000, 1000)
        return bool(self._write_single(_ADDR_RELATIVE + joint - 1, offset))

    def set_all_relative(self, offsets: list[int]) -> bool:
        """Move all 17 joints by relative angle-step offsets (each -1000 to 1000)."""
        if len(offsets) != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} offsets, got {len(offsets)}")
        for i, o in enumerate(offsets):
            self._check_range(f"offsets[{i}]", o, -1000, 1000)
        return bool(self._write_multiple(_ADDR_RELATIVE, offsets))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, joint: int) -> bool:
        """Zero-position calibration for one joint motor (~1 s)."""
        self._check_joint(joint)
        return bool(self._write_single(_ADDR_SINGLE_CAL + joint - 1, 1))

    def calibrate_all(self) -> bool:
        """Zero-position calibration for all joints (whole hand, ~13 s)."""
        return bool(self._write_single(_ADDR_ALL_CAL, 1))

    def calibrate_all_steppers(self) -> bool:
        """Zero-position calibration for stepper joints only (~1 s)."""
        return bool(self._write_single(_ADDR_ALL_STEP_CAL, 1))

    # ------------------------------------------------------------------
    # Telemetry / getters
    # ------------------------------------------------------------------

    def get_init_state(self) -> Union[int, bool]:
        """Device initialisation status. Returns 1 when OK, False on failure."""
        return self._read_registers(_ADDR_INIT_STATE, 1)

    def get_bootloader_version(self) -> Union[int, bool]:
        """Bootloader version number."""
        return self._read_registers(_ADDR_BOOTLOADER_VER, 1)

    def get_hardware_version(self) -> Union[int, bool]:
        """Hardware revision number."""
        return self._read_registers(_ADDR_HW_VER, 1)

    def get_software_version(self) -> Union[int, bool]:
        """Firmware version number."""
        return self._read_registers(_ADDR_SW_VER, 1)

    def get_errors(self) -> Union[List[int], bool]:
        """9-element list of error codes (0 = no error)."""
        return self._read_registers(_ADDR_ERROR, 9)

    def get_voltage(self) -> Optional[float]:
        """Supply voltage in volts, or None on failure."""
        raw = self._read_registers(_ADDR_VOLTAGE, 1)
        if raw is False:
            return None
        return raw * 0.001

    def get_stall_states(self) -> Union[List[int], bool]:
        """17-element list of stall flags (1 = stalled, 0 = normal)."""
        return self._read_registers(_ADDR_STALL, NUM_JOINTS)

    def get_angles(self) -> Union[List[int], bool]:
        """17-element list of current angle steps for all joints."""
        return self._read_registers(_ADDR_ANGLE, NUM_JOINTS)

    def get_fingertip_torques(self) -> Union[List[int], bool]:
        """5-element list of fingertip skin torque readings."""
        return self._read_registers(_ADDR_FINGERTIP, 5)

    def get_moving_range(self) -> Union[List[int], bool]:
        """17-element list of joint travel ranges."""
        return self._read_registers(_ADDR_MOVING_RANGE, NUM_JOINTS)

    # ------------------------------------------------------------------
    # Vendor-compatible aliases
    # ------------------------------------------------------------------

    set_error_clear = clear_errors
    set_power_off_save = save_config
    set_factory_data_reset = factory_reset
    set_single_motor_speed = set_speed
    set_all_motor_speed = set_all_speeds
    set_single_motor_current = set_current
    set_all_motor_current = set_all_currents
    set_single_motor_stop = stop
    set_all_motor_stop = stop_all
    set_single_motor_absolute = set_absolute
    set_all_motor_absolute = set_all_absolute
    set_single_motor_relative = set_relative
    set_all_motor_relative = set_all_relative
    set_single_motor_calibration = calibrate
    set_all_motor_calibration = calibrate_all
    set_all_step_motor_calibration = calibrate_all_steppers
    get_initialize_state = get_init_state
    get_device_error = get_errors
    get_device_voltage = get_voltage
    get_motor_locked_state = get_stall_states
    get_motor_real_angle = get_angles
    get_fingertip_skin_moment = get_fingertip_torques
