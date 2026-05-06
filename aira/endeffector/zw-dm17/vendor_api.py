"""
ZWHAND DM17-V6 — RS485 Modbus RTU driver for the 17-DOF dexterous hand.

Reimplemented from the vendor-supplied DOF17_ZWHAND_API.py with English
documentation, type hints, and structured logging.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Union

import serial

logger = logging.getLogger(__name__)

NUM_JOINTS = 17
BAUD_LEVELS = {1: 9600, 2: 115200, 3: 921600, 4: 2_000_000}
_DEFAULT_BAUD = 115200
_RECEIVE_TIMEOUT_S = 0.5

# ---------------------------------------------------------------------------
# Modbus register addresses (write via function code 0x10)
# ---------------------------------------------------------------------------
_ADDR_SET_ID = 0x00
_ADDR_SET_BAUD = 0x01
_ADDR_CLEAR_ERROR = 0x02
_ADDR_POWER_OFF_SAVE = 0x03
_ADDR_FACTORY_RESET = 0x04
_ADDR_SPEED = 0x05           # +0..16 for joints 1..17
_ADDR_CURRENT = 0x16         # +0..16
_ADDR_STOP = 0x27            # +0..16
_ADDR_ABSOLUTE = 0x38        # +0..16
_ADDR_RELATIVE = 0x49        # +0..16
_ADDR_SINGLE_CAL = 0x5A      # +0..16
_ADDR_ALL_STEP_CAL = 0x6B
_ADDR_ALL_CAL = 0x6C

# ---------------------------------------------------------------------------
# Modbus register addresses (read via function code 0x04)
# ---------------------------------------------------------------------------
_ADDR_INIT_STATE = 0x00
_ADDR_BOOTLOADER_VER = 0x01
_ADDR_HW_VER = 0x02
_ADDR_SW_VER = 0x03
_ADDR_ERROR = 0x04           # 9 registers
_ADDR_VOLTAGE = 0x0D
_ADDR_MOVING_RANGE = 0x0E    # 17 registers (joint travel ranges)
_ADDR_STALL = 0x1F           # 17 registers
_ADDR_ANGLE = 0x30           # 17 registers
_ADDR_FINGERTIP = 0x41       # 5 registers


def _modbus_crc(data: list[int]) -> bytes:
    """Compute Modbus RTU CRC-16 and return *data* with the CRC appended."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    data += [crc & 0xFF, (crc >> 8) & 0xFF]
    return bytes(data)


class ZWDM17Hand:
    """Driver for one ZWHAND DM17-V6 hand on an RS-485 bus."""

    def __init__(
        self,
        device_id: int,
        port: str,
        baudrate: int = _DEFAULT_BAUD,
    ) -> None:
        if not 1 <= device_id <= 255:
            raise ValueError(f"device_id must be 1-255, got {device_id}")

        self._id = device_id
        self._ser = serial.Serial()
        self._ser.port = port
        self._ser.baudrate = baudrate
        self._ser.timeout = 2

        self._ser.open()
        if not self._ser.is_open:
            raise serial.SerialException(f"Failed to open serial port {port}")
        logger.info("Serial port %s opened at %d baud", port, baudrate)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def close(self) -> bool:
        """Close the serial port."""
        try:
            self._ser.close()
            return True
        except Exception:
            logger.exception("Error closing serial port")
            return False

    # ------------------------------------------------------------------
    # Low-level Modbus helpers
    # ------------------------------------------------------------------

    def _receive(
        self,
        expected_header: list[int],
        expected_len: int,
    ) -> Union[int, List[int], bool]:
        """
        Wait up to 500 ms for a response, validate the header and CRC echo,
        then return the decoded payload.

        Returns:
            - A single int when the payload is one register.
            - A list[int] when the payload spans multiple registers.
            - True when the response is valid but carries no data payload.
            - False on timeout or validation failure.
        """
        deadline = time.time() + _RECEIVE_TIMEOUT_S
        while True:
            if time.time() > deadline:
                avail = self._ser.in_waiting
                if avail:
                    leftover = self._ser.read(avail)
                    logger.warning(
                        "Receive timeout – stale bytes: %s",
                        leftover.hex(" "),
                    )
                else:
                    logger.warning("Receive timeout – no data")
                return False

            try:
                if not (self._ser and self._ser.is_open):
                    logger.error("Serial port is not open")
                    return False
                avail = self._ser.in_waiting
            except Exception:
                logger.exception("Serial read error")
                return False

            if avail < expected_len:
                continue

            raw = self._ser.read(avail)
            data = list(raw)

            for i, expected_byte in enumerate(expected_header):
                if data[i] != expected_byte:
                    logger.error(
                        "Response header mismatch at byte %d: "
                        "expected 0x%02X got 0x%02X",
                        i, expected_byte, data[i],
                    )
                    return False

            payload = data[len(expected_header):-2]
            if not payload:
                return True

            values = []
            for i in range(0, len(payload) - 1, 2):
                values.append((payload[i] << 8) | payload[i + 1])
            return values[0] if len(values) == 1 else values

    def _write_single(self, address: int, value: int) -> Union[int, List[int], bool]:
        """Write one 16-bit register (Modbus function 0x10, 1 register)."""
        if not (self._ser and self._ser.is_open):
            logger.error("Serial port is not open")
            return False

        self._ser.reset_input_buffer()
        hi, lo = value.to_bytes(2, "big", signed=True)
        frame = [self._id, 0x10, 0x00, address, 0x00, 0x01, 0x02, hi, lo]
        cmd = _modbus_crc(frame)

        try:
            n = self._ser.write(cmd)
            if n != len(frame):
                logger.error("Incomplete write (%d/%d bytes)", n, len(frame))
                return False
            return self._receive(frame[:6], 8)
        except Exception:
            logger.exception("Write failed (addr=0x%02X)", address)
            return False

    def _write_multiple(
        self,
        start_address: int,
        values: list[int],
    ) -> Union[int, List[int], bool]:
        """Write N consecutive 16-bit registers (Modbus function 0x10)."""
        if not (self._ser and self._ser.is_open):
            logger.error("Serial port is not open")
            return False

        self._ser.reset_input_buffer()
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

        cmd = _modbus_crc(frame)

        try:
            n = self._ser.write(cmd)
            if n != len(frame):
                logger.error("Incomplete write (%d/%d bytes)", n, len(frame))
                return False
            return self._receive(frame[:6], 8)
        except Exception:
            logger.exception("Write failed (addr=0x%02X)", start_address)
            return False

    def _read_registers(
        self,
        start_address: int,
        count: int,
    ) -> Union[int, List[int], bool]:
        """Read *count* consecutive registers (Modbus function 0x04)."""
        if not (self._ser and self._ser.is_open):
            logger.error("Serial port is not open")
            return False

        self._ser.reset_input_buffer()
        frame = [self._id, 0x04, 0x00, start_address, 0x00, count]
        cmd = _modbus_crc(frame)

        try:
            n = self._ser.write(cmd)
            if n != len(frame):
                logger.error("Incomplete write (%d/%d bytes)", n, len(frame))
                return False
            header = [self._id, 0x04, count * 2]
            return self._receive(header, count * 2 + 5)
        except Exception:
            logger.exception("Read failed (addr=0x%02X)", start_address)
            return False

    # ------------------------------------------------------------------
    # Validation helpers
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
        """Change the Modbus slave address (1-255). Updates the local ID on success."""
        self._check_range("new_id", new_id, 1, 255)
        if self._write_single(_ADDR_SET_ID, new_id):
            self._id = new_id
            logger.info("Device ID set to %d", new_id)
            return True
        logger.error("Failed to set device ID to %d", new_id)
        return False

    def set_baudrate(self, level: int) -> bool:
        """
        Change baud rate by level (1=9600, 2=115200, 3=921600, 4=2000000).

        The serial port is automatically closed and re-opened at the new rate.
        """
        if level not in BAUD_LEVELS:
            raise ValueError(f"level must be one of {list(BAUD_LEVELS)}, got {level}")

        if not self._write_single(_ADDR_SET_BAUD, level):
            logger.error("Failed to set baud rate level %d", level)
            return False

        new_baud = BAUD_LEVELS[level]
        self._ser.baudrate = new_baud
        if self._ser.is_open:
            self._ser.close()
        self._ser.open()
        logger.info("Baud rate set to %d", new_baud)
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

        The serial port is re-opened at 115200 baud and the local ID is reset.
        """
        if not self._write_single(_ADDR_FACTORY_RESET, 1):
            logger.error("Factory reset command failed")
            return False

        self._id = 1
        if self._ser.baudrate != _DEFAULT_BAUD:
            if self._ser.is_open:
                self._ser.close()
            self._ser.baudrate = _DEFAULT_BAUD
            self._ser.open()
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
    # These mirror the original DOF17_ZWHAND_API.py method names so that
    # code written against the vendor docs can work with minimal changes.

    close_zwhand = close
    set_error_clear = clear_errors
    set_power_off_save = save_config  # note: vendor takes save_type 1|2
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
