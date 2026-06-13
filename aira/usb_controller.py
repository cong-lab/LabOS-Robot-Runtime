"""Simple serial controllers for USB-attached lab devices."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Optional


DEFAULT_USB_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200


@dataclass
class _SerialCommandController:
    """Open a serial port and send small ASCII commands."""

    port: str = DEFAULT_USB_PORT
    baudrate: int = DEFAULT_BAUDRATE
    timeout: float = 1.0
    reset_delay: float = 2.0
    terminator: bytes = b"\n"
    _serial: Optional[object] = field(default=None, init=False, repr=False)

    def connect(self):
        if self._serial is None:
            try:
                import serial  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("pyserial is required. Install with: pip install pyserial") from exc
            self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            # Many USB serial boards reset when opened; let firmware boot before commands.
            if self.reset_delay > 0:
                time.sleep(self.reset_delay)
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
        return self._serial

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def send(self, command: str, *, response_timeout: float = 0.0) -> str:
        ser = self.connect()
        ser.write(command.encode("ascii") + self.terminator)
        ser.flush()
        if response_timeout <= 0:
            return ""
        return self.read_response(response_timeout=response_timeout)

    def read_response(self, *, response_timeout: float = 2.0) -> str:
        ser = self.connect()
        deadline = time.time() + response_timeout
        chunks = bytearray()
        while time.time() < deadline:
            waiting = getattr(ser, "in_waiting", 0)
            chunk = ser.read(waiting or 1)
            if chunk:
                chunks.extend(chunk)
            else:
                time.sleep(0.02)
        return bytes(chunks).decode("utf-8", errors="replace").strip()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class VortexPowerController(_SerialCommandController):
    """Power control for the vortex controller."""

    def vortex_on(self) -> str:
        return self.send("on", response_timeout=2.0)

    def vortex_off(self) -> str:
        return self.send("off", response_timeout=2.0)


class TubeHolderController(_SerialCommandController):
    """Open/close control for the tube holder."""

    def grasp_tube(self) -> None:
        self.send("85")

    def release_tube(self) -> None:
        self.send("60")
