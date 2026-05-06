# ZWHAND DM17-V6 — RS485 Modbus Python Driver

Python driver for the **ZWHAND DM17-V6** 17-DOF dexterous robotic hand,
communicating over RS485 via the Modbus RTU protocol.

Reimplemented from the vendor-supplied `DOF17_ZWHAND_API.py` with English
documentation, type hints, and structured logging.

## Requirements

- Python >= 3.9
- `pyserial`

```bash
pip install pyserial
```

## Quick Start

```python
from aira.endeffector.zw_dm17.api import ZWDM17Hand

# Connect (device_id=1, serial port, baud rate)
hand = ZWDM17Hand(device_id=1, port="/dev/ttyUSB0", baudrate=115200)

# Check that the device booted correctly
if not hand.get_init_state():
    raise RuntimeError("Hand failed to initialise")

# Calibrate all joints to zero
hand.calibrate_all()

# Move all joints to position 500 (range 0-1000)
hand.set_all_absolute([500] * 17)

# Read back actual positions
print(hand.get_angles())

# Emergency-stop everything
hand.stop_all()

# Disconnect
hand.close()
```

## API Reference

### Constructor

#### `ZWDM17Hand(device_id, port, baudrate=115200)`

Open a serial connection and prepare the hand for commands.

| Parameter   | Type  | Default  | Description                                  |
|-------------|-------|----------|----------------------------------------------|
| `device_id` | `int` | —        | Modbus slave ID of the hand (1-255)          |
| `port`      | `str` | —        | Serial port, e.g. `"/dev/ttyUSB0"`, `"COM3"` |
| `baudrate`  | `int` | `115200` | Baud rate. Supported: 9600, 115200, 921600, 2000000 |

---

### Connection

| Method    | Returns | Description                      |
|-----------|---------|----------------------------------|
| `close()` | `bool` | Close the serial port gracefully |

---

### Device Configuration

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `set_id(new_id)` | `new_id: int` (1-255) | `bool` | Change the Modbus slave ID |
| `set_baudrate(level)` | `level: int` (1-4) | `bool` | Change baud rate. Levels: 1=9600, 2=115200, 3=921600, 4=2000000. Reopens the port automatically. |
| `clear_errors()` | — | `bool` | Clear the device error register |
| `save_config()` | — | `bool` | Persist configuration (ID, baud rate) across power cycles |
| `save_params()` | — | `bool` | Persist motion parameters (speed, current) across power cycles |
| `factory_reset()` | — | `bool` | Restore factory defaults (ID→1, baud→115200). Reopens the port. |

---

### Motion — Speed & Current

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `set_speed(joint, speed)` | `joint: int` (1-17), `speed: int` (1-100) | `bool` | Set speed for one joint motor |
| `set_all_speeds(speed)` | `speed: int` (1-100) | `bool` | Set uniform speed for all 17 joints |
| `set_current(joint, current)` | `joint: int` (1-17), `current: int` (1-100) | `bool` | Set current limit for one joint motor |
| `set_all_currents(current)` | `current: int` (1-100) | `bool` | Set uniform current limit for all 17 joints |

---

### Motion — Position Control

All positions are expressed in **angle steps** (integer range 0-1000 for absolute,
-1000 to 1000 for relative).

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `set_absolute(joint, position)` | `joint: int` (1-17), `position: int` (0-1000) | `bool` | Move one joint to an absolute position |
| `set_all_absolute(positions)` | `positions: list[int]` (len 17, each 0-1000) | `bool` | Move all joints to absolute positions |
| `set_relative(joint, offset)` | `joint: int` (1-17), `offset: int` (-1000 to 1000) | `bool` | Move one joint by a relative offset from current position |
| `set_all_relative(offsets)` | `offsets: list[int]` (len 17, each -1000 to 1000) | `bool` | Move all joints by relative offsets |

---

### Emergency Stop

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `stop(joint)` | `joint: int` (1-17) | `bool` | Emergency-stop one joint motor |
| `stop_all()` | — | `bool` | Emergency-stop all 17 joint motors |

---

### Calibration

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `calibrate(joint)` | `joint: int` (1-17) | `bool` | Zero-position calibration for one joint (~1 s) |
| `calibrate_all()` | — | `bool` | Zero-position calibration for all joints (~13 s) |
| `calibrate_all_steppers()` | — | `bool` | Zero-position calibration for stepper joints only (~1 s) |

---

### Telemetry / Getters

| Method | Returns | Description |
|--------|---------|-------------|
| `get_init_state()` | `int \| False` | Device initialisation status (1 = OK) |
| `get_bootloader_version()` | `int \| False` | Bootloader version number |
| `get_hardware_version()` | `int \| False` | Hardware revision number |
| `get_software_version()` | `int \| False` | Firmware version number |
| `get_errors()` | `list[int] \| False` | 9-element list of error codes (0 = no error) |
| `get_voltage()` | `float \| False` | Supply voltage in volts |
| `get_stall_states()` | `list[int] \| False` | 17-element list: 1 = stalled, 0 = normal |
| `get_angles()` | `list[int] \| False` | 17-element list of current angle steps |
| `get_fingertip_torques()` | `list[int] \| False` | 5-element list of fingertip skin torque values |

---

## Joint Mapping

The hand has **17 degrees of freedom** addressed as joints 1-17.
Angle steps range from **0** (fully open / zero) to **1000** (fully closed / max).

## Notes

- After calling `set_all_absolute()` or any motion command, allow time for the
  motors to reach their target before reading back angles (typically 0.4-0.6 s).
- `calibrate_all()` takes approximately 13 seconds to complete.
- Use `save_config()` / `save_params()` to persist settings across power cycles.
- The device communicates at 115200 baud by default (factory setting).
