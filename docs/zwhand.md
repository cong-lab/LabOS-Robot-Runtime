# ZWHAND DM17 Guide

This guide covers the ZWHAND DM17-V6 dexterous hand when it is connected through the xArm tool-end RS485 bus. The runtime treats the hand as an end-effector named `zw-dm17`.

The normal setup order is:

1. Wire and power the hand.
2. Scan the xArm tool RS485 bus to find the DM17 device ID and baudrate.
3. Add the hand to `configs/robot_mapping.json`.
4. Run zero-position calibration.
5. Record and replay hand poses.

## How The Driver Connects

The DM17 driver lives under `aira/endeffector/zw-dm17/`. It does not open a USB serial port directly. Instead, it sends Modbus RTU frames through the connected xArm's tool-end RS485 controller.

Important details:

- The transport uses xArm tool GPIO Modbus with host ID `9`.
- The driver uses transparent transmission mode so raw Modbus RTU frames and CRCs pass through the xArm SDK.
- The hand state format is:

```json
{
  "type": "zw-dm17",
  "angles": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

`angles` must contain 17 integer values. Values are normally in the `0` to `1000` range.

## Scan For The Hand

Run commands from the repository root.

Scan the arm configured in `configs/robot_mapping.json`:

```bash
PYTHONPATH=. python scripts/scan_zwdm17.py --arm left
```

Scan a direct xArm IP:

```bash
PYTHONPATH=. python scripts/scan_zwdm17.py --ip 192.168.0.3
```

Limit the device ID range when you expect a low ID:

```bash
PYTHONPATH=. python scripts/scan_zwdm17.py --arm left --start 1 --end 10
```

Try another baudrate if the default does not respond:

```bash
PYTHONPATH=. python scripts/scan_zwdm17.py --arm left --baud 9600
PYTHONPATH=. python scripts/scan_zwdm17.py --arm left --baud 921600
PYTHONPATH=. python scripts/scan_zwdm17.py --arm left --baud 2000000
```

Successful output reports one or more device IDs:

```text
Found 1 device(s) at baud 115200:
  Device ID: 1
```

Use the reported ID and baudrate in `configs/robot_mapping.json`.

## Configure Robot Mapping

Add the DM17 under the arm's `end_effector` block:

```json
{
  "left": {
    "ip": "192.168.0.3",
    "has_camera": false,
    "camera_device": null,
    "camera_calibration": null,
    "camera_intrinsics": null,
    "camera_distortion": null,
    "handeye_data": null,
    "tare": null,
    "on_linear_rail": true,
    "end_effector": {
      "type": "zw-dm17",
      "device_id": 1,
      "baudrate": 115200
    }
  }
}
```

Fields:

- `type`: must be `zw-dm17`.
- `device_id`: Modbus device ID found by `scripts/scan_zwdm17.py`.
- `baudrate`: RS485 baudrate that responded during scan.

The runtime uses this mapping in `record_location.py`, `record_hand.py`, `gotoee.py`, YAML `hand_position` steps, and hand calibration.

## Calibrate The Hand

Use `scripts/calibrate_hand.py` for zero-position calibration.

Calibrate one configured arm:

```bash
PYTHONPATH=. python scripts/calibrate_hand.py --arm left
```

Calibrate every configured arm whose end-effector type is `zw-dm17`:

```bash
PYTHONPATH=. python scripts/calibrate_hand.py --all
```

Calibrate only stepper joints:

```bash
PYTHONPATH=. python scripts/calibrate_hand.py --arm left --steppers-only
```

What calibration does:

1. Connects to the xArm in `configs/robot_mapping.json`.
2. Connects to the DM17 using `device_id` and `baudrate`.
3. Checks initialization state and clears active hand errors.
4. Runs zero-position calibration.
5. Reads back joint angles.
6. Runs a grasp/open test.
7. Saves hand calibration data with a power-off save command.

Keep the hand clear of fixtures and samples during calibration. The script closes and opens the hand as part of the post-calibration test.

## Record Hand Positions

Use `record_hand.py` to record a hand pose into a location file:

```bash
python record_hand.py --arm left locations/hand/repeater-grip.json
```

Use the visual editor for a browser-based 3D hand pose workflow:

```bash
python record_hand.py --arm left --visual locations/hand/repeater-grip.json
```

Seed the visual editor from an existing hand pose:

```bash
python record_hand.py --arm left --visual \
  --start locations/hand/repeater-grip.json \
  locations/hand/pipette-grip.json
```

You can also record a robot arm pose and hand state together:

```bash
python record_location.py hand/repeater-grip --arm left --visual
```

Saved hand locations are referenced without the `locations/` prefix or `.json` suffix. For example:

```text
locations/hand/repeater-grip.json -> hand/repeater-grip
```

## Replay Hand Positions

Move only the hand/end-effector to a saved state:

```bash
python gotoee.py hand/repeater-grip --arm left
```

Update only the end-effector state in an existing location:

```bash
python gotoee.py hand/repeater-grip --arm left --save
python gotoee.py hand/repeater-grip --arm left --visual
```

Use a recorded hand pose in YAML:

```yaml
- step: hand_position
  arm: left
  location: hand/repeater-grip
```

Or provide raw DM17 angles directly:

```yaml
- step: hand_position
  arm: left
  angles: [61, 754, 591, 1000, 743, 0, 1000, 978, 0, 1000, 886, 0, 713, 112, 214, 112, 1000]
```

Prefer recorded location files for reusable hand poses. Raw angle lists are useful for quick tests or generated protocols.

## Troubleshooting

### Scan finds no devices

Check:

- The hand has external power.
- RS485 A/B are wired to the xArm tool connector.
- The xArm IP or `--arm` target is correct.
- No other process is using the same xArm connection.
- The baudrate is correct. Try `--baud 9600`, `--baud 921600`, and `--baud 2000000`.
- The scan range includes the device ID.

### Mapping is configured but connection fails

Confirm `device_id` and `baudrate` match the scan result. Then rerun with verbose logging:

```bash
PYTHONPATH=. python scripts/calibrate_hand.py --arm left --verbose
```

The driver reads initialization state, voltage, firmware versions, and error registers during connection.

### Calibration fails or errors persist

Make sure the hand is unobstructed and powered. Rerun calibration after clearing any physical obstruction:

```bash
PYTHONPATH=. python scripts/calibrate_hand.py --arm left
```

If only the stepper joints need resetting, use:

```bash
PYTHONPATH=. python scripts/calibrate_hand.py --arm left --steppers-only
```

### YAML `hand_position` fails

Check that:

- The arm has `end_effector.type: "zw-dm17"` in `configs/robot_mapping.json`.
- The location exists under `locations/`.
- The location contains an `end_effector` object with `type: "zw-dm17"` and 17 `angles`.
- The hand can be reached by `gotoee.py` outside the protocol.
