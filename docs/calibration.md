# Robot Calibration Guide

This guide describes the calibration process for the vision based protocols.

The normal order is:

1. Capture checkerboard images.
2. Solve camera intrinsics.
3. Collect hand-eye samples with an ArUco marker.
4. Solve hand-eye calibration.
5. Confirm runtime config points at the new calibration files.

## Prerequisites

Run commands from the repository root:

```bash
python scripts/calibrate.py --help
```

You need:

- A RealSense camera connected to the robot/toolhead.
- `pyrealsense2`, `opencv-python-headless`, and `numpy` installed from `requirements.txt`.
- A printed checkerboard for intrinsics calibration. The default is `7 x 9` inner corners with `20 mm` squares.
- A printed ArUco marker for hand-eye calibration. The default dictionary is `DICT_5X5_100`.
- Robot access over the configured IP address.

The current right-arm runtime config in `configs/robot_mapping.json` expects:

```text
calibration_images/calibration_matrix.npy
calibration_images/distortion_coefficients.npy
configs/handeye_calibration_data.json
configs/handeye_calibration_result.json
configs/tare.json
```

## Command Summary

```bash
# 1. Capture checkerboard images
python scripts/calibrate.py --mode capture --output calibration_images

# 2. Solve camera intrinsics
python scripts/calibrate.py --mode intrinsics \
  --input calibration_images \
  --output calibration_images \
  --checkerboard 7 9 \
  --square-size 20 \
  --visualize

# 3. Collect hand-eye samples
python scripts/calibrate.py --mode handeye \
  --ip 192.168.0.2 \
  --intrinsics calibration_images \
  --output configs/handeye_calibration_data.json \
  --aruco-dict DICT_5X5_100 \
  --aruco-size 0.04

# 4. Solve hand-eye transform
python scripts/calibrate.py --mode handeye_solve \
  --input configs/handeye_calibration_data.json \
  --output configs/handeye_calibration_result.json \
  --intrinsics calibration_images/calibration_matrix.npy
```

Adjust `--ip`, `--checkerboard`, `--square-size`, and `--aruco-size` to match your robot and printed calibration targets.

## Step 1: Capture Checkerboard Images

Use the calibration capture mode:

```bash
python scripts/calibrate.py --mode capture --output calibration_images
```

This calls `aira.vision.calibrate.capture.run_capture`. The preview shows:

- Live color image.
- Optional depth view.
- Checkerboard detection status.
- Capture count.

Controls:

- `SPACE`: save a calibration image.
- `D`: toggle depth view.
- `C`: toggle checkerboard overlay.
- `Q` or `ESC`: quit.

Output files:

```text
calibration_images/calib_<timestamp>.png
calibration_images/depth_<timestamp>.png
```

Only the `calib_*.png` images are used for intrinsics. The depth images are useful for visual inspection.

Capture many views of the checkerboard:

- Fill different parts of the image.
- Tilt and rotate the board.
- Include near, middle, and far distances.
- Avoid blur, glare, and partially hidden corners.

At least 3 valid images are required by the solver, but a practical calibration should use more.

## Step 2: Solve Camera Intrinsics

Run:

```bash
python scripts/calibrate.py --mode intrinsics \
  --input calibration_images \
  --output calibration_images \
  --checkerboard 7 9 \
  --square-size 20 \
  --visualize
```

This calls `aira.vision.calibrate.intrinsics.calibrate_camera`. It finds checkerboard corners in the captured images and computes the camera matrix and distortion coefficients.

Important arguments:

- `--checkerboard 7 9`: number of inner checkerboard corners, columns then rows.
- `--square-size 20`: square size in millimeters.
- `--visualize`: show detected checkerboard corners while solving.
- `--pattern`: image glob, default `*.png`.

Output files:

```text
calibration_images/calibration_matrix.npy
calibration_images/distortion_coefficients.npy
calibration_images/intrinsics.txt
```

`intrinsics.txt` is the human-readable summary. The `.npy` files are consumed by hand-eye calibration and runtime vision.

Check the printed calibration quality. Lower reprojection error is better. If the solver reports too few valid images, recapture clearer checkerboard images.

## Step 3: Collect Hand-Eye Samples

Hand-eye calibration estimates the transform from camera coordinates into the robot tool frame (`T_cam_to_tool`). It needs:

- Camera intrinsics from Step 2.
- A visible ArUco marker.
- Robot tool poses recorded at multiple viewpoints.

Run:

```bash
python scripts/calibrate.py --mode handeye \
  --ip 192.168.0.2 \
  --intrinsics calibration_images \
  --output configs/handeye_calibration_data.json \
  --aruco-dict DICT_5X5_100 \
  --aruco-size 0.04
```

This calls `aira.vision.calibrate.handeye.run_handeye_data_collection`.

### Phase 1: Z=0 Reference

The script asks you to place the ArUco marker on the table and move the gripper/toolhead to the table. This records `z0_reference` in the output metadata. Runtime Z-level helpers use this reference for table-relative behavior.

### Phase 2: Find Starting Height

The robot moves upward in tool-frame steps until the ArUco marker is visible. The camera preview shows whether the marker is detected.

### Phase 3: Manual Data Collection

The robot enters manual mode. Move the toolhead through a variety of poses while keeping the ArUco marker visible.

Controls:

- `SPACE`: record a sample when the marker is detected.
- `Q`: stop collection.

Each sample stores:

- Robot `toolhead_pose`.
- Tool-frame motion delta from the previous sample.
- ArUco pose in camera coordinates.
- Timestamp.

Output file:

```text
configs/handeye_calibration_data.json
```

Collect a broad set of poses:

- Different X/Y positions.
- Different heights.
- Different roll/pitch/yaw orientations.
- Avoid all samples being nearly identical.

The solver needs at least 3 valid samples, but more samples usually produce a better result.

## Step 4: Solve Hand-Eye Calibration

Run:

```bash
python scripts/calibrate.py --mode handeye_solve \
  --input configs/handeye_calibration_data.json \
  --output configs/handeye_calibration_result.json \
  --intrinsics calibration_images/calibration_matrix.npy
```

This calls `aira.vision.calibrate.handeye.run_handeye_solve`. It loads the hand-eye samples, reconstructs the robot and marker poses, tries multiple OpenCV hand-eye methods, chooses the best result, verifies it, and writes the final transform.

Output file:

```text
configs/handeye_calibration_result.json
```

The result includes:

- `calibration.T_cam_to_tool`: 4x4 camera-to-tool transform.
- `calibration.translation_mm`: translation component in millimeters.
- `calibration.rotation_euler_deg`: rotation component.
- `verification.position_error_mm`: average consistency error.
- `verification.rotation_error_deg`: average rotation consistency.
- `verification.quality`: quality label.

Quality interpretation:

- `EXCELLENT`: good to use.
- `GOOD`: usually acceptable.
- `ACCEPTABLE`: usable, but consider collecting better samples.
- `POOR`: recapture hand-eye data with more varied/cleaner samples.

## Step 5: Confirm Runtime Configuration

Runtime vision loads calibration through `aira.vision.singletons.calibration`. The right-arm entry in `configs/robot_mapping.json` should point at the active calibration files:

```json
{
  "camera_calibration": "configs/handeye_calibration_result.json",
  "camera_intrinsics": "calibration_images/calibration_matrix.npy",
  "camera_distortion": "calibration_images/distortion_coefficients.npy",
  "handeye_data": "configs/handeye_calibration_data.json",
  "tare": "configs/tare.json"
}
```

If you save calibration files elsewhere, update `configs/robot_mapping.json` to match.

`configs/tare.json` contains an optional `[dx, dy, dz]` correction applied in the tool/end-effector frame. Leave it as `[0, 0, 0]` until you have measured a systematic offset.

## Optional: Depth Calibration

Depth calibration is separate from robot hand-eye calibration. Use it for RealSense device health, tare, reset, and on-chip calibration.

Examples:

```bash
# Check depth health / enter interactive depth mode
python scripts/calibrate.py --mode depth --health

# Show live depth preview
python scripts/calibrate.py --mode depth --preview

# Run RealSense on-chip calibration
python scripts/calibrate.py --mode depth --on-chip

# Reset RealSense depth calibration to factory
python scripts/calibrate.py --mode depth --reset
```

Use these carefully. `--on-chip`, `--tare`, and `--reset` can modify device calibration after confirmation.

## `scripts/capture.py` vs Calibration Capture

Use this for calibration:

```bash
python scripts/calibrate.py --mode capture --output calibration_images
```

`scripts/capture.py` is a general RealSense image/depth capture utility. It saves files like `color_<timestamp>.png`, `depth_<timestamp>.png`, and `depth_<timestamp>.npy`. It does not show checkerboard detection and is not the recommended entrypoint for intrinsics calibration.

## Troubleshooting

### `pyrealsense2` is missing

Install dependencies from `requirements.txt` in the robot runtime environment. The calibration modes require RealSense access.

### Camera does not start

Check that the RealSense is connected and not already opened by another process. The shared `RealSenseCamera` tries multiple fallback resolutions, but only one process can usually own the device.

### Checkerboard is not found

Confirm:

- `--checkerboard` matches the inner-corner count, not the number of squares.
- `--square-size` matches the printed square size.
- The whole checkerboard is visible and in focus.
- Lighting avoids glare and motion blur.

### Intrinsics solve says too few images

The solver needs at least 3 images with detected checkerboard corners. Capture more views and rerun `--mode intrinsics --visualize`.

### Hand-eye collection says intrinsics are missing

Make sure these files exist:

```text
calibration_images/calibration_matrix.npy
calibration_images/distortion_coefficients.npy
```

If they are stored elsewhere, pass the directory with `--intrinsics`.

### ArUco is not detected

Confirm:

- The printed marker uses the same dictionary as `--aruco-dict`.
- `--aruco-size` matches the printed marker side length in meters.
- The marker is flat, visible, and not overexposed.
- The marker remains in the camera frame while collecting samples.

### Hand-eye quality is poor

Collect more samples with more pose diversity. Avoid recording many samples from almost the same tool pose. Include changes in position and orientation while keeping the marker reliably detected.

### Runtime vision moves to the wrong place

Check the full calibration chain:

1. `calibration_images/calibration_matrix.npy`
2. `calibration_images/distortion_coefficients.npy`
3. `configs/handeye_calibration_data.json`
4. `configs/handeye_calibration_result.json`
5. `configs/tare.json`
6. `configs/robot_mapping.json`

If the transform is consistent but offset, tune `configs/tare.json`. If the motion is rotated/flipped or highly inconsistent, redo hand-eye calibration.
