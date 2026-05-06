#!/usr/bin/env python3
"""
Interactive MediaPipe calibration for ZWHAND DM17 retargeting.

Guides the user through a series of hand poses, captures the IK-solved
joint angles for each, and saves per-joint observed min/max ranges to
``configs/mediapipe_calibration.json``.  The retargeter then uses these
observed ranges (instead of raw URDF limits) so that the user's full
range of motion maps to the full 0-1000 motor range.

Usage::

    PYTHONPATH=. python scripts/mediapipe_calibrate.py
    PYTHONPATH=. python scripts/mediapipe_calibrate.py --output configs/mediapipe_calibration.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = str(ROOT / "hand_landmarker.task")

draw_utils = mp.tasks.vision.drawing_utils
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections

logger = logging.getLogger(__name__)

# ── Calibration pose definitions ──────────────────────────────────────

POSES = [
    {
        "name": "open",
        "title": "OPEN HAND",
        "description": "Fully extend all fingers, thumb out.",
        "instruction": "Spread your hand wide open, fingers straight.",
    },
    {
        "name": "fist",
        "title": "CLOSED FIST",
        "description": "Make a tight fist, thumb wrapped over fingers.",
        "instruction": "Curl all fingers tightly into a fist.",
    },
    {
        "name": "thumb_opposition",
        "title": "THUMB OPPOSITION",
        "description": "Touch your thumb to your pinky base, fingers open.",
        "instruction": "Bring your thumb across to touch your pinky base.",
    },
    {
        "name": "spread",
        "title": "SPREAD FINGERS",
        "description": "Spread all fingers as wide apart as possible.",
        "instruction": "Spread fingers apart like a fan, thumb out.",
    },
    {
        "name": "hook",
        "title": "HOOK GRIP",
        "description": "Curl only the DIP/PIP joints, keep MCP straight.",
        "instruction": "Make a hook shape: curl fingertips while keeping knuckles straight.",
    },
]

CAPTURE_FRAMES = 30
COUNTDOWN_SECONDS = 3


def draw_status(frame: np.ndarray, lines: list[str], color=(0, 255, 0)) -> None:
    """Draw multi-line status text on the frame."""
    y = 40
    for line in lines:
        cv2.putText(frame, line, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 35


def capture_pose(
    landmarker,
    cap: cv2.VideoCapture,
    retargeter,
    pose: dict,
    timestamp_ms_ref: list[int],
) -> np.ndarray | None:
    """
    Guide the user through one pose capture.

    Returns an (N, 17) array of IK joint angles (radians) collected
    over N valid frames, or None if the user skipped.
    """

    # Phase 1: show instructions, wait for SPACE
    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms_ref[0])
        timestamp_ms_ref[0] += 33

        if result.hand_landmarks:
            for hl in result.hand_landmarks:
                draw_utils.draw_landmarks(
                    frame, hl, HandLandmarksConnections.HAND_CONNECTIONS)

        draw_status(frame, [
            f"Pose: {pose['title']}",
            pose["instruction"],
            "",
            "Hold the pose and press SPACE to capture",
            "Press S to skip this pose, Q to quit",
        ])

        cv2.imshow("MediaPipe Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("s"):
            return np.empty((0, 17))
        if key == ord("q") or key == 27:
            return None

    # Phase 2: countdown
    t_start = time.monotonic()
    while time.monotonic() - t_start < COUNTDOWN_SECONDS:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms_ref[0])
        timestamp_ms_ref[0] += 33

        if result.hand_landmarks:
            for hl in result.hand_landmarks:
                draw_utils.draw_landmarks(
                    frame, hl, HandLandmarksConnections.HAND_CONNECTIONS)

        remaining = COUNTDOWN_SECONDS - (time.monotonic() - t_start)
        draw_status(frame, [
            f"Pose: {pose['title']}",
            f"Capturing in {remaining:.1f}s ...",
            "Hold steady!",
        ], color=(0, 200, 255))

        cv2.imshow("MediaPipe Calibration", frame)
        cv2.waitKey(1)

    # Phase 3: capture CAPTURE_FRAMES valid IK frames
    collected: list[np.ndarray] = []
    attempts = 0
    max_attempts = CAPTURE_FRAMES * 5

    while len(collected) < CAPTURE_FRAMES and attempts < max_attempts:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms_ref[0])
        timestamp_ms_ref[0] += 33
        attempts += 1

        if result.hand_landmarks:
            for hl in result.hand_landmarks:
                draw_utils.draw_landmarks(
                    frame, hl, HandLandmarksConnections.HAND_CONNECTIONS)

        if result.hand_world_landmarks:
            world_lm = result.hand_world_landmarks[0]
            retargeter.retarget(world_lm)
            angles = np.array(retargeter._prev_angles, copy=True)
            collected.append(angles)

        progress = len(collected) / CAPTURE_FRAMES
        draw_status(frame, [
            f"Pose: {pose['title']}",
            f"Capturing: {len(collected)}/{CAPTURE_FRAMES}",
            f"[{'#' * int(progress * 30)}{'-' * (30 - int(progress * 30))}]",
        ], color=(0, 255, 100))

        cv2.imshow("MediaPipe Calibration", frame)
        cv2.waitKey(1)

    if not collected:
        logger.warning("No valid frames captured for pose %r", pose["name"])
        return np.empty((0, 17))

    return np.array(collected)

    # Phase 4: review (show results, let user accept or redo)
    # handled by the caller


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate MediaPipe-to-DM17 joint range mapping",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(ROOT / "configs" / "mediapipe_calibration.json"),
        help="Output calibration JSON path",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _rt_mod = importlib.import_module("aira.endeffector.zw-dm17.retarget")
    retargeter = _rt_mod.DM17Retargeter()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    timestamp_ms_ref = [0]
    pose_data: dict[str, dict] = {}
    all_angles: list[np.ndarray] = []

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            for pose in POSES:
                while True:
                    result = capture_pose(
                        landmarker, cap, retargeter, pose, timestamp_ms_ref,
                    )

                    if result is None:
                        logger.info("Calibration cancelled by user.")
                        return 1

                    if result.shape[0] == 0:
                        logger.info("Skipped pose %r", pose["name"])
                        break

                    mean = result.mean(axis=0)
                    std = result.std(axis=0)
                    all_angles.append(result)

                    logger.info(
                        "Pose %r: captured %d frames, mean angles (deg): %s",
                        pose["name"], result.shape[0],
                        [f"{np.degrees(a):.1f}" for a in mean],
                    )

                    pose_data[pose["name"]] = {
                        "description": pose["description"],
                        "mean_angles_rad": mean.tolist(),
                        "std_angles_rad": std.tolist(),
                    }

                    # Review: accept or redo
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB, data=rgb)
                        landmarker.detect_for_video(
                            mp_image, timestamp_ms_ref[0])
                        timestamp_ms_ref[0] += 33

                        lines = [
                            f"Pose: {pose['title']} - CAPTURED",
                            f"Frames: {result.shape[0]}",
                            "",
                        ]
                        for j in range(17):
                            dm17 = retargeter._pb_to_dm17[j]
                            lines.append(
                                f"  DM17 J{dm17:2d}: "
                                f"{np.degrees(mean[j]):6.1f} +/- "
                                f"{np.degrees(std[j]):4.1f} deg"
                            )
                        lines.append("")
                        lines.append("SPACE=accept  R=redo  Q=quit")

                        draw_status(frame, lines, color=(255, 255, 255))
                        cv2.imshow("MediaPipe Calibration", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(" "):
                            break
                        if key == ord("r"):
                            break
                        if key == ord("q") or key == 27:
                            return 1

                    if key == ord(" "):
                        break
                    # else key == 'r', loop back to redo

    finally:
        cap.release()
        cv2.destroyAllWindows()
        retargeter.close()

    if not all_angles:
        logger.error("No poses captured. Calibration aborted.")
        return 1

    # Compute per-joint observed min/max across all frames of all poses
    stacked = np.concatenate(all_angles, axis=0)
    observed_min = stacked.min(axis=0).tolist()
    observed_max = stacked.max(axis=0).tolist()

    # Ensure min < max for every joint (add small epsilon if degenerate)
    for i in range(17):
        if observed_max[i] - observed_min[i] < 1e-4:
            observed_max[i] = observed_min[i] + 0.01

    calibration = {
        "version": 1,
        "poses": pose_data,
        "joint_ranges": {
            "observed_min": observed_min,
            "observed_max": observed_max,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)

    logger.info("Calibration saved to %s", out_path)

    print("\n--- Calibration Summary ---")
    print(f"Poses captured: {len(pose_data)}")
    print(f"Total frames:   {stacked.shape[0]}")
    print(f"\nPer-joint observed ranges (degrees):")
    for j in range(17):
        dm17 = retargeter._pb_to_dm17[j]
        lo_deg = np.degrees(observed_min[j])
        hi_deg = np.degrees(observed_max[j])
        urdf_lo = np.degrees(retargeter._lower[j])
        urdf_hi = np.degrees(retargeter._upper[j])
        print(
            f"  DM17 J{dm17:2d}: "
            f"observed [{lo_deg:6.1f}, {hi_deg:6.1f}] deg  "
            f"(URDF [{urdf_lo:6.1f}, {urdf_hi:6.1f}])"
        )

    print(f"\nSaved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
