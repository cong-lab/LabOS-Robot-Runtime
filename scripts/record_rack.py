#!/usr/bin/env python3
"""Record a rack-hole geometry model from multiple RealSense/YOLO captures."""

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aira.vision.rack import build_rack_model_from_points, save_rack_model
from aira.vision.singletons import calibration, camera, yolo_for_object
from aira.vision.vision import RACK_HOLE_DIAMETER_MM, analyze_rack, draw_rack_overlay


def _cluster_points(points: List[np.ndarray], threshold_mm: float) -> List[np.ndarray]:
    clusters: List[List[np.ndarray]] = []
    for point in points:
        if not np.isfinite(point).all():
            continue
        best_idx = None
        best_dist = float("inf")
        for i, cluster in enumerate(clusters):
            center = np.mean(np.array(cluster), axis=0)
            dist = float(np.linalg.norm(point - center))
            if dist < best_dist:
                best_idx = i
                best_dist = dist
        if best_idx is not None and best_dist <= threshold_mm:
            clusters[best_idx].append(point)
        else:
            clusters.append([point])
    return [np.mean(np.array(cluster), axis=0) for cluster in clusters]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a rack geometry model from visible rack holes.")
    parser.add_argument("--name", required=True, help="Rack model name, e.g. orange-3-row-offset-rack")
    parser.add_argument("--out", default=str(ROOT / "configs" / "racks"), help="Rack model output directory")
    parser.add_argument("--min-frames", type=int, default=3, help="Minimum SPACE captures before saving")
    parser.add_argument("--cluster-mm", type=float, default=18.0, help="Point clustering radius across captures")
    parser.add_argument("--hole-diameter-mm", type=float, default=RACK_HOLE_DIAMETER_MM)
    args = parser.parse_args()

    cam = camera()
    model = yolo_for_object("rack hole")
    K = calibration()["K"]
    captures: List[np.ndarray] = []
    window = "Record Rack"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Controls: SPACE capture rack holes, Q/ESC save and quit")
    capture_count = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
            scene = analyze_rack(frame, model, K, getattr(model, "names", None), conf=0.2)
            display = draw_rack_overlay(frame, scene)
            cv2.putText(
                display,
                f"{len(captures)} capture(s), {len(scene.holes)} hole(s) visible | SPACE capture | Q save",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and scene.holes:
                capture_count += 1
                captures.extend([hole.p_cam_mm.copy() for hole in scene.holes])
                print(f"Captured {len(scene.holes)} hole(s); total raw points={len(captures)}")
            elif key in (ord("q"), 27):
                break
    finally:
        cv2.destroyWindow(window)

    if capture_count < args.min_frames:
        print(f"Need at least {args.min_frames} captures; got {capture_count}")
        return 1

    clustered = _cluster_points(captures, args.cluster_mm)
    if len(clustered) < 3:
        print(f"Need at least 3 distinct rack holes after clustering; got {len(clustered)}")
        return 1

    rack_model, _plane = build_rack_model_from_points(
        args.name,
        clustered,
        hole_diameter_mm=args.hole_diameter_mm,
    )
    path = save_rack_model(rack_model, Path(args.out))
    print(f"Saved rack model with {len(rack_model.holes_xy_mm)} hole(s) to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
