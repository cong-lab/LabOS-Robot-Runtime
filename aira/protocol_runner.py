"""
YAML protocol runner: run robot procedures from YAML with queryable status.

- run_protocol(path): start protocol in background thread (non-blocking).
- join_protocol(timeout): wait for protocol to finish.
- get_protocol_status(): return current step, progress %, state (for MCP).

Step types: load_home, home, set_tcp_limits, set_joint_limits, start_vision,
  go_to, move, move_joint, move_joint_absolute, tool_position, move_to_object,
  z_level, z_level_object, grip, hand_position, handoff, qr_align, sleep,
  wait_until_visible, run (subprotocol), repeat (loop
  with optional grid mode), parallel, random_choice, dispense_circle,
  python_call, stop, move_world, move_other.

Steps support an optional ``arm`` field to target a specific arm (e.g. arm: left).
``go_to`` auto-detects from the location file's ``arm`` metadata.
``move_to_object`` also accepts ``camera_arm`` for cross-arm vision.

Protocols can define top-level "args" with defaults. Steps can use "{{arg_name}}".
The "run" step runs another YAML file (relative to protocols/ or current file) with optional args.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
import importlib.util
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    class _MissingNumpy:
        def __getattr__(self, name: str) -> Any:
            raise ImportError("numpy is required for real robot/vision protocol steps") from None

    np = _MissingNumpy()  # type: ignore[assignment]

BASE = Path(__file__).resolve().parent.parent
PROTOCOLS_DIR = BASE / "protocols"

# Global status (thread-safe writes)
_status: Dict[str, Any] = {
    "state": "idle",
    "protocol_name": "",
    "current_step_index": 0,
    "current_step_name": "",
    "current_step_description": None,
    "total_steps": 0,
    "progress_pct": 0.0,
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_status_lock = threading.Lock()
_status_changed_event = threading.Event()
_protocol_thread: Optional[threading.Thread] = None
_protocol_stop = threading.Event()
_mock_mode = False


def _load_objects() -> Dict[str, Any]:
    """Load object presets from configs/objects.yaml (includes default_confidence). Single source of truth for yolo_class, shape, confidence, pick_type."""
    path = BASE / "configs" / "objects.yaml"
    if path.exists():
        try:
            import yaml
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {
        "default_confidence": 0.25,
        "50ml eppendorf": {
            "shape": {"type": "circle", "diameter": 33, "location": "center"},
            "yolo_class": "50Ml eppendorf cap",
            "pick_type": "toolhead_close",
        },
    }


def _object_presets_only(objects: Dict[str, Any]) -> Dict[str, Any]:
    """Return only preset entries (exclude default_confidence)."""
    return {k: v for k, v in objects.items() if isinstance(v, dict) and k != "default_confidence"}


def _substitute_args(obj: Any, args: Dict[str, Any]) -> Any:
    """Replace ``{{key}}`` placeholders with values from *args*.

    When a string is exactly ``"{{key}}"`` (the whole value), the raw arg
    value is returned (preserving type -- list, number, etc.).
    When ``{{key}}`` appears as part of a larger string, it is replaced
    with ``str(value)`` as before.  Recursively processes dicts and lists.
    """
    if args is None or not args:
        return obj
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith("{{") and stripped.endswith("}}") and stripped.count("{{") == 1:
            key = stripped[2:-2].strip()
            if key in args:
                return args[key]
        for k, v in args.items():
            obj = obj.replace("{{" + str(k) + "}}", str(v))
        return obj
    if isinstance(obj, dict):
        return {key: _substitute_args(val, args) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_substitute_args(item, args) for item in obj]
    return obj


def _update_status(**kwargs: Any) -> None:
    with _status_lock:
        for k, v in kwargs.items():
            if k in _status:
                _status[k] = v
    _status_changed_event.set()


def is_mock_mode() -> bool:
    """Return whether the protocol runner should avoid real arm/vision calls."""
    return _mock_mode or os.environ.get("MOCK_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_arm_name(step: Dict[str, Any]) -> Optional[str]:
    """Extract the ``arm`` field from a step dict. None means default arm."""
    val = step.get("arm")
    if val is not None:
        return str(val).strip() or None
    return None


def _step_with_default_arm(step: Dict[str, Any], args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply a branch-level default arm to a step when one is present."""
    out = dict(step)
    if "arm" not in out and args and args.get("_default_arm"):
        out["arm"] = args["_default_arm"]
    return out


def _run_python_helper(module_name: str, function_name: str, args: Optional[Dict[str, Any]]) -> None:
    """Run a helper from protocols/helpers/<module>.py."""
    safe_module = module_name.replace(".", "/").strip("/")
    helper_path = PROTOCOLS_DIR / "helpers" / f"{safe_module}.py"
    if not helper_path.exists():
        raise FileNotFoundError(f"Protocol helper not found: {helper_path}")

    spec = importlib.util.spec_from_file_location(f"protocol_helper_{safe_module.replace('/', '_')}", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load protocol helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Helper function not found: {module_name}.{function_name}")
    fn(args or {})


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse YAML/template bool-like values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _as_float_pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("desired_position must be [x, y]")
    return float(value[0]), float(value[1])


def _resolve_handoff_model_path(model_ref: str) -> str:
    """Resolve a YOLOE model ref while preserving plain names YOLO can auto-load."""
    ref = str(model_ref or "yoloe-11l-seg.pt").strip()
    if not ref:
        ref = "yoloe-11l-seg.pt"
    p = Path(ref)
    if p.is_absolute():
        return str(p)
    root_candidate = BASE / ref
    if root_candidate.exists():
        return str(root_candidate)
    weights_candidate = BASE / "weights" / ref
    if weights_candidate.exists():
        return str(weights_candidate)
    return ref


def _edge_distance(a_bbox: Tuple[float, float, float, float], b_bbox: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a_bbox
    bx1, by1, bx2, by2 = b_bbox
    return min(
        abs(ax1 - bx1), abs(ax1 - bx2),
        abs(ax2 - bx1), abs(ax2 - bx2),
        abs(ay1 - by1), abs(ay1 - by2),
        abs(ay2 - by1), abs(ay2 - by2),
    )


def _qr_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    return float(abs(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))) / 2.0)


def _qr_rotation_degrees(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    vec = np.array([pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]], dtype=np.float64)
    if np.linalg.norm(vec) < 1e-9:
        raise ValueError("QR code points have zero-length edge")
    return float(-np.degrees(np.arctan2(vec[1], vec[0])))


def _filter_angle_measurements(
    measurements: List[float],
    min_count: int = 6,
    max_deviation: float = 5.0,
) -> Tuple[Optional[float], List[float]]:
    if len(measurements) < min_count:
        return None, []
    for center_angle in measurements:
        filtered = []
        for angle in measurements:
            diff = abs(angle - center_angle)
            if diff > 180:
                diff = 360 - diff
            if diff <= max_deviation:
                filtered.append(angle)
        if len(filtered) >= min_count:
            return float(np.mean(filtered)), filtered
    return None, []


def _run_qr_align_step(step: Dict[str, Any], arm_name: Optional[str]) -> None:
    """Detect a QR code angle and rotate the tool around Z to align it."""
    import time

    import cv2
    from aira.robot import arm

    a = arm(name=arm_name)
    raw_arm = a.arm

    camera_device = int(step.get("camera_device", step.get("source", 0)))
    target_degrees = float(step.get("target_degrees", -90))
    num_measurements = int(step.get("num_measurements", 9))
    min_count = int(step.get("min_count", 6))
    max_deviation = float(step.get("max_deviation", 5.0))
    min_area = float(step.get("min_area", 400))
    max_area = float(step.get("max_area", 8000))
    max_rotation_step = float(step.get("max_rotation_step", 90))
    tolerance = float(step.get("tolerance_degrees", 0.5))
    speed = float(step.get("speed", 10))
    acc = float(step.get("acc", 5))
    wait_ms = int(step.get("wait_ms", 100))
    timeout_seconds = float(step.get("timeout_seconds", 45))
    display = _as_bool(step.get("display"), True)
    window_name = str(step.get("window_name", "QR Angle Collection"))

    cap = cv2.VideoCapture(camera_device)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"qr_align could not open camera device {camera_device}")

    qr = cv2.QRCodeDetector()
    angle_measurements: List[float] = []
    started = time.monotonic()

    try:
        while len(angle_measurements) < num_measurements:
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            if time.monotonic() - started > timeout_seconds:
                raise RuntimeError("qr_align timed out collecting QR measurements")

            ret, img = cap.read()
            if not ret or img is None:
                time.sleep(0.05)
                continue

            ret_qr, all_qr_points = qr.detect(img)
            if ret_qr and all_qr_points is not None:
                for qr_points in all_qr_points:
                    area = _qr_area(qr_points)
                    if min_area < area < max_area:
                        rotation_degrees = _qr_rotation_degrees(qr_points)
                        angle_measurements.append(rotation_degrees)
                        print(
                            f"QR measurement {len(angle_measurements)}/{num_measurements}: "
                            f"area={area:.1f}, angle={rotation_degrees:.2f}"
                        )
                        if display:
                            for p in qr_points:
                                cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
                            cv2.putText(
                                img,
                                f"Measurement {len(angle_measurements)}/{num_measurements}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            )
                            cv2.putText(
                                img,
                                f"Angle: {rotation_degrees:.2f}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            )
                        break

            if display:
                cv2.imshow(window_name, img)
                key = cv2.waitKey(wait_ms)
                if key == 27 or key == ord("q"):
                    raise RuntimeError("qr_align interrupted by user")
            elif wait_ms > 0:
                time.sleep(wait_ms / 1000.0)
    finally:
        cap.release()
        if display:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

    if not angle_measurements:
        raise RuntimeError("qr_align did not collect valid QR measurements")

    filtered_angle, filtered = _filter_angle_measurements(angle_measurements, min_count, max_deviation)
    if filtered_angle is None:
        filtered_angle = float(np.mean(angle_measurements))
        angle_std = float(np.std(angle_measurements))
        if angle_std > max_deviation:
            print(f"qr_align warning: high angle standard deviation {angle_std:.2f} deg")
    else:
        print(f"qr_align using {len(filtered)} filtered measurements")

    rotation_needed = target_degrees - filtered_angle
    if rotation_needed > 180:
        rotation_needed -= 360
    elif rotation_needed < -180:
        rotation_needed += 360
    print(f"qr_align target={target_degrees:.2f}, measured={filtered_angle:.2f}, rotate={rotation_needed:.2f}")

    raw_arm.set_mode(1)
    raw_arm.set_state(0)
    time.sleep(0.1)
    remaining_rotation = rotation_needed
    while abs(remaining_rotation) > tolerance:
        if _protocol_stop.is_set():
            raise RuntimeError("Stopped by user")
        next_move = max(-max_rotation_step, min(max_rotation_step, remaining_rotation))
        code = raw_arm.set_servo_cartesian(
            mvpose=[0, 0, 0, 0, 0, next_move],
            speed=speed,
            mvacc=acc,
            is_tool_coord=True,
            is_radian=False,
            relative=True,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"qr_align rotation returned code {code}")
        remaining_rotation -= next_move
        time.sleep(1.0)
    raw_arm.set_mode(0)
    raw_arm.set_state(0)


def _run_handoff_step(step: Dict[str, Any], arm_name: Optional[str]) -> None:
    """Use YOLOE + velocity control to align with a handoff object and grip it."""
    import time

    import cv2
    from ultralytics import YOLOE
    from aira.robot import arm

    a = arm(name=arm_name)
    raw_arm = a.arm

    model_ref = _resolve_handoff_model_path(str(step.get("model", "yoloe-11l-seg.pt")))
    yolo_classes = step.get("yolo_classes") or ["orange circle", "orange cap", "hand", "person"]
    if not isinstance(yolo_classes, list) or not yolo_classes:
        raise ValueError("handoff step requires non-empty 'yolo_classes'")
    yolo_classes = [str(c) for c in yolo_classes]

    object_name = str(step.get("object", "orange cap")).strip()
    object_classes = step.get("object_classes") or ["orange circle", "orange cap", "50ml orange cap centrifuge tube"]
    if not isinstance(object_classes, list):
        object_classes = [object_classes]
    object_classes = {str(c).strip() for c in object_classes if str(c).strip()}
    if object_name:
        object_classes.add(object_name)
    hand_class = str(step.get("hand_class", "hand")).strip()

    desired_x, desired_y = _as_float_pair(step.get("desired_position"), (581.0, 656.0))
    desired_area = float(step.get("desired_area", 16000))
    area_tolerance = float(step.get("area_tolerance", 2000))
    position_tolerance_px = float(step.get("position_tolerance_px", 20))
    pixel_to_mm_factor = float(step.get("pixel_to_mm_factor", 0.15))
    area_to_mm_factor = float(step.get("area_to_mm_factor", 0.001))
    max_xy_per_frame = float(step.get("max_xy_per_frame", 12.0))
    max_z_per_frame = float(step.get("max_z_per_frame", 10.0))
    velocity_duration = float(step.get("velocity_duration", 0.5))
    settle_seconds = float(step.get("settle_seconds", 0.2))
    history_frames = max(1, int(step.get("history_frames", 10)))
    required_hits = max(1, int(step.get("required_hits", 5)))
    timeout_seconds = float(step.get("timeout_seconds", 60))
    confidence = float(step.get("confidence", 0.03))
    grip_state = float(step.get("grip_state", 200))
    display = _as_bool(step.get("display"), True)
    require_hand = _as_bool(step.get("require_hand"), False)
    camera_device = int(step.get("camera_device", 2))
    frame_width = int(step.get("frame_width", 1280))
    frame_height = int(step.get("frame_height", 800))
    window_name = str(step.get("window_name", "YOLO Handoff"))

    if required_hits > history_frames:
        raise ValueError("handoff required_hits cannot exceed history_frames")

    print(f"Loading YOLOE model for handoff: {model_ref}")
    model = YOLOE(model_ref)
    model.set_classes(yolo_classes, model.get_text_pe(yolo_classes))

    cap = cv2.VideoCapture(camera_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"handoff could not open camera device {camera_device}")

    frame_history: List[bool] = []
    started = time.monotonic()
    succeeded = False

    try:
        raw_arm.set_mode(5)
        raw_arm.set_state(0)
        time.sleep(0.5)
        raw_arm.set_mode(5)
        time.sleep(1.5)

        while not succeeded:
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            if time.monotonic() - started > timeout_seconds:
                raise RuntimeError("handoff timed out before reaching the target area")

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            results = model.predict(frame, conf=confidence, verbose=False)
            targets: List[Dict[str, Any]] = []
            hands: List[Dict[str, Any]] = []

            for result in results or []:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    class_name = yolo_classes[class_id] if class_id < len(yolo_classes) else str(class_id)
                    x1, y1, x2, y2 = map(float, boxes.xyxy[i].cpu().numpy())
                    det = {
                        "name": class_name,
                        "bbox": (x1, y1, x2, y2),
                        "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        "area": float((x2 - x1) * (y2 - y1)),
                    }
                    if class_name in object_classes:
                        targets.append(det)
                    elif class_name == hand_class:
                        hands.append(det)

            best_target = None
            if targets and hands:
                best_target = min(
                    targets,
                    key=lambda target: min(_edge_distance(target["bbox"], hand["bbox"]) for hand in hands),
                )
            elif targets and not require_hand:
                best_target = max(targets, key=lambda target: target["area"])

            is_on_target = False
            if best_target is not None:
                center_x, center_y = best_target["center"]
                area = float(best_target["area"])
                relative_x = center_x - desired_x
                relative_y = center_y - desired_y
                distance = math.sqrt(relative_x ** 2 + relative_y ** 2)
                is_on_target = distance <= position_tolerance_px and abs(area - desired_area) <= area_tolerance

                move_x = relative_x * pixel_to_mm_factor
                move_y = -relative_y * pixel_to_mm_factor
                move_z = -(area - desired_area) * area_to_mm_factor
                move_x = max(-max_xy_per_frame, min(max_xy_per_frame, move_x))
                move_y = max(-max_xy_per_frame, min(max_xy_per_frame, move_y))
                move_z = max(-max_z_per_frame, min(max_z_per_frame, move_z))

                if abs(move_x) > 0.1 or abs(move_y) > 0.1 or abs(move_z) > 0.1:
                    raw_arm.vc_set_cartesian_velocity(
                        speeds=[int(move_y), int(move_x), int(move_z), 0, 0, 0],
                        is_radian=False,
                        is_tool_coord=True,
                        duration=velocity_duration,
                    )
                    time.sleep(settle_seconds)

                print(
                    f"handoff target={best_target['name']} "
                    f"relative=({relative_x:.1f}, {relative_y:.1f}) "
                    f"area={area:.1f} on_target={is_on_target}"
                )

            frame_history.append(is_on_target)
            if len(frame_history) > history_frames:
                frame_history.pop(0)
            if len(frame_history) >= history_frames and sum(frame_history) >= required_hits:
                succeeded = True

            if display:
                im_annot = results[0].plot() if results else frame.copy()
                if best_target is not None:
                    center_x, center_y = best_target["center"]
                    cv2.circle(im_annot, (center_x, center_y), 20, (0, 255, 0), 3)
                cv2.circle(im_annot, (int(desired_x), int(desired_y)), 10, (0, 0, 255), 2)
                cv2.imshow(window_name, im_annot)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise RuntimeError("handoff interrupted by user")

        a.set_gripper_position(grip_state, wait=True)
    finally:
        try:
            raw_arm.vc_set_cartesian_velocity(
                speeds=[0, 0, 0, 0, 0, 0],
                is_radian=False,
                is_tool_coord=True,
                duration=0.1,
            )
        except Exception:
            pass
        try:
            raw_arm.set_mode(0)
            raw_arm.set_state(0)
        except Exception:
            pass
        cap.release()
        if display:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass


def _run_failure_steps(failure_steps: List[Dict[str, Any]], protocol_path: Optional[Path] = None) -> None:
    """Execute failure block steps. Supports the same core handlers as the
    main protocol plus ``go_to`` and ``run`` so that cleanup sequences
    (e.g. sending an arm home, turning off equipment) can be expressed
    in the failure block.
    """
    import yaml as _yaml
    from aira.robot import arm, load_location
    objects = _load_objects()

    for s in failure_steps:
        step_type = (s.get("step") or "").strip().lower()
        if step_type == "stop":
            continue
        arm_name = _resolve_arm_name(s)
        try:
            a = arm(name=arm_name)
            if step_type == "grip":
                pos = s.get("state", 800)
                a.set_gripper_position(float(pos), wait=True)
            elif step_type == "set_tcp_limits":
                a.set_tcp_limits(
                    max_speed=float(s["max_speed"]) if s.get("max_speed") is not None else None,
                    max_acc=float(s["max_acc"]) if s.get("max_acc") is not None else None,
                    jerk=float(s["jerk"]) if s.get("jerk") is not None else None,
                )
            elif step_type == "set_joint_limits":
                a.set_joint_limits(
                    max_speed=float(s["max_speed"]) if s.get("max_speed") is not None else None,
                    max_acc=float(s["max_acc"]) if s.get("max_acc") is not None else None,
                    jerk=float(s["jerk"]) if s.get("jerk") is not None else None,
                )
            elif step_type == "load_home":
                f = s.get("file", "home.json")
                a.load_ref_frame(BASE / f if not Path(f).is_absolute() else f)
            elif step_type == "home":
                a.home()
            elif step_type == "go_to":
                location = (s.get("location") or s.get("go_to") or "").strip()
                if location:
                    speed = float(s.get("speed", 100))
                    acc = float(s.get("acc", 500))
                    a.go_to(location, speed=speed, acc=acc, wait=True)
            elif step_type == "sleep":
                import time
                secs = float(s.get("seconds") or s.get("sleep", 0))
                if secs > 0:
                    time.sleep(secs)
            elif step_type == "run":
                file_ref = (s.get("file") or "").strip()
                if file_ref:
                    sub_path = _resolve_protocol_path(file_ref, protocol_path)
                    if sub_path.exists():
                        with open(sub_path, "r") as f:
                            sub_data = _yaml.safe_load(f) or {}
                        sub_steps = sub_data.get("protocol") or []
                        file_args = dict(sub_data.get("args") or {})
                        run_args = dict(s.get("args") or {})
                        merged = {**file_args, **run_args}
                        for sub_step in sub_steps:
                            _execute_step(sub_step, objects, merged, sub_path)
            elif step_type == "move":
                rel = s.get("relative", [0, 0, 0])
                dx = float(rel[0]) if len(rel) > 0 else 0
                dy = float(rel[1]) if len(rel) > 1 else 0
                dz = float(rel[2]) if len(rel) > 2 else 0
                rx = float(rel[3]) if len(rel) > 3 else 0
                ry = float(rel[4]) if len(rel) > 4 else 0
                rz = float(rel[5]) if len(rel) > 5 else 0
                speed = float(s.get("speed", 100))
                acc = float(s.get("acc", 500))
                if str(s.get("frame", "tool")).strip().lower() == "base":
                    a.base_move(dx, dy, dz, speed=speed, acc=acc)
                else:
                    a.tool_move(dx, dy, dz, rx, ry, rz, speed=speed, acc=acc)
            elif step_type == "move_joint":
                rel = s.get("relative", [0] * 7)
                d_j = [float(rel[i]) if i < len(rel) else 0.0 for i in range(7)]
                a.joint_move(d_j1=d_j[0], d_j2=d_j[1], d_j3=d_j[2], d_j4=d_j[3], d_j5=d_j[4], d_j6=d_j[5], d_j7=d_j[6])
            elif step_type == "hand_position":
                angles = s.get("angles")
                location = (s.get("location") or "").strip()
                if location:
                    loc_data = load_location(location)
                    ee_state = loc_data.get("end_effector")
                elif angles is not None:
                    ee_state = {"type": "zw-dm17", "angles": [int(v) for v in angles]}
                else:
                    ee_state = None
                if ee_state:
                    ee = a.end_effector()
                    if ee is None:
                        ee = a.connect_end_effector()
                    if ee is not None:
                        ee.load_state_dict(ee_state)
            elif step_type == "parallel":
                branches = s.get("branches") or []
                from aira.robot import execute_parallel

                tasks = []
                for branch in branches:
                    branch_steps = branch.get("protocol") or branch.get("steps") or []
                    branch_args = dict(branch.get("args") or {})
                    branch_arm = branch.get("arm")
                    if branch_arm is not None:
                        branch_args["_default_arm"] = str(branch_arm).strip()

                    def _run_branch(steps=branch_steps, bargs=branch_args):
                        for sub_step in steps:
                            _execute_step(sub_step, objects, bargs, protocol_path)

                    tasks.append((_run_branch, (), {}))
                execute_parallel(tasks)
            elif step_type == "python_call":
                module_name = (s.get("module") or "").strip()
                function_name = (s.get("function") or "").strip()
                if module_name and function_name:
                    _run_python_helper(module_name, function_name, dict(s.get("args") or {}))
        except Exception:
            pass


def _resolve_protocol_path(file_ref: str, relative_to: Optional[Path] = None) -> Path:
    """Resolve a protocol file path. file_ref can be 'name', 'name.yaml', or 'subfolder/name.yaml'."""
    p = Path(file_ref)
    if not p.suffix or p.suffix.lower() != ".yaml":
        p = Path(str(p) + ".yaml")
    if p.is_absolute() and p.exists():
        return p
    base = relative_to.parent if (relative_to and relative_to.is_file()) else PROTOCOLS_DIR
    candidate = base / p
    if candidate.exists():
        return candidate
    if not candidate.exists() and (PROTOCOLS_DIR / p).exists():
        return PROTOCOLS_DIR / p
    return candidate


def _execute_step(
    step: Dict[str, Any],
    objects: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
    protocol_path: Optional[Path] = None,
) -> None:
    """Execute a single protocol step. Raises on error.

    Steps may include ``arm: left|right`` to target a specific arm and
    ``camera_arm: ...`` for cross-arm vision in move_to_object.

    For ``go_to``, if the location file has ``"arm": "both"`` the step
    moves both arms in parallel (or sequentially with ``parallel: false``).
    """
    import yaml
    from aira.robot import arm, move_to_object, load_location, execute_parallel

    step = _substitute_args(dict(step), args) if args else dict(step)
    step = _step_with_default_arm(step, args)
    arm_name = _resolve_arm_name(step)
    a = arm(name=arm_name)
    step_type = (step.get("step") or "").strip().lower()

    # ------------------------------------------------------------------
    if step_type == "load_home":
        f = step.get("file", "home.json")
        path = Path(f)
        if not path.is_absolute():
            path = BASE / path
        a.load_ref_frame(path)

    elif step_type == "home":
        a.home()

    elif step_type == "set_tcp_limits":
        max_speed = step.get("max_speed")
        max_acc = step.get("max_acc")
        jerk = step.get("jerk")
        a.set_tcp_limits(
            max_speed=float(max_speed) if max_speed is not None else None,
            max_acc=float(max_acc) if max_acc is not None else None,
            jerk=float(jerk) if jerk is not None else None,
        )

    elif step_type == "set_joint_limits":
        max_speed = step.get("max_speed")
        max_acc = step.get("max_acc")
        jerk = step.get("jerk")
        a.set_joint_limits(
            max_speed=float(max_speed) if max_speed is not None else None,
            max_acc=float(max_acc) if max_acc is not None else None,
            jerk=float(jerk) if jerk is not None else None,
        )

    elif step_type == "start_vision":
        from aira.robot import start_vision_display, warmup_vision
        warmup_vision(arm_name)
        start_vision_display()

    # ------------------------------------------------------------------
    elif step_type == "go_to":
        location = (step.get("location") or step.get("go_to") or "").strip()
        if not location:
            raise ValueError("go_to step requires 'location' or 'go_to'")
        speed = float(step.get("speed", 100))
        acc = float(step.get("acc", 500))
        offset = step.get("offset")
        if offset is not None:
            offset = [float(v) for v in offset]

        loc_data = load_location(location)
        loc_arm = loc_data.get("arm")

        if loc_arm == "both" and arm_name is None:
            parallel = step.get("parallel", True)
            arm_names = [k for k in loc_data if k != "arm"]
            if parallel and len(arm_names) > 1:
                tasks = []
                for name in arm_names:
                    tasks.append((
                        lambda n=name: arm(name=n).go_to(location, speed=speed, acc=acc, wait=True, offset=offset),
                        (), {},
                    ))
                results = execute_parallel(tasks)
                for code in results:
                    if code != 0:
                        raise RuntimeError(f"go_to (parallel) returned code {code}")
            else:
                for name in arm_names:
                    code = arm(name=name).go_to(location, speed=speed, acc=acc, wait=True, offset=offset)
                    if code != 0:
                        arm(name=name).clear_error()
                        raise RuntimeError(f"go_to returned code {code} for arm '{name}'")
        else:
            code = a.go_to(location, speed=speed, acc=acc, wait=True, offset=offset)
            if code != 0:
                if a.check_error():
                    a.clear_error()
                raise RuntimeError(f"go_to returned code {code}")

    # ------------------------------------------------------------------
    elif step_type == "z_level":
        height = float(step.get("height") or step.get("z_level", 0))
        code = a.z_level(
            height,
            speed=float(step.get("speed", 100)),
            acc=float(step.get("acc", 500)),
            wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"z_level returned code {code}")

    elif step_type == "move":
        rel = step.get("relative", [0, 0, 0])
        dx = float(rel[0]) if len(rel) > 0 else 0
        dy = float(rel[1]) if len(rel) > 1 else 0
        dz = float(rel[2]) if len(rel) > 2 else 0
        rx = float(rel[3]) if len(rel) > 3 else 0
        ry = float(rel[4]) if len(rel) > 4 else 0
        rz = float(rel[5]) if len(rel) > 5 else 0
        speed = float(step.get("speed", 100))
        acc = float(step.get("acc", 500))
        frame = str(step.get("frame", "tool")).strip().lower()
        if frame == "base":
            code = a.base_move(dx, dy, dz, speed=speed, acc=acc, wait=True)
        elif frame == "tool":
            code = a.tool_move(dx, dy, dz, rx, ry, rz, speed=speed, acc=acc, wait=True)
        else:
            raise ValueError("move step frame must be 'tool' or 'base'")
        if code != 0:
            if a.check_error():
                a.clear_error()  
            raise RuntimeError(f"{frame}_move returned code {code}")

    elif step_type == "move_joint":
        rel = step.get("relative", [0] * 7)
        d_j = [float(rel[i]) if i < len(rel) else 0.0 for i in range(7)]
        speed = float(step.get("speed", 100))
        acc = float(step.get("acc", 500))
        code = a.joint_move(
            d_j1=d_j[0], d_j2=d_j[1], d_j3=d_j[2], d_j4=d_j[3],
            d_j5=d_j[4], d_j6=d_j[5], d_j7=d_j[6],
            speed=speed, acc=acc, wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"joint_move returned code {code}")

    elif step_type == "move_joint_absolute":
        angles_raw = step.get("angles")
        if angles_raw is None:
            raise ValueError("move_joint_absolute requires 'angles'")
        angles = [float(v) for v in angles_raw]
        preserve_current = _as_bool(step.get("preserve_current"), len(angles) < 7)
        if preserve_current and len(angles) < 7:
            code, current = a.get_joint_angles()
            if code != 0 or not current:
                raise RuntimeError(f"get_joint_angles returned code {code}")
            target = list(current)
            for idx, val in enumerate(angles):
                target[idx] = val
        else:
            target = angles
        code = a.arm.set_servo_angle(
            angle=target,
            is_radian=_as_bool(step.get("is_radian"), False),
            speed=float(step.get("speed", 30)),
            mvacc=float(step.get("acc", step.get("mvacc", 500))),
            wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"move_joint_absolute returned code {code}")

    elif step_type == "tool_position":
        rel = step.get("relative", [0, 0, 0, 0, 0, 0])
        vals = [float(rel[i]) if i < len(rel) else 0.0 for i in range(6)]
        code = a.set_tool_position(
            x=vals[0],
            y=vals[1],
            z=vals[2],
            roll=vals[3],
            pitch=vals[4],
            yaw=vals[5],
            speed=float(step.get("speed", 50)),
            mvacc=float(step.get("acc", step.get("mvacc", 500))),
            wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"tool_position returned code {code}")

    # ------------------------------------------------------------------
    elif step_type == "move_to_object":
        presets = _object_presets_only(objects)
        obj_name = (step.get("object") or "").strip()
        if not obj_name or obj_name not in presets:
            raise ValueError(f"Unknown object '{obj_name}' (not in configs/objects.yaml)")
        preset = objects[obj_name]
        shape = preset.get("shape", {})
        yolo_class = preset.get("yolo_class")
        conf_threshold = float(preset.get("confidence", objects.get("default_confidence", 0.25)))
        camera_arm = step.get("camera_arm")
        if camera_arm is not None:
            camera_arm = str(camera_arm).strip()
        kwargs: Dict[str, Any] = {
            "shape": shape,
            "yolo_class": yolo_class,
            "pick_type": step.get("pick_type") or preset.get("pick_type", "toolhead_close"),
            "conf_threshold": conf_threshold,
            "average_frames": step.get("average_frames", 5),
            "repeat": step.get("repeat", 3),
            "repeat_skip_mm": float(step.get("repeat_skip_mm", 3.0)),
            "speed": float(step.get("speed", 100)),
            "acc": float(step.get("acc", 500)),
            "display": step.get("display", True),
            "use_robot": True,
            "arm_name": arm_name,
            "camera_arm": camera_arm,
            "object_name": obj_name,
            "min_frames": int(step.get("min_frames", 1)),
            "iou_threshold": float(step.get("iou_threshold", 0.7)),
        }
        off = step.get("offset")
        if off is not None and len(off) >= 2:
            kwargs["offset"] = tuple(float(x) for x in off[:3]) if len(off) >= 3 else (float(off[0]), float(off[1]))
        result = move_to_object(**kwargs)
        if not result.get("success"):
            raise RuntimeError(result.get("error", "move_to_object failed"))
        if step.get("raise_on_not_found") and result.get("moves_done", 0) == 0:
            from aira.robot import ObjectNotFoundError
            raise ObjectNotFoundError(
                f"Object '{obj_name}' not detected"
            )

    elif step_type == "z_level_object":
        presets = _object_presets_only(objects)
        obj_name = (step.get("object") or "").strip()
        if not obj_name or obj_name not in presets:
            raise ValueError(f"Unknown object '{obj_name}' (not in configs/objects.yaml)")
        preset = objects[obj_name]
        z_offset = float(step.get("z_offset", 10.0))
        average_frames = int(step.get("average_frames", 5))
        pick_type = step.get("pick_type") or preset.get("pick_type", "toolhead_close")
        code = a.z_level_object(
            obj_name,
            z_offset=z_offset,
            average_frames=average_frames,
            pick_type=pick_type,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"z_level_object returned code {code}")

    # ------------------------------------------------------------------
    elif step_type == "grip":
        pos = step.get("state", 0)
        speed = step.get("speed")
        code = a.set_gripper_position(float(pos), wait=True, speed=float(speed) if speed is not None else None)
        if code != 0:
            raise RuntimeError(f"set_gripper_position returned code {code}")

    elif step_type == "hand_position":
        angles = step.get("angles")
        location = (step.get("location") or "").strip()
        if location:
            loc_data = load_location(location)
            ee_state = loc_data.get("end_effector")
            if ee_state is None:
                raise ValueError(f"Location '{location}' has no 'end_effector' key")
        elif angles is not None:
            ee_state = {"type": "zw-dm17", "angles": [int(v) for v in angles]}
        else:
            raise ValueError("hand_position step requires 'angles' or 'location'")
        ee = a.end_effector()
        if ee is None:
            ee = a.connect_end_effector()
        if ee is None:
            raise RuntimeError(f"Arm '{arm_name}' has no dexterous end-effector")
        if not ee.load_state_dict(ee_state):
            raise RuntimeError("Failed to set hand position")

    elif step_type == "handoff":
        _run_handoff_step(step, arm_name)

    elif step_type == "qr_align":
        _run_qr_align_step(step, arm_name)

    elif step_type == "sleep":
        import time
        secs = float(step.get("seconds") or step.get("sleep", 0))
        if secs > 0:
            time.sleep(secs)

    # ------------------------------------------------------------------
    elif step_type == "wait_until_visible":
        import time as _wuv_time
        from aira.robot import see_object as _see_object

        obj_name_wuv = (step.get("object") or "").strip()
        if not obj_name_wuv:
            raise ValueError("wait_until_visible requires 'object'")
        min_hits = int(step.get("min_frames", 5))
        delay_after = float(step.get("delay", 3.0))
        poll_interval = float(step.get("poll_interval", 0.2))

        hits = 0
        first_poll = True
        seen_on_first = False
        while hits < min_hits:
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            if _see_object(obj_name_wuv):
                hits += 1
                if first_poll:
                    seen_on_first = True
            first_poll = False
            _wuv_time.sleep(poll_interval)

        if delay_after > 0 and not seen_on_first:
            _wuv_time.sleep(delay_after)

    # ------------------------------------------------------------------
    elif step_type == "move_world":
        from aira.coords import world_to_base
        pos = step.get("position")
        if pos is None or len(pos) < 3:
            raise ValueError("move_world requires 'position' [x, y, z]")
        p_world = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        target_arm = arm_name
        p_base = world_to_base(p_world, target_arm or "default")
        ori = step.get("orientation")
        if ori and len(ori) >= 3:
            roll, pitch, yaw = float(ori[0]), float(ori[1]), float(ori[2])
        else:
            _, cur_pose = a.get_position()
            roll, pitch, yaw = float(cur_pose[3]), float(cur_pose[4]), float(cur_pose[5])
        code = a._ctrl.move_to_absolute(
            x=float(p_base[0]), y=float(p_base[1]), z=float(p_base[2]),
            roll=roll, pitch=pitch, yaw=yaw,
            speed=float(step.get("speed", 100)),
            mvacc=float(step.get("acc", 500)),
            wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"move_world returned code {code}")

    elif step_type == "move_other":
        from aira.coords import base_to_base
        pos = step.get("position")
        if pos is None or len(pos) < 3:
            raise ValueError("move_other requires 'position' [x, y, z]")
        ref_arm = step.get("reference_arm")
        if not ref_arm:
            raise ValueError("move_other requires 'reference_arm'")
        ref_arm = str(ref_arm).strip()
        target_arm = arm_name
        p_source = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        p_target = base_to_base(p_source, from_arm=ref_arm, to_arm=target_arm or "default")
        ori = step.get("orientation")
        if ori and len(ori) >= 3:
            roll, pitch, yaw = float(ori[0]), float(ori[1]), float(ori[2])
        else:
            _, cur_pose = a.get_position()
            roll, pitch, yaw = float(cur_pose[3]), float(cur_pose[4]), float(cur_pose[5])
        code = a._ctrl.move_to_absolute(
            x=float(p_target[0]), y=float(p_target[1]), z=float(p_target[2]),
            roll=roll, pitch=pitch, yaw=yaw,
            speed=float(step.get("speed", 100)),
            mvacc=float(step.get("acc", 500)),
            wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"move_other returned code {code}")

    # ------------------------------------------------------------------
    elif step_type == "parallel":
        branches = step.get("branches") or []
        if not isinstance(branches, list) or not branches:
            raise ValueError("parallel step requires a non-empty 'branches' list")

        tasks = []
        for branch_idx, branch in enumerate(branches):
            if not isinstance(branch, dict):
                raise ValueError("parallel branches must be dictionaries")
            branch_steps = branch.get("protocol") or branch.get("steps") or []
            if not isinstance(branch_steps, list):
                raise ValueError("parallel branch 'protocol' must be a list")
            branch_args = dict(args or {})
            branch_args.update(dict(branch.get("args") or {}))
            branch_arm = branch.get("arm")
            if branch_arm is not None:
                branch_args["_default_arm"] = str(branch_arm).strip()
            branch_description = branch.get("description") or f"branch {branch_idx + 1}"

            def _run_branch(
                steps: List[Dict[str, Any]] = branch_steps,
                bargs: Dict[str, Any] = branch_args,
                desc: str = str(branch_description),
            ) -> None:
                for branch_step_idx, branch_step in enumerate(steps):
                    if _protocol_stop.is_set():
                        raise RuntimeError("Stopped by user")
                    _update_status(
                        current_step_description=f"{desc} (step {branch_step_idx + 1}/{len(steps)})",
                    )
                    _execute_step(branch_step, objects, bargs, protocol_path)

            tasks.append((_run_branch, (), {}))

        execute_parallel(tasks)

    elif step_type == "random_choice":
        var_name = (step.get("var") or "").strip()
        choices = step.get("choices")
        if not var_name:
            raise ValueError("random_choice requires 'var'")
        if not isinstance(choices, list) or not choices:
            raise ValueError("random_choice requires a non-empty 'choices' list")
        if args is None:
            raise ValueError("random_choice requires a mutable args context")
        args[var_name] = random.choice(choices)

    elif step_type == "dispense_circle":
        arms = step.get("arms") or ["right", "left"]
        if len(arms) < 2:
            raise ValueError("dispense_circle requires two arms")
        tip_arm = arm(name=str(arms[0]).strip())
        plunger_arm = arm(name=str(arms[1]).strip())
        n_steps = int(step.get("n_steps", 9))
        radius_mm = float(step.get("radius_mm", 30.0))
        z_top = float(step.get("z_top"))
        z_bottom = float(step.get("z_bottom"))
        speed = float(step.get("speed", 80))
        acc = float(step.get("acc", 150))
        step_depth = (z_top - z_bottom) / n_steps
        prev_x, prev_y = 0.0, 0.0

        for i in range(n_steps + 1):
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            push_z = z_top - step_depth * i
            code = plunger_arm.z_level(push_z, speed=speed, acc=acc)
            if code != 0:
                raise RuntimeError(f"dispense_circle plunger z_level returned code {code}")
            if i == n_steps:
                break

            slice_lo = 2 * math.pi * i / n_steps
            slice_hi = 2 * math.pi * (i + 1) / n_steps
            r = radius_mm * math.sqrt(random.random())
            theta = random.uniform(slice_lo, slice_hi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            dx = x - prev_x
            dy = y - prev_y
            results = execute_parallel([
                (lambda: tip_arm.base_move(dx, dy, 0, speed=speed, acc=acc), (), {}),
                (lambda: plunger_arm.base_move(dx, dy, 0, speed=speed, acc=acc), (), {}),
            ])
            for code in results:
                if code != 0:
                    raise RuntimeError(f"dispense_circle base_move returned code {code}")
            prev_x, prev_y = x, y

    # ------------------------------------------------------------------
    elif step_type == "repeat":
        from aira.robot import ObjectNotFoundError
        count = int(step.get("count", 1))
        var_name = str(step.get("var", "i"))
        offset_step_raw = step.get("offset_step")
        offset_var = str(step.get("offset_var", "loop_offset"))
        stop_on_not_found = bool(step.get("stop_on_not_found", False))
        inner_steps = step.get("protocol", [])
        description = (step.get("description") or "repeat").strip()

        # Grid mode: ``columns`` sets how many items per row.  After
        # *columns* items the column offset resets and the row offset is
        # added to advance to the next row.  This lets a single repeat
        # loop traverse an NxM grid of tubes/wells.
        columns_raw = step.get("columns")
        columns = int(columns_raw) if columns_raw is not None else None
        y_offset_step_raw = step.get("y_offset_step")

        if offset_step_raw is not None:
            offset_step_vec = [float(v) for v in offset_step_raw]
        else:
            offset_step_vec = None

        if y_offset_step_raw is not None:
            y_offset_step_vec = [float(v) for v in y_offset_step_raw]
        else:
            y_offset_step_vec = None

        # Multiple named offsets: each entry in ``offsets`` produces its
        # own variable with independent step / y_step sizes while sharing
        # the same iteration index and column count.
        #
        #   offsets:
        #     right_offset:
        #       step: [40, 0, 0]
        #       y_step: [0, 60, 0]
        #     left_offset:
        #       step: [25, 0, 0]
        #       y_step: [0, 25, 0]
        extra_offsets_raw = step.get("offsets") or {}

        def _compute_grid_offset(
            iteration_idx: int,
            col_step: list,
            row_step: list | None,
            n_columns: int | None,
        ) -> list:
            if n_columns is not None and n_columns > 0 and row_step is not None:
                col = iteration_idx % n_columns
                row = iteration_idx // n_columns
                result = [v * col for v in col_step]
                for k, yv in enumerate(row_step):
                    if k < len(result):
                        result[k] += yv * row
                    else:
                        result.append(yv * row)
                return result
            return [v * iteration_idx for v in col_step]

        for iteration in range(count):
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            _update_status(
                current_step_description=f"{description} (iteration {iteration + 1}/{count})",
            )
            iter_args = dict(args or {})
            iter_args[var_name] = iteration

            if offset_step_vec is not None:
                iter_args[offset_var] = _compute_grid_offset(
                    iteration, offset_step_vec, y_offset_step_vec, columns,
                )

            for oname, ocfg in extra_offsets_raw.items():
                if not isinstance(ocfg, dict):
                    continue
                o_step_raw = ocfg.get("step")
                if o_step_raw is None:
                    continue
                o_step = [float(v) for v in o_step_raw]
                o_ystep_raw = ocfg.get("y_step")
                o_ystep = [float(v) for v in o_ystep_raw] if o_ystep_raw else None
                o_cols_raw = ocfg.get("columns")
                o_cols = int(o_cols_raw) if o_cols_raw is not None else columns
                iter_args[str(oname)] = _compute_grid_offset(
                    iteration, o_step, o_ystep, o_cols,
                )

            try:
                for j, inner_step in enumerate(inner_steps):
                    if _protocol_stop.is_set():
                        raise RuntimeError("Stopped by user")
                    _execute_step(inner_step, objects, iter_args, protocol_path)
            except ObjectNotFoundError:
                if stop_on_not_found:
                    break
                raise

    # ------------------------------------------------------------------
    elif step_type == "run":
        file_ref = (step.get("file") or "").strip()
        if not file_ref:
            raise ValueError("run step requires 'file' (e.g. subprotocol/pickuptube.yaml)")
        sub_path = _resolve_protocol_path(file_ref, protocol_path)
        if not sub_path.exists():
            raise FileNotFoundError(f"Subprotocol not found: {sub_path}")
        with open(sub_path, "r") as f:
            sub_data = yaml.safe_load(f) or {}
        sub_steps = sub_data.get("protocol") or []
        file_args = dict(sub_data.get("args") or {})
        run_args = dict(step.get("args") or {})
        merged_args = {**dict(args or {}), **file_args, **run_args}
        if arm_name is not None:
            merged_args["_default_arm"] = arm_name
        run_description = (step.get("description") or sub_path.stem).strip()
        for i, sub_step in enumerate(sub_steps):
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            _update_status(current_step_description=f"{run_description} (step {i + 1}/{len(sub_steps)})")
            _execute_step(sub_step, objects, merged_args, sub_path)

    elif step_type == "python_call":
        module_name = (step.get("module") or "").strip()
        function_name = (step.get("function") or "").strip()
        if not module_name or not function_name:
            raise ValueError("python_call requires 'module' and 'function'")
        helper_args = dict(args or {})
        helper_args.update(dict(step.get("args") or {}))
        _run_python_helper(module_name, function_name, helper_args)

    elif step_type == "stop":
        raise RuntimeError(step.get("description") or "Stop step")

    else:
        raise ValueError(f"Unknown step type: {step_type}")


def _mock_step_description(step: Dict[str, Any]) -> str:
    """Create a concise mock status message for a protocol step."""
    step_type = (step.get("step") or "step").strip()
    for key in ("location", "object", "file", "state"):
        value = step.get(key)
        if value is not None:
            return f"[mock] {step_type} {value}"
    return f"[mock] {step_type}"


def _execute_step_mock(
    step: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
    protocol_path: Optional[Path] = None,
) -> None:
    """Execute a protocol step without touching robot, camera, or end-effector hardware."""
    import time
    import yaml

    step = _substitute_args(dict(step), args) if args else dict(step)
    step = _step_with_default_arm(step, args)
    step_type = (step.get("step") or "").strip().lower()

    if step_type == "stop":
        raise RuntimeError(step.get("description") or "Stop step")

    if step_type == "sleep":
        secs = float(step.get("seconds") or step.get("sleep", 0))
        time.sleep(min(max(secs, 0.0), 0.3))
        return

    if step_type == "run":
        file_ref = (step.get("file") or "").strip()
        if not file_ref:
            return
        sub_path = _resolve_protocol_path(file_ref, protocol_path)
        if not sub_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {sub_path}")
        with open(sub_path, "r") as f:
            sub_data = yaml.safe_load(f) or {}
        sub_steps = sub_data.get("protocol") or []
        file_args = dict(sub_data.get("args") or {})
        run_args = dict(step.get("args") or {})
        merged = {**file_args, **run_args}
        arm_name = _resolve_arm_name(step)
        if arm_name is not None:
            merged["_default_arm"] = arm_name
        run_description = (step.get("description") or sub_path.stem).strip()
        for i, sub_step in enumerate(sub_steps):
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            _update_status(current_step_description=f"[mock] {run_description} (step {i + 1}/{len(sub_steps)})")
            _execute_step_mock(sub_step, merged, sub_path)
        return

    if step_type == "parallel":
        branches = step.get("branches") or []
        for branch_idx, branch in enumerate(branches):
            branch_steps = branch.get("protocol") or branch.get("steps") or []
            branch_args = dict(branch.get("args") or {})
            branch_arm = branch.get("arm")
            if branch_arm is not None:
                branch_args["_default_arm"] = str(branch_arm).strip()
            branch_description = branch.get("description") or f"branch {branch_idx + 1}"
            for branch_step_idx, branch_step in enumerate(branch_steps):
                if _protocol_stop.is_set():
                    raise RuntimeError("Stopped by user")
                _update_status(
                    current_step_description=f"[mock] {branch_description} (step {branch_step_idx + 1}/{len(branch_steps)})",
                )
                _execute_step_mock(branch_step, branch_args, protocol_path)
        return

    if step_type == "repeat":
        count = int(step.get("count", 1))
        inner_steps = step.get("protocol", [])
        description = (step.get("description") or "repeat").strip()
        for iteration in range(max(0, count)):
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            _update_status(
                current_step_description=f"[mock] {description} (iteration {iteration + 1}/{count})",
            )
            iter_args = dict(args or {})
            iter_args["i"] = iteration
            for inner_step in inner_steps:
                _execute_step_mock(inner_step, iter_args, protocol_path)
        return

    _update_status(current_step_description=step.get("major_description") or step.get("description") or _mock_step_description(step))
    time.sleep(0.3)


def _protocol_loop(protocol_path: Path, mock: bool = False) -> None:
    global _status
    import yaml
    with open(protocol_path, "r") as f:
        data = yaml.safe_load(f) or {}
    protocol_steps = data.get("protocol") or []
    failure_steps = data.get("failure") or []
    protocol_name = protocol_path.stem
    total = len(protocol_steps)
    args = dict(data.get("args") or {})
    objects = _load_objects()
    _update_status(
        state="running",
        protocol_name=protocol_name,
        total_steps=total,
        progress_pct=0.0,
        error=None,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
    )
    _protocol_stop.clear()
    try:
        if mock or is_mock_mode():
            _update_status(current_step_description="[mock] Starting protocol without hardware")
            arm = None
        else:
            from aira.robot import arm, start_vision_display, warmup_vision

            arm().set_position_mode()

            _update_status(current_step_description="Warming up vision (camera + YOLO + calibration)…")
            warmup_vision()
            start_vision_display()

        for i, step in enumerate(protocol_steps):
            if _protocol_stop.is_set():
                _update_status(
                    state="failed",
                    error="Stopped by user",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                if mock or is_mock_mode():
                    for failure_step in failure_steps:
                        _execute_step_mock(failure_step, args, protocol_path)
                else:
                    _run_failure_steps(failure_steps, protocol_path)
                stop_protocol()
                if not (mock or is_mock_mode()):
                    try:
                        arm().clear_error()
                        arm().set_position_mode()
                    except Exception:
                        pass
                return
            step_type = (step.get("step") or "").strip().lower()
            _update_status(
                current_step_index=i,
                current_step_name=step_type,
                current_step_description=step.get("major_description") or step.get("description"),
                progress_pct=(i / total * 100.0) if total else 0.0,
            )
            if step_type == "stop":
                _update_status(
                    state="failed",
                    error=step.get("description") or "Stop step",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                if mock or is_mock_mode():
                    for failure_step in failure_steps:
                        _execute_step_mock(failure_step, args, protocol_path)
                else:
                    _run_failure_steps(failure_steps, protocol_path)
                stop_protocol()
                if not (mock or is_mock_mode()):
                    try:
                        arm().clear_error()
                        arm().set_position_mode()
                    except Exception:
                        pass
                return
            try:
                if mock or is_mock_mode():
                    _execute_step_mock(step, args, protocol_path)
                else:
                    _execute_step(step, objects, args, protocol_path)
            except Exception as e:
                _update_status(
                    state="failed",
                    error=str(e),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                if mock or is_mock_mode():
                    for failure_step in failure_steps:
                        _execute_step_mock(failure_step, args, protocol_path)
                else:
                    _run_failure_steps(failure_steps, protocol_path)
                stop_protocol()
                if not (mock or is_mock_mode()):
                    try:
                        arm().clear_error()
                        arm().set_position_mode()
                    except Exception:
                        pass
                return
        _update_status(
            state="finished",
            progress_pct=100.0,
            current_step_index=total,
            current_step_name="",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        _update_status(
            state="failed",
            error=str(e),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if mock or is_mock_mode():
            for failure_step in failure_steps:
                _execute_step_mock(failure_step, args, protocol_path)
        else:
            _run_failure_steps(failure_steps, protocol_path)
        stop_protocol()
        if not (mock or is_mock_mode()):
            try:
                from aira.robot import arm
                arm().clear_error()
                arm().set_position_mode()
            except Exception:
                pass


def run_protocol(path: str | Path, mock: bool = False) -> bool:
    """
    Start protocol from YAML file in a background thread (non-blocking).
    Path can be 'test', 'vortexing', 'subfolder/pickuptube', or 'test.yaml'.
    Lookup: protocols/<path>.yaml (with subfolders). Returns False if already running, True if started.
    """
    global _protocol_thread
    with _status_lock:
        if _status.get("state") == "running":
            return False
    p = Path(path)
    if not p.suffix or p.suffix.lower() != ".yaml":
        p = Path(str(p) + ".yaml")
    if not p.is_absolute():
        candidate = PROTOCOLS_DIR / p
        if not candidate.exists():
            candidate = PROTOCOLS_DIR / p.name
        p = candidate
    if not p.exists():
        raise FileNotFoundError(f"Protocol not found: {p}")
    global _mock_mode
    _mock_mode = bool(mock) or is_mock_mode()
    _protocol_thread = threading.Thread(target=_protocol_loop, args=(p, _mock_mode), daemon=True)
    _protocol_thread.start()
    return True


def join_protocol(timeout: Optional[float] = None) -> bool:
    """
    Block until the protocol thread finishes or timeout.
    Returns True if finished (state is finished or failed), False if timeout.
    """
    if _protocol_thread is None or not _protocol_thread.is_alive():
        return True
    _protocol_thread.join(timeout=timeout)
    return not _protocol_thread.is_alive()


def get_protocol_status() -> Dict[str, Any]:
    """Return a copy of the current protocol status for MCP or other callers."""
    with _status_lock:
        return dict(_status)


def get_status_changed_event() -> threading.Event:
    """Return the event set whenever status is updated (for MCP to wait on)."""
    return _status_changed_event


def get_protocol_status_formatted() -> str:
    """
    Return a short, LLM-friendly status string: which protocol is running (or "waiting"),
    current step description, and what comes next (or "will finish up soon").
    """
    import yaml
    with _status_lock:
        state = _status.get("state") or "idle"
        protocol_name = (_status.get("protocol_name") or "").strip()
        current_step_index = int(_status.get("current_step_index") or 0)
        total_steps = int(_status.get("total_steps") or 0)
        current_step_description = _status.get("current_step_description")
        error = _status.get("error")

    if state in ("idle", "waiting", ""):
        return "Waiting. No protocol is currently running. You can start a protocol with start_protocol."

    if state == "failed":
        return (
            f"Protocol '{protocol_name}' failed. Error: {error or 'Unknown'}. "
            "No protocol is running now (waiting)."
        )

    if state == "finished":
        return (
            f"Protocol '{protocol_name}' has finished. No protocol is running now (waiting)."
        )

    if state != "running":
        return f"Status: {state}. Protocol: {protocol_name or 'none'} (waiting)."

    # Running: build message with current step and next step
    current_desc = (current_step_description or "running").strip()
    if not current_desc and protocol_name:
        current_desc = f"step {current_step_index + 1} of {total_steps}"

    next_part = "will finish up soon."
    protocol_path = PROTOCOLS_DIR / f"{protocol_name}.yaml"
    if protocol_path.exists() and total_steps > 0:
        try:
            with open(protocol_path, "r") as f:
                data = yaml.safe_load(f) or {}
            steps = data.get("protocol") or []
            if current_step_index + 1 < len(steps):
                next_step = steps[current_step_index + 1]
                next_desc = (next_step.get("description") or next_step.get("step") or "").strip()
                if next_desc:
                    next_part = f"next: {next_desc}."
                else:
                    next_part = f"next: step {current_step_index + 2} of {total_steps}."
        except Exception:
            pass

    return (
        f"Running protocol '{protocol_name}'. "
        f"Current step: {current_desc}. "
        f"{next_part}"
    )


def stop_protocol() -> None:
    """Request the running protocol to stop after the current step. Runs failure block then sets state failed."""
    _protocol_stop.set()


def _protocol_metadata_from_data(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Extract brief and major steps from loaded YAML data. Prefer protocol[].major_description; else step_descriptions or steps."""
    brief = (data.get("brief") or "").strip()
    steps = [str(s.get("major_description")).strip() for s in (data.get("protocol") or []) if s.get("major_description")]
    if not steps:
        steps_raw = data.get("step_descriptions") or data.get("steps") or []
        steps = [str(s).strip() for s in steps_raw if s]
    return {"name": name, "brief": brief, "steps": steps}


def list_protocols() -> List[Dict[str, Any]]:
    """
    Scan protocols/ recursively for *.yaml and return name, brief, and major steps for each.
    Protocol name is the path relative to protocols/ without .yaml (e.g. vortexing, steps/rack/pick_up_from_rack).
    """
    import yaml
    out: List[Dict[str, Any]] = []
    for path in sorted(PROTOCOLS_DIR.rglob("*.yaml")):
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            rel = path.relative_to(PROTOCOLS_DIR)
            name = str(rel.with_suffix("")).replace("\\", "/")
            if "test" in name.lower() or name.startswith("steps/"):
                continue
            out.append(_protocol_metadata_from_data(data, name))
        except Exception:
            continue
    return out


def describe_protocol(protocol_name: str) -> Dict[str, Any]:
    """
    Load a protocol by name and return its brief and major steps only (no low-level step breakdown).
    Uses same path resolution as run_protocol. Raises FileNotFoundError if not found.
    """
    import yaml
    p = Path(protocol_name.strip())
    if not p.suffix or p.suffix.lower() != ".yaml":
        p = Path(str(p) + ".yaml")
    if not p.is_absolute():
        candidate = PROTOCOLS_DIR / p
        if not candidate.exists():
            candidate = PROTOCOLS_DIR / p.name
        p = candidate
    if not p.exists():
        raise FileNotFoundError(f"Protocol not found: {p}")
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    name = str(p.relative_to(PROTOCOLS_DIR).with_suffix("")).replace("\\", "/")
    return _protocol_metadata_from_data(data, name)
