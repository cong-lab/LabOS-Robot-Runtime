from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from aira.robot import ObjectNotFoundError, XArmFailure, arm
from aira.usb_controller import VortexPowerController, TubeHolderController
from aira.vision.vision import (
    CannotPickUpError,
    CannotPlaceError,
    analyze_rack,
    draw_rack_overlay,
)

AVAILABLE_COLORS = {
    "orange": "#fb836b",
    "blue": "#468abd",
    # "green": "#add37a",
    # "yellow": "#fdd582",
    "red": "#aa2938",
}

_BOX_COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 165, 255), (255, 165, 0), (128, 0, 255), (255, 128, 0),
]

PICKUP_ADJACENCY_FACTOR = 1.3
PLACEMENT_SPACING_FACTOR = 1.0
MOTION_SPEEDUP_FACTOR = 1.3
_RACK_MODEL_CACHE = {}
_LAST_PICKUP_XY = None
_LAST_PICKUP_TARGET = None


def _speed(value: float) -> float:
    return float(value) * MOTION_SPEEDUP_FACTOR


def _motion_kwargs(kwargs: dict, default_speed: float, default_acc: float | None = None) -> dict:
    scaled = dict(kwargs)
    scaled["speed"] = _speed(float(scaled.get("speed", default_speed)))
    if default_acc is not None:
        scaled["acc"] = _speed(float(scaled.get("acc", default_acc)))
    return scaled


def _check_xarm(code, action: str):
    if code not in (0, None):
        raise XArmFailure(action, int(code))
    return code


def _go_to(a, location: str, **kwargs):
    kwargs = _motion_kwargs(kwargs, default_speed=250, default_acc=600)
    return _check_xarm(a.go_to(location, **kwargs), f"go_to({location})")


def _z_level(a, height: float, **kwargs):
    kwargs = _motion_kwargs(kwargs, default_speed=200, default_acc=600)
    return _check_xarm(a.z_level(height, **kwargs), f"z_level({height})")


def _set_gripper(a, position: float, **kwargs):
    if "speed" in kwargs and kwargs["speed"] is not None:
        kwargs = dict(kwargs)
        kwargs["speed"] = _speed(float(kwargs["speed"]))
    return _check_xarm(a.set_gripper_position(position, **kwargs), f"set_gripper_position({position})")


def _tool_move(a, action: str, **kwargs):
    kwargs = _motion_kwargs(kwargs, default_speed=250, default_acc=600)
    return _check_xarm(a.tool_move(**kwargs), action)


def _base_move(a, action: str, **kwargs):
    kwargs = _motion_kwargs(kwargs, default_speed=250, default_acc=600)
    return _check_xarm(a.base_move(**kwargs), action)


def _draw_boxes(
    frame: np.ndarray,
    results,
    names,
    conf_threshold: float = 0.2,
    *,
    model=None,
    K=None,
) -> np.ndarray:
    """Draw YOLO bounding boxes (no masks) on a copy of the frame."""
    out = frame.copy()
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            cls_id = int(boxes.cls[i])
            color = _BOX_COLORS[cls_id % len(_BOX_COLORS)]
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{label} {conf:.2f}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if model is not None and K is not None:
        try:
            scene = analyze_rack(
                frame,
                model,
                K,
                names,
                results=results,
                palette=AVAILABLE_COLORS,
                conf=conf_threshold,
            )
            out = draw_rack_overlay(out, scene)
        except Exception:
            pass
    return out


def _load_rack_model(rack_model_name: str | None):
    if not rack_model_name:
        return None
    if rack_model_name not in _RACK_MODEL_CACHE:
        from aira.vision.rack import load_rack_model

        try:
            _RACK_MODEL_CACHE[rack_model_name] = load_rack_model(rack_model_name)
        except Exception as exc:
            print(f"Rack model '{rack_model_name}' could not be loaded; falling back to live detections: {exc}")
            _RACK_MODEL_CACHE[rack_model_name] = None
    return _RACK_MODEL_CACHE[rack_model_name]


def _rack_scene_from_camera(conf: float = 0.2, rack_model_name: str | None = None):
    from aira.vision.singletons import calibration, camera, yolo_for_object

    cam = camera()
    ok, frame = cam.read()
    if not ok or frame is None:
        return None, None
    model = yolo_for_object("rack hole")
    cal = calibration()
    scene = analyze_rack(
        frame,
        model,
        cal["K"],
        getattr(model, "names", None),
        palette=AVAILABLE_COLORS,
        rack_model=_load_rack_model(rack_model_name),
        conf=conf,
    )
    return scene, frame


def _save_last_pickup_xy(a):
    global _LAST_PICKUP_XY
    code, pose = a.get_position()
    if code != 0 or not pose or len(pose) < 2:
        raise CannotPickUpError("cannot pick up tube: failed to save pickup xy position")
    _LAST_PICKUP_XY = (float(pose[0]), float(pose[1]))
    print(f"Saved pickup XY: x={_LAST_PICKUP_XY[0]:.1f}, y={_LAST_PICKUP_XY[1]:.1f}")


def _save_last_pickup_target(target):
    global _LAST_PICKUP_TARGET
    _LAST_PICKUP_TARGET = target
    if target is not None and target.plane_center_px is not None:
        x, y = target.plane_center_px
        print(f"Saved pickup target pixel: x={x:.1f}, y={y:.1f}")


def _return_to_last_pickup_xy(a):
    if _LAST_PICKUP_XY is None:
        print("No saved pickup XY; using current rack-watch position")
        return
    _base_move(
        a,
        "return_to_last_pickup_xy",
        x=_LAST_PICKUP_XY[0],
        y=_LAST_PICKUP_XY[1],
        speed=150,
        acc=300,
        wait=True,
    )


def _run_vortex_command_async(vortex: VortexPowerController, command: str, close: bool = False):
    """Run a vortex controller command without blocking robot motion."""
    def _worker():
        try:
            getattr(vortex, command)()
        finally:
            if close:
                vortex.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


class SideBySideRecorder:
    """Record webcam + RealSense (with YOLO bbox overlay) side by side to one mp4.

    The RealSense feed is read from the aira camera singleton's pipeline (the
    same one move_to_object uses) instead of opening /dev/video* with OpenCV --
    opening the V4L2 nodes directly blocks pyrealsense2 from starting.
    """

    def __init__(self, output_path: Path, webcam_name: str = "USB Live camera",
                 fps: float = 10.0, panel_size: tuple = (640, 480)):
        self.output_path = output_path
        self.webcam_name = webcam_name
        self.fps = fps
        self.panel_size = panel_size
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        while self.thread.is_alive():
            try:
                self.thread.join(timeout=3.0)
                break
            except KeyboardInterrupt:
                # Keep waiting; aborting a join can crash the interpreter.
                continue

    def _read_realsense(self, cam):
        """Read a color frame from the shared RealSense pipeline (no align needed)."""
        pipeline = getattr(cam, "_pipeline", None)
        if pipeline is None:
            ok, frame = cam.read()
            return frame if ok else None
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            cf = frames.get_color_frame()
            return np.asanyarray(cf.get_data()) if cf else None
        except Exception:
            return None

    def _run(self):
        try:
            from aira.vision.singletons import calibration, camera, yolo_default_weights
            from ultralytics import YOLO
            cam = camera()  # singleton must already be initialized by main thread
            # Own model instance: sharing the singleton model across threads
            # races with move_to_object's predict() calls.
            model = YOLO(yolo_default_weights())
            names = model.names
            K = calibration()["K"]
        except Exception as e:
            print(f"Recorder: RealSense/YOLO init failed: {e}")
            return

        webcam = None
        webcam_idx = _find_capture_device(self.webcam_name)
        if webcam_idx is not None:
            webcam = cv2.VideoCapture(webcam_idx)
            if not webcam.isOpened():
                webcam.release()
                webcam = None
        if webcam is None:
            print(f"Recorder: webcam '{self.webcam_name}' not available; recording RealSense only")

        pw, ph = self.panel_size
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (pw * 2, ph),
        )
        if not writer.isOpened():
            if webcam is not None:
                webcam.release()
            print(f"Recorder: failed to open writer for {self.output_path}")
            return

        print(f"Recording side-by-side video to {self.output_path} ({pw * 2}x{ph} @ {self.fps:.1f} fps)")
        blank = np.zeros((ph, pw, 3), dtype=np.uint8)
        last_web = blank.copy()
        last_rs = blank.copy()
        period = 1.0 / self.fps
        try:
            while not self.stop_event.is_set():
                t0 = time.time()

                if webcam is not None:
                    ok, wf = webcam.read()
                    if ok and wf is not None:
                        last_web = cv2.resize(wf, (pw, ph))

                rs_frame = self._read_realsense(cam)
                if rs_frame is not None:
                    try:
                        results = model.predict(rs_frame, conf=0.2, imgsz=640, verbose=False)
                        rs_frame = _draw_boxes(rs_frame, results, names, model=model, K=K)
                    except Exception:
                        pass
                    last_rs = cv2.resize(rs_frame, (pw, ph))

                writer.write(np.hstack((last_web, last_rs)))

                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            writer.release()
            if webcam is not None:
                webcam.release()
            print(f"Stopped recording {self.output_path}")


def _normalize_device_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _video_devices():
    devices = []
    for entry in Path("/sys/class/video4linux").glob("video*"):
        try:
            index = int(entry.name.replace("video", ""))
            name = (entry / "name").read_text(errors="ignore").strip()
        except Exception:
            continue
        devices.append((index, name))
    return sorted(devices)


def _find_capture_device(name_substring: str):
    target = _normalize_device_name(name_substring)
    for index, name in _video_devices():
        if target not in _normalize_device_name(name):
            continue
        cap = cv2.VideoCapture(index)
        opened = cap.isOpened()
        cap.release()
        if opened:
            return index
    return None


def screw_motion(a, total_deg: float, total_dz_mm: float, steps: int = 6, speed: float = 80):
    """Rotate the TCP about the vertical axis while moving in z at the same time.

    Uses xArm relative cartesian moves (set_position with relative=True), which
    interpolate translation and rotation together in one blended motion.
    RPY yaw is bounded to +/-180 deg (shortest path), so a full 360 deg
    rotation is split into *steps* increments (each still move+rotate).
    """
    dz = total_dz_mm / steps
    dyaw = total_deg / steps
    for _ in range(steps):
        code = a.arm.set_position(
            z=dz,
            yaw=dyaw,
            relative=True,
            speed=_speed(speed),
            mvacc=_speed(300),
            is_radian=False,
            wait=True,
        )
        _check_xarm(code, "screw_motion")


def _check_pickup_clearance(color_of_tube: str, rack_model: str | None = None):
    scene, frame = _rack_scene_from_camera(rack_model_name=rack_model)
    if scene is None or frame is None:
        raise CannotPickUpError(f"cannot pick up {color_of_tube} tube: rack scene could not be analyzed")
    target_px = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
    target = scene.target_cap(color=color_of_tube, target_px=target_px)
    if target is None:
        raise CannotPickUpError(f"cannot pick up {color_of_tube} tube: target cap was not visible")
    target_hole = scene.model_hole_for_cap(target)
    _save_last_pickup_target(target_hole or target)
    target_for_spacing = target_hole or target
    target_center = target_for_spacing.plane_center_px or target_for_spacing.center_px
    candidates = scene.occupied_holes() if target_hole is not None else scene.caps
    pitch_px = scene.pitch_px or max(target.radius_px * 2.0, 1.0)
    approach_threshold = pitch_px * PICKUP_ADJACENCY_FACTOR
    row_tolerance = pitch_px * 0.6
    diagonal_tolerance = pitch_px * 0.45
    nearest_blocking = None

    for candidate in candidates:
        if candidate is target or candidate is target_hole:
            continue
        candidate_center = candidate.plane_center_px or candidate.center_px
        dx = abs(float(candidate_center[0] - target_center[0]))
        dy = abs(float(candidate_center[1] - target_center[1]))
        # Skip residual duplicate detections of the same cap top.
        if dx < max(target.radius_px, candidate.radius_px) * 0.6 and dy < max(target.radius_px, candidate.radius_px) * 0.6:
            continue
        same_row = dy <= row_tolerance and dx < approach_threshold
        diagonal = (
            abs(dx - dy) <= diagonal_tolerance
            and dx < approach_threshold
            and dy < approach_threshold
        )
        if not same_row and not diagonal:
            continue
        distance = max(dx, dy)
        if nearest_blocking is None or distance < nearest_blocking[0]:
            nearest_blocking = (distance, dx, dy, "camera X" if same_row else "45 degree diagonal")

    if nearest_blocking is not None:
        distance, dx, dy, direction = nearest_blocking
        raise CannotPickUpError(
            f"cannot pick up {color_of_tube} tube: adjacent tube is too close along {direction} "
            f"(dx={dx:.1f}, dy={dy:.1f}, limit={approach_threshold:.1f}, "
            f"row_tol={row_tolerance:.1f}, diag_tol={diagonal_tolerance:.1f})"
        )


def _center_on_tube_for_pickup(a, color_of_tube: str, repeat: int = 2):
    result = a.move_to_object(
        "50ml eppendorf cap top",
        offset=[-2, 0],
        pick_type="toolhead_close",
        min_frames=2,
        repeat=repeat,
        repeat_skip_mm=1.0,
        speed=_speed(200),
        acc=_speed(600),
        available=AVAILABLE_COLORS,
        filter=color_of_tube,
        raise_on_not_found=True,
    )
    if result.get("moves_done", 0) == 0:
        raise CannotPickUpError(f"cannot pick up {color_of_tube} tube: target was not found")
    _save_last_pickup_xy(a)


def _ensure_pickup_clearance_with_rotation_retries(a, color_of_tube: str, rack_model: str | None = None):
    last_error = None
    for attempt_label, yaw_delta in (
        ("initial", 0),
        ("rotated 45", 45),
        ("rotated 90", 45),
        ("rotated 135", 45),
    ):
        if yaw_delta:
            print(f"Pickup clearance failed; rotating toolhead +{yaw_delta} degrees and retrying")
            _tool_move(
                a,
                f"pickup_retry_rotate_yaw({yaw_delta})",
                yaw=yaw_delta,
                speed=240,
                acc=600,
                wait=True,
            )
            try:
                _center_on_tube_for_pickup(a, color_of_tube, repeat=2)
            except ObjectNotFoundError as exc:
                print(f"Tube not visible after {attempt_label}; continuing rotation retries")
                last_error = CannotPickUpError(
                    f"cannot pick up {color_of_tube} tube: target not visible after {attempt_label}"
                )
                continue
        try:
            _check_pickup_clearance(color_of_tube, rack_model=rack_model)
            print(f"Pickup clearance OK ({attempt_label})")
            return
        except CannotPickUpError as exc:
            last_error = exc
    raise last_error if last_error is not None else CannotPickUpError("cannot pick up tube: clearance check failed")


def _safe_clear_and_home_after_pickup_error(a):
    try:
        _z_level(a, 150)
    except Exception as exc:
        print(f"Failed to clear upward after pickup error: {exc}")
    try:
        _go_to(a, 'awe_home')
    except Exception as exc:
        print(f"Failed to go home after pickup error: {exc}")


def _restore_pickup_yaw(a, original_yaw: float):
    code, pose = a.get_position()
    if code != 0 or not pose or len(pose) < 6:
        print("Could not restore toolhead yaw: failed to read pose")
        return
    yaw_delta = float(original_yaw) - float(pose[5])
    if abs(yaw_delta) > 1.0:
        print(f"Restoring pickup yaw by {yaw_delta:.1f} degrees")
        _tool_move(
            a,
            "restore_pickup_yaw",
            yaw=yaw_delta,
            speed=240,
            acc=600,
            wait=True,
        )


def pickUpTube(color_of_tube: str, rack_model: str | None = None):
    """Locate a tube with the given cap *color* in the rack, descend and grab it.

    Leaves the arm holding the tube just above the rack.
    """
    a = arm()
    _go_to(a, 'awe_home')
    _set_gripper(a, 650)
    _go_to(a, 'awe_rack_watch')
    _z_level(a, 155)
    code, pose = a.get_position()
    original_yaw = float(pose[5]) if code == 0 and pose and len(pose) >= 6 else 0.0

    try:
        # find the respective tube (filtered by cap color)
        _center_on_tube_for_pickup(a, color_of_tube, repeat=4)
        _ensure_pickup_clearance_with_rotation_retries(a, color_of_tube, rack_model=rack_model)

        # move down to tube and close the gripper on it
        _z_level(a, 100)
        _set_gripper(a, 275)  # grab tube
        _z_level(a, 150)
        _restore_pickup_yaw(a, original_yaw)
    except CannotPickUpError:
        _safe_clear_and_home_after_pickup_error(a)
        raise


def _approach_vortex_genie(a):
    """Move the held tube to the vortex genie and seat it in the cup (no vortexing)."""
    _z_level(a, 340)
    print("Moving to vortex genie 2 with one vision step")
    a.move_to_object(
        'vortex genie 2',
        offset=[0, 0],
        pick_type='ranked',
        repeat=1,
        min_frames=1,
        repeat_skip_mm=0.0,
        speed=_speed(200),
        acc=_speed(600),
        raise_on_not_found=True,
    )
    a.move_to_object(
        'vortex genie hole',
        offset=[-1, 0],
        pick_type='ranked',
        speed=_speed(200),
        acc=_speed(600),
        raise_on_not_found=True,
    )
    _z_level(a, 254)


def vortexTube():
    """Vortex an already-held tube, then present it to the watch camera."""
    a = arm()
    vortex = VortexPowerController()
    try:
        print("Turning vortex on (async)")
        _run_vortex_command_async(vortex, "vortex_on")
        _approach_vortex_genie(a)
        time.sleep(2.0)
    finally:
        print("Turning vortex off (async)")
        _run_vortex_command_async(vortex, "vortex_off", close=True)
    _z_level(a, 340)

    # go to camera watch position and let the camera observe the tube
    # a.go_to('awe_cam_watch')
    # time.sleep(3.0)


def _move_to_rack_detection(a, detection, speed: float = 100, acc: float = 200):
    from aira.vision.singletons import calibration
    from aira.vision.vision import camera_to_tool

    cal = calibration()
    tare = np.array(cal.get("tare_mm", (0.0, 0.0, 0.0)), dtype=np.float64)
    p_tool = camera_to_tool(detection.p_cam_mm, cal["T_cam_to_tool"]) + tare
    if not np.isfinite(p_tool).all():
        raise CannotPlaceError("cannot place tube: selected rack target has invalid geometry")
    _check_xarm(
        a.tool_move(
            float(p_tool[0]),
            float(p_tool[1]),
            0,
            0,
            0,
            0,
            speed=_speed(speed),
            acc=_speed(acc),
            wait=True,
        ),
        "move_to_rack_detection",
    )


def _pick_spaced_rack_hole(
    min_spacing_factor: float = PLACEMENT_SPACING_FACTOR,
    rack_model: str | None = None,
):
    """Pick an empty rack hole at least one rack pitch away from existing tubes."""
    scene, _frame = _rack_scene_from_camera(conf=0.2, rack_model_name=rack_model)
    if scene is None:
        raise CannotPlaceError("cannot place tube: rack scene could not be analyzed")
    if not scene.empty_holes():
        raise CannotPlaceError("cannot place tube: no empty rack holes visible")

    spaced = scene.spaced_empty_holes(min_spacing_factor=min_spacing_factor)
    if not spaced:
        raise CannotPlaceError(
            f"cannot place tube: no empty hole is at least {min_spacing_factor:.1f} rack pitch from another tube"
        )
    best = None
    if _LAST_PICKUP_TARGET is not None:
        candidate = min(spaced, key=lambda hole: scene.distance(hole, _LAST_PICKUP_TARGET))
        original_tolerance = max(scene.pitch_mm * 0.45, 12.0) if scene.plane is not None else max(scene.pitch_px * 0.45, 12.0)
        original_dist = scene.distance(candidate, _LAST_PICKUP_TARGET)
        if original_dist <= original_tolerance:
            best = candidate
            print(f"placeDownTube: using original pickup hole (distance {original_dist:.1f})")
        else:
            print(f"placeDownTube: original pickup hole not available (nearest distance {original_dist:.1f})")
    if best is None:
        best = max(spaced, key=lambda hole: scene.nearest_cap_distance(hole))
    target = best.plane_center_px or best.center_px
    print(
        f"placeDownTube: {len(scene.holes)} empty hole(s), {len(scene.caps)} tube(s); "
        f"pitch~{scene.pitch_mm:.1f}mm, chosen clearance {scene.nearest_cap_distance(best):.1f}"
    )
    return best, (float(target[0]), float(target[1])), scene


def placeDownTube(rack_model: str | None = None):
    """Place the held tube into an empty rack hole, keeping at least one
    hole-spacing from any tube already in the rack (regardless of cap color)."""
    a = arm()
    _go_to(a, 'awe_rack_watch')
    _return_to_last_pickup_xy(a)
    rack_object = "rack hole"

    # choose a well-spaced empty hole; fail early if none is available
    target, target_px, scene = _pick_spaced_rack_hole(rack_model=rack_model)

    if scene.model_holes and any(target is hole for hole in scene.model_holes):
        _move_to_rack_detection(a, target, speed=100, acc=200)
    else:
        _z_level(a, 250)
        a.move_to_object(
            rack_object,
            offset=[0, 0],
            pick_type=target_px,
            speed=_speed(100),
            acc=_speed(200),
            raise_on_not_found=True,
        )
    _z_level(a, 150)
    _z_level(a, 116, speed=200, acc=200)
    _set_gripper(a, 600, speed=5000)  # open gripper to drop tube
    _z_level(a, 175, speed=100, acc=150)
    _go_to(a, 'awe_home')


def vortexAndTimeRecovery(
    color_of_tube: str,
    rack_model: str | None = None,
    vortex_seconds: float = 3.0,
    timeout: float = 120.0,
    poll_interval: float = 0.5,
    debounce: int = 3,
    settle_seconds: float = 1.0,
    save_dir: str | None = None,
):
    """Pick up a tube, capture its original color, vortex it to trigger the
    color change, then time how long it takes to return to the original color
    using the wrist-cam VLM judge. Places the tube back into the rack when the
    color has reverted (or on timeout).

    Returns the recovery time in seconds (from vortex-off), or ``None`` on timeout.

    State machine:
      pick up -> seat in vortexer -> capture baseline -> relay ON (vortex)
      -> relay OFF (t0) -> poll wrist cam -> VLM says reverted -> place back.
    """
    from aira.vision.color_recovery import wait_for_color_recovery
    from aira.vision.singletons import camera

    a = arm()
    cam = camera()

    save_path = Path(save_dir) if save_dir else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    poll_idx = [0]

    def grab():
        ok, frame = cam.read()
        return frame if ok else None

    def grab_and_save():
        frame = grab()
        if frame is not None and save_path is not None:
            cv2.imwrite(str(save_path / f"poll_{poll_idx[0]:03d}.jpg"), frame)
            poll_idx[0] += 1
        return frame

    # 1. pick up the requested tube
    pickUpTube(color_of_tube, rack_model=rack_model)

    # 2. seat the tube in the vortex genie (NOT vortexing yet) and register the
    #    baseline (original settled color) from the same pose we will observe from.
    _approach_vortex_genie(a)
    time.sleep(0.5)
    baseline = grab()
    if baseline is None:
        raise RuntimeError("cannot time color recovery: failed to capture baseline wrist-cam frame")
    if save_path is not None:
        cv2.imwrite(str(save_path / "baseline.jpg"), baseline)

    # 3. vortex to trigger the color change, then break the relay -> t0 (kinetic zero)
    vortex = VortexPowerController()
    try:
        print(f"Vortexing {color_of_tube} tube for {vortex_seconds:.1f}s to trigger color change")
        _run_vortex_command_async(vortex, "vortex_on")
        time.sleep(vortex_seconds)
    finally:
        _run_vortex_command_async(vortex, "vortex_off", close=True)
    t0 = time.monotonic()  # kinetic zero stays at vortex-off
    # Let the liquid stop sloshing/bubbling before we trust any verdict; t0 is
    # unchanged so the reported time is still measured from vortex-off.
    if settle_seconds > 0:
        time.sleep(settle_seconds)
    print(f"Vortex stopped; timing color recovery for {color_of_tube} tube "
          f"(settle {settle_seconds:.1f}s, debounce {debounce}, timeout {timeout:.0f}s)")

    # 4. observe in place until the VLM reports the color returned to baseline
    def _log_sample(elapsed, reverted, reply):
        print(f"  t={elapsed:5.1f}s reverted={reverted} vlm={reply!r}")

    recovered, elapsed = wait_for_color_recovery(
        grab_and_save if save_path is not None else grab,
        baseline,
        t0=t0,
        poll_interval=poll_interval,
        debounce=debounce,
        timeout=timeout,
        on_sample=_log_sample,
    )
    if recovered:
        print(f"=== {color_of_tube} tube color recovered in {elapsed:.1f}s ===")
    else:
        print(f"=== {color_of_tube} tube color recovery TIMED OUT after {elapsed:.1f}s ===")

    # 5. advance: lift clear and place the tube back into the rack
    _z_level(a, 340)
    placeDownTube(rack_model=rack_model)
    return elapsed if recovered else None


def vortexColors(colors, rack_model: str | None = None):
    """Vortex a list of tubes in order: pick up, vortex, and place each back.

    Aborts the whole sequence on the first failure (fail-fast). All colors are
    validated up front so an invalid batch never starts any robot motion.
    """
    normalized = []
    for color in colors:
        c = color.strip().lower()
        if c not in AVAILABLE_COLORS:
            available = ", ".join(sorted(AVAILABLE_COLORS))
            raise ValueError(f"Unsupported tube color '{color}'. Available colors: {available}.")
        normalized.append(c)

    for index, c in enumerate(normalized, start=1):
        print(f"=== Vortexing {c} tube ({index}/{len(normalized)}) ===")
        pickUpTube(c, rack_model=rack_model)
        vortexTube()
        placeDownTube(rack_model=rack_model)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AWE vortexing demo / color-recovery timing.")
    parser.add_argument(
        "--measure-recovery",
        metavar="COLOR",
        help="Pick up COLOR tube, vortex it, and time how long its color takes to "
             "return to baseline using the wrist-cam VLM judge. Runs without the recorder.",
    )
    parser.add_argument(
        "--colors",
        default="red,blue,orange",
        help="Comma-separated tube colors for the default vortexing demo.",
    )
    parser.add_argument(
        "--vortex-seconds",
        type=float,
        default=3.0,
        help="How long to vortex before timing recovery (--measure-recovery only).",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save the baseline + every polled wrist-cam frame for diagnosis "
             "(--measure-recovery only).",
    )
    args = parser.parse_args()

    # Initialize the RealSense singleton in the main thread first, so the
    # recorder thread and move_to_object don't race to create it.
    from aira.vision.singletons import camera
    camera()

    # Color-recovery timing mode: no recorder (avoids a second concurrent
    # camera reader contending with the poll loop).
    if args.measure_recovery:
        color = args.measure_recovery.strip().lower()
        save_dir = None
        if args.save_frames:
            save_dir = str(Path("output_videos")
                           / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                           / f"recovery_{color}")
        elapsed = vortexAndTimeRecovery(
            color,
            vortex_seconds=args.vortex_seconds,
            save_dir=save_dir,
        )
        if elapsed is not None:
            print(f"Recovery time: {elapsed:.1f}s")
        if save_dir:
            print(f"Diagnostic frames saved under: {save_dir}")
        return

    output_dir = Path("output_videos") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    recorder = SideBySideRecorder(output_dir / "side_by_side.mp4")
    recorder.start()

    try:
        a = arm()
        if True:
            # Vortex each color in turn (pick up -> vortex -> place back per tube).
            vortexColors([c.strip().lower() for c in args.colors.split(",") if c.strip()])

        if False:
            # go to open tube
            a.go_to('awe_tube_open_1')
            a.z_level(160)  # down to drop tube
            a.set_gripper_position(350)  # set gripper to open tube/drop in tube holder

        if False:
            holder = TubeHolderController()
            try:
                holder.grasp_tube()  # clamp tube in holder
                a.set_gripper_position(275)  # grab the cap

                # unscrew: up 6mm while rotating +360 deg (RZ positive)
                screw_motion(a, total_deg=360, total_dz_mm=6)

                # lift cap clear (166 -> 186), wait, then back down to 166
                a.z_level(186)
                time.sleep(1.0)
                a.z_level(166)

                # screw cap back on: rotate the other way while moving down 6mm
                screw_motion(a, total_deg=-360, total_dz_mm=-6)

                # release tube from holder and lift it out
                holder.release_tube()
                a.z_level(250)
            finally:
                holder.close()
    except KeyboardInterrupt:
        print("Interrupted by user; stopping recorder...")
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()