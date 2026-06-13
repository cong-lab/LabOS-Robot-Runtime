"""
Arm-aware singletons for camera, YOLO model, and hand-eye calibration.

camera(arm_name)      -- per-arm camera (RealSense or OpenCV)
calibration(arm_name) -- per-arm hand-eye calibration, intrinsics, tare
yolo()                -- active (or default) YOLO model
yolo_for_object(name) -- YOLO model whose classes cover *name*

When arm_name is None the "default" (or only) arm's resources are returned,
preserving backward compatibility with single-arm setups.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from aira.utils.paths import get_project_root

BASE = get_project_root()
CALIBRATION_IMAGE_WIDTH = 1280
CALIBRATION_IMAGE_HEIGHT = 720
CONFIGS_PATH = BASE / "configs"
BASE_MODEL = BASE / "weights" / "robot-segmentation.pt"

# ---------------------------------------------------------------------------
# Camera -- keyed by arm name
# ---------------------------------------------------------------------------

_camera_singletons: Dict[str, Any] = {}
_camera_default_use_cv: Optional[bool] = None
_camera_default_device: int = 0


class _CvCamera:
    def __init__(self, cap: Any):
        self._cap = cap

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        return ret, np.asarray(frame) if ret and frame is not None else None

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if ret and frame is not None:
            return np.asarray(frame), None
        return None, None

    def stop(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class _RealSenseCamera:
    def __init__(self, pipeline: Any, align: Any):
        self._pipeline = pipeline
        self._align = align

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self._align.process(frames)
            cf = aligned.get_color_frame()
            if not cf:
                return False, None
            return True, np.asanyarray(cf.get_data())
        except Exception:
            return False, None

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self._align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf:
                return None, None
            color_image = np.asanyarray(cf.get_data())
            depth_image = np.asanyarray(df.get_data()) if df else None
            return color_image, depth_image
        except Exception:
            return None, None

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


def _resolve_arm_name(arm_name: Optional[str]) -> str:
    """Map None -> 'default' (or the first arm in robot_mapping.json)."""
    if arm_name is not None:
        return arm_name
    try:
        from aira.robot import get_arm_names
        names = get_arm_names()
        return names[0] if names else "default"
    except Exception:
        return "default"


def _arm_config(arm_name: str) -> Dict[str, Any]:
    """Return robot_mapping config for arm_name, or sensible defaults."""
    try:
        from aira.robot import load_robot_mapping
        mapping = load_robot_mapping()
        if arm_name in mapping:
            return mapping[arm_name]
        if mapping:
            return next(iter(mapping.values()))
    except Exception:
        pass
    return {
        "has_camera": True,
        "camera_device": 0,
        "camera_calibration": "configs/handeye_calibration_result.json",
        "camera_intrinsics": "calibration_images/calibration_matrix.npy",
        "camera_distortion": "calibration_images/distortion_coefficients.npy",
        "handeye_data": "configs/handeye_calibration_data.json",
        "tare": "configs/tare.json",
    }


def camera(
    use_cv_cap: Optional[bool] = None,
    cv_device: int = 0,
    arm_name: Optional[str] = None,
) -> Any:
    """Return camera singleton for the given arm.

    On first call for an arm, creates the camera from robot_mapping.json config.
    Raises RuntimeError if the arm has no camera (``has_camera: false``).
    """
    global _camera_default_use_cv, _camera_default_device
    key = _resolve_arm_name(arm_name)

    if key in _camera_singletons:
        return _camera_singletons[key]

    cfg = _arm_config(key)
    if not cfg.get("has_camera", True):
        raise RuntimeError(f"Camera not available for arm '{key}'")

    device = cfg.get("camera_device", cv_device)
    force_cv = use_cv_cap if use_cv_cap is not None else _camera_default_use_cv
    if force_cv is None:
        force_cv = False
    if use_cv_cap is not None:
        _camera_default_use_cv = use_cv_cap
    _camera_default_device = device if device is not None else cv_device

    if force_cv:
        import cv2
        cap = cv2.VideoCapture(device if device is not None else _camera_default_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIBRATION_IMAGE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIBRATION_IMAGE_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam (device={device})")
        cam = _CvCamera(cap)
        _camera_singletons[key] = cam
        return cam

    try:
        import pyrealsense2 as rs
    except ImportError:
        raise RuntimeError("pyrealsense2 not installed")
    pipeline = rs.pipeline()

    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            if dev.supports(rs.camera_info.usb_type_descriptor):
                usb = dev.get_info(rs.camera_info.usb_type_descriptor)
                if usb and not usb.startswith("3"):
                    logger.warning(
                        "RealSense is on USB %s, not USB 3.x; trying lower-fps camera modes",
                        usb,
                    )
                break
    except Exception:
        pass

    # Prefer preserving calibration color resolution; lower FPS first for USB 2.0.
    stream_options = [
        (CALIBRATION_IMAGE_WIDTH, CALIBRATION_IMAGE_HEIGHT, 640, 480, 30),
        (CALIBRATION_IMAGE_WIDTH, CALIBRATION_IMAGE_HEIGHT, 640, 480, 15),
        (CALIBRATION_IMAGE_WIDTH, CALIBRATION_IMAGE_HEIGHT, 640, 480, 6),
        (848, 480, 848, 480, 15),
        (848, 480, 848, 480, 6),
        (640, 480, 640, 480, 15),
        (640, 480, 640, 480, 6),
    ]
    started = False
    last_error = None
    for color_w, color_h, depth_w, depth_h, fps in stream_options:
        try:
            config = rs.config()
            config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
            pipeline.start(config)
            for _ in range(30):
                pipeline.wait_for_frames(timeout_ms=5000)
            logger.info(
                "RealSense camera started for arm '%s': color=%sx%s depth=%sx%s fps=%s",
                key, color_w, color_h, depth_w, depth_h, fps,
            )
            if color_w != CALIBRATION_IMAGE_WIDTH or color_h != CALIBRATION_IMAGE_HEIGHT:
                logger.warning(
                    "RealSense color resolution %sx%s differs from calibration resolution %sx%s; "
                    "move_to_object depth estimates may be less accurate",
                    color_w, color_h, CALIBRATION_IMAGE_WIDTH, CALIBRATION_IMAGE_HEIGHT,
                )
            started = True
            break
        except RuntimeError as exc:
            last_error = exc
            try:
                pipeline.stop()
            except Exception:
                pass

    if not started:
        raise RuntimeError(f"Failed to start RealSense camera for arm '{key}': {last_error}")
    align = rs.align(rs.stream.color)
    cam = _RealSenseCamera(pipeline, align)
    _camera_singletons[key] = cam
    return cam


# ---------------------------------------------------------------------------
# YOLO -- multi-model with object-class lookup
# ---------------------------------------------------------------------------
#
# vision_models in configs/objects.yaml maps object preset names to weights:
#
#   vision_models:
#     - eppendorf:
#         weights: robot-segmentation.pt
#         classes: [50ml eppendorf, vortex genie hole, rack hole]
#     - 14ml tube:
#         weights: segment14ml.pt
#         classes: [14ml tube, 14ml rack hole, vortex genie hole]
#
# yolo()                    -> active (or default) model
# yolo_for_object(name)     -> model whose classes list contains *name*
#
# The "active" model is tracked so the vision display loop automatically
# follows model switches triggered by yolo_for_object().

_yolo_models: Dict[str, Any] = {}
_yolo_active_weights: Optional[str] = None
_object_to_weights: Dict[str, str] = {}
_default_weights_path: Optional[str] = None
_all_weights_paths: List[str] = []
_vision_config_loaded: bool = False


def _load_vision_model_config() -> None:
    """Parse ``vision_models`` from objects.yaml to build object -> weights mapping."""
    global _default_weights_path, _vision_config_loaded
    if _vision_config_loaded:
        return
    _vision_config_loaded = True

    objects_path = CONFIGS_PATH / "objects.yaml"
    if not objects_path.exists():
        return
    try:
        import yaml
        with open(objects_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return

    vision_models = data.get("vision_models")
    if not vision_models or not isinstance(vision_models, list):
        return

    for entry in vision_models:
        if not isinstance(entry, dict):
            continue
        for _model_name, model_cfg in entry.items():
            if not isinstance(model_cfg, dict):
                continue
            weights = model_cfg.get("weights")
            if not weights:
                continue
            weights_resolved = str(BASE / "weights" / weights)
            _all_weights_paths.append(weights_resolved)
            for cls_name in model_cfg.get("classes", []):
                if cls_name not in _object_to_weights:
                    _object_to_weights[cls_name] = weights_resolved
            if _default_weights_path is None:
                _default_weights_path = weights_resolved


def yolo(model_path: Optional[str] = None) -> Any:
    """Return a cached YOLO model and mark it as the *active* model.

    *model_path* can be an absolute path or a filename inside ``weights/``.
    With no argument the *active* model is returned (the last one loaded or
    selected via ``yolo_for_object``).  If no model has been activated yet,
    the first entry in ``vision_models`` is used, falling back to
    ``BASE_MODEL``.
    """
    global _yolo_active_weights
    _load_vision_model_config()

    if model_path is not None:
        p = Path(model_path)
        path = str(p if p.is_absolute() else BASE / "weights" / model_path)
    else:
        path = _yolo_active_weights or _default_weights_path or str(BASE_MODEL)

    if path in _yolo_models:
        _yolo_active_weights = path
        return _yolo_models[path]

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed")

    logger.info("Loading YOLO model: %s", path)
    model = YOLO(path)
    _yolo_models[path] = model
    _yolo_active_weights = path
    return model


def yolo_for_object(object_name: str) -> Any:
    """Return the YOLO model whose weights cover *object_name*.

    Looks up ``vision_models`` in ``configs/objects.yaml`` to find which
    weights file lists this object in its ``classes``.  Falls back to the
    default model when the object isn't mapped.

    The returned model becomes the *active* model so that the vision
    display loop automatically reflects the switch.
    """
    _load_vision_model_config()
    weights_path = _object_to_weights.get(object_name)
    if weights_path:
        return yolo(weights_path)
    return yolo()


def yolo_active_weights() -> Optional[str]:
    """Return the resolved path of the currently active YOLO model, or None."""
    return _yolo_active_weights


def yolo_default_weights() -> str:
    """Return the resolved path of the default (first) vision-model weights."""
    _load_vision_model_config()
    return _default_weights_path or str(BASE_MODEL)


def yolo_all_weights() -> List[str]:
    """Return resolved paths for every model listed in ``vision_models``."""
    _load_vision_model_config()
    return list(_all_weights_paths) if _all_weights_paths else [str(BASE_MODEL)]


def yolo_weights_for_object(object_name: str) -> Optional[str]:
    """Return the resolved weights path for *object_name*, or None if unmapped."""
    _load_vision_model_config()
    return _object_to_weights.get(object_name)


# ---------------------------------------------------------------------------
# Calibration -- keyed by arm name
# ---------------------------------------------------------------------------

_calibration_singletons: Dict[str, Dict[str, Any]] = {}


def calibration(
    calibration_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    distortion_path: Optional[str] = None,
    tare_path: Optional[str] = None,
    handeye_data_path: Optional[str] = None,
    arm_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return calibration data for the given arm: T_cam_to_tool, K, D, tare_mm, z0_reference.

    Paths are resolved from robot_mapping.json config for the arm, falling back
    to the explicit arguments or legacy defaults.
    """
    key = _resolve_arm_name(arm_name)

    if key in _calibration_singletons:
        return _calibration_singletons[key]

    cfg = _arm_config(key)

    calib_p = calibration_path or cfg.get("camera_calibration") or "configs/handeye_calibration_result.json"
    calib_path_obj = Path(calib_p)
    if not calib_path_obj.is_absolute():
        calib_path_obj = BASE / calib_path_obj

    intr_p = intrinsics_path or cfg.get("camera_intrinsics") or "calibration_images/calibration_matrix.npy"
    matrix_path = Path(intr_p)
    if not matrix_path.is_absolute():
        matrix_path = BASE / matrix_path

    dist_p = distortion_path or cfg.get("camera_distortion") or "calibration_images/distortion_coefficients.npy"
    dist_path = Path(dist_p)
    if not dist_path.is_absolute():
        dist_path = BASE / dist_path

    tare_p = tare_path or cfg.get("tare") or "configs/tare.json"
    tare_path_obj = Path(tare_p)
    if not tare_path_obj.is_absolute():
        tare_path_obj = BASE / tare_path_obj

    hdata_p = handeye_data_path or cfg.get("handeye_data") or "configs/handeye_calibration_data.json"
    data_path = Path(hdata_p)
    if not data_path.is_absolute():
        data_path = BASE / data_path

    if not calib_path_obj.exists() or not matrix_path.exists() or not dist_path.exists():
        raise FileNotFoundError(
            f"Calibration or intrinsics files not found for arm '{key}': "
            f"calib={calib_path_obj}, K={matrix_path}, D={dist_path}"
        )

    with open(calib_path_obj, "r") as f:
        data = json.load(f)
    T_cam_to_tool = np.array(data["calibration"]["T_cam_to_tool"], dtype=np.float64)
    K = np.load(str(matrix_path))
    D = np.load(str(dist_path))

    tare_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    if tare_path_obj.exists():
        try:
            with open(tare_path_obj, "r") as f:
                t = json.load(f)
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                tare_mm = (float(t[0]), float(t[1]), float(t[2]))
        except Exception:
            pass

    z0_reference = None
    if data_path.exists():
        try:
            with open(data_path, "r") as f:
                hdata = json.load(f)
            z0_reference = hdata.get("metadata", {}).get("z0_reference")
            if z0_reference is not None:
                z0_reference = list(z0_reference)
        except Exception:
            pass

    result = {
        "T_cam_to_tool": T_cam_to_tool,
        "K": K,
        "D": D,
        "tare_mm": tare_mm,
        "z0_reference": z0_reference,
    }
    _calibration_singletons[key] = result
    return result


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------

def reset_camera(arm_name: Optional[str] = None) -> None:
    """Stop and remove camera singleton(s)."""
    if arm_name is not None:
        key = arm_name
        cam = _camera_singletons.pop(key, None)
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass
    else:
        for cam in _camera_singletons.values():
            try:
                cam.stop()
            except Exception:
                pass
        _camera_singletons.clear()


def reset_yolo() -> None:
    global _yolo_active_weights, _vision_config_loaded
    _yolo_models.clear()
    _yolo_active_weights = None
    _vision_config_loaded = False


def reset_calibration(arm_name: Optional[str] = None) -> None:
    if arm_name is not None:
        _calibration_singletons.pop(arm_name, None)
    else:
        _calibration_singletons.clear()
