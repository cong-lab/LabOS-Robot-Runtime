"""
Vision: background YOLO detection, visible_objects, object_within, and helpers.

- visible_objects() -> list of detections (bbox_xyxy, class_id, class_name, conf)
- object_within(obj_a, obj_b, proportion) -> overlap check
- parse_shape, object_point_3d_camera, pick_detection, camera_to_tool, etc.
"""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

DEFAULT_CIRCLE_DIAMETER_MM = 33.0
RACK_HOLE_DIAMETER_MM = 28.0
EPPENDORF_CAP_DIAMETER_MM = 33.0
RACK_PITCH_FALLBACK_MM = 38.0

_vision_thread: Optional[threading.Thread] = None
_vision_stop = threading.Event()
_visible: List[Dict[str, Any]] = []
_visible_lock = threading.Lock()
_vision_started = False
_vision_conf = 0.25
_vision_imgsz = 896
_vision_show_window = True


def _bbox_from_obj(obj: Union[Dict[str, Any], Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        return float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
    if isinstance(obj, dict) and "bbox_xyxy" in obj:
        b = obj["bbox_xyxy"]
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    raise ValueError("object must have 'bbox_xyxy' or be (x1,y1,x2,y2)")


def _bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def object_within(
    obj_a: Union[Dict[str, Any], Tuple[float, float, float, float]],
    obj_b: Union[Dict[str, Any], Tuple[float, float, float, float]],
    proportion: float = 0.5,
) -> bool:
    ba = _bbox_from_obj(obj_a)
    bb = _bbox_from_obj(obj_b)
    area_a = _bbox_area(*ba)
    if area_a <= 0:
        return False
    inter = _bbox_intersection_area(ba, bb)
    return (inter / area_a) >= proportion


def camera_to_tool(p_cam_mm: np.ndarray, T_cam_to_tool: np.ndarray) -> np.ndarray:
    """Transform point from camera frame to tool (EE) frame.
    Thin wrapper around coords.camera_to_ee (without tare; tare is applied by callers)."""
    from aira.coords import camera_to_ee
    return camera_to_ee(p_cam_mm, T_cam_to_tool, tare=(0.0, 0.0, 0.0))


class RackError(RuntimeError):
    """Base class for rack geometry/protocol validation failures."""


class CannotPickUpError(RackError):
    """Raised when a requested tube cannot be safely picked up."""


class CannotPlaceError(RackError):
    """Raised when no safe rack placement is available."""


@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray


@dataclass
class RackDetection:
    class_name: str
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    center_px: Tuple[float, float]
    radius_px: float
    p_cam_mm: np.ndarray
    plane_point_cam_mm: Optional[np.ndarray] = None
    plane_center_px: Optional[Tuple[float, float]] = None
    plane_radius_px: Optional[float] = None
    color: Optional[str] = None
    occupied: bool = False
    model_index: Optional[int] = None


@dataclass
class RackScene:
    holes: List[RackDetection]
    caps: List[RackDetection]
    vortex_holes: List[RackDetection]
    model_holes: List[RackDetection]
    plane: Optional[Plane]
    pitch_mm: float
    pitch_px: float
    K: np.ndarray

    def distance(self, a: RackDetection, b: RackDetection) -> float:
        if self.plane is None:
            return _dist_px(a.center_px, b.center_px)
        ap = a.plane_point_cam_mm if a.plane_point_cam_mm is not None else a.p_cam_mm
        bp = b.plane_point_cam_mm if b.plane_point_cam_mm is not None else b.p_cam_mm
        if np.isfinite(ap).all() and np.isfinite(bp).all():
            return float(np.linalg.norm(ap - bp))
        return _dist_px(a.center_px, b.center_px)

    def occupied_holes(self) -> List[RackDetection]:
        return [hole for hole in self.model_holes if hole.occupied] if self.model_holes else []

    def empty_holes(self) -> List[RackDetection]:
        if self.model_holes:
            return [hole for hole in self.model_holes if not hole.occupied]
        return self.holes

    def nearest_cap_distance(self, item: RackDetection, exclude: Optional[RackDetection] = None) -> float:
        occupied = self.occupied_holes()
        if occupied:
            return min(
                (self.distance(item, hole) for hole in occupied if hole is not exclude),
                default=float("inf"),
            )
        return min(
            (self.distance(item, cap) for cap in self.caps if cap is not exclude),
            default=float("inf"),
        )

    def target_cap(self, color: Optional[str] = None, target_px: Optional[Tuple[float, float]] = None) -> Optional[RackDetection]:
        candidates = self.caps
        if color:
            wanted = color.strip().lower()
            filtered = [cap for cap in candidates if (cap.color or "").lower() == wanted]
            if filtered:
                candidates = filtered
        if not candidates:
            return None
        if target_px is None:
            return max(candidates, key=lambda cap: cap.conf)
        return min(candidates, key=lambda cap: _dist_px(cap.center_px, target_px))

    def model_hole_for_cap(self, cap: RackDetection) -> Optional[RackDetection]:
        if not self.model_holes:
            return None
        return min(self.model_holes, key=lambda hole: self.distance(hole, cap))

    def spaced_empty_holes(self, min_spacing_factor: float = 1.0) -> List[RackDetection]:
        min_clear = self.pitch_mm * min_spacing_factor if self.plane is not None else self.pitch_px * min_spacing_factor
        return [hole for hole in self.empty_holes() if self.nearest_cap_distance(hole) >= min_clear]


def build_plane(points_3d: List[np.ndarray]) -> Optional[Plane]:
    """Fit a least-squares plane through 3D camera-frame points."""
    pts = np.array([p for p in points_3d if np.isfinite(p).all()], dtype=np.float64)
    if pts.shape[0] < 3:
        return None
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm <= 1e-9:
        return None
    normal = normal / norm
    if normal[2] > 0:
        normal = -normal
    return Plane(point=centroid, normal=normal)


def project_along_ray_to_plane(p_cam_mm: np.ndarray, plane: Plane) -> Optional[np.ndarray]:
    """Project a camera-frame point onto *plane* along the camera ray."""
    denom = float(np.dot(plane.normal, p_cam_mm))
    if abs(denom) <= 1e-9:
        return None
    scale = float(np.dot(plane.normal, plane.point) / denom)
    if scale <= 0:
        return None
    projected = p_cam_mm * scale
    return projected if np.isfinite(projected).all() else None


def project_orthogonal_to_plane(p_cam_mm: np.ndarray, plane: Plane) -> Optional[np.ndarray]:
    """Drop a camera-frame point onto *plane* along the plane normal."""
    if not np.isfinite(p_cam_mm).all():
        return None
    projected = p_cam_mm - float(np.dot(plane.normal, p_cam_mm - plane.point)) * plane.normal
    return projected if np.isfinite(projected).all() else None


def point_to_pixel(p_cam_mm: np.ndarray, K: np.ndarray) -> Optional[Tuple[float, float]]:
    if not np.isfinite(p_cam_mm).all() or p_cam_mm[2] <= 0:
        return None
    u = K[0, 0] * p_cam_mm[0] / p_cam_mm[2] + K[0, 2]
    v = K[1, 1] * p_cam_mm[1] / p_cam_mm[2] + K[1, 2]
    return float(u), float(v)


def apparent_radius_px(diameter_mm: float, depth_mm: float, K: np.ndarray) -> float:
    if depth_mm <= 0:
        return 0.0
    return float((K[0, 0] + K[1, 1]) * 0.25 * diameter_mm / depth_mm)


def load_classes_from_yaml(yaml_path: Path) -> Optional[List[str]]:
    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        if isinstance(names, list):
            return names
    except Exception:
        pass
    return None


def load_tare_json(path: Path) -> Optional[Tuple[float, float, float]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, (list, tuple)) and len(data) >= 3:
            return (float(data[0]), float(data[1]), float(data[2]))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _parse_mm(value: Union[int, float, str]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s.endswith("mm"):
        return float(s[:-2].strip())
    return float(s)


def parse_shape(shape: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(shape)
    stype = (out.get("type") or "circle").lower()
    out["type"] = stype
    out["location"] = (out.get("location") or "center").lower()
    if stype == "circle":
        d = out.get("diameter") or out.get("diameter_mm")
        out["diameter_mm"] = _parse_mm(d) if d is not None else DEFAULT_CIRCLE_DIAMETER_MM
    elif stype == "square":
        s = out.get("side") or out.get("side_mm")
        out["side_mm"] = _parse_mm(s) if s is not None else 24.0
    elif stype == "rect":
        out["width_mm"] = _parse_mm(out.get("width") or out.get("width_mm") or 40)
        out["height_mm"] = _parse_mm(out.get("height") or out.get("height_mm") or 30)
    return out


def _bbox_point_uv(bbox_xyxy: Tuple[float, float, float, float], location: str) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if location == "center":
        return cx, cy
    if location == "tl":
        return x1, y1
    if location == "tr":
        return x2, y1
    if location == "bl":
        return x1, y2
    if location == "br":
        return x2, y2
    return cx, cy


def object_point_3d_camera(
    bbox_xyxy: Tuple[float, float, float, float],
    shape: Dict[str, Any],
    K: np.ndarray,
) -> np.ndarray:
    shape = parse_shape(shape)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x1, y1, x2, y2 = bbox_xyxy
    w_px = x2 - x1
    h_px = y2 - y1
    if w_px <= 0 or h_px <= 0:
        return np.full(3, np.nan)
    stype = shape["type"]
    location = shape["location"]
    u, v = _bbox_point_uv(bbox_xyxy, location)
    if stype == "circle":
        size_mm = shape["diameter_mm"]
        size_px = max(w_px, h_px)
    elif stype == "square":
        size_mm = shape["side_mm"]
        size_px = max(w_px, h_px)
    else:
        size_mm = (shape.get("width_mm", 40) * shape.get("height_mm", 30)) ** 0.5
        size_px = (w_px * h_px) ** 0.5
    if size_px <= 0:
        return np.full(3, np.nan)
    Z_cam_mm = fx * (size_mm / size_px)
    x_cam_mm = (u - cx) * Z_cam_mm / fx
    y_cam_mm = (v - cy) * Z_cam_mm / fy
    return np.array([x_cam_mm, y_cam_mm, Z_cam_mm], dtype=np.float64)


def _dist_px(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _normalize_class_name(name: Any) -> str:
    return str(name or "").strip().lower()


def _is_rack_hole_class(name: str) -> bool:
    lowered = _normalize_class_name(name)
    return "rack" in lowered and "hole" in lowered


def _is_eppendorf_cap_top_class(name: str) -> bool:
    lowered = _normalize_class_name(name)
    return "eppendorf" in lowered and "cap top" in lowered


def _is_vortex_hole_class(name: str) -> bool:
    lowered = _normalize_class_name(name)
    return "vortex" in lowered and "hole" in lowered


def _hex_to_hsv(hex_color: str) -> np.ndarray:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        return np.array([0, 0, 0], dtype=np.float32)
    bgr = np.uint8([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)


def _classify_color(
    frame: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    palette: Optional[Dict[str, str]],
) -> Optional[str]:
    if not palette:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(w, int(x2)), min(h, int(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    roi = frame[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > 40
    val_mask = hsv[:, :, 2] > 40
    mask = sat_mask & val_mask
    pixels = hsv[mask] if np.any(mask) else hsv.reshape(-1, 3)
    if pixels.size == 0:
        return None
    sample = np.median(pixels, axis=0).astype(np.float32)
    best_name = None
    best_score = float("inf")
    for name, hex_color in palette.items():
        target = _hex_to_hsv(hex_color)
        hue_delta = abs(float(sample[0] - target[0]))
        hue_delta = min(hue_delta, 180.0 - hue_delta)
        sat_delta = abs(float(sample[1] - target[1])) / 4.0
        val_delta = abs(float(sample[2] - target[2])) / 8.0
        score = hue_delta + sat_delta + val_delta
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def _class_names_from_model(model: Any, names: Optional[Union[Dict[int, str], List[str]]] = None) -> Dict[int, str]:
    raw = names if names is not None else getattr(model, "names", {})
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    return {}


def _median_nearest_distance(points: List[Union[np.ndarray, Tuple[float, float]]]) -> float:
    if len(points) < 2:
        return 0.0
    vals: List[float] = []
    for i, p in enumerate(points):
        nearest = min(
            (float(np.linalg.norm(np.array(p) - np.array(q))) for j, q in enumerate(points) if j != i),
            default=0.0,
        )
        if nearest > 0:
            vals.append(nearest)
    if not vals:
        return 0.0
    vals.sort()
    return vals[len(vals) // 2]


def _dedupe_cap_tops(caps: List[RackDetection]) -> List[RackDetection]:
    """Collapse duplicate generic/color cap-top detections of the same tube."""
    deduped: List[RackDetection] = []
    for cap in sorted(caps, key=lambda item: item.conf, reverse=True):
        duplicate = False
        for kept in deduped:
            if np.isfinite(cap.p_cam_mm).all() and np.isfinite(kept.p_cam_mm).all():
                dist = float(np.linalg.norm(cap.p_cam_mm - kept.p_cam_mm))
                duplicate = dist <= EPPENDORF_CAP_DIAMETER_MM * 0.6
            else:
                duplicate = _dist_px(cap.center_px, kept.center_px) <= max(cap.radius_px, kept.radius_px) * 0.6
            if duplicate:
                # Preserve color information if the higher-confidence generic
                # detection did not classify cleanly but the duplicate did.
                if kept.color is None and cap.color is not None:
                    kept.color = cap.color
                break
        if not duplicate:
            deduped.append(cap)
    return deduped


def analyze_rack(
    frame: np.ndarray,
    model: Any,
    K: np.ndarray,
    names: Optional[Union[Dict[int, str], List[str]]] = None,
    *,
    results: Optional[Any] = None,
    palette: Optional[Dict[str, str]] = None,
    rack_model: Optional[Any] = None,
    conf: float = 0.2,
) -> RackScene:
    """Analyze rack holes and tube caps, projecting tube tops onto the rack plane."""
    class_names = _class_names_from_model(model, names)
    if results is None:
        results = model.predict(frame, conf=conf, imgsz=640, verbose=False)

    holes: List[RackDetection] = []
    caps: List[RackDetection] = []
    vortex_holes: List[RackDetection] = []
    model_holes: List[RackDetection] = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            det_conf = float(boxes.conf[i])
            if det_conf < conf:
                continue
            cls_id = int(boxes.cls[i])
            class_name = class_names.get(cls_id, str(cls_id))
            lowered = _normalize_class_name(class_name)
            is_hole = _is_rack_hole_class(lowered)
            is_cap = _is_eppendorf_cap_top_class(lowered)
            is_vortex_hole = _is_vortex_hole_class(lowered)
            if not is_hole and not is_cap and not is_vortex_hole:
                continue
            x1, y1, x2, y2 = map(float, boxes.xyxy[i].cpu().numpy())
            bbox = (x1, y1, x2, y2)
            center_px = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if is_vortex_hole:
                diameter_mm = 50.0
            elif is_hole:
                diameter_mm = RACK_HOLE_DIAMETER_MM
            else:
                diameter_mm = EPPENDORF_CAP_DIAMETER_MM
            p_cam = object_point_3d_camera(bbox, {"type": "circle", "diameter": diameter_mm, "location": "center"}, K)
            radius_px = max(1.0, max(x2 - x1, y2 - y1) / 2.0)
            det = RackDetection(
                class_name=class_name,
                bbox_xyxy=bbox,
                conf=det_conf,
                center_px=center_px,
                radius_px=radius_px,
                p_cam_mm=p_cam,
                color=_classify_color(frame, bbox, palette) if is_cap else None,
            )
            if is_vortex_hole:
                vortex_holes.append(det)
            elif is_hole:
                det.plane_point_cam_mm = p_cam
                det.plane_center_px = center_px
                det.plane_radius_px = radius_px
                holes.append(det)
            else:
                caps.append(det)

    caps = _dedupe_cap_tops(caps)
    plane = build_plane([hole.p_cam_mm for hole in holes])
    if plane is None and rack_model is not None and len(caps) >= 3:
        # When the rack is mostly full, empty-hole detections can disappear.
        # The cap-top plane is not the rack surface, but it preserves the rack's
        # 2D layout well enough to register the optional model as a geometry hint.
        plane = build_plane([cap.p_cam_mm for cap in caps])
    observed_hole_radius_px = float(np.median([hole.radius_px for hole in holes])) if holes else 0.0
    for cap in caps:
        if plane is not None:
            projected = project_orthogonal_to_plane(cap.p_cam_mm, plane)
            if projected is not None:
                cap.plane_point_cam_mm = projected
                cap.plane_center_px = point_to_pixel(projected, K)
                cap.plane_radius_px = observed_hole_radius_px or apparent_radius_px(
                    RACK_HOLE_DIAMETER_MM, float(projected[2]), K
                )
        if cap.plane_center_px is None:
            cap.plane_center_px = cap.center_px
        if cap.plane_radius_px is None:
            cap.plane_radius_px = cap.radius_px * (RACK_HOLE_DIAMETER_MM / EPPENDORF_CAP_DIAMETER_MM)

    pitch_mm = _median_nearest_distance([hole.p_cam_mm for hole in holes])
    if pitch_mm <= 0:
        pitch_mm = RACK_PITCH_FALLBACK_MM
    pitch_px = _median_nearest_distance([hole.center_px for hole in holes])
    if pitch_px <= 0:
        pitch_px = max((hole.radius_px * 2.0 for hole in holes), default=RACK_PITCH_FALLBACK_MM)
    if rack_model is not None and plane is not None:
        try:
            from aira.vision.rack import plane_basis, points_to_plane_xy, register_rack_model, xy_to_plane_points

            registration_points = [hole.p_cam_mm for hole in holes]
            if len(registration_points) < 2:
                registration_points = [
                    cap.plane_point_cam_mm if cap.plane_point_cam_mm is not None else cap.p_cam_mm
                    for cap in caps
                ]
            basis = plane_basis(registration_points, plane)
            if basis is not None:
                observed_xy = points_to_plane_xy(registration_points, plane, basis)
                registration = register_rack_model(observed_xy, rack_model.holes_xy_mm)
                if registration is not None:
                    model_live_xy = (registration.R @ rack_model.holes_xy_mm.T).T + registration.t
                    model_points = xy_to_plane_points(model_live_xy, basis)
                    model_pitch = float(getattr(rack_model, "pitch_mm", 0.0) or pitch_mm)
                    occupancy_threshold = max(model_pitch * 0.5, RACK_HOLE_DIAMETER_MM * 0.75)
                    for idx, point in enumerate(model_points):
                        center_px = point_to_pixel(point, K)
                        if center_px is None:
                            continue
                        radius_px = observed_hole_radius_px or apparent_radius_px(
                            float(getattr(rack_model, "hole_diameter_mm", RACK_HOLE_DIAMETER_MM)),
                            float(point[2]),
                            K,
                        )
                        model_hole = RackDetection(
                            class_name=f"rack model {rack_model.name} hole",
                            bbox_xyxy=(
                                center_px[0] - radius_px,
                                center_px[1] - radius_px,
                                center_px[0] + radius_px,
                                center_px[1] + radius_px,
                            ),
                            conf=1.0,
                            center_px=center_px,
                            radius_px=radius_px,
                            p_cam_mm=point,
                            plane_point_cam_mm=point,
                            plane_center_px=center_px,
                            plane_radius_px=radius_px,
                            model_index=idx,
                        )
                        model_hole.occupied = any(
                            cap.plane_point_cam_mm is not None
                            and float(np.linalg.norm(point - cap.plane_point_cam_mm)) <= occupancy_threshold
                            for cap in caps
                        )
                        model_holes.append(model_hole)
                    if model_pitch > 0:
                        pitch_mm = model_pitch
        except Exception:
            model_holes = []
    return RackScene(
        holes=holes,
        caps=caps,
        vortex_holes=vortex_holes,
        model_holes=model_holes,
        plane=plane,
        pitch_mm=pitch_mm,
        pitch_px=pitch_px,
        K=K,
    )


def _draw_dashed_circle(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    segments: int = 24,
) -> None:
    if radius <= 0:
        return
    for i in range(segments):
        if i % 2:
            continue
        a0 = 2.0 * np.pi * i / segments
        a1 = 2.0 * np.pi * (i + 1) / segments
        p0 = (int(center[0] + radius * np.cos(a0)), int(center[1] + radius * np.sin(a0)))
        p1 = (int(center[0] + radius * np.cos(a1)), int(center[1] + radius * np.sin(a1)))
        cv2.line(frame, p0, p1, color, thickness)


def draw_rack_overlay(frame: np.ndarray, scene: RackScene) -> np.ndarray:
    """Draw estimated cap, rack-hole, and projected filled-hole circles."""
    out = frame.copy()
    for hole in scene.model_holes:
        c = (int(round(hole.center_px[0])), int(round(hole.center_px[1])))
        r = max(2, int(round(hole.radius_px)))
        color = (0, 0, 255) if hole.occupied else (180, 180, 180)
        cv2.circle(out, c, r, color, 1)
        label = "model occupied" if hole.occupied else "model empty"
        cv2.putText(out, label, (c[0] - r, c[1] + r + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    for hole in scene.holes:
        c = (int(round(hole.center_px[0])), int(round(hole.center_px[1])))
        r = max(2, int(round(hole.radius_px)))
        cv2.circle(out, c, r, (255, 255, 0), 2)
        cv2.putText(out, "hole", (c[0] - r, max(12, c[1] - r - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    for hole in scene.vortex_holes:
        c = (int(round(hole.center_px[0])), int(round(hole.center_px[1])))
        r = max(2, int(round(hole.radius_px)))
        cv2.circle(out, c, r, (255, 0, 255), 2)
        cv2.putText(out, "vortex hole", (c[0] - r, max(12, c[1] - r - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    for cap in scene.caps:
        c = (int(round(cap.center_px[0])), int(round(cap.center_px[1])))
        r = max(2, int(round(cap.radius_px)))
        cv2.circle(out, c, r, (0, 165, 255), 2)
        label = f"cap {cap.color}" if cap.color else "cap"
        cv2.putText(out, label, (c[0] - r, max(12, c[1] - r - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        if cap.plane_center_px is None:
            continue
        pc = (int(round(cap.plane_center_px[0])), int(round(cap.plane_center_px[1])))
        pr = max(2, int(round(cap.plane_radius_px or r)))
        _draw_dashed_circle(out, pc, pr, (0, 0, 255), 2)
        cv2.putText(out, "filled", (pc[0] - pr, pc[1] + pr + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return out


def pick_detection(
    detections: List[Dict[str, Any]],
    pick_type: Union[str, Tuple[float, float]],
    image_shape: Tuple[int, int],
    T_cam_to_tool: np.ndarray,
    tare_arr: np.ndarray,
) -> Optional[Dict[str, Any]]:
    if not detections:
        return None
    # Support tuple (px_x, px_y) to pick detection closest to that pixel location
    if isinstance(pick_type, tuple) and len(pick_type) == 2:
        target_px = pick_type
        def key_px(d):
            b = d["bbox_xyxy"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            return (cx - target_px[0]) ** 2 + (cy - target_px[1]) ** 2
        return min(detections, key=key_px)
    pick = (pick_type or "toolhead_close").lower()
    H, W = image_shape[:2]
    cam_center = (W / 2.0, H / 2.0)
    if pick == "toolhead_close":
        def key(d):
            p = d.get("p_tool_mm")
            if p is None or not np.isfinite(p).all():
                return float("inf")
            return np.sqrt(p[0] ** 2 + p[1] ** 2)
        detections = sorted(detections, key=key)
        return detections[0] if key(detections[0]) != float("inf") else None
    if pick == "camera_center":
        def key(d):
            b = d["bbox_xyxy"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            return (cx - cam_center[0]) ** 2 + (cy - cam_center[1]) ** 2
        return min(detections, key=key)
    if pick == "largest":
        def key(d):
            b = d["bbox_xyxy"]
            return -((b[2] - b[0]) * (b[3] - b[1]))
        return min(detections, key=key)
    if pick == "highest_confidence":
        return max(detections, key=lambda d: d.get("conf", 0.0))
    if pick == "ranked":
        # Rank by area (1 = largest) and by confidence (1 = highest); pick lowest average rank.
        def area(d):
            b = d["bbox_xyxy"]
            return (b[2] - b[0]) * (b[3] - b[1])
        def conf(d):
            return d.get("conf", 0.0)
        by_area = sorted(detections, key=area, reverse=True)
        by_conf = sorted(detections, key=conf, reverse=True)
        rank_area = {id(d): i + 1 for i, d in enumerate(by_area)}
        rank_conf = {id(d): i + 1 for i, d in enumerate(by_conf)}
        def avg_rank(d):
            return (rank_area[id(d)] + rank_conf[id(d)]) / 2.0
        return min(detections, key=avg_rank)
    if pick == "ranked_vertical":
        # 1. Compute centre-x and average width of all detections
        cxs: List[float] = []
        widths: List[float] = []
        for d in detections:
            b = d["bbox_xyxy"]
            cxs.append((b[0] + b[2]) / 2.0)
            widths.append(b[2] - b[0])
        avg_w = sum(widths) / len(widths) if widths else 1.0
        col_tol = 0.75 * avg_w

        # 2. Cluster into vertical columns by cx proximity
        indexed = sorted(range(len(detections)), key=lambda i: cxs[i])
        columns: List[List[int]] = []
        for idx in indexed:
            if columns:
                col_avg_cx = sum(cxs[j] for j in columns[-1]) / len(columns[-1])
                if abs(cxs[idx] - col_avg_cx) <= col_tol:
                    columns[-1].append(idx)
                    continue
            columns.append([idx])

        def _tool_dist_sq(j: int) -> float:
            p = detections[j].get("p_tool_mm")
            if p is None or not np.isfinite(p).all():
                return float("inf")
            return float(p[0]) ** 2 + float(p[1]) ** 2

        # 3. Rank columns by avg tool-frame distance of their 3 closest members
        def _col_score(col: List[int]) -> float:
            dists = sorted(_tool_dist_sq(j) for j in col)
            top = dists[:3]
            return sum(top) / len(top) if top else float("inf")
        columns.sort(key=_col_score)

        # 4. In the winning column, pick the single closest to toolhead
        best_idx = min(columns[0], key=_tool_dist_sq)
        return detections[best_idx]
    if pick == "tl":
        detections = sorted(detections, key=lambda d: (d["bbox_xyxy"][1], d["bbox_xyxy"][0]))
        return detections[0]
    if pick == "tr":
        detections = sorted(detections, key=lambda d: (d["bbox_xyxy"][1], -d["bbox_xyxy"][2]))
        return detections[0]
    if pick == "bl":
        detections = sorted(detections, key=lambda d: (-d["bbox_xyxy"][3], d["bbox_xyxy"][0]))
        return detections[0]
    if pick == "br":
        detections = sorted(detections, key=lambda d: (-d["bbox_xyxy"][3], -d["bbox_xyxy"][2]))
        return detections[0]
    return detections[0]


def resolve_class_to_index(classes: List[str], yolo_class: Optional[Union[str, int]]) -> int:
    if yolo_class is None:
        return 0
    if isinstance(yolo_class, int):
        return max(0, min(yolo_class, len(classes) - 1))
    name = str(yolo_class).strip().lower()
    for i, c in enumerate(classes):
        if name == str(c).strip().lower():
            return i
    for i, c in enumerate(classes):
        if name in c.lower():
            return i
    return 0


def _detection_loop(
    conf: float = 0.25,
    imgsz: int = 896,
    poll_interval: float = 0.05,
    show_window: bool = True,
) -> None:
    global _visible
    from aira.vision.singletons import camera, yolo
    cam = camera()
    model = yolo()
    class_names = getattr(model, "names", None)
    if class_names is not None and isinstance(class_names, dict):
        names_list = [class_names.get(i, str(i)) for i in sorted(class_names.keys())]
    else:
        names_list = []
    window_name = "Vision"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while not _vision_stop.is_set():
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(poll_interval)
            continue
        try:
            results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        except Exception:
            time.sleep(poll_interval)
            continue
        out: List[Dict[str, Any]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(float, box)
                cls_id = int(boxes.cls[i])
                conf_val = float(boxes.conf[i])
                name = names_list[cls_id] if cls_id < len(names_list) else str(cls_id)
                out.append({
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": name,
                    "conf": conf_val,
                })
        with _visible_lock:
            _visible = out
        if show_window and frame is not None:
            disp = frame.copy()
            for d in out:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                label = f"{d['class_name']} {d['conf']:.2f}"
                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(disp, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(disp, f"Detections: {len(out)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_name, disp)
            cv2.waitKey(1)
        time.sleep(poll_interval)
    if show_window:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


def visible_objects(
    *,
    conf: float = 0.25,
    imgsz: int = 896,
    show_window: bool = True,
) -> List[Dict[str, Any]]:
    global _vision_thread, _vision_started, _vision_conf, _vision_imgsz, _vision_show_window
    if not _vision_started:
        with _visible_lock:
            if not _vision_started:
                _vision_conf = conf
                _vision_imgsz = imgsz
                _vision_show_window = show_window
                _vision_stop.clear()
                _vision_thread = threading.Thread(
                    target=_detection_loop,
                    kwargs={
                        "conf": _vision_conf,
                        "imgsz": _vision_imgsz,
                        "show_window": _vision_show_window,
                    },
                    daemon=True,
                )
                _vision_thread.start()
                _vision_started = True
    with _visible_lock:
        return list(_visible)


def stop_vision() -> None:
    global _vision_started
    _vision_stop.set()
    if _vision_thread is not None and _vision_thread.is_alive():
        _vision_thread.join(timeout=2.0)
    try:
        cv2.destroyWindow("Vision")
    except Exception:
        pass
    _vision_started = False
