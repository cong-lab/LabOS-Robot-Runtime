import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from aira.vision.vision import Plane, build_plane, project_orthogonal_to_plane

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RACK_DIR = ROOT / "configs" / "racks"


@dataclass
class RackModel:
    name: str
    hole_diameter_mm: float
    holes_xy_mm: np.ndarray

    @property
    def pitch_mm(self) -> float:
        if len(self.holes_xy_mm) < 2:
            return 0.0
        vals = []
        for i, point in enumerate(self.holes_xy_mm):
            nearest = min(
                (
                    float(np.linalg.norm(point - other))
                    for j, other in enumerate(self.holes_xy_mm)
                    if j != i
                ),
                default=0.0,
            )
            if nearest > 0:
                vals.append(nearest)
        if not vals:
            return 0.0
        vals.sort()
        return vals[len(vals) // 2]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "hole_diameter_mm": float(self.hole_diameter_mm),
            "holes_xy_mm": self.holes_xy_mm.astype(float).tolist(),
        }

    @classmethod
    def from_json(cls, data: dict) -> "RackModel":
        return cls(
            name=str(data["name"]),
            hole_diameter_mm=float(data.get("hole_diameter_mm", 28.0)),
            holes_xy_mm=np.array(data.get("holes_xy_mm", []), dtype=np.float64),
        )


@dataclass
class RackRegistration:
    R: np.ndarray
    t: np.ndarray
    inliers: List[Tuple[int, int]]
    error_mm: float


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-")
    if not value:
        raise ValueError("rack model name cannot be empty")
    return value


def rack_model_path(name: str, rack_dir: Path = DEFAULT_RACK_DIR) -> Path:
    return rack_dir / f"{_safe_name(name)}.json"


def save_rack_model(model: RackModel, rack_dir: Path = DEFAULT_RACK_DIR) -> Path:
    rack_dir.mkdir(parents=True, exist_ok=True)
    path = rack_model_path(model.name, rack_dir)
    path.write_text(json.dumps(model.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


def load_rack_model(name: str, rack_dir: Path = DEFAULT_RACK_DIR) -> RackModel:
    path = rack_model_path(name, rack_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    return RackModel.from_json(data)


def plane_basis(points_3d: List[np.ndarray], plane: Optional[Plane] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pts = np.array([p for p in points_3d if np.isfinite(p).all()], dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    if plane is None:
        plane = build_plane(list(pts))
        if plane is None:
            return None
    projected = []
    for point in pts:
        p = project_orthogonal_to_plane(point, plane)
        if p is not None:
            projected.append(p)
    if len(projected) < 2:
        return None
    projected_arr = np.array(projected, dtype=np.float64)
    origin = projected_arr.mean(axis=0)
    centered = projected_arr - origin
    _, _, vh = np.linalg.svd(centered)
    x_axis = vh[0]
    x_axis = x_axis - float(np.dot(x_axis, plane.normal)) * plane.normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1e-9:
        return None
    x_axis = x_axis / x_norm
    y_axis = np.cross(plane.normal, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm <= 1e-9:
        return None
    y_axis = y_axis / y_norm
    return origin, x_axis, y_axis


def points_to_plane_xy(points_3d: List[np.ndarray], plane: Plane, basis: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    origin, x_axis, y_axis = basis
    xy = []
    for point in points_3d:
        projected = project_orthogonal_to_plane(point, plane)
        if projected is None:
            continue
        rel = projected - origin
        xy.append([float(np.dot(rel, x_axis)), float(np.dot(rel, y_axis))])
    return np.array(xy, dtype=np.float64)


def xy_to_plane_points(xy: np.ndarray, basis: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> List[np.ndarray]:
    origin, x_axis, y_axis = basis
    return [origin + float(p[0]) * x_axis + float(p[1]) * y_axis for p in np.asarray(xy, dtype=np.float64)]


def build_rack_model_from_points(
    name: str,
    points_3d: List[np.ndarray],
    hole_diameter_mm: float = 28.0,
) -> Tuple[RackModel, Plane]:
    plane = build_plane(points_3d)
    if plane is None:
        raise ValueError("at least 3 valid rack-hole points are required to build a rack model")
    basis = plane_basis(points_3d, plane)
    if basis is None:
        raise ValueError("could not derive rack plane basis")
    xy = points_to_plane_xy(points_3d, plane, basis)
    if xy.shape[0] < 3:
        raise ValueError("at least 3 projected rack-hole points are required")
    model = RackModel(name=_safe_name(name), hole_diameter_mm=hole_diameter_mm, holes_xy_mm=xy)
    return model, plane


def _procrustes(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    return R, t


def _nearest_match_errors(transformed_model_xy: np.ndarray, observed_xy: np.ndarray) -> List[Tuple[float, int, int]]:
    matches = []
    for mi, point in enumerate(transformed_model_xy):
        dists = np.linalg.norm(observed_xy - point, axis=1)
        oi = int(np.argmin(dists))
        matches.append((float(dists[oi]), mi, oi))
    return sorted(matches, key=lambda item: item[0])


def register_rack_model(
    observed_xy: np.ndarray,
    model_xy: np.ndarray,
    *,
    tolerance_mm: Optional[float] = None,
) -> Optional[RackRegistration]:
    observed = np.asarray(observed_xy, dtype=np.float64)
    model = np.asarray(model_xy, dtype=np.float64)
    if observed.shape[0] < 2 or model.shape[0] < 2:
        return None
    if tolerance_mm is None:
        pitch = RackModel("tmp", 28.0, model).pitch_mm
        tolerance_mm = max(8.0, pitch * 0.35) if pitch > 0 else 12.0

    best: Optional[RackRegistration] = None
    observed_pairs = [(i, j) for i in range(len(observed)) for j in range(i + 1, len(observed))]
    model_pairs = [(i, j) for i in range(len(model)) for j in range(i + 1, len(model))]
    for m0, m1 in model_pairs:
        mv = model[m1] - model[m0]
        md = float(np.linalg.norm(mv))
        if md <= 1e-6:
            continue
        ma = np.arctan2(mv[1], mv[0])
        for o0, o1 in observed_pairs:
            ov = observed[o1] - observed[o0]
            od = float(np.linalg.norm(ov))
            if od <= 1e-6 or abs(od - md) > tolerance_mm:
                continue
            oa = np.arctan2(ov[1], ov[0])
            theta = oa - ma
            R = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                dtype=np.float64,
            )
            for flip in (False, True):
                candidate_R = -R if flip else R
                t = observed[o0] - (candidate_R @ model[m0])
                transformed = (candidate_R @ model.T).T + t
                raw_matches = _nearest_match_errors(transformed, observed)
                used_model = set()
                used_observed = set()
                inliers = []
                for err, mi, oi in raw_matches:
                    if err > tolerance_mm or mi in used_model or oi in used_observed:
                        continue
                    used_model.add(mi)
                    used_observed.add(oi)
                    inliers.append((mi, oi))
                if len(inliers) < 2:
                    continue
                src = np.array([model[mi] for mi, _ in inliers], dtype=np.float64)
                dst = np.array([observed[oi] for _, oi in inliers], dtype=np.float64)
                refined_R, refined_t = _procrustes(src, dst)
                transformed_refined = (refined_R @ model.T).T + refined_t
                refined_matches = _nearest_match_errors(transformed_refined, observed)
                used_model.clear()
                used_observed.clear()
                refined_inliers = []
                errors = []
                for err, mi, oi in refined_matches:
                    if err > tolerance_mm or mi in used_model or oi in used_observed:
                        continue
                    used_model.add(mi)
                    used_observed.add(oi)
                    refined_inliers.append((mi, oi))
                    errors.append(err)
                if len(refined_inliers) < 2:
                    continue
                mean_error = float(np.mean(errors)) if errors else float("inf")
                candidate = RackRegistration(refined_R, refined_t, refined_inliers, mean_error)
                if best is None:
                    best = candidate
                    continue
                if len(candidate.inliers) > len(best.inliers) or (
                    len(candidate.inliers) == len(best.inliers) and candidate.error_mm < best.error_mm
                ):
                    best = candidate
    return best
