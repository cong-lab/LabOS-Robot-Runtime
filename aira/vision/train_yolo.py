#!/usr/bin/env python3
"""
YOLO Training Script (detect + segment)

- Detect: fine-tunes RT-DETR large (or YOLO-World) on custom COCO (bbox only).
- Segment: fine-tunes YOLO segment model (e.g. yolov8l-seg) with COCO polygons;
  mask loss uses BCE + Dice (Ultralytics default).

Usage:
    python train_yolo.py
    python train_yolo.py --model rtdetr-l.pt --epochs 50
    python train_yolo.py --task segment --segment-model yolov8l-seg.pt --split-dataset
    python train_yolo.py --data dataset.yaml --batch 16
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random
import math

from aira.utils.paths import get_project_root

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: pyyaml not available")

try:
    from ultralytics import YOLOWorld, YOLO, RTDETR
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False
    sys.exit(1)


def load_coco_categories(json_path: Path) -> Dict[int, str]:
    """Load category mapping from COCO JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping: COCO category ID -> name
    categories = {}
    for cat in data['categories']:
        categories[cat['id']] = cat['name']
    
    return categories


def estimate_class_weights_from_coco(
    coco_json: Path, *, min_weight: float = 0.5, max_weight: float = 2.0
) -> tuple[Dict[str, float], Dict[str, int], float]:
    """Estimate per-class weights from COCO annotations using inverse-sqrt frequency."""
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    counts_by_id: Dict[int, int] = {}
    for ann in data.get("annotations", []):
        cid = ann.get("category_id")
        counts_by_id[cid] = counts_by_id.get(cid, 0) + 1

    nonzero_counts = [counts_by_id.get(c["id"], 0) for c in categories if counts_by_id.get(c["id"], 0) > 0]
    median_count = float(sorted(nonzero_counts)[len(nonzero_counts) // 2]) if nonzero_counts else 1.0

    class_weights: Dict[str, float] = {}
    class_counts: Dict[str, int] = {}
    for cat in categories:
        name = cat["name"]
        count = counts_by_id.get(cat["id"], 0)
        class_counts[name] = count
        if count > 0:
            weight = math.sqrt(median_count / float(count))
            weight = max(min_weight, min(max_weight, weight))
        else:
            # Keep neutral until class has labeled samples.
            weight = 1.0
        class_weights[name] = round(weight, 3)

    return class_weights, class_counts, median_count


def write_class_weighting_yaml_from_dataset(
    dataset_dir: Path, output_yaml: Path, *, min_weight: float = 0.5, max_weight: float = 2.0
) -> Path:
    """Generate class weighting config from <dataset_dir>/annotations.json."""
    if not YAML_AVAILABLE:
        raise RuntimeError("pyyaml is required to write class weighting yaml")

    coco_json = dataset_dir / "annotations.json"
    if not coco_json.exists():
        raise FileNotFoundError(f"Could not find annotations file for weighting: {coco_json}")

    class_weights, class_counts, median_count = estimate_class_weights_from_coco(
        coco_json, min_weight=min_weight, max_weight=max_weight
    )
    payload = {
        "version": 1,
        "dataset": str(dataset_dir),
        "source_annotations": str(coco_json),
        "weighting_method": "inverse_sqrt_frequency_clamped",
        "normalization": "median_frequency",
        "clamp": {"min": float(min_weight), "max": float(max_weight)},
        "median_count_nonzero": float(median_count),
        "class_counts": class_counts,
        "class_weights": class_weights,
    }
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"Created class weighting config: {output_yaml}")
    return output_yaml


def _class_names_from_data_yaml(data_yaml: Path) -> List[str]:
    """Read class names from Ultralytics data yaml preserving class index order."""
    if not YAML_AVAILABLE:
        return []
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        def _to_int(k):
            try:
                return int(k)
            except Exception:
                return k
        return [str(v) for _, v in sorted(names.items(), key=lambda kv: _to_int(kv[0]))]
    return []


def _load_class_weight_vector(weighting_yaml: Path, class_names: List[str]):
    """Load class weights aligned to class_names; defaults missing classes to 1.0."""
    if not YAML_AVAILABLE or not weighting_yaml.exists() or not class_names:
        return None
    with open(weighting_yaml, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    weight_map = payload.get("class_weights", {})
    if not isinstance(weight_map, dict):
        return None
    weights = [float(weight_map.get(name, 1.0)) for name in class_names]
    if not any(abs(w - 1.0) > 1e-9 for w in weights):
        return None
    if not TORCH_AVAILABLE:
        return None
    return torch.tensor(weights, dtype=torch.float32)


def _install_class_weighting(model, class_weight_vector, class_names: List[str]) -> None:
    """Inject class-wise BCE pos_weight into Ultralytics criterion at init time."""
    if class_weight_vector is None:
        return
    if not TORCH_AVAILABLE:
        print("Class weighting requested but torch unavailable; skipping.")
        return
    if not hasattr(model, "model") or not hasattr(model.model, "init_criterion"):
        print("Class weighting not supported for this model wrapper; skipping.")
        return

    original_init_criterion = model.model.init_criterion

    def _wrapped_init_criterion():
        criterion = original_init_criterion()
        if not hasattr(criterion, "bce"):
            print("Class weighting: criterion has no BCE cls loss; skipping.")
            return criterion
        device = getattr(criterion, "device", None)
        if device is None:
            try:
                device = next(model.model.parameters()).device
            except Exception:
                device = "cpu"
        pos_weight = class_weight_vector.to(device)
        criterion.bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        preview = ", ".join(
            f"{n}={float(w):.3f}" for n, w in zip(class_names[:6], pos_weight[:6])
        )
        if len(class_names) > 6:
            preview += ", ..."
        print(f"Applied class weighting (pos_weight) for {len(class_names)} classes: {preview}")
        return criterion

    model.model.init_criterion = _wrapped_init_criterion


def _discover_dataset_paths(dataset_dir: Path) -> Tuple[Path, str]:
    """Auto-discover annotations JSON and images subdirectory inside a dataset dir.

    Returns (annotations_path, images_subdir) where images_subdir is the
    relative subdirectory under <dataset_dir>/images/ (e.g. "Train"), or
    empty string if images are directly in images/.
    """
    def _pick_annotations(json_files: List[Path]) -> Path:
        """Prefer unsplit source annotations over *_train/*_val splits."""
        if not json_files:
            raise FileNotFoundError("No annotations JSON files found")

        by_name = {p.name: p for p in json_files}
        preferred = [
            "annotations.json",
            "annotations_combined.json",
            "annotations_all.json",
        ]
        for name in preferred:
            if name in by_name:
                return by_name[name]

        unsplit = [
            p for p in json_files
            if not p.stem.endswith("_train") and not p.stem.endswith("_val")
        ]
        if unsplit:
            return sorted(unsplit)[0]

        # Fallback: only split files are present.
        return sorted(json_files)[0]

    ann_dir = dataset_dir / "annotations"
    if ann_dir.is_dir():
        json_files = sorted(ann_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {ann_dir}")
        annotations_path = _pick_annotations(json_files)
    else:
        json_files = sorted(dataset_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No annotations JSON found in {dataset_dir} or {ann_dir}"
            )
        annotations_path = _pick_annotations(json_files)
    print(f"Auto-discovered annotations: {annotations_path}")

    images_root = dataset_dir / "images"
    images_subdir = ""
    if images_root.is_dir():
        subdirs = [d for d in sorted(images_root.iterdir()) if d.is_dir()]
        top_level_images = (
            list(images_root.glob("*.png")) + list(images_root.glob("*.jpg"))
        )
        if subdirs:
            # Choose subdir by matching annotation filenames to actual files.
            # This avoids accidentally selecting an old split folder (e.g. Train)
            # when current annotations point to another folder (e.g. default).
            try:
                with open(annotations_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                ann_names = [
                    Path(img.get("file_name", "")).name
                    for img in ann.get("images", [])
                    if img.get("file_name")
                ]
            except Exception:
                ann_names = []

            best_subdir = None
            best_hits = -1
            if ann_names:
                for d in subdirs:
                    hits = 0
                    for name in ann_names[:200]:
                        if (d / name).exists():
                            hits += 1
                    if hits > best_hits:
                        best_hits = hits
                        best_subdir = d

            if best_subdir is not None and best_hits > 0:
                images_subdir = best_subdir.name
                print(
                    f"Auto-discovered images subdirectory: images/{images_subdir}/ "
                    f"(matched {best_hits}/{min(len(ann_names), 200)} annotation filenames)"
                )
            elif top_level_images:
                images_subdir = ""
                print("Auto-discovered images at top-level images/ (no subdir match)")
            else:
                images_subdir = subdirs[0].name
                print(
                    f"Auto-discovered images subdirectory: images/{images_subdir}/ "
                    f"(fallback: first subdir)"
                )

    return annotations_path, images_subdir


def split_train_val(
    annotations_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Path, Path]:
    """
    Split COCO dataset into train and validation sets.
    
    Returns paths to train and val JSON files.
    """
    print(f"Loading annotations from: {annotations_path}")
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Get unique image IDs
    image_ids = list(set(img['id'] for img in data['images']))
    random.seed(seed)
    random.shuffle(image_ids)
    
    # Split
    split_idx = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])
    
    print(f"  Total images: {len(image_ids)}")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/len(image_ids)*100:.1f}%)")
    print(f"  Val: {len(val_ids)} ({len(val_ids)/len(image_ids)*100:.1f}%)")
    
    # Create train dataset
    train_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': [img for img in data['images'] if img['id'] in train_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in train_ids]
    }
    
    # Create val dataset
    val_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': [img for img in data['images'] if img['id'] in val_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in val_ids]
    }
    
    # Save split datasets
    dataset_dir = annotations_path.parent
    train_json = dataset_dir / 'annotations_train.json'
    val_json = dataset_dir / 'annotations_val.json'
    
    with open(train_json, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_json, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"  Saved train annotations: {train_json}")
    print(f"  Saved val annotations: {val_json}")
    
    return train_json, val_json


def coco_to_yolo(
    coco_json: Path,
    images_dir: Path,
    labels_dir: Path,
    categories: Dict[int, str]
):
    """
    Convert COCO format annotations to YOLO format (txt files).
    
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    """
    print(f"Converting COCO to YOLO format...")
    print(f"  Input: {coco_json}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    
    # Create labels directory
    labels_dir.mkdir(exist_ok=True, parents=True)
    
    # Load COCO data
    with open(coco_json, 'r') as f:
        data = json.load(f)
    
    # Create mapping: COCO category ID -> YOLO class ID (0-indexed)
    # COCO categories might start at 1, but we need 0-indexed for YOLO
    sorted_cats = sorted(categories.items())
    coco_to_yolo_id = {coco_id: yolo_id for yolo_id, (coco_id, _) in enumerate(sorted_cats)}
    
    print(f"  Category mapping: {coco_to_yolo_id}")
    
    # Create image ID -> image info mapping
    image_map = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Convert each image's annotations
    labels_created = 0
    total_annotations = 0
    images_with_labels = 0
    
    for image_id, image_info in image_map.items():
        image_filename = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Create label file path (same name as image, but .txt)
        label_filename = Path(image_filename).stem + '.txt'
        label_path = labels_dir / label_filename
        
        # Get annotations for this image
        annotations = annotations_by_image.get(image_id, [])
        
        # Write YOLO format labels
        label_lines = []
        for ann in annotations:
            # Get YOLO class ID
            coco_cat_id = ann['category_id']
            yolo_class_id = coco_to_yolo_id.get(coco_cat_id)
            
            if yolo_class_id is None:
                print(f"  Warning: Category ID {coco_cat_id} not found in mapping (available: {list(coco_to_yolo_id.keys())})")
                continue
            
            # Get bounding box (COCO format: [x, y, width, height] in pixels)
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to YOLO format (normalized center coordinates)
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            norm_width = w / image_width
            norm_height = h / image_height
            
            # Clamp to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            # Write: class_id center_x center_y width height
            label_lines.append(f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        # Write label file (even if empty, to match all images)
        with open(label_path, 'w') as f:
            f.writelines(label_lines)
        
        if len(label_lines) > 0:
            labels_created += 1
            images_with_labels += 1
            total_annotations += len(label_lines)
        else:
            # Create empty label file for images without annotations
            labels_created += 1
    
    print(f"  Created {labels_created} label files ({images_with_labels} with annotations, {labels_created - images_with_labels} empty)")
    print(f"  Total annotations converted: {total_annotations}")
    return labels_dir


def _segmentation_to_polygon(ann: dict, image_width: int, image_height: int) -> List[Tuple[float, float]]:
    """
    Get normalized polygon from COCO annotation (polygon or RLE).
    Returns list of (x, y) normalized 0-1; at least 3 points for YOLO segment.
    """
    seg = ann.get('segmentation')
    if not seg:
        # Fallback: bbox as 4-point polygon
        bbox = ann.get('bbox', [0, 0, 1, 1])
        x, y, w, h = bbox
        pts = [
            (x / image_width, y / image_height),
            ((x + w) / image_width, y / image_height),
            ((x + w) / image_width, (y + h) / image_height),
            (x / image_width, (y + h) / image_height),
        ]
        return [(max(0, min(1, p[0])), max(0, min(1, p[1]))) for p in pts]

    # RLE format: {"counts": ..., "size": [h, w]} - fallback to bbox (no pycocotools required)
    if isinstance(seg, dict):
        bbox = ann.get('bbox', [0, 0, 1, 1])
        x, y, w, h = bbox
        pts = [(x / image_width, y / image_height), ((x + w) / image_width, y / image_height),
               ((x + w) / image_width, (y + h) / image_height), (x / image_width, (y + h) / image_height)]
        return [(max(0, min(1, p[0])), max(0, min(1, p[1]))) for p in pts]

    # Polygon format: list of lists [x1,y1,x2,y2,...]
    polygons = seg if isinstance(seg[0], (list, tuple)) else [seg]
    best = []
    for poly in polygons:
        if isinstance(poly, (list, tuple)) and len(poly) >= 6:  # 3 points = 6 numbers
            coords = []
            for i in range(0, len(poly), 2):
                if i + 1 < len(poly):
                    nx = max(0, min(1, poly[i] / image_width))
                    ny = max(0, min(1, poly[i + 1] / image_height))
                    coords.append((nx, ny))
            if len(coords) >= 3:
                best = coords
                break
    if not best and ann.get('bbox'):
        x, y, w, h = ann['bbox']
        best = [(x / image_width, y / image_height), ((x + w) / image_width, y / image_height),
                ((x + w) / image_width, (y + h) / image_height), (x / image_width, (y + h) / image_height)]
        best = [(max(0, min(1, p[0])), max(0, min(1, p[1]))) for p in best]
    return best


def coco_to_yolo_segment(
    coco_json: Path,
    images_dir: Path,
    labels_dir: Path,
    categories: Dict[int, str]
):
    """
    Convert COCO format annotations to YOLO segment format (txt files).

    YOLO segment format: class_id x1 y1 x2 y2 ... xn yn (normalized 0-1).
    Minimum 3 points per polygon. Uses BCE + Dice loss in Ultralytics segment training.
    """
    print(f"Converting COCO to YOLO segment format...")
    print(f"  Input: {coco_json}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")

    labels_dir.mkdir(exist_ok=True, parents=True)

    with open(coco_json, 'r') as f:
        data = json.load(f)

    sorted_cats = sorted(categories.items())
    coco_to_yolo_id = {coco_id: yolo_id for yolo_id, (coco_id, _) in enumerate(sorted_cats)}
    print(f"  Category mapping: {coco_to_yolo_id}")

    image_map = {img['id']: img for img in data['images']}
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    labels_created = 0
    total_annotations = 0
    images_with_labels = 0
    skipped_no_poly = 0

    for image_id, image_info in image_map.items():
        image_filename = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        label_filename = Path(image_filename).stem + '.txt'
        label_path = labels_dir / label_filename

        annotations = annotations_by_image.get(image_id, [])
        label_lines = []

        for ann in annotations:
            coco_cat_id = ann['category_id']
            yolo_class_id = coco_to_yolo_id.get(coco_cat_id)
            if yolo_class_id is None:
                continue

            polygon = _segmentation_to_polygon(ann, image_width, image_height)
            if len(polygon) < 3:
                skipped_no_poly += 1
                continue

            # YOLO segment line: class_id x1 y1 x2 y2 ... xn yn (interleaved x,y)
            flat = [str(yolo_class_id)]
            for p in polygon:
                flat.append(f"{p[0]:.6f}")
                flat.append(f"{p[1]:.6f}")
            label_lines.append(" ".join(flat) + "\n")

        with open(label_path, 'w') as f:
            f.writelines(label_lines)

        if len(label_lines) > 0:
            labels_created += 1
            images_with_labels += 1
            total_annotations += len(label_lines)
        else:
            labels_created += 1

    if skipped_no_poly:
        print(f"  Skipped {skipped_no_poly} annotations (no valid polygon).")
    print(f"  Created {labels_created} label files ({images_with_labels} with annotations)")
    print(f"  Total segment annotations: {total_annotations}")
    return labels_dir


def create_yolo_dataset_yaml(
    dataset_dir: Path,
    train_json: Path,
    val_json: Path,
    categories: Dict[int, str],
    output_yaml: Path,
    task: str = "detect",
    images_subdir: str = "",
):
    """Create Ultralytics-compatible YAML dataset file and convert COCO to YOLO format.
    task: 'detect' -> bbox labels; 'segment' -> polygon labels (BCE + Dice mask loss).
    images_subdir: subdirectory under images/ where images reside (e.g. 'Train').
    """
    sorted_cats = sorted(categories.items())
    class_names = [cat[1] for cat in sorted_cats]

    images_rel = f"images/{images_subdir}" if images_subdir else "images"
    labels_rel = f"labels/{images_subdir}" if images_subdir else "labels"
    images_dir = dataset_dir / images_rel
    labels_dir = dataset_dir / labels_rel

    def _clear_existing_labels_and_cache(target_labels_dir: Path) -> None:
        """Remove stale label txt files and Ultralytics cache files before conversion."""
        target_labels_dir.mkdir(exist_ok=True, parents=True)
        stale_txt = list(target_labels_dir.glob("*.txt"))
        for p in stale_txt:
            p.unlink(missing_ok=True)

        # Ultralytics often creates cache files alongside labels directory,
        # e.g. labels/Train.cache or labels.cache.
        cache_candidates = [
            dataset_dir / "labels.cache",
            dataset_dir / "labels" / "labels.cache",
            dataset_dir / "labels" / f"{images_subdir}.cache" if images_subdir else None,
            target_labels_dir.with_suffix(".cache"),
        ]
        for c in cache_candidates:
            if c is not None and c.exists():
                print(f"Removing stale cache file: {c}")
                c.unlink(missing_ok=True)

    _clear_existing_labels_and_cache(labels_dir)
    
    # Load both train and val to get all images
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    
    # Combine annotations (each image only appears in one set)
    all_images = {img['id']: img for img in train_data['images'] + val_data['images']}
    all_annotations = train_data['annotations'] + val_data['annotations']
    
    # Create combined data structure for conversion
    combined_data = {
        'images': list(all_images.values()),
        'annotations': all_annotations
    }
    
    # Save temporary combined JSON for conversion
    temp_json = dataset_dir / 'annotations_combined.json'
    with open(temp_json, 'w') as f:
        json.dump(combined_data, f)
    
    # Convert to YOLO format (detect or segment)
    if task == "segment":
        coco_to_yolo_segment(temp_json, images_dir, labels_dir, categories)
    else:
        coco_to_yolo(temp_json, images_dir, labels_dir, categories)
    
    # Clean up temp file
    temp_json.unlink()
    
    yaml_content = f"""# YOLO-World Dataset Configuration
# Auto-generated from COCO annotations

# Dataset path
path: {dataset_dir.absolute()}
train: {images_rel}
val: {images_rel}

# Number of classes
nc: {len(class_names)}

# Class names
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset YAML: {output_yaml}")
    return output_yaml


def train_detect(
    model_name: str = "rtdetr-l.pt",
    init_weights: str = "",
    data_yaml: str = "dataset.yaml",
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 896,
    lr0: float = 2e-4,
    device: str = "auto",
    project: str = "runs/detect",
    name: str = "yolo_world_train",
    close_mosaic: int = 10,
    val_interval: int = 5,
    save_period: int = 10,
    class_weighting_yaml: str = "",
    **kwargs
):
    """
    Train detection model on custom dataset.
    Supports RT-DETR, YOLO-World, and plain YOLO checkpoints.
    """
    print("\n" + "="*60)
    use_rtdetr = "rtdetr" in model_name.lower()
    use_yolo_world = ("world" in model_name.lower()) and not use_rtdetr
    if use_rtdetr:
        print("RT-DETR FINE-TUNING")
    elif use_yolo_world:
        print("YOLO-WORLD FINE-TUNING")
    else:
        print("YOLO DETECTION FINE-TUNING")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Learning rate: {lr0}")
    if device == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = "0"
            print(f"CUDA detected: Using device {device}")
        else:
            device = "cpu"
            print(f"No CUDA available: Using CPU")
    else:
        print(f"Using specified device: {device}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    if init_weights:
        print(f"Loading seg checkpoint directly for detect-only training: {init_weights}")
        model = YOLO(init_weights)
        model.task = "detect"
        model.overrides["task"] = "detect"
        print(f"  Forced task=detect on seg model ({sum(p.numel() for p in model.model.parameters()):,} params loaded)")
    elif use_rtdetr:
        print(f"Loading RT-DETR model: {model_name}")
        model = RTDETR(model_name)
    elif use_yolo_world:
        print(f"Loading YOLO-World model: {model_name}")
        model = YOLOWorld(model_name)
        import yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = [data_config['names'][i] for i in sorted(data_config['names'].keys())]
        print(f"Setting custom vocabulary: {class_names}")
        model.set_classes(class_names)
    else:
        print(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name)

    if class_weighting_yaml:
        class_names = _class_names_from_data_yaml(Path(data_yaml))
        class_weight_vector = _load_class_weight_vector(Path(class_weighting_yaml), class_names)
        if class_weight_vector is not None:
            _install_class_weighting(model, class_weight_vector, class_names)
        else:
            print(f"Class weighting not applied from: {class_weighting_yaml}")

    # Training arguments
    train_args = {
        'task': 'detect',
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'lr0': lr0,
        'device': device,
        'project': project,
        'name': name,
        'close_mosaic': close_mosaic,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': save_period,
        'patience': 1000,  # Early stopping patience
        'optimizer': 'AdamW',  # Good for small datasets
        'weight_decay': 0.05,
        'warmup_epochs': 3,  # Warmup for small dataset
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,  # HSV-Saturation augmentation
        'hsv_v': 0.4,  # HSV-Value augmentation
        'degrees': 45,  # Rotation degrees (0 for lab equipment)
        'translate': 0.15,  # Translation
        'scale': 0.5,  # Scale augmentation
        'shear': 5.0,  # Shear degrees
        'perspective': 0.001,  # Perspective
        'flipud': 0.5,  # Vertical flip probability
        'fliplr': 0.5,  # Horizontal flip probability
        'mosaic': 1.0,  # Mosaic augmentation probability
        'mixup': 0.1,  # MixUp augmentation probability
        'copy_paste': 0.1,  # Copy-paste augmentation probability
    }
    
    # Add any additional kwargs
    train_args.update(kwargs)
    
    print("\nStarting training...")
    print(f"Training arguments: {train_args}\n")
    
    # Train model
    results = model.train(**train_args)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved to: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last model saved to: {results.save_dir / 'weights' / 'last.pt'}")
    print("="*60)
    
    return results


def train_segment(
    model_name: str = "yolov8l-seg.pt",
    data_yaml: str = "dataset.yaml",
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 896,
    lr0: float = 2e-4,
    device: str = "auto",
    project: str = "runs/segment",
    name: str = "yolo_seg_train",
    close_mosaic: int = 10,
    val_interval: int = 5,
    save_period: int = 10,
    class_weighting_yaml: str = "",
    **kwargs
):
    """
    Train YOLO segment model on custom dataset (instance segmentation).
    Uses BCE + Dice mask loss (Ultralytics default for segment task).
    """
    print("\n" + "="*60)
    print("YOLO SEGMENT TRAINING (BCE + Dice mask loss)")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Learning rate: {lr0}")
    if device == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = "0"
            print(f"CUDA detected: Using device {device}")
        else:
            device = "cpu"
            print(f"No CUDA available: Using CPU")
    else:
        print(f"Using specified device: {device}")
    print("="*60 + "\n")

    model = YOLO(model_name)

    if class_weighting_yaml:
        class_names = _class_names_from_data_yaml(Path(data_yaml))
        class_weight_vector = _load_class_weight_vector(Path(class_weighting_yaml), class_names)
        if class_weight_vector is not None:
            _install_class_weighting(model, class_weight_vector, class_names)
        else:
            print(f"Class weighting not applied from: {class_weighting_yaml}")

    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'lr0': lr0,
        'device': device,
        'project': project,
        'name': name,
        'close_mosaic': close_mosaic,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': save_period,
        'patience': 1000,
        'optimizer': 'AdamW',
        'weight_decay': 0.05,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 45,
        'translate': 0.15,
        'scale': 0.5,
        'shear': 5.0,
        'perspective': 0.001,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'overlap_mask': False,  # avoid index error when a crop has 0 instances (mosaic)
    }
    train_args.update(kwargs)

    print("Starting segment training (mask loss includes Dice)...\n")
    results = model.train(**train_args)

    print("\n" + "="*60)
    print("SEGMENT TRAINING COMPLETE")
    print("="*60)
    print(f"Best model: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {results.save_dir / 'weights' / 'last.pt'}")
    print("="*60)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLO on custom COCO dataset (detect or segment)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_yolo.py
  python train_yolo.py --model rtdetr-l.pt --epochs 50
  python train_yolo.py --task segment --segment-model yolov8l-seg.pt --split-dataset
  python train_yolo.py --task segment --detect-only --segment-model weights/robot-segmentation.pt --dataset-dir datasets/gtcdata --split-dataset
  python train_yolo.py --data dataset.yaml --batch 8 --imgsz 640
  python train_yolo.py --split-dataset  # Auto-split train/val
  python train_yolo.py --task segment --segment-model weights/robot-segmentation.pt --dataset-dir datasets/14mldataset --split-dataset
        """
    )
    
    parser.add_argument('--task', type=str, choices=('detect', 'segment'), default='detect',
                       help='Task: detect (bbox) or segment (masks, BCE+Dice) (default: segment)')
    parser.add_argument('--model', type=str, default='rtdetr-l.pt',
                       help='Pretrained detect model: rtdetr-l.pt (default) or yolov8l-worldv2.pt')
    parser.add_argument('--segment-model', type=str, default='yolo26l-seg.pt',
                       help='Pretrained segment model when --task segment (default: yolov8l-seg.pt)')
    _root = get_project_root()
    parser.add_argument('--data', type=str, default=str(_root / 'configs' / 'dataset.yaml'),
                       help='Dataset YAML file (default: configs/dataset.yaml)')
    parser.add_argument('--annotations', type=str, default=str(_root / 'dataset' / 'annotations.json'),
                       help='COCO annotations JSON (default: dataset/annotations.json)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size per GPU (default: 16)')
    parser.add_argument('--imgsz', type=int, default=1024,
                       help='Image size for training (default: 768)')
    parser.add_argument('--lr0', type=float, default=5e-4,
                       help='Initial learning rate (default: 5e-4)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, cpu, or device ID (default: auto)')
    parser.add_argument('--project', type=str, default=str(_root / 'runs' / 'detect'),
                       help='Project directory (default: runs/detect)')
    parser.add_argument('--name', type=str, default='yolo_world_train',
                       help='Experiment name (default: yolo_world_train)')
    parser.add_argument('--close-mosaic', type=int, default=10,
                       help='Epochs before end to disable mosaic (default: 10)')
    parser.add_argument('--val-interval', type=int, default=5,
                       help='Validate every N epochs (default: 5)')
    parser.add_argument('--split-dataset', action='store_true',
                       help='Automatically split dataset into train/val (80/20)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for dataset split (default: 42)')
    parser.add_argument('--dataset-dir', type=str, default=None,
                       help='Dataset directory with annotations/ and images/ subdirs. '
                            'Auto-discovers annotations JSON and images layout.')
    parser.add_argument('--detect-only', action='store_true',
                       help='When --task segment, ignore masks and train detection only '
                            'using bbox labels (useful when segmentation masks are bad).')
    parser.add_argument('--class-weighting-yaml', type=str, default=str(_root / 'configs' / 'class_weighting.yaml'),
                       help='Path to class weighting yaml used to scale BCE class loss per class.')
    parser.add_argument('--class-weighting-dataset-dir', type=str, default=str(_root / 'dataset'),
                       help='Dataset dir used to estimate class weights into --class-weighting-yaml.')
    parser.add_argument('--disable-class-weighting', action='store_true',
                       help='Disable per-class BCE weighting even if weighting config exists.')
    parser.add_argument('--freeze-class-weighting', action='store_true',
                       help='Use --class-weighting-yaml as-is (do not auto-regenerate from dataset).')
    
    args = parser.parse_args()
    
    if not ULTRALYTICS_AVAILABLE:
        return 1
    
    effective_task = "detect" if (args.task == "segment" and args.detect_only) else args.task
    if args.task == "segment" and args.detect_only:
        print("\nDetect-only mode enabled for segment task: using bbox labels and detection loss only.")

    class_weighting_yaml = ""
    if not args.disable_class_weighting:
        weight_yaml_path = Path(args.class_weighting_yaml)
        if not weight_yaml_path.is_absolute():
            weight_yaml_path = _root / weight_yaml_path
        weighting_dataset_dir = Path(args.class_weighting_dataset_dir)
        if not weighting_dataset_dir.is_absolute():
            weighting_dataset_dir = _root / weighting_dataset_dir
        try:
            if args.freeze_class_weighting:
                print(f"Freeze class weighting enabled: keeping existing file {weight_yaml_path}")
            elif weighting_dataset_dir.exists():
                write_class_weighting_yaml_from_dataset(weighting_dataset_dir, weight_yaml_path)
            if weight_yaml_path.exists():
                class_weighting_yaml = str(weight_yaml_path)
                print(f"Using class weighting config: {class_weighting_yaml}")
        except Exception as exc:
            print(f"Warning: failed to prepare class weighting config: {exc}")

    # Resolve --dataset-dir: auto-discover annotations and images layout
    images_subdir = ""
    if args.dataset_dir:
        ds_dir = Path(args.dataset_dir)
        if not ds_dir.is_absolute():
            ds_dir = _root / ds_dir
        if not ds_dir.is_dir():
            print(f"Error: Dataset directory not found: {ds_dir}")
            return 1
        annotations_path, images_subdir = _discover_dataset_paths(ds_dir)
        args.annotations = str(annotations_path)
        if args.data == str(_root / 'configs' / 'dataset.yaml'):
            args.data = str(ds_dir / 'dataset.yaml')
    
    # Handle dataset splitting and conversion if requested
    if args.split_dataset:
        annotations_path = Path(args.annotations)
        if not annotations_path.exists():
            print(f"Error: Annotations file not found: {annotations_path}")
            return 1
        
        if args.dataset_dir:
            dataset_dir = ds_dir
        else:
            dataset_dir = annotations_path.parent
        ann_parent = annotations_path.parent
        sibling_train = ann_parent / "annotations_train.json"
        sibling_val = ann_parent / "annotations_val.json"

        # If user points at an already split file and both siblings exist,
        # reuse the existing split to avoid repeatedly shrinking the dataset.
        if (
            annotations_path.stem.endswith("_train")
            and sibling_train.exists()
            and sibling_val.exists()
        ):
            print("Detected existing split annotations; reusing annotations_train/annotations_val.")
            train_json, val_json = sibling_train, sibling_val
        elif (
            annotations_path.stem.endswith("_val")
            and sibling_train.exists()
            and sibling_val.exists()
        ):
            print("Detected existing split annotations; reusing annotations_train/annotations_val.")
            train_json, val_json = sibling_train, sibling_val
        else:
            train_json, val_json = split_train_val(
                annotations_path,
                train_ratio=args.train_ratio,
                seed=args.seed
            )
        
        # Load categories
        categories = load_coco_categories(annotations_path)
        
        # Create updated YAML with split annotations and convert to YOLO format
        data_yaml = Path(args.data)
        create_yolo_dataset_yaml(
            dataset_dir,
            train_json,
            val_json,
            categories,
            data_yaml,
            task=effective_task,
            images_subdir=images_subdir,
        )
        
        # Verify labels were created
        labels_dir = dataset_dir / 'labels' / images_subdir if images_subdir else dataset_dir / 'labels'
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            print(f"\nVerification: Found {len(label_files)} label files in {labels_dir}")
            if len(label_files) > 0:
                # Check first label file
                sample_label = label_files[0]
                with open(sample_label, 'r') as f:
                    content = f.read().strip()
                    if content:
                        print(f"  Sample label ({sample_label.name}): {len(content.split(chr(10)))} annotations")
                    else:
                        print(f"  Warning: {sample_label.name} is empty")
        else:
            print(f"  Error: Labels directory not created at {labels_dir}")
            return 1
    
    # Verify dataset YAML exists
    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        print(f"Error: Dataset YAML not found: {data_yaml_path}")
        print("Run with --split-dataset to create it automatically")
        return 1
    
    # Verify labels exist - if not, try to create them
    # Parse dataset path from YAML
    import yaml
    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    dataset_path = yaml_data.get('path', 'dataset')
    dataset_dir = Path(dataset_path)
    if not dataset_dir.is_absolute():
        dataset_dir = data_yaml_path.parent / dataset_path
    
    # Recover images_subdir from YAML if not already set via --dataset-dir
    train_rel = yaml_data.get('train', 'images')
    if not images_subdir:
        train_parts = Path(train_rel).parts
        if len(train_parts) > 1 and train_parts[0] == 'images':
            images_subdir = str(Path(*train_parts[1:]))

    labels_dir = dataset_dir / 'labels' / images_subdir if images_subdir else dataset_dir / 'labels'
    
    # Delete cache file if it exists (forces refresh)
    cache_file = dataset_dir / "labels.cache"
    if cache_file.exists():
        print(f"Removing stale cache file: {cache_file}")
        cache_file.unlink()
    
    def labels_are_segment_format(labels_path: Path) -> bool:
        """True if at least one line in any label file has > 6 values (segment polygon)."""
        for p in list(labels_path.glob("*.txt"))[:20]:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 6:
                        return True
        return False
    
    # For segment task: ensure labels are segment format (polygons), not detect (5 cols)
    if effective_task == "segment" and labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        if label_files and not labels_are_segment_format(labels_dir):
            print(f"\nSegment task requires polygon labels; current labels are detect format (5 columns).")
            print("Re-converting from COCO annotations to segment format...")
            ann_parent = Path(args.annotations).parent
            train_json = ann_parent / "annotations_train.json"
            val_json = ann_parent / "annotations_val.json"
            annotations_path = Path(args.annotations)
            if not annotations_path.exists():
                annotations_path = dataset_dir / "annotations.json"
            resolved_images_dir = dataset_dir / "images" / images_subdir if images_subdir else dataset_dir / "images"
            if train_json.exists() and val_json.exists():
                categories = load_coco_categories(train_json)
                with open(train_json, 'r') as f:
                    train_data = json.load(f)
                with open(val_json, 'r') as f:
                    val_data = json.load(f)
                all_images = {img['id']: img for img in train_data['images'] + val_data['images']}
                combined_data = {'images': list(all_images.values()), 'annotations': train_data['annotations'] + val_data['annotations']}
                temp_json = dataset_dir / "annotations_combined_segment.json"
                with open(temp_json, 'w') as f:
                    json.dump(combined_data, f)
                coco_to_yolo_segment(temp_json, resolved_images_dir, labels_dir, categories)
                temp_json.unlink(missing_ok=True)
            elif annotations_path.exists():
                categories = load_coco_categories(annotations_path)
                coco_to_yolo_segment(annotations_path, resolved_images_dir, labels_dir, categories)
            else:
                print("Error: Need annotations_train.json + annotations_val.json or annotations.json to re-convert. Run with --split-dataset --task segment first.")
                return 1
            if cache_file.exists():
                cache_file.unlink(missing_ok=True)
    
    if not labels_dir.exists() or len(list(labels_dir.glob("*.txt"))) == 0:
        print(f"\nWarning: Labels not found in {labels_dir}")
        print("Attempting to create labels from COCO annotations...")
        
        # Try to find annotations file
        annotations_path = Path(args.annotations)
        if not annotations_path.exists():
            # Try default location
            annotations_path = dataset_dir / "annotations.json"
        
        if annotations_path.exists():
            categories = load_coco_categories(annotations_path)
            
            # Create labels from original annotations (detect or segment format)
            images_dir = dataset_dir / "images" / images_subdir if images_subdir else dataset_dir / "images"
            if images_dir.exists():
                if effective_task == "segment":
                    coco_to_yolo_segment(annotations_path, images_dir, labels_dir, categories)
                else:
                    coco_to_yolo(annotations_path, images_dir, labels_dir, categories)
                
                # Verify labels were created
                label_files = list(labels_dir.glob("*.txt"))
                if len(label_files) == 0:
                    print(f"Error: Failed to create label files")
                    return 1
                print(f"Successfully created {len(label_files)} label files")
            else:
                print(f"Error: Images directory not found: {images_dir}")
                return 1
        else:
            print(f"Error: Could not find annotations file: {annotations_path}")
            print("Please run with --split-dataset to create labels")
            return 1
    
    # Train model (detect or segment)
    try:
        if effective_task == "segment":
            results = train_segment(
                model_name=args.segment_model,
                data_yaml=args.data,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                lr0=args.lr0,
                device=args.device,
                project=args.project if args.project != "runs/detect" else "runs/segment",
                name=args.name if args.name != "yolo_world_train" else "yolo_seg_train",
                close_mosaic=args.close_mosaic,
                val_interval=args.val_interval,
                class_weighting_yaml=class_weighting_yaml,
            )
        else:
            detect_model = args.model
            init_weights = ""
            if args.task == "segment" and args.detect_only:
                init_weights = args.segment_model
                print(f"Detect-only: loading '{init_weights}' with forced detect trainer")
            results = train_detect(
                model_name=detect_model,
                init_weights=init_weights,
                data_yaml=args.data,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                lr0=args.lr0,
                device=args.device,
                project=args.project,
                name=args.name,
                close_mosaic=args.close_mosaic,
                val_interval=args.val_interval,
                class_weighting_yaml=class_weighting_yaml,
            )
        
        print("\nTraining completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

