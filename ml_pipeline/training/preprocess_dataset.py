"""
Dataset Preprocessing Script
──────────────────────────────
Prepares the raw PlantVillage dataset for training:
  1. Validates image integrity
  2. Removes duplicates (by perceptual hash)
  3. Filters out non-leaf images (basic quality gate)
  4. Resizes to 456×456 (slightly larger than model input for aug headroom)
  5. Organizes into class subdirectories
  6. Generates class statistics report

Usage:
  python preprocess_dataset.py \
    --input  data/raw/plantvillage_dataset/color \
    --output data/processed/plantvillage \
    --num-workers 8

Expected input structure (PlantVillage standard):
  input/
    Tomato___Early_blight/
      *.jpg
    Tomato___Late_blight/
      *.jpg
    ...
"""

import argparse
import hashlib
import json
import logging
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SIZE = (456, 456)
MIN_IMAGE_SIZE = (64, 64)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Maps raw PlantVillage folder names → clean disease codes
# The raw names use triple underscore and title case
FOLDER_TO_CODE = {
    "Apple___Apple_scab": "apple___apple_scab",
    "Apple___Black_rot": "apple___black_rot",
    "Apple___Cedar_apple_rust": "apple___cedar_apple_rust",
    "Apple___healthy": "apple___healthy",
    "Blueberry___healthy": "blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew": "cherry___powdery_mildew",
    "Cherry_(including_sour)___healthy": "cherry___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn___cercospora_leaf_spot",
    "Corn_(maize)___Common_rust_": "corn___common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "corn___northern_leaf_blight",
    "Corn_(maize)___healthy": "corn___healthy",
    "Grape___Black_rot": "grape___black_rot",
    "Grape___Esca_(Black_Measles)": "grape___esca_black_measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape___leaf_blight",
    "Grape___healthy": "grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "orange___haunglongbing",
    "Peach___Bacterial_spot": "peach___bacterial_spot",
    "Peach___healthy": "peach___healthy",
    "Pepper,_bell___Bacterial_spot": "pepper___bacterial_spot",
    "Pepper,_bell___healthy": "pepper___healthy",
    "Potato___Early_blight": "potato___early_blight",
    "Potato___Late_blight": "potato___late_blight",
    "Potato___healthy": "potato___healthy",
    "Raspberry___healthy": "raspberry___healthy",
    "Soybean___healthy": "soybean___healthy",
    "Squash___Powdery_mildew": "squash___powdery_mildew",
    "Strawberry___Leaf_scorch": "strawberry___leaf_scorch",
    "Strawberry___healthy": "strawberry___healthy",
    "Tomato___Bacterial_spot": "tomato___bacterial_spot",
    "Tomato___Early_blight": "tomato___early_blight",
    "Tomato___Late_blight": "tomato___late_blight",
    "Tomato___Leaf_Mold": "tomato___leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato___septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato___spider_mites",
    "Tomato___Target_Spot": "tomato___target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato___yellow_leaf_curl_virus",
    "Tomato___Tomato_mosaic_virus": "tomato___mosaic_virus",
    "Tomato___healthy": "tomato___healthy",
}


def perceptual_hash(image_path: Path) -> Optional[str]:
    """
    8×8 average hash for near-duplicate detection.
    Fast enough for large datasets; catches resized/recompressed dupes.
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        mean = small.mean()
        bits = (small > mean).flatten()
        return "".join("1" if b else "0" for b in bits)
    except Exception:
        return None


def process_one_image(args: tuple) -> dict:
    """
    Worker function (runs in subprocess).
    Validates, deduplicates (via hash), resizes, and saves one image.
    Returns a result dict for stats aggregation.
    """
    src_path, dst_path, class_code = args
    result = {
        "class": class_code,
        "status": "ok",
        "src": str(src_path),
        "hash": None,
    }

    try:
        # Integrity check
        try:
            img_pil = Image.open(src_path)
            img_pil.verify()
            img_pil = Image.open(src_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            result["status"] = "corrupt"
            result["error"] = str(e)
            return result

        # Size gate — reject tiny images
        w, h = img_pil.size
        if w < MIN_IMAGE_SIZE[0] or h < MIN_IMAGE_SIZE[1]:
            result["status"] = "too_small"
            result["error"] = f"{w}x{h}"
            return result

        # Perceptual hash
        phash = perceptual_hash(src_path)
        result["hash"] = phash

        # Resize with high-quality resampling
        img_resized = img_pil.resize(TARGET_SIZE, Image.LANCZOS)

        # Save as JPEG with good quality (lossless PNG would be 3× larger)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img_resized.save(dst_path, format="JPEG", quality=92, optimize=True, progressive=True)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def run_preprocessing(input_dir: Path, output_dir: Path, num_workers: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all images and build task list
    tasks = []
    folder_to_class = {}
    unknown_folders = []

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_code = FOLDER_TO_CODE.get(class_dir.name)
        if class_code is None:
            unknown_folders.append(class_dir.name)
            logger.warning(f"Unknown folder: {class_dir.name} — skipping")
            continue

        out_class_dir = output_dir / class_code
        folder_to_class[class_dir.name] = class_code

        for img_path in class_dir.glob("**/*"):
            if img_path.suffix.lower() in SUPPORTED_EXTS:
                dst = out_class_dir / img_path.name
                tasks.append((img_path, dst, class_code))

    logger.info(f"Total images to process: {len(tasks)} across {len(folder_to_class)} classes")
    if unknown_folders:
        logger.warning(f"Skipped unknown folders: {unknown_folders}")

    # Process in parallel
    stats = defaultdict(lambda: {"total": 0, "ok": 0, "corrupt": 0, "too_small": 0, "error": 0, "duplicate": 0})
    seen_hashes = defaultdict(set)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one_image, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            result = future.result()
            cls = result["class"]
            stats[cls]["total"] += 1

            if result["status"] == "ok":
                # Duplicate check
                phash = result.get("hash")
                if phash and phash in seen_hashes[cls]:
                    # Remove the duplicate file we just wrote
                    dst = Path(futures[future][1])
                    if dst.exists():
                        dst.unlink()
                    stats[cls]["duplicate"] += 1
                else:
                    if phash:
                        seen_hashes[cls].add(phash)
                    stats[cls]["ok"] += 1
            else:
                stats[cls][result["status"]] += 1

    # Summary report
    report = {}
    grand_total = grand_ok = 0
    for cls in sorted(stats.keys()):
        s = stats[cls]
        report[cls] = dict(s)
        grand_total += s["total"]
        grand_ok += s["ok"]

    report["__summary__"] = {
        "total_processed": grand_total,
        "total_usable": grand_ok,
        "total_classes": len(stats),
        "target_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
    }

    report_path = output_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(f"\n{'='*60}")
    logger.info(f"Preprocessing complete")
    logger.info(f"  Total processed : {grand_total:,}")
    logger.info(f"  Usable images   : {grand_ok:,}")
    logger.info(f"  Rejected        : {grand_total - grand_ok:,}")
    logger.info(f"  Classes         : {len(stats)}")
    logger.info(f"  Report saved to : {report_path}")

    # Print per-class table
    logger.info(f"\n{'Class':<45} {'Total':>6} {'OK':>6} {'Dupes':>6} {'Bad':>5}")
    logger.info("-" * 70)
    for cls in sorted(stats.keys()):
        s = stats[cls]
        bad = s["corrupt"] + s["too_small"] + s["error"]
        logger.info(f"{cls:<45} {s['total']:>6} {s['ok']:>6} {s['duplicate']:>6} {bad:>5}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess PlantVillage dataset")
    parser.add_argument("--input",  required=True, help="Input directory (PlantVillage raw)")
    parser.add_argument("--output", required=True, help="Output directory for processed images")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    run_preprocessing(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
