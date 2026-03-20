"""
Model Download Script
──────────────────────
Downloads pre-trained model weights from S3 or a public URL.
Run once during initial setup, or to update model versions.

Usage:
  # Download from S3 (production)
  python scripts/download_models.py --source s3

  # Download from public URL (demo / first-time setup)
  python scripts/download_models.py --source url

  # List available models
  python scripts/download_models.py --list
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")

# ── Model catalogue ───────────────────────────────────────────────────────────
# In production, update these to point to your S3 bucket and real checksums.
MODELS = {
    "efficientnet_b4_plantvillage": {
        "filename": "efficientnet_b4_plantvillage.pth",
        "version": "v1.2.0",
        "description": "EfficientNet-B4 trained on PlantVillage (38 classes, ~98% acc)",
        "size_mb": 72,
        "sha256": "REPLACE_WITH_REAL_CHECKSUM",
        "s3_key": "models/efficientnet_b4_plantvillage_v1.2.0.pth",
        "public_url": "",  # Set to public URL for demo downloads
    },
    "mobilenet_v3_plantvillage": {
        "filename": "mobilenet_v3_plantvillage.pth",
        "version": "v1.0.0",
        "description": "MobileNetV3-Large trained on PlantVillage (38 classes, ~96% acc)",
        "size_mb": 21,
        "sha256": "REPLACE_WITH_REAL_CHECKSUM",
        "s3_key": "models/mobilenet_v3_plantvillage_v1.0.0.pth",
        "public_url": "",
    },
}


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    if expected_sha256 == "REPLACE_WITH_REAL_CHECKSUM":
        logger.warning("Checksum verification skipped — placeholder hash in catalogue.")
        return True
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        logger.error(f"Checksum mismatch! Expected {expected_sha256}, got {actual}")
        return False
    return True


def download_from_s3(s3_key: str, dest: Path):
    import boto3
    from botocore.exceptions import ClientError

    # Reads AWS credentials from environment / ~/.aws/credentials
    s3 = boto3.client("s3")
    bucket = "plant-disease-models"

    logger.info(f"Downloading s3://{bucket}/{s3_key} → {dest}")
    try:
        file_size = s3.head_object(Bucket=bucket, Key=s3_key)["ContentLength"]
        logger.info(f"File size: {file_size / 1024 / 1024:.1f} MB")

        from tqdm import tqdm
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
            s3.download_file(
                bucket, s3_key, str(dest),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise FileNotFoundError(f"Model not found in S3: s3://{bucket}/{s3_key}")
        raise


def download_from_url(url: str, dest: Path):
    import urllib.request
    logger.info(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)


def download_model(name: str, source: str = "s3"):
    spec = MODELS.get(name)
    if spec is None:
        logger.error(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / spec["filename"]

    # Skip if already downloaded and checksum matches
    if dest.exists():
        logger.info(f"Found existing file: {dest}")
        if verify_checksum(dest, spec["sha256"]):
            logger.info(f"✓ {name} already up to date")
            return
        else:
            logger.warning("Checksum failed — re-downloading...")

    logger.info(f"Downloading {name} v{spec['version']} ({spec['size_mb']} MB)...")

    if source == "s3":
        download_from_s3(spec["s3_key"], dest)
    elif source == "url":
        if not spec["public_url"]:
            logger.error(f"No public URL configured for {name}")
            sys.exit(1)
        download_from_url(spec["public_url"], dest)
    else:
        raise ValueError(f"Unknown source: {source}")

    if not verify_checksum(dest, spec["sha256"]):
        dest.unlink(missing_ok=True)
        logger.error("Downloaded file failed checksum — deleted. Please try again.")
        sys.exit(1)

    logger.info(f"✓ {name} downloaded and verified → {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download plant disease model weights")
    parser.add_argument("--source", choices=["s3", "url"], default="s3")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Download specific model only")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        print("-" * 60)
        for name, spec in MODELS.items():
            print(f"  {name}")
            print(f"    Version : {spec['version']}")
            print(f"    Size    : {spec['size_mb']} MB")
            print(f"    Info    : {spec['description']}")
            local = MODEL_DIR / spec["filename"]
            print(f"    Local   : {'✓ Present' if local.exists() else '✗ Not downloaded'}")
            print()
        return

    if args.model:
        download_model(args.model, args.source)
    else:
        logger.info("Downloading all models...")
        for name in MODELS:
            download_model(name, args.source)

    logger.info("\nAll models ready. You can now start the server.")


if __name__ == "__main__":
    main()
