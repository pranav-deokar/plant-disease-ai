"""
Model Registration Script
──────────────────────────
After training, register a new model version in the database
and optionally promote it to active production status.

Usage:
  # Register only
  python scripts/register_model.py \
    --name efficientnet_b4_plantvillage \
    --version v1.3.0 \
    --checkpoint models/efficientnet_b4_best.pth \
    --mlflow-run-id abc123def456

  # Register and immediately activate
  python scripts/register_model.py \
    --name efficientnet_b4_plantvillage \
    --version v1.3.0 \
    --checkpoint models/efficientnet_b4_best.pth \
    --activate
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def register(args):
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
    from sqlalchemy import select, update
    from app.models.database_models import ModelVersion
    from app.ml.models.model_manager import NUM_CLASSES, MODEL_REGISTRY
    from app.core.config import settings

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as db:
        # Check for existing version
        existing = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_name == args.name,
                ModelVersion.version == args.version,
            )
        )
        if existing.scalar_one_or_none():
            logger.warning(f"Version {args.version} already registered for {args.name}. Use --force to overwrite.")
            if not args.force:
                return

        # Load checkpoint to extract architecture info
        spec = MODEL_REGISTRY.get(args.name)
        if spec is None:
            logger.error(f"Unknown model name: {args.name}. Must be one of: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            logger.error(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)

        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        param_count = sum(t.numel() for t in state_dict.values())
        logger.info(f"Checkpoint loaded: {param_count:,} parameters")

        # Parse metrics from MLflow if run ID provided
        val_accuracy = args.val_accuracy
        val_f1 = args.val_f1
        if args.mlflow_run_id:
            try:
                import mlflow
                mlflow.set_tracking_uri(args.mlflow_uri)
                run = mlflow.get_run(args.mlflow_run_id)
                metrics = run.data.metrics
                val_accuracy = val_accuracy or metrics.get("val_acc")
                val_f1 = val_f1 or metrics.get("val_f1_macro")
                logger.info(f"Pulled metrics from MLflow: acc={val_accuracy:.4f}, f1={val_f1:.4f}")
            except Exception as e:
                logger.warning(f"Failed to pull MLflow metrics: {e}")

        # Upload weights to S3
        artifact_uri = args.artifact_uri
        if not artifact_uri and args.upload_to_s3:
            logger.info("Uploading checkpoint to S3...")
            import aioboto3
            session = aioboto3.Session()
            s3_key = f"models/{args.name}/{args.version}/{checkpoint.name}"
            async with session.client("s3") as s3:
                with open(checkpoint, "rb") as f:
                    await s3.put_object(
                        Bucket="plant-disease-models",
                        Key=s3_key,
                        Body=f.read(),
                    )
            artifact_uri = f"s3://plant-disease-models/{s3_key}"
            logger.info(f"Uploaded to: {artifact_uri}")

        # Deactivate current active model if activating new one
        if args.activate:
            await db.execute(
                update(ModelVersion)
                .where(ModelVersion.model_name == args.name, ModelVersion.is_active == True)
                .values(is_active=False, deprecated_at=datetime.now(timezone.utc))
            )
            logger.info(f"Deactivated previous active version of {args.name}")

        import uuid as uuid_mod
        mv = ModelVersion(
            id=uuid_mod.uuid4(),
            model_name=args.name,
            version=args.version,
            architecture=spec["class"].__name__,
            mlflow_run_id=args.mlflow_run_id,
            artifact_uri=artifact_uri or str(checkpoint.resolve()),
            num_classes=NUM_CLASSES,
            input_size=spec["input_size"],
            val_accuracy=val_accuracy,
            val_f1_macro=val_f1,
            test_accuracy=args.test_accuracy,
            is_active=args.activate,
            is_shadow=args.shadow,
            deployed_at=datetime.now(timezone.utc) if (args.activate or args.shadow) else None,
            deployment_notes=args.notes,
            total_predictions=0,
        )
        db.add(mv)
        await db.commit()
        await db.refresh(mv)

        logger.info(f"\n{'='*50}")
        logger.info(f"✓ Model registered successfully")
        logger.info(f"  ID          : {mv.id}")
        logger.info(f"  Name        : {mv.model_name}")
        logger.info(f"  Version     : {mv.version}")
        logger.info(f"  Architecture: {mv.architecture}")
        logger.info(f"  Val Accuracy: {mv.val_accuracy}")
        logger.info(f"  Val F1 Macro: {mv.val_f1_macro}")
        logger.info(f"  Active      : {mv.is_active}")
        logger.info(f"  Shadow      : {mv.is_shadow}")
        logger.info(f"  Artifact    : {mv.artifact_uri}")

        if args.activate:
            logger.info(f"\n🟢 {args.name} v{args.version} is now ACTIVE in production.")
            logger.info("   Restart the API to load the new weights, or use the hot-swap endpoint:")
            logger.info(f"   POST /api/v1/admin/models/{mv.id}/activate")

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Register a trained model version")
    parser.add_argument("--name",       required=True, help="Model name (must match MODEL_REGISTRY key)")
    parser.add_argument("--version",    required=True, help="Version string e.g. v1.3.0")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth weights file")
    parser.add_argument("--mlflow-run-id", default="", help="MLflow run ID to pull metrics from")
    parser.add_argument("--mlflow-uri",    default="http://localhost:5000")
    parser.add_argument("--val-accuracy",  type=float, default=None)
    parser.add_argument("--val-f1",        type=float, default=None)
    parser.add_argument("--test-accuracy", type=float, default=None)
    parser.add_argument("--artifact-uri",  default="", help="S3/GCS URI of model artifact")
    parser.add_argument("--activate",  action="store_true", help="Immediately set as active production model")
    parser.add_argument("--shadow",    action="store_true", help="Deploy in shadow mode (A/B testing)")
    parser.add_argument("--upload-to-s3", action="store_true", help="Upload checkpoint to S3 before registering")
    parser.add_argument("--force",     action="store_true", help="Overwrite existing version registration")
    parser.add_argument("--notes",     default="", help="Deployment notes / changelog")
    args = parser.parse_args()

    asyncio.run(register(args))


if __name__ == "__main__":
    main()
