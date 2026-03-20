"""
Celery Task Definitions
────────────────────────
Async tasks for:
  - process_prediction_async: background image prediction
  - trigger_retraining:       kick off model retraining when feedback threshold reached
  - cleanup_old_images:       scheduled job to purge expired images from S3
  - generate_daily_report:    admin stats report
"""

import asyncio
import logging
from pathlib import Path

from celery import shared_task
from celery.utils.log import get_task_logger

from app.tasks.celery_app import celery_app

logger = get_task_logger(__name__)


def run_async(coro):
    """Run an async coroutine in a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    name="tasks.process_prediction_async",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    queue="predictions",
    acks_late=True,
)
def process_prediction_async(
    self,
    image_bytes: bytes,
    user_id: str | None,
    original_filename: str | None,
    crop_hint: str | None = None,
    top_k: int = 5,
):
    """
    Async prediction task. Called when client sets async_mode=true.
    Saves result to DB so client can poll GET /predictions/{task_id}.
    """
    from app.db.database import AsyncSessionLocal
    from app.ml.models.model_manager import ModelManager
    from app.services.prediction_service import PredictionService
    from app.services.storage_service import StorageService
    from app.services.advisory_service import AdvisoryService

    async def _run():
        async with AsyncSessionLocal() as db:
            # ModelManager needs to be reloaded in worker process
            model_manager = ModelManager()
            await model_manager.load_models()

            service = PredictionService(
                model_manager=model_manager,
                storage=StorageService(),
                advisory=AdvisoryService(db=db),
                db=db,
            )
            result = await service.predict(
                image_bytes=image_bytes,
                user_id=user_id,
                original_filename=original_filename,
                crop_hint=crop_hint,
                top_k=top_k,
            )
            await model_manager.unload_models()
            return result.prediction_id

    try:
        prediction_id = run_async(_run())
        logger.info(f"Async prediction complete: {prediction_id}")
        return {"status": "completed", "prediction_id": prediction_id}
    except Exception as exc:
        logger.error(f"Prediction task failed: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    name="tasks.trigger_retraining",
    queue="retraining",
)
def trigger_retraining(feedback_threshold: int = 500):
    """
    Check if enough new labeled feedback has accumulated.
    If so, kick off a retraining job (could submit to GPU cluster).
    """
    from app.db.database import AsyncSessionLocal
    from sqlalchemy import select, func
    from app.models.database_models import PredictionFeedback

    async def _check():
        async with AsyncSessionLocal() as db:
            count = await db.scalar(
                select(func.count()).select_from(PredictionFeedback)
                .where(PredictionFeedback.used_for_retraining == False)
            )
            return count

    new_feedback = run_async(_check())
    logger.info(f"New feedback samples: {new_feedback}")

    if new_feedback >= feedback_threshold:
        logger.info(f"Threshold reached ({new_feedback} >= {feedback_threshold}). Submitting retraining job.")
        submit_training_job.delay()
        return {"retraining_triggered": True, "samples": new_feedback}

    return {"retraining_triggered": False, "samples": new_feedback}


@celery_app.task(name="tasks.submit_training_job", queue="retraining")
def submit_training_job():
    """
    Submit a training job to the compute cluster.
    In production: can use AWS Batch, GCP Vertex AI, or a local GPU.
    """
    import subprocess
    logger.info("Starting model retraining...")

    result = subprocess.run([
        "python", "ml_pipeline/training/train.py",
        "--model", "efficientnet_b4",
        "--epochs", "10",             # Fine-tuning, not full training
        "--lr", "5e-5",               # Lower LR for fine-tuning
    ], capture_output=True, text=True, timeout=14400)   # 4h timeout

    if result.returncode == 0:
        logger.info("Retraining completed successfully")
        # TODO: evaluate new model and auto-promote if metrics improve
        return {"status": "success"}
    else:
        logger.error(f"Retraining failed: {result.stderr}")
        return {"status": "failed", "error": result.stderr[:500]}


@celery_app.task(
    name="tasks.cleanup_old_images",
    queue="retraining",
)
def cleanup_old_images(older_than_days: int = 90):
    """Periodic task: purge S3 images older than N days for non-premium users."""
    from app.services.storage_service import StorageService
    from datetime import datetime, timedelta, timezone

    async def _cleanup():
        storage = StorageService()
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        deleted = await storage.delete_objects_older_than("uploads/", cutoff)
        return deleted

    deleted = run_async(_cleanup())
    logger.info(f"Cleaned up {deleted} old image objects from S3")
    return {"deleted_objects": deleted}


@celery_app.task(name="tasks.generate_daily_report", queue="retraining")
def generate_daily_report():
    """Generate and email daily prediction statistics to admins."""
    from app.db.database import AsyncSessionLocal
    from sqlalchemy import select, func, text
    from app.models.database_models import Prediction
    from datetime import datetime, timedelta, timezone

    async def _report():
        async with AsyncSessionLocal() as db:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            total = await db.scalar(
                select(func.count()).select_from(Prediction)
                .where(Prediction.created_at >= yesterday)
            )
            avg_conf = await db.scalar(
                select(func.avg(Prediction.confidence))
                .where(Prediction.created_at >= yesterday)
            )
            return {"predictions_24h": total, "avg_confidence": float(avg_conf or 0)}

    stats = run_async(_report())
    logger.info(f"Daily report: {stats}")
    # TODO: send via email service (SendGrid / SES)
    return stats
