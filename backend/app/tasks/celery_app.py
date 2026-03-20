"""
Celery Application Configuration
──────────────────────────────────
Defines the Celery app with Redis as broker and result backend.
Includes periodic task schedule (Celery Beat).
"""

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

celery_app = Celery(
    "plant_disease_ai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.prediction_tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task behavior
    task_acks_late=True,            # only ack after task completes (safer)
    worker_prefetch_multiplier=1,   # one task at a time per worker (for heavy ML)
    task_reject_on_worker_lost=True,

    # Result expiry
    result_expires=86400,           # 24h

    # Queue routing
    task_routes={
        "tasks.process_prediction_async": {"queue": "predictions"},
        "tasks.trigger_retraining": {"queue": "retraining"},
        "tasks.submit_training_job": {"queue": "retraining"},
        "tasks.cleanup_old_images": {"queue": "retraining"},
        "tasks.generate_daily_report": {"queue": "retraining"},
    },

    # Periodic tasks (Celery Beat)
    beat_schedule={
        "check-retraining-threshold": {
            "task": "tasks.trigger_retraining",
            "schedule": crontab(hour="*/6"),    # every 6 hours
            "kwargs": {"feedback_threshold": 500},
        },
        "cleanup-old-images": {
            "task": "tasks.cleanup_old_images",
            "schedule": crontab(hour=2, minute=0),  # 2am daily
            "kwargs": {"older_than_days": 90},
        },
        "daily-admin-report": {
            "task": "tasks.generate_daily_report",
            "schedule": crontab(hour=6, minute=0),  # 6am daily
        },
    },
)
