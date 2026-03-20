"""
Prometheus Metrics Middleware
──────────────────────────────
Exposes custom metrics at /metrics alongside the default FastAPI instrumentation:

Custom metrics:
  - plant_disease_predictions_total          — counter by disease_code, severity
  - plant_disease_confidence_histogram        — distribution of prediction confidence
  - plant_disease_processing_ms_histogram     — end-to-end inference latency
  - plant_disease_avg_confidence              — gauge for dashboard
  - plant_disease_deployment_timestamp        — gauge for deployment annotations
"""

from prometheus_client import Counter, Gauge, Histogram, Info, make_asgi_app
from fastapi import FastAPI

# ── Metric definitions ─────────────────────────────────────────────────────────

predictions_total = Counter(
    "plant_disease_predictions_total",
    "Total number of disease predictions",
    labelnames=["disease_code", "severity", "model_name"],
)

confidence_histogram = Histogram(
    "plant_disease_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)

processing_ms_histogram = Histogram(
    "plant_disease_processing_ms",
    "End-to-end prediction processing time in milliseconds",
    buckets=[50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
)

avg_confidence_gauge = Gauge(
    "plant_disease_avg_confidence",
    "Rolling average prediction confidence (last 1000 predictions)",
)

image_quality_histogram = Histogram(
    "plant_disease_image_quality",
    "Distribution of uploaded image quality scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

model_info = Info(
    "plant_disease_model",
    "Currently active model metadata",
)

deployment_timestamp = Gauge(
    "plant_disease_deployment_timestamp",
    "Unix timestamp of the last deployment",
)


# ── Helper to record a prediction ─────────────────────────────────────────────

def record_prediction_metrics(
    disease_code: str,
    severity: str,
    model_name: str,
    confidence: float,
    processing_ms: int,
    image_quality_score: float,
):
    """Called by PredictionService after each successful prediction."""
    predictions_total.labels(
        disease_code=disease_code,
        severity=severity,
        model_name=model_name,
    ).inc()

    confidence_histogram.observe(confidence)
    processing_ms_histogram.observe(processing_ms)
    image_quality_histogram.observe(image_quality_score)

    # Update rolling average gauge (approximate — no lock needed)
    current = avg_confidence_gauge._value.get()  # type: ignore
    avg_confidence_gauge.set(current * 0.99 + confidence * 0.01)


def setup_metrics(app: FastAPI):
    """Mount Prometheus metrics endpoint and set deployment timestamp."""
    import time
    deployment_timestamp.set(time.time())

    # Mount /metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
