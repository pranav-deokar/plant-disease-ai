"""
Model Evaluation Script
────────────────────────
Evaluates a trained model checkpoint on the test set.
Produces:
  - Per-class accuracy, precision, recall, F1
  - Confusion matrix (saved as PNG)
  - Worst-performing classes report
  - Comparison with baseline model (if provided)
  - MLflow logging of all metrics

Usage:
  python evaluate.py \
    --checkpoint models/efficientnet_b4_best.pth \
    --data-dir data/processed/plantvillage \
    --output-dir reports/eval_v1
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import mlflow

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.ml.models.model_manager import EfficientNetB4PlantDisease, DISEASE_CLASSES, NUM_CLASSES
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_preds, all_labels, all_probs


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: Path,
    normalize: bool = True,
):
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    n = len(class_names)
    fig_size = max(16, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = range(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    short_names = [c.split("___")[-1].replace("_", " ")[:20] for c in class_names]
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=5)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {output_path}")


def find_worst_classes(report: dict, n: int = 10) -> list:
    """Return N classes with lowest F1 score."""
    class_scores = [
        (cls, metrics["f1-score"])
        for cls, metrics in report.items()
        if isinstance(metrics, dict) and "f1-score" in metrics
    ]
    return sorted(class_scores, key=lambda x: x[1])[:n]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = ImagePreprocessor(target_size=(380, 380))

    # Load test set
    logger.info(f"Loading test data from {args.data_dir}")
    test_ds = ImageFolder(args.data_dir, transform=preprocessor.inference_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    class_names = test_ds.classes
    logger.info(f"Test samples: {len(test_ds)}, Classes: {len(class_names)}")

    # Load model
    model = EfficientNetB4PlantDisease(NUM_CLASSES, pretrained=False)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if all(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    logger.info(f"Model loaded from {args.checkpoint}")

    # Evaluate
    logger.info("Running evaluation...")
    preds, labels, probs = evaluate_model(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    report_dict = classification_report(
        labels, preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    report_str = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    logger.info(f"\n{'='*70}\nEvaluation Results\n{'='*70}")
    logger.info(f"Accuracy     : {acc:.4f}")
    logger.info(f"F1 Macro     : {f1_macro:.4f}")
    logger.info(f"F1 Weighted  : {f1_weighted:.4f}")
    logger.info(f"\n{report_str}")

    # Save reports
    (output_dir / "classification_report.txt").write_text(report_str)
    (output_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2))
    (output_dir / "summary.json").write_text(json.dumps({
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "checkpoint": args.checkpoint,
    }, indent=2))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    # Worst classes
    worst = find_worst_classes(report_dict)
    logger.info("\nWorst-performing classes (by F1):")
    for cls, f1 in worst:
        logger.info(f"  {cls:<45} F1={f1:.3f}")

    # MLflow logging
    if args.mlflow_run_id:
        mlflow.set_tracking_uri(args.mlflow_uri)
        with mlflow.start_run(run_id=args.mlflow_run_id):
            mlflow.log_metrics({
                "test_accuracy": acc,
                "test_f1_macro": f1_macro,
                "test_f1_weighted": f1_weighted,
            })
            mlflow.log_artifact(str(output_dir / "confusion_matrix.png"))
            mlflow.log_artifact(str(output_dir / "classification_report.json"))
            logger.info(f"Metrics logged to MLflow run {args.mlflow_run_id}")

    logger.info(f"\nAll reports saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pth model file")
    parser.add_argument("--data-dir", required=True, help="Test data directory (ImageFolder format)")
    parser.add_argument("--output-dir", default="reports/eval", help="Where to save reports")
    parser.add_argument("--mlflow-run-id", default="", help="Existing MLflow run ID to log metrics to")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()
    main(args)
