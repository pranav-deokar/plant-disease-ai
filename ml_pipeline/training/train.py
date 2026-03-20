"""
Model Training Script
──────────────────────
Full training pipeline for plant disease classification.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

# ── Dummy env vars so config loads without errors ──────────────────────────────
os.environ.setdefault("SECRET_KEY", "training-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/plant_disease")
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

# ── Imports ────────────────────────────────────────────────────────────────────
from torchvision import transforms
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from app.ml.models.model_manager import (
    EfficientNetB4PlantDisease, MobileNetV3PlantDisease,
    DISEASE_CLASSES, NUM_CLASSES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model": "mobilenet_v3",
    "data_dir": "data/processed/plantvillage",
    "output_dir": "models",
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "dropout": 0.4,
    "num_workers": 0,
    "pin_memory": False,
    "use_amp": True,
    "grad_clip": 1.0,
    "patience": 7,
    "seed": 42,
    "mlflow_experiment": "plant_disease_detection",
    "mlflow_tracking_uri": "file:///C:/Users/prana/Downloads/plant_disease_ai_final/plant_disease_ai/mlruns",
}


# ── Simple Preprocessor ────────────────────────────────────────────────────────

def get_preprocessor():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class SimplePreprocessor:
        def __init__(self):
            self.train_transform = train_transform
            self.inference_transform = val_transform

    return SimplePreprocessor()


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_datasets(data_dir: str, preprocessor):
    import pickle
    from sklearn.model_selection import train_test_split

    data_dir = Path(data_dir)
    cache_file = data_dir / "dataset_split_cache.pkl"

    print(f"Loading dataset from: {data_dir}", flush=True)

    full_dataset = ImageFolder(data_dir, transform=None)
    print(full_dataset.classes)

    if cache_file.exists():
        print("Loading cached dataset split...", flush=True)
        with open(cache_file, "rb") as f:
            train_idx, val_idx, test_idx = pickle.load(f)
    else:
        print("Building dataset split (first run)...", flush=True)
        indices = list(range(len(full_dataset)))
        labels = full_dataset.targets
        train_idx, tmp_idx, _, tmp_labels = train_test_split(
            indices, labels, test_size=0.25, stratify=labels, random_state=42
        )
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.4, stratify=tmp_labels, random_state=42
        )
        with open(cache_file, "wb") as f:
            pickle.dump((train_idx, val_idx, test_idx), f)
        print("Dataset split cached.", flush=True)

    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_ds = TransformedSubset(full_dataset, train_idx, preprocessor.train_transform)
    val_ds   = TransformedSubset(full_dataset, val_idx,   preprocessor.inference_transform)
    test_ds  = TransformedSubset(full_dataset, test_idx,  preprocessor.inference_transform)

    print(f"Dataset sizes → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}", flush=True)

    return train_ds, val_ds, test_ds, full_dataset.classes


def make_balanced_sampler(dataset) -> WeightedRandomSampler:
    labels = [dataset.dataset.targets[i] for i in dataset.indices]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, grad_clip, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={loss.item():.4f} acc={correct/total:.3f} ({elapsed:.1f}s)",
                flush=True
            )

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return running_loss / n, accuracy, f1_macro, all_preds, all_labels


# ── Main Training Function ─────────────────────────────────────────────────────

def train(config: dict):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}", flush=True)

    # MLflow
    print("Setting up MLflow...", flush=True)
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])
    print("MLflow ready!", flush=True)

    # Preprocessor
    preprocessor = get_preprocessor()

    # Data
    print("Loading dataset...", flush=True)
    train_ds, val_ds, test_ds, class_names = build_datasets(config["data_dir"], preprocessor)
    print(f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}", flush=True)
    

    print("Creating DataLoaders...", flush=True)
    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"] * 2, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"] * 2, shuffle=False, num_workers=0)
    print("DataLoaders ready!", flush=True)

    # Model
    print("Loading model...", flush=True)
    if config["model"] == "efficientnet_b4":
        model = EfficientNetB4PlantDisease(NUM_CLASSES, pretrained=True, dropout=config["dropout"])
    else:
        model = MobileNetV3PlantDisease(NUM_CLASSES, pretrained=True, dropout=config["dropout"])
    model = model.to(device)
    print(f"Model loaded on {device}!", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": config["learning_rate"] / 5},
        {"params": head_params,     "lr": config["learning_rate"]},
    ], weight_decay=config["weight_decay"])

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    scaler = torch.amp.GradScaler("cuda") if config["use_amp"] and device.type == "cuda" else None

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / f"{config['model']}_best.pth"

    print("Starting training loop...", flush=True)

    with mlflow.start_run() as run:
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("device", str(device))

        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, config["epochs"] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion,
                scaler, device, config["grad_clip"], epoch,
            )

            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1_macro": val_f1,
            }, step=epoch)

            print(
                f"Epoch {epoch:3d}/{config['epochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}",
                flush=True
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ✓ New best F1={best_f1:.4f} — checkpoint saved!", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch}", flush=True)
                    break

        # Final test evaluation
        print("Running final test evaluation...", flush=True)
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"\n{'='*60}\n"
            f"TEST RESULTS\n"
            f"  Accuracy : {test_acc:.4f}\n"
            f"  F1 Macro : {test_f1:.4f}\n"
            f"  Loss     : {test_loss:.4f}\n"
            f"{'='*60}",
            flush=True
        )

        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc, "test_f1_macro": test_f1})

        print(f"Best checkpoint saved to: {best_ckpt}", flush=True)
        print(f"MLflow run ID: {run.info.run_id}", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plant disease detection model")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"], choices=["efficientnet_b4", "mobilenet_v3"])
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG}
    config.update({
        "model": args.model,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "use_amp": not args.no_amp,
    })

    train(config)
