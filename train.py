"""
Multi-Model Breast Cancer Detection using CBIS-DDSM Dataset
============================================================
Trains and compares 3 architectures for binary classification (Benign vs Malignant):
  1. VGG16           - Classic deep CNN
  2. ConvNeXt-Small  - Modern ConvNet (ImageNet-22k pretrained)
  3. Swin-Small      - Swin Transformer (ImageNet-22k pretrained)

Target: 93-94% accuracy on test set.

Key optimizations:
  - Smart image path resolution using dicom_info.csv (picks ROI crops, avoids masks)
  - Higher resolution (384x384)
  - Strong augmentation + Mixup
  - Class-weighted loss
  - Discriminative learning rates (backbone 0.1x, head 1x)
  - Cosine annealing with warm restarts
  - Gradient clipping + Mixed precision (AMP)
  - Test-Time Augmentation (TTA) for final evaluation
  - Patient-level train/val split to prevent data leakage
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    auc,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import timm


# ============================================================
# 1. CONFIGURATION
# ============================================================
CONFIG = {
    "data_dir": "archive",
    "img_size": 384,
    "batch_size": 12,  # Reduced for Swin/ConvNeXt on 8GB VRAM
    "num_epochs": 60,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "num_classes": 2,  # Binary: Benign vs Malignant
    "patience": 15,
    "save_dir": "saved_models",
    "results_dir": "results",
    "mixup_alpha": 0.2,
    "grad_clip": 1.0,
    "use_tta": True,
    "use_amp": True,
}

# The 3 models to train and compare
MODEL_NAMES = ["vgg16", "convnext_small", "swin_small"]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. SMART IMAGE PATH RESOLUTION
# ============================================================
def build_dicom_lookup(data_dir):
    """
    Build lookup from dicom_info.csv:
      PatientName -> { 'cropped images': jpeg_path, 'ROI mask images': ..., 'full mammogram images': ... }
    This lets us pick the correct file (cropped ROI, not mask) for each CSV row.
    """
    dicom_csv = os.path.join(data_dir, "csv", "dicom_info.csv")
    if not os.path.exists(dicom_csv):
        print("WARNING: dicom_info.csv not found, falling back to UID matching")
        return None

    print("Building DICOM lookup table from dicom_info.csv...")
    df = pd.read_csv(dicom_csv)
    lookup = defaultdict(dict)

    for _, row in df.iterrows():
        patient_name = str(row.get("PatientName", "")).strip()
        series_desc = str(row.get("SeriesDescription", "")).strip()
        image_path = str(row.get("image_path", "")).strip()

        if patient_name and series_desc and image_path:
            if image_path.startswith("CBIS-DDSM/"):
                image_path = image_path[len("CBIS-DDSM/") :]
            lookup[patient_name][series_desc] = image_path

    print(f"  Built lookup for {len(lookup)} patient entries")
    return lookup


def find_image_path_smart(row, jpeg_dir, dicom_lookup, all_jpeg_dirs):
    """
    Find actual JPEG file from CSV metadata.
    Strategy 1: Use dicom_info lookup for exact cropped ROI path.
    Strategy 2: UID-based fallback, pick largest file (ROI > mask).
    """
    cropped_path = str(row.get("cropped image file path", "")).strip().strip('"')
    full_path = str(row.get("image file path", "")).strip().strip('"')

    # Extract case name (e.g., "Mass-Training_P_00001_LEFT_CC_1")
    case_name = None
    if cropped_path and cropped_path != "nan":
        case_name = cropped_path.split("/")[0]

    # Strategy 1: dicom_info lookup
    if dicom_lookup and case_name:
        entry = dicom_lookup.get(case_name, {})
        for desc_key in ["cropped images", "full mammogram images"]:
            if desc_key in entry:
                img_path = entry[desc_key]
                full_img_path = os.path.join(os.path.dirname(jpeg_dir), img_path)
                if os.path.exists(full_img_path):
                    return img_path
                alt_path = os.path.join(CONFIG["data_dir"], img_path)
                if os.path.exists(alt_path):
                    return img_path

    # Strategy 2: UID-based fallback
    path_candidates = []
    if cropped_path and cropped_path != "nan":
        path_candidates.append(cropped_path)
    if full_path and full_path != "nan":
        path_candidates.append(full_path)

    for candidate in path_candidates:
        parts = candidate.replace("\\", "/").split("/")
        for part in parts:
            if part.startswith("1.3.6.1"):
                if part in all_jpeg_dirs:
                    uid_dir = os.path.join(jpeg_dir, part)
                    jpg_files = [f for f in os.listdir(uid_dir) if f.endswith(".jpg")]
                    if jpg_files:
                        if len(jpg_files) > 1:
                            jpg_files.sort(
                                key=lambda f: os.path.getsize(os.path.join(uid_dir, f)),
                                reverse=True,
                            )
                        rel_path = os.path.join("jpeg", part, jpg_files[0])
                        return rel_path.replace("\\", "/")

    return None


# ============================================================
# 3. DATA PREPARATION
# ============================================================
def generate_csv_files():
    """
    Load CBIS-DDSM CSV files, resolve image paths, create train/val/test CSVs.
    Uses patient-level splitting to prevent data leakage.
    """
    print("=" * 60)
    print("Generating Train/Val/Test CSV Files")
    print("=" * 60)

    csv_dir = os.path.join(CONFIG["data_dir"], "csv")
    jpeg_dir = os.path.join(CONFIG["data_dir"], "jpeg")

    # Force regeneration
    for name in ["train.csv", "val.csv", "test.csv"]:
        p = os.path.join(CONFIG["data_dir"], name)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Deleted old {name}")

    # Load CSV files
    print("\nLoading CSV metadata files...")
    mass_train = pd.read_csv(
        os.path.join(csv_dir, "mass_case_description_train_set.csv")
    )
    mass_test = pd.read_csv(os.path.join(csv_dir, "mass_case_description_test_set.csv"))
    calc_train = pd.read_csv(
        os.path.join(csv_dir, "calc_case_description_train_set.csv")
    )
    calc_test = pd.read_csv(os.path.join(csv_dir, "calc_case_description_test_set.csv"))

    print(f"  Mass train: {len(mass_train)}, Mass test: {len(mass_test)}")
    print(f"  Calc train: {len(calc_train)}, Calc test: {len(calc_test)}")

    train_df = pd.concat([mass_train, calc_train], ignore_index=True)
    test_df = pd.concat([mass_test, calc_test], ignore_index=True)

    # Build lookup
    dicom_lookup = build_dicom_lookup(CONFIG["data_dir"])
    all_jpeg_dirs = set(os.listdir(jpeg_dir))
    print(f"  Total jpeg directories: {len(all_jpeg_dirs)}")

    def resolve_paths(df, desc):
        rows = []
        not_found = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Resolving {desc}"):
            img_path = find_image_path_smart(row, jpeg_dir, dicom_lookup, all_jpeg_dirs)
            if img_path:
                pathology = str(row["pathology"]).upper().strip()
                patient_id = str(row["patient_id"]).strip()
                rows.append(
                    {
                        "image_path": img_path,
                        "pathology": pathology,
                        "patient_id": patient_id,
                    }
                )
            else:
                not_found += 1
        print(f"  {desc}: {len(rows)} found, {not_found} missing")
        return pd.DataFrame(rows)

    train_data = resolve_paths(train_df, "train images")
    test_data = resolve_paths(test_df, "test images")

    # Deduplicate
    before_train, before_test = len(train_data), len(test_data)
    train_data = train_data.drop_duplicates(subset=["image_path"]).reset_index(
        drop=True
    )
    test_data = test_data.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    print(
        f"\nDeduplication: train {before_train}->{len(train_data)}, test {before_test}->{len(test_data)}"
    )

    # Binary labels
    train_data["label"] = train_data["pathology"].apply(
        lambda x: 1 if "MALIGNANT" in x else 0
    )
    test_data["label"] = test_data["pathology"].apply(
        lambda x: 1 if "MALIGNANT" in x else 0
    )

    print(f"\nClass distribution (train):")
    print(f"  Benign:    {(train_data['label'] == 0).sum()}")
    print(f"  Malignant: {(train_data['label'] == 1).sum()}")

    # Patient-level train/val split
    patient_labels = train_data.groupby("patient_id")["label"].max().reset_index()
    train_patients, val_patients = train_test_split(
        patient_labels["patient_id"],
        test_size=0.15,
        stratify=patient_labels["label"],
        random_state=42,
    )

    train_split = train_data[train_data["patient_id"].isin(set(train_patients))].copy()
    val_split = train_data[train_data["patient_id"].isin(set(val_patients))].copy()

    for split_name, split_data in [
        ("train", train_split),
        ("val", val_split),
        ("test", test_data),
    ]:
        output = split_data[["image_path", "pathology"]].copy()
        output.to_csv(
            os.path.join(CONFIG["data_dir"], f"{split_name}.csv"), index=False
        )

    print(f"\nFinal splits:")
    print(
        f"  Train: {len(train_split)} ({(train_split['label'] == 0).sum()} B, {(train_split['label'] == 1).sum()} M)"
    )
    print(
        f"  Val:   {len(val_split)} ({(val_split['label'] == 0).sum()} B, {(val_split['label'] == 1).sum()} M)"
    )
    print(
        f"  Test:  {len(test_data)} ({(test_data['label'] == 0).sum()} B, {(test_data['label'] == 1).sum()} M)"
    )
    print("CSV files saved!")


# ============================================================
# 4. DATASET CLASS
# ============================================================
class BreastCancerDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_frame = pd.read_csv(csv_file)
        self.label_map = {"BENIGN": 0, "MALIGNANT": 1, "BENIGN_WITHOUT_CALLBACK": 0}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data_frame.iloc[idx]["image_path"])
        label = self.label_map.get(self.data_frame.iloc[idx]["pathology"].upper(), 0)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"WARNING: Failed to load {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 5. TRANSFORMS
# ============================================================
def get_train_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ]
    )


def get_val_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_tta_transforms(img_size):
    """Test-Time Augmentation: 5 deterministic views averaged at inference."""
    return [
        # 1. Original
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        # 2. Horizontal flip
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        # 3. Vertical flip
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        # 4. Center crop from slightly larger
        transforms.Compose(
            [
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        # 5. Both flips
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    ]


# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================
def get_model(model_name, num_classes=2, pretrained=True):
    """
    Create model architecture.
    - vgg16:          torchvision VGG16, ImageNet-1K pretrained
    - convnext_small: timm ConvNeXt-Small, ImageNet-22K -> 1K fine-tuned (384px)
    - swin_small:     timm Swin-Small, ImageNet-22K -> 1K fine-tuned
    """
    if model_name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        # Replace final classifier layer with dropout + 2-class head
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(4096, num_classes)
        )

    elif model_name == "convnext_small":
        # ConvNeXt-Small pretrained on ImageNet-22K, fine-tuned on 1K at 384px
        # This is one of the strongest CNN architectures available
        model = timm.create_model(
            "convnext_small.fb_in22k_ft_in1k_384",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.4,
            drop_path_rate=0.2,
        )

    elif model_name == "swin_small":
        # Swin Transformer Small, ImageNet-22K -> 1K fine-tuned
        # State-of-the-art vision transformer for image classification
        model = timm.create_model(
            "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.4,
            drop_path_rate=0.3,
        )

    else:
        raise ValueError(
            f"Model {model_name} not supported. Choose from: {MODEL_NAMES}"
        )

    return model


# ============================================================
# 7. TRAINING UTILITIES
# ============================================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_optimizer(model, model_name, config):
    """
    Discriminative learning rates: backbone gets 0.1x the head learning rate.
    This prevents catastrophic forgetting of pretrained features.
    """
    if model_name == "vgg16":
        backbone_params = list(model.features.parameters())
        classifier_params = list(model.classifier.parameters())

    elif model_name == "convnext_small":
        # timm models: everything except 'head' is backbone
        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if "head" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

    elif model_name == "swin_small":
        # timm swin: 'head' is the classifier
        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if "head" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

    else:
        raise ValueError(f"Model {model_name} not supported")

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": config["learning_rate"] * 0.1},
            {"params": classifier_params, "lr": config["learning_rate"]},
        ],
        weight_decay=config["weight_decay"],
    )

    return optimizer


# ============================================================
# 8. TRAINING AND VALIDATION LOOPS
# ============================================================
def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    use_mixup=True,
    mixup_alpha=0.2,
    grad_clip=1.0,
    use_amp=True,
):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend((labels_a if use_mixup else labels).cpu().numpy())

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, use_amp=True):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=use_amp):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_probs, all_labels


def train_model(model_name, train_loader, val_loader, config, device, class_weights):
    print(f"\n{'=' * 60}")
    print(f"Training {model_name.upper()}")
    print(f"{'=' * 60}\n")

    model = get_model(model_name, num_classes=config["num_classes"])
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Parameters: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable"
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model, model_name, config)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    early_stopping = EarlyStopping(patience=config["patience"])
    scaler = GradScaler(enabled=config["use_amp"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(config["save_dir"], f"{model_name}_best.pth")

    for epoch in range(config["num_epochs"]):
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"\nEpoch {epoch + 1}/{config['num_epochs']} | LR: backbone={lrs[0]:.6f}, head={lrs[1]:.6f}"
        )

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_mixup=True,
            mixup_alpha=config["mixup_alpha"],
            grad_clip=config["grad_clip"],
            use_amp=config["use_amp"],
        )

        val_loss, val_acc, _, _, _ = validate_epoch(
            model, val_loader, criterion, device, use_amp=config["use_amp"]
        )

        scheduler.step(epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            print(f"  >>> Best model saved (Val Acc: {val_acc:.4f})")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\n  Early stopping triggered!")
            break

    print(f"\nTraining completed for {model_name}. Best Val Acc: {best_val_acc:.4f}")

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    return history, best_model_path


# ============================================================
# 9. EVALUATION (with optional TTA)
# ============================================================
def evaluate_model(
    model,
    dataloader,
    device,
    use_tta=False,
    tta_transforms=None,
    data_dir=None,
    csv_file=None,
):
    model.eval()

    if use_tta and tta_transforms and data_dir and csv_file:
        print("  Using Test-Time Augmentation (TTA) with 5 views...")
        return evaluate_with_tta(model, device, tta_transforms, data_dir, csv_file)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            with autocast(enabled=CONFIG["use_amp"]):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    return compute_metrics(all_labels, all_preds, all_probs)


def evaluate_with_tta(model, device, tta_transforms, data_dir, csv_file):
    """Apply TTA: inference with 5 transforms, average probabilities."""
    df = pd.read_csv(csv_file)
    label_map = {"BENIGN": 0, "MALIGNANT": 1, "BENIGN_WITHOUT_CALLBACK": 0}
    all_labels = [
        label_map.get(row["pathology"].upper(), 0) for _, row in df.iterrows()
    ]

    n_samples = len(df)
    avg_probs = np.zeros((n_samples, 2))

    for t_idx, transform in enumerate(tta_transforms):
        dataset = BreastCancerDataset(data_dir, csv_file, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
        )

        sample_idx = 0
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                with autocast(enabled=CONFIG["use_amp"]):
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                batch_size = probs.shape[0]
                avg_probs[sample_idx : sample_idx + batch_size] += probs
                sample_idx += batch_size

    avg_probs /= len(tta_transforms)
    all_preds = np.argmax(avg_probs, axis=1).tolist()
    all_probs_list = avg_probs[:, 1].tolist()

    return compute_metrics(all_labels, all_preds, all_probs_list)


def compute_metrics(all_labels, all_preds, all_probs):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    try:
        roc_auc_val = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
    except Exception:
        roc_auc_val = 0.0
        fpr, tpr = None, None

    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_curve, precision_curve)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
    }


# ============================================================
# 10. VISUALIZATION
# ============================================================
def plot_training_history(all_histories, results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training History Comparison", fontsize=16, fontweight="bold")

    for model_name, history in all_histories.items():
        axes[0, 0].plot(history["train_loss"], label=model_name, linewidth=2)
        axes[0, 1].plot(history["val_loss"], label=model_name, linewidth=2)
        axes[1, 0].plot(history["train_acc"], label=model_name, linewidth=2)
        axes[1, 1].plot(history["val_acc"], label=model_name, linewidth=2)

    titles = [
        "Training Loss",
        "Validation Loss",
        "Training Accuracy",
        "Validation Accuracy",
    ]
    ylabels = ["Loss", "Loss", "Accuracy", "Accuracy"]
    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabels[idx])
        ax.set_title(titles[idx], fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "training_history.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved training_history.png")


def plot_roc_curves(all_metrics, results_dir):
    plt.figure(figsize=(10, 8))
    for model_name, metrics in all_metrics.items():
        if metrics["fpr"] is not None:
            plt.plot(
                metrics["fpr"],
                metrics["tpr"],
                label=f"{model_name} (AUC={metrics['roc_auc']:.3f})",
                linewidth=2,
            )
    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(results_dir, "roc_curves.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved roc_curves.png")


def plot_pr_curves(all_metrics, results_dir):
    plt.figure(figsize=(10, 8))
    for model_name, metrics in all_metrics.items():
        plt.plot(
            metrics["recall_curve"],
            metrics["precision_curve"],
            label=f"{model_name} (AUC={metrics['pr_auc']:.3f})",
            linewidth=2,
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - All Models", fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(results_dir, "precision_recall_curves.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved precision_recall_curves.png")


def plot_confusion_matrices(all_metrics, results_dir):
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    class_names = ["Benign", "Malignant"]
    for idx, (model_name, metrics) in enumerate(all_metrics.items()):
        cm = metrics["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx],
        )
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
        axes[idx].set_title(
            f"{model_name.upper()}\nAcc: {metrics['accuracy']:.3f}", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "confusion_matrices.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved confusion_matrices.png")


def plot_bar_comparison(comparison_df, results_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
    metrics_to_plot = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
    ]
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        data = comparison_df.sort_values(metric, ascending=False)
        bars = ax.bar(
            range(len(data)),
            data[metric],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(data))),
        )
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data["Model"], rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "metrics_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved metrics_comparison.png")


# ============================================================
# 11. MAIN
# ============================================================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    print(f"\nModels to train: {MODEL_NAMES}")

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # --- Data Preparation ---
    generate_csv_files()

    train_csv = os.path.join(CONFIG["data_dir"], "train.csv")
    val_csv = os.path.join(CONFIG["data_dir"], "val.csv")
    test_csv = os.path.join(CONFIG["data_dir"], "test.csv")

    train_transform = get_train_transform(CONFIG["img_size"])
    val_transform = get_val_transform(CONFIG["img_size"])

    train_dataset = BreastCancerDataset(
        CONFIG["data_dir"], train_csv, transform=train_transform
    )
    val_dataset = BreastCancerDataset(
        CONFIG["data_dir"], val_csv, transform=val_transform
    )
    test_dataset = BreastCancerDataset(
        CONFIG["data_dir"], test_csv, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    print(
        f"\nDataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # --- Class Weights ---
    train_df_temp = pd.read_csv(train_csv)
    label_map = {"BENIGN": 0, "MALIGNANT": 1, "BENIGN_WITHOUT_CALLBACK": 0}
    train_labels = np.array(
        [label_map.get(r["pathology"].upper(), 0) for _, r in train_df_temp.iterrows()]
    )
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts.astype(float)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Class counts: Benign={class_counts[0]}, Malignant={class_counts[1]}")
    print(
        f"Class weights: Benign={class_weights[0]:.4f}, Malignant={class_weights[1]:.4f}"
    )

    # --- Train All Models ---
    all_histories = {}
    model_paths = {}

    for model_name in MODEL_NAMES:
        try:
            start_time = time.time()
            history, model_path = train_model(
                model_name,
                train_loader,
                val_loader,
                CONFIG,
                device,
                class_weights_tensor,
            )
            elapsed = time.time() - start_time
            print(f"  Training time: {elapsed / 60:.1f} minutes")

            all_histories[model_name] = history
            model_paths[model_name] = model_path

            pd.DataFrame(history).to_csv(
                os.path.join(CONFIG["results_dir"], f"{model_name}_history.csv"),
                index=False,
            )
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print("ALL MODELS TRAINING COMPLETED!")
    print(f"{'=' * 60}")

    # --- Evaluate All Models ---
    all_metrics = {}
    tta_transforms = (
        get_tta_transforms(CONFIG["img_size"]) if CONFIG["use_tta"] else None
    )

    for model_name in MODEL_NAMES:
        if model_name not in model_paths:
            continue

        print(f"\nEvaluating {model_name.upper()}...")
        model = get_model(model_name, num_classes=CONFIG["num_classes"])
        checkpoint = torch.load(
            model_paths[model_name], map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        if CONFIG["use_tta"]:
            metrics = evaluate_model(
                model,
                test_loader,
                device,
                use_tta=True,
                tta_transforms=tta_transforms,
                data_dir=CONFIG["data_dir"],
                csv_file=test_csv,
            )
        else:
            metrics = evaluate_model(model, test_loader, device)

        all_metrics[model_name] = metrics

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

        # Classification report
        print(f"\n  Classification Report:")
        print(
            classification_report(
                metrics["labels"],
                metrics["predictions"],
                target_names=["Benign", "Malignant"],
            )
        )

        # Free memory
        del model
        torch.cuda.empty_cache()

    # --- Comparison Summary ---
    comparison_data = []
    for model_name, metrics in all_metrics.items():
        comparison_data.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc"],
            }
        )
    comparison_df = pd.DataFrame(comparison_data).sort_values(
        "Accuracy", ascending=False
    )

    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(comparison_df.to_string(index=False))
    print(f"{'=' * 80}")

    comparison_df.to_csv(
        os.path.join(CONFIG["results_dir"], "model_comparison.csv"), index=False
    )

    # --- Plots ---
    if all_histories:
        plot_training_history(all_histories, CONFIG["results_dir"])
    if all_metrics:
        plot_roc_curves(all_metrics, CONFIG["results_dir"])
        plot_pr_curves(all_metrics, CONFIG["results_dir"])
        plot_confusion_matrices(all_metrics, CONFIG["results_dir"])
    if len(comparison_data) > 0:
        plot_bar_comparison(comparison_df, CONFIG["results_dir"])

    # --- Save JSON results ---
    for model_name, metrics in all_metrics.items():
        results = {
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "Confusion Matrix": metrics["confusion_matrix"].tolist(),
        }
        with open(
            os.path.join(CONFIG["results_dir"], f"{model_name}_results.json"), "w"
        ) as f:
            json.dump(results, f, indent=4)

    # --- Inference Example ---
    if all_metrics:
        best_model_name = comparison_df.iloc[0]["Model"]
        print(
            f"\nBest model: {best_model_name.upper()} (Accuracy={comparison_df.iloc[0]['Accuracy']:.4f})"
        )

        best_model = get_model(best_model_name, num_classes=CONFIG["num_classes"])
        checkpoint = torch.load(
            model_paths[best_model_name], map_location=device, weights_only=True
        )
        best_model.load_state_dict(checkpoint["model_state_dict"])
        best_model = best_model.to(device)
        best_model.eval()

        sample_image, sample_label = test_dataset[0]
        with torch.no_grad():
            output = best_model(sample_image.unsqueeze(0).to(device))
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)

        class_names = ["Benign", "Malignant"]
        print(f"\nSample Inference:")
        print(
            f"  Prediction: {class_names[pred.item()]} (confidence: {probs[0][pred].item():.4f})"
        )
        print(f"  True Label: {class_names[sample_label]}")

    print(f"\nAll results saved in: {CONFIG['results_dir']}/")
    print(f"All models saved in: {CONFIG['save_dir']}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
