import os
import random
import json
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight


# ===================================================
# KONFIGÜRASYON
# ===================================================
DATASET_DIR = "dataset"  # train/validation/test klasörlerini içerir
OUTPUT_DIR = "output_short"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3  # erken durdurma için (val loss iyileşmezse)
MIN_DELTA = 1e-3  # val loss iyileşme eşiği
NUM_WORKERS = 2  # Windows'ta düşük tutmak stabil olur
SEED = 42
# Zor sınıflara ekstra ağırlık çarpanı
CLASS_WEIGHT_OVERRIDES = {
    "metal": 1.2,
    "plastic": 1.4,
    "glass": 1.2,
    "shoes": 1.2,
    "paper": 1.2,
}


# ===================================================
# YARDIMCI FONKSİYONLAR
# ===================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """CUDA kullanılabilirse CUDA, değilse CPU döndürür."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA kullanılabilir! GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Versiyonu: {torch.version.cuda}")
        print(f"   GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA bulunamadı, CPU kullanılacak.")
    return device


def build_transforms(img_size: Tuple[int, int]) -> Dict[str, transforms.Compose]:
    """Train için data augmentation, val/test için sadece resize + normalize."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def get_dataloaders(dataset_dir: str) -> Tuple[Dict[str, DataLoader], List[str], List[int]]:
    """ImageFolder ile train/validation/test dataloaders üretir."""
    tfs = build_transforms(IMG_SIZE)
    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=tfs["train"]),
        "val": datasets.ImageFolder(os.path.join(dataset_dir, "validation"), transform=tfs["val"]),
        "test": datasets.ImageFolder(os.path.join(dataset_dir, "test"), transform=tfs["test"]),
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)
    print(f"🔎 Sınıflar ({num_classes}): {class_names}")

    dataloaders = {
        phase: DataLoader(
            image_datasets[phase],
            batch_size=BATCH_SIZE,
            shuffle=True if phase == "train" else False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        for phase in ["train", "val", "test"]
    }

    return dataloaders, class_names, num_classes


class SimpleCNN(nn.Module):
    """Önceki TF mimarisine benzer 4 konv blok + FC."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (IMG_SIZE[0] // 16) * (IMG_SIZE[1] // 16), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def compute_weights(train_dataset) -> torch.Tensor:
    targets = np.array(train_dataset.targets)
    classes = np.unique(targets)
    class_weights = compute_class_weight("balanced", classes=classes, y=targets)
    print(f"⚖️ Sınıf ağırlıkları: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def plot_history(history: Dict[str, List[float]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_acc"], label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    return cm


def plot_roc_curves(y_true, y_probs, class_names, output_dir: str):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

    return roc_auc


def build_report_table(report: Dict, class_names: List[str]) -> str:
    """Classification report sözlüğünü okunabilir tablo string'ine çevirir."""
    rows = []
    for cls in class_names:
        metrics = report[cls]
        rows.append({
            "Sınıf": cls,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1-score"],
            "Destek": int(metrics["support"]),
        })

    for key, label in [("macro avg", "Macro Avg"), ("weighted avg", "Weighted Avg")]:
        metrics = report[key]
        rows.append({
            "Sınıf": label,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1-score"],
            "Destek": int(metrics["support"]),
        })

    rows.append({
        "Sınıf": "Accuracy",
        "Precision": report.get("accuracy", 0.0),
        "Recall": "",
        "F1": "",
        "Destek": int(report["macro avg"]["support"]),
    })

    df = pd.DataFrame(rows, columns=["Sınıf", "Precision", "Recall", "F1", "Destek"])
    for col in ["Precision", "Recall", "F1"]:
        df[col] = df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (float, int)) and x != "" else x)
    return df.to_string(index=False)


def get_top_confusions(cm: np.ndarray, class_names: List[str], top_k: int = 5) -> List[tuple]:
    """En çok karıştırılan sınıfları (off-diagonal) döndürür."""
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat = cm_copy.flatten()
    indices = flat.argsort()[::-1]

    results = []
    for idx in indices:
        count = flat[idx]
        if count <= 0:
            break
        true_idx = idx // cm.shape[1]
        pred_idx = idx % cm.shape[1]
        results.append((class_names[true_idx], class_names[pred_idx], int(count)))
        if len(results) >= top_k:
            break
    return results


def main():
    set_seed(SEED)
    device = get_device()
    print(f"🖥️ Eğitim cihazı: {device}")
    
    # CUDA optimizasyonları
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # CUDA optimizasyonu
        print("🚀 CUDA optimizasyonları aktif!")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dataloaders & class info
    dataloaders, class_names, num_classes = get_dataloaders(DATASET_DIR)

    # Model
    model = SimpleCNN(num_classes=num_classes).to(device)
    print(model)

    # Class weights
    class_weights = compute_weights(dataloaders["train"].dataset).to(device)
    # Zorlanan sınıflar için manuel çarpanlar ekle
    if CLASS_WEIGHT_OVERRIDES:
        class_weights_scaled = class_weights.clone()
        for idx, name in enumerate(class_names):
            if name in CLASS_WEIGHT_OVERRIDES:
                class_weights_scaled[idx] *= CLASS_WEIGHT_OVERRIDES[name]
        class_weights = class_weights_scaled
        print(f"🧭 Uygulanan ağırlık çarpanları: {CLASS_WEIGHT_OVERRIDES}")
        print(f"📏 Nihai sınıf ağırlıkları: {class_weights.cpu().numpy()}")

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, dataloaders["train"], device)
        val_loss, val_acc = evaluate(model, criterion, dataloaders["val"], device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Erken durdurma (val loss iyileşmezse)
        improved = val_loss < (best_val_loss - MIN_DELTA)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            torch.save(best_state, os.path.join(OUTPUT_DIR, "best_model.pt"))
            print("  ✅ En iyi model kaydedildi (val_loss iyileşti)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  ⏸ Erken durdurma tetiklendi (patience={PATIENCE}, val_loss iyileşmedi)")
                break

    # Eğitim grafikleri
    plot_history(history, OUTPUT_DIR)

    # En iyi modeli yükle
    if best_state:
        model.load_state_dict(best_state["model_state"])
        print(f"En iyi model epoch {best_state['epoch']} (val_loss={best_state['val_loss']:.4f}, val_acc={best_state['val_acc']:.4f}) yüklendi.")

    # Test değerlendirme
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    test_acc = (all_preds == all_labels).mean()
    print(f"\n✅ Test Doğruluğu: {test_acc:.4f}")

    # Confusion matrix
    cm = plot_confusion_matrix(all_labels, all_preds, class_names, OUTPUT_DIR)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    pd.DataFrame(report).to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
    print("📄 classification_report.csv kaydedildi")
    print("\n📊 Classification Report (konsol özeti):")
    print(build_report_table(report, class_names))

    # ROC curves
    roc_auc = plot_roc_curves(all_labels, all_probs, class_names, OUTPUT_DIR)
    print("\n🔎 ROC-AUC (sınıf bazında):")
    for i, name in enumerate(class_names):
        print(f"  - {name}: {roc_auc.get(i, float('nan')):.3f}")

    # Confusion highlights
    top_confusions = get_top_confusions(cm, class_names, top_k=5)
    if top_confusions:
        print("\n❗ En sık karışan sınıflar (ilk 5):")
        for true_label, pred_label, count in top_confusions:
            print(f"  - {true_label} -> {pred_label}: {count} örnek")

    # Özet json
    summary = {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "history": history,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("📝 summary.json kaydedildi")


if __name__ == "__main__":
    main()

