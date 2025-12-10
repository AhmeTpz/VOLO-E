import os
import json
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import classification_report

# Mevcut yardımcılar ve model tanımı
from train_pytorch import (
    set_seed,
    get_device,
    get_dataloaders,
    SimpleCNN,
    compute_weights,
    plot_history,
    plot_confusion_matrix,
    plot_roc_curves,
    build_report_table,
    get_top_confusions,
)


# ======================
# Konfigürasyon (fine-tune)
# ======================
DATASET_DIR = "dataset"
BASE_CKPT = os.path.join("output_model", "best_model.pt")
OUTPUT_DIR = "output_ekstra"

EPOCHS = 10
PATIENCE = 3
MIN_DELTA = 1e-3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

# Confusion matrix gözlemlerine göre sınıf ağırlığı çarpanları
# (FN oranına göre yaklaşık; en çok karışanlar daha yüksek)
CLASS_WEIGHT_OVERRIDES: Dict[str, float] = {
    "glass": 3.2,
    "plastic": 3.6,
    "shoes": 3.6,
    "paper": 2.2,
    "metal": 2.0,
    "trash": 1.4,
    "battery": 1.5,
}


def apply_overrides(class_weights: torch.Tensor, class_names: List[str]) -> torch.Tensor:
    """Sınıf ağırlıklarına manuel çarpan uygula."""
    scaled = class_weights.clone()
    for idx, name in enumerate(class_names):
        if name in CLASS_WEIGHT_OVERRIDES:
            scaled[idx] *= CLASS_WEIGHT_OVERRIDES[name]
    return scaled


def train_finetune():
    set_seed(42)
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dataloaders ve sınıf isimleri
    dataloaders, class_names, num_classes = get_dataloaders(DATASET_DIR)

    # Modeli oluştur ve checkpoint yükle
    model = SimpleCNN(num_classes=num_classes).to(device)
    if os.path.isfile(BASE_CKPT):
        ckpt = torch.load(BASE_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"✅ Checkpoint yüklendi: {BASE_CKPT} (epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc'):.4f})")
    else:
        print(f"⚠️ Checkpoint bulunamadı: {BASE_CKPT}, sıfırdan başlıyor.")

    # Sınıf ağırlıkları
    class_weights = compute_weights(dataloaders["train"].dataset).to(device)
    class_weights = apply_overrides(class_weights, class_names)
    print(f"🧭 Fine-tune ağırlık çarpanları uygulandı: {CLASS_WEIGHT_OVERRIDES}")
    print(f"📏 Nihai sınıf ağırlıkları: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Basit epoch döngüsü
    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Fine-tune] Epoch {epoch}/{EPOCHS}")
        # Train
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            tr_loss += loss.item() * inputs.size(0)
            tr_correct += torch.sum(preds == labels).item()
            tr_total += inputs.size(0)

        train_loss = tr_loss / tr_total
        train_acc = tr_correct / tr_total

        # Val
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss_sum += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels).item()
                val_total += inputs.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

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
            torch.save(best_state, os.path.join(OUTPUT_DIR, "best_model_finetune.pt"))
            print("  ✅ En iyi fine-tune modeli kaydedildi (val_loss iyileşti)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  ⏸ Fine-tune erken durdurma tetiklendi (patience={PATIENCE})")
                break

    plot_history(history, OUTPUT_DIR)

    if best_state:
        model.load_state_dict(best_state["model_state"])
        print(f"En iyi fine-tune epoch {best_state['epoch']} (val_loss={best_state['val_loss']:.4f}, val_acc={best_state['val_acc']:.4f}) yüklendi.")

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
    print(f"\n✅ Fine-tune Test Doğruluğu: {test_acc:.4f}")

    cm = plot_confusion_matrix(all_labels, all_preds, class_names, OUTPUT_DIR)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    pd.DataFrame(report_dict).to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
    print("📄 classification_report.csv kaydedildi (fine-tune)")
    print("\n📊 Classification Report (konsol özeti):")
    print(build_report_table(report_dict, class_names))

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
        "best_val_loss": best_val_loss,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "history": history,
        "weight_overrides": CLASS_WEIGHT_OVERRIDES,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("📝 summary.json kaydedildi (fine-tune)")


if __name__ == "__main__":
    train_finetune()

