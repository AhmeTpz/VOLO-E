"""
Kaggle veri setlerini indirip tekleştirir, sınıf bazlı sınırlar uygular ve
train/validation/test şeklinde böler.

Yeni bölme kuralı (sınıf başına):
- Önce 75 görsel test için ayrılır (75 veya eldeki sayı kadar).
- Kalan görseller 80/20 oranında train/validation olarak bölünür.

Kaynaklar:
- sumn2u/garbage-classification-v2
- feyzazkefe/trashnet

Hedef sınıflar:
- battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash
"""

import argparse
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List

TARGET_CLASSES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]

SOURCES = {
    "garbage-classification-v2": "sumn2u/garbage-classification-v2",
    "trashnet": "feyzazkefe/trashnet",
}


def run(cmd: List[str]):
    print(f"⚙️  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_and_extract(raw_dir: Path, slug: str, name: str):
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{name}.zip"
    if not zip_path.exists():
        run(["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir)])
    # unzip
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir / name)


def collect_class_dirs(source_root: Path) -> Dict[str, List[Path]]:
    class_map = {cls: [] for cls in TARGET_CLASSES}
    for cls_dir in source_root.rglob("*"):
        if cls_dir.is_dir():
            cls_name = cls_dir.name.lower()
            if cls_name in class_map:
                imgs = [p for p in cls_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                if imgs:
                    class_map[cls_name].extend(imgs)
    return class_map


def split_with_reserved(files: List[Path], reserve: int = 75):
    """
    Her sınıftan önce reserve adet (varsayılan 75) görseli ayır,
    kalanını 80/20 oranında train/validation olarak böl.
    Test setine ayrılanlar: reserved.
    """
    random.shuffle(files)
    if not files:
        return [], [], []

    reserve_n = min(reserve, len(files))
    reserved = files[:reserve_n]  # test
    remaining = files[reserve_n:]

    if not remaining:
        return [], [], reserved

    train_n = int(len(remaining) * 0.8)
    train_files = remaining[:train_n]
    val_files = remaining[train_n:]
    return train_files, val_files, reserved


def copy_split(train_files: List[Path], val_files: List[Path], test_files: List[Path], out_dir: Path, cls: str):
    for split_name, split_files in [("train", train_files), ("validation", val_files), ("test", test_files)]:
        dest = out_dir / split_name / cls
        dest.mkdir(parents=True, exist_ok=True)
        for src in split_files:
            shutil.copy2(src, dest / src.name)


def prepare(out_dir: Path, raw_dir: Path):
    random.seed(42)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged: Dict[str, List[Path]] = {cls: [] for cls in TARGET_CLASSES}

    # 1) İndir ve topla
    for name, slug in SOURCES.items():
        download_and_extract(raw_dir, slug, name)
        collected = collect_class_dirs(raw_dir / name)
        for cls, paths in collected.items():
            merged[cls].extend(paths)
        print(f"✅ {name}: sınıflar -> " + ", ".join([f"{c}:{len(collected[c])}" for c in TARGET_CLASSES if collected[c]]))

    # 2) Sınırla ve böl
    for cls, files in merged.items():
        if not files:
            print(f"⚠️  {cls}: kaynaklarda bulunamadı, atlanıyor.")
            continue
        train_f, val_f, test_f = split_with_reserved(files, reserve=75)
        copy_split(train_f, val_f, test_f, out_dir, cls)
        print(f"📦 {cls}: train={len(train_f)}, val={len(val_f)}, test={len(test_f)} (toplam={len(train_f)+len(val_f)+len(test_f)})")


def main():
    parser = argparse.ArgumentParser(description="Kaggle veri setlerini indirip tekleştirir ve böler.")
    parser.add_argument("--out_dir", default="dataset", help="Çıktı kök klasörü (train/validation/test)")
    parser.add_argument("--raw_dir", default="data_raw", help="Ham indirilen veri klasörü")
    args = parser.parse_args()

    prepare(Path(args.out_dir), Path(args.raw_dir))
    print("\n✅ Tamamlandı. Eğitim için 'dataset/train|validation|test' hazır.")
    print("Not: Kaggle API için 'kaggle.json' veya KAGGLE_USERNAME/KAGGLE_KEY gerekir.")


if __name__ == "__main__":
    main()

