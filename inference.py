"""
Model inference modülü
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
import numpy as np

from main_training import SimpleCNN, IMG_SIZE

# ----------------------------------
# Yardımcılar
# ----------------------------------
DEFAULT_DETECTOR = Path(__file__).resolve().parent / "models" / "detector" / "best_yolo.pt"

# Dedektör için ölçek sınırı (YOLO 600px eğitimi, 960px inference upper bound)
MAX_DET_SIDE = 960

# Sınıflandırma crop ön-işleme eşikleri
MIN_CROP_SKIP = 32   # çok küçük crop'u atla
MIN_CROP_UPSCALE = None  # küçük crop'u büyütme (kapalı)

# CLAHE ayarları (aydınlık/kontrast için hafif iyileştirme)
CLAHE_ENABLED = False  # İstersen kapat
CLAHE_CLIP = 1.3
CLAHE_GRID = (8, 8)

def generate_random_light_color() -> Tuple[int, int, int]:
    """Açık renkli random BGR renk üret (görünürlük için)"""
    # Açık renkler için yüksek değerler kullan
    # BGR formatında
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)
    return (b, g, r)  # BGR formatı


def calculate_font_scale(image_width: int, image_height: int) -> float:
    """Görsel boyutuna göre (genişlik+yükseklik) dinamik font ölçeği hesapla (lineer yok)."""
    image_side = max(image_width, image_height)  # dikine uzun görselleri de kapsa

    if image_side < 1024:
        return 0.3
    if image_side < 1920:
        return 1.0
    if image_side < 2560:
        return 1.5
    if image_side < 3840:
        return 3.0
    if image_side < 7680:
        return 4.5
    return 7.0


def calculate_thickness(image_width: int, image_height: int) -> int:
    """Görsel boyutuna göre dinamik çizgi kalınlığı hesapla (OR mantığı: max(w,h))."""
    image_side = max(image_width, image_height)

    # Düşük çözünürlükte 1, yükseklerde daha kalın
    if image_side < 1024:
        return 1
    if image_side < 1920:
        return 2
    if image_side < 2560:
        return 3
    if image_side < 3840:
        return 5
    if image_side < 7680:
        return 7
    return 8


def load_class_names(summary_path: Path) -> List[str]:
    """Sınıf isimlerini summary.json'dan yükle"""
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)["class_names"]


def load_classifier(model_path: Path, class_names: List[str], device: torch.device):
    """Sınıflandırıcı modelini yükle"""
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def get_classifier_transform():
    """Sınıflandırıcı için transform"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def resize_for_detection(img_bgr: np.ndarray, max_side: int = MAX_DET_SIDE) -> Tuple[np.ndarray, float]:
    """Dedektör için görüntüyü orantılı küçült; ölçek faktörünü döndür."""
    h, w = img_bgr.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return img_bgr, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_crop_for_classifier(crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
    """Crop'u kare pad + CLAHE + normalize ile sınıflandırmaya hazırla."""
    h, w = crop_bgr.shape[:2]
    # Çok küçük crop'u atla
    if min(h, w) < MIN_CROP_SKIP:
        return None
    # Küçük crop'u büyüt (kapalı)
    if MIN_CROP_UPSCALE and min(h, w) < MIN_CROP_UPSCALE:
        scale = MIN_CROP_UPSCALE / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = crop_bgr.shape[:2]

    # CLAHE (sadece L kanalında) - opsiyonel
    if CLAHE_ENABLED:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        crop_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Kare pad
    side = max(h, w)
    pad_color = (128, 128, 128)
    squared = np.full((side, side, 3), pad_color, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    squared[y0:y0 + h, x0:x0 + w] = crop_bgr

    # Modele uygun boyutlandırma
    squared = cv2.resize(squared, IMG_SIZE, interpolation=cv2.INTER_AREA)
    squared_rgb = cv2.cvtColor(squared, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(squared_rgb)

    norm_tf = get_classifier_transform()
    tensor = norm_tf(pil_img).unsqueeze(0)
    return tensor


def detect_and_classify(
    image_path: Path,
    detector_path: Path,
    classifier_path: Path,
    summary_path: Path,
    conf_thres: float,
) -> Dict[str, Any]:
    """
    Görsel üzerinde tespit ve sınıflandırma yap
    
    Returns:
        Dict with keys:
            - image: BGR görsel (annotated)
            - detections: List of detection dicts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(summary_path)
    classifier = load_classifier(classifier_path, class_names, device)
    detector = YOLO(str(detector_path))

    # Görseli oku ve dedektör için ölçekle
    image_bgr_orig = cv2.imread(str(image_path))
    det_image, scale = resize_for_detection(image_bgr_orig, MAX_DET_SIDE)

    det_results = detector(det_image, conf=conf_thres, imgsz=MAX_DET_SIDE, verbose=False)[0]
    h, w = image_bgr_orig.shape[:2]
    detections = []
    
    # Dinamik font ve çizgi kalınlığı (OR mantığı: max(w, h))
    font_scale = calculate_font_scale(w, h)
    thickness = calculate_thickness(w, h)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box, det_conf in zip(det_results.boxes.xyxy, det_results.boxes.conf):
        x1, y1, x2, y2 = box.tolist()

        # Ölçek geri al (det_image -> orijinal)
        x1, y1, x2, y2 = [coord / scale for coord in [x1, y1, x2, y2]]

        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        x1i, y1i = max(0, x1i), max(0, y1i)
        x2i, y2i = min(w - 1, x2i), min(h - 1, y2i)

        crop = image_bgr_orig[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue

        inp = preprocess_crop_for_classifier(crop)
        if inp is None:
            continue
        inp = inp.to(device)

        with torch.no_grad():
            logits = classifier(inp)
            probs = F.softmax(logits, dim=1)[0]
            cls_idx = int(torch.argmax(probs))
            cls_prob = float(probs[cls_idx])
            cls_name = class_names[cls_idx]

        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "det_conf": float(det_conf),
                "cls": cls_name,
                "cls_prob": cls_prob,
            }
        )

        # Her tespit için random açık renk üret
        color = generate_random_light_color()
        
        label = f"{cls_name} {cls_prob:.2f}"
        
        # Çerçeve çiz
        cv2.rectangle(image_bgr_orig, (x1i, y1i), (x2i, y2i), color, thickness)
        
        # Yazı için arka plan (okunabilirlik için)
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x1i
        text_y = max(y1i - 5, text_height + baseline + 5)
        
        # Arka plan dikdörtgeni (padding ile)
        padding = 3
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_height - baseline - padding
        bg_x2 = text_x + text_width + padding
        bg_y2 = text_y + baseline + padding
        
        # Arka planı çiz
        cv2.rectangle(
            image_bgr_orig,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1
        )
        
        # Yazıyı çiz (siyah renk, okunabilirlik için)
        cv2.putText(
            image_bgr_orig,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),  # Siyah yazı
            thickness,
            cv2.LINE_AA
        )

    return {"image": image_bgr_orig, "detections": detections}

