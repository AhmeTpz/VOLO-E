import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

from main_training import SimpleCNN, IMG_SIZE


def load_class_names(summary_path: Path) -> List[str]:
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["class_names"]


def load_classifier(model_path: Path, class_names: List[str], device: torch.device) -> torch.nn.Module:
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def get_classifier_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def detect_and_classify(
    image_path: Path,
    detector: YOLO,
    classifier: torch.nn.Module,
    class_names: List[str],
    device: torch.device,
    conf_thres: float,
) -> Dict[str, Any]:
    det_results = detector(str(image_path), conf=conf_thres, verbose=False)[0]
    image_bgr = cv2.imread(str(image_path))
    h, w = image_bgr.shape[:2]
    tfm = get_classifier_transform()
    detections = []

    for box, det_conf in zip(det_results.boxes.xyxy, det_results.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        x1i, y1i = max(0, x1i), max(0, y1i)
        x2i, y2i = min(w - 1, x2i), min(h - 1, y2i)

        crop = image_bgr[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        inp = tfm(pil_img).unsqueeze(0).to(device)

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

        label = f"{cls_name} {cls_prob:.2f}"
        cv2.rectangle(image_bgr, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1i, max(y1i - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return {"image": image_bgr, "detections": detections}


def save_outputs(result: Dict[str, Any], out_dir: Path, image_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / f"{image_name}_annotated.png"
    out_json = out_dir / f"{image_name}_detections.json"
    cv2.imwrite(str(out_img), result["image"])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result["detections"], f, ensure_ascii=False, indent=2)
    print(f"✅ Kaydedildi: {out_img}")
    print(f"✅ Kaydedildi: {out_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO dedektör + sınıflandırıcı inference pipeline")
    parser.add_argument("--image", required=True, help="Girdi görsel yolu")
    parser.add_argument("--detector", default="models/detector/best.pt", help="YOLO dedektör ağırlığı")
    parser.add_argument("--classifier", default="models/classifier/best_model.pt", help="Sınıflandırıcı ağırlığı")
    parser.add_argument("--summary", default="models/classifier/summary.json", help="Sınıf isimleri için summary.json")
    parser.add_argument("--out_dir", default="output_infer", help="Çıktı klasörü")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO tespit conf eşiği")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    detector_path = Path(args.detector)
    classifier_path = Path(args.classifier)
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Cihaz: {device}")

    class_names = load_class_names(summary_path)
    classifier = load_classifier(classifier_path, class_names, device)
    detector = YOLO(str(detector_path))

    result = detect_and_classify(
        image_path=image_path,
        detector=detector,
        classifier=classifier,
        class_names=class_names,
        device=device,
        conf_thres=args.conf,
    )
    save_outputs(result, out_dir, image_path.stem)
    print(f"🔎 Toplam tespit: {len(result['detections'])}")


if __name__ == "__main__":
    main()

