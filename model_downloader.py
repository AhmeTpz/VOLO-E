"""
Google Drive klasöründeki modelleri indirip `models/classifier` klasörüne kaydeder.
Gereksinim: pip install gdown
"""

from pathlib import Path
import sys

try:
    import gdown
except ImportError:
    sys.stderr.write(
        "gdown yüklü değil. Kurmak için: pip install gdown\n"
    )
    sys.exit(1)

# Google Drive klasör ID'si (linkten alınmış)
FOLDER_ID = "1KQQYBz87y4dVUX4vaiaaiMJ8igR6_SBg"


def main() -> None:
    # İndirilen her şeyi models/classifier/model içine koy
    target_dir = Path(__file__).parent / "models" / "classifier" / "model"
    target_dir.mkdir(parents=True, exist_ok=True)

    # gdown.download_folder tüm içerikleri indirir
    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    print(f"İndiriliyor: {url}")
    print(f"Hedef klasör: {target_dir}")

    gdown.download_folder(
        url=url,
        output=str(target_dir),
        quiet=False,
        use_cookies=False,
    )

    print("İndirme tamamlandı.")


if __name__ == "__main__":
    main()

