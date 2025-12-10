import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from inference import detect_and_classify, DEFAULT_DETECTOR

# ----------------------------------
# Yardımcılar
# ----------------------------------
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
# Görsel penceresi için sabit boyutlar
RESULT_WINDOW_WIDTH = 1080
RESULT_WINDOW_HEIGHT = 720
# Maskot boyutu
MASCOT_SIZE = (224, 300)  # (width, height) - 56x75'in 3 katı

# Renk paleti (arka plan görseline uyumlu açık/bej tonları)
BG_COLOR = "#f6f2e8"
PANEL_COLOR = "#f0e8d8"
TEXT_COLOR = "#3b3b3b"
ACCENT_COLOR = "#c9a43b"
BUTTON_BG = "#d8c9a6"
BUTTON_FG = "#4a3c2c"
RUN_BG = "#5aa469"
RUN_FG = "#ffffff"


def resize_image_to_fit(img_bgr, max_width, max_height):
    """Görseli belirtilen boyutlara sığdır (aspect ratio korunarak)"""
    h, w = img_bgr.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def show_image_in_window(img_bgr, title="Sonuç"):
    """Görseli sabit boyutlu pencerede göster"""
    # Görseli pencere boyutuna sığdır
    img_resized = resize_image_to_fit(img_bgr, RESULT_WINDOW_WIDTH, RESULT_WINDOW_HEIGHT)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    win = tk.Toplevel()
    win.title(title)
    win.geometry(f"{RESULT_WINDOW_WIDTH}x{RESULT_WINDOW_HEIGHT}+100+100")
    
    tk_img = ImageTk.PhotoImage(pil_img)
    lbl = tk.Label(win, image=tk_img, bg=PANEL_COLOR)
    lbl.image = tk_img
    lbl.pack(expand=True, fill="both")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VOLO-E Akıllı Geri Dönüşüm Paneli")
        self.root.geometry("720x960")
        self.root.configure(bg=BG_COLOR)

        # Arka plan görseli
        bg_path = ASSETS_DIR / "background.png"
        if bg_path.exists():
            bg_img = Image.open(bg_path).resize((720, 960))
            self.bg_photo = ImageTk.PhotoImage(bg_img)
            self.bg_label = tk.Label(self.root, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.bg_label.lower()

        self.classifier_path: Path | None = None
        self.summary_path: Path | None = None
        self.image_path: Path | None = None
        self.conf_value = tk.DoubleVar(value=0.25)

        self.mascot_images = self.load_mascots()
        self.current_mascot_idx = 0
        self.mascot_photo = None

        self.build_ui()
        self.set_mascot(0, instant=True)  # 1. görsel

    # ---------------- Mascot ----------------
    def load_mascots(self):
        """Maskot görsellerini yükle ve küçük boyuta getir"""
        imgs = []
        for i in range(1, 7):
            p = ASSETS_DIR / f"{i}.png"
            img = Image.open(p).convert("RGBA")
            # Maskot boyutunu küçült
            img_resized = img.resize(MASCOT_SIZE, Image.Resampling.LANCZOS)
            imgs.append(img_resized)
        return imgs

    def blend_transition(self, target_idx, steps=8, delay=40):
        if target_idx == self.current_mascot_idx:
            return
        src = self.mascot_images[self.current_mascot_idx]
        tgt = self.mascot_images[target_idx]
        
        def animate_step(step):
            if step > steps:
                self.current_mascot_idx = target_idx
                return
            alpha = step / steps
            frame = Image.blend(src, tgt, alpha)
            self.mascot_photo = ImageTk.PhotoImage(frame)
            self.mascot_label.configure(image=self.mascot_photo)
            self.root.after(delay, lambda: animate_step(step + 1))
        
        animate_step(0)

    def set_mascot(self, idx, instant=False):
        idx = max(0, min(idx, len(self.mascot_images) - 1))
        if instant:
            frame = self.mascot_images[idx]
            self.mascot_photo = ImageTk.PhotoImage(frame)
            self.mascot_label.configure(image=self.mascot_photo)
            self.current_mascot_idx = idx
        else:
            self.blend_transition(idx)

    # ---------------- UI ----------------
    def build_ui(self):
        # Üst başlık
        top_bar = tk.Frame(self.root, bg=PANEL_COLOR, bd=0, highlightthickness=0)
        top_bar.pack(fill="x", pady=10, padx=14)
        title = tk.Label(
            top_bar,
            text="VOLO-E Akıllı Geri Dönüşüm",
            fg=TEXT_COLOR,
            bg=PANEL_COLOR,
            font=("Arial", 18, "bold"),
        )
        title.pack(side="right")

        # Ana içerik alanı
        main = tk.Frame(self.root, bg=PANEL_COLOR, bd=0, highlightthickness=0)
        main.pack(fill="both", expand=True, padx=14, pady=10)

        # Sol taraf: Maskot (sol altta)
        left_frame = tk.Frame(main, bg=PANEL_COLOR)
        left_frame.pack(side="left", fill="y", padx=(0, 20))
        self.mascot_label = tk.Label(left_frame, bg=PANEL_COLOR)
        self.mascot_label.pack(side="bottom", anchor="sw", padx=10, pady=10)

        # Merkez: Kontroller
        center_frame = tk.Frame(main, bg=PANEL_COLOR)
        center_frame.pack(side="left", fill="both", expand=True)
        
        # Kontrolleri merkeze hizalamak için boşluk ekle
        spacer_top = tk.Frame(center_frame, bg=PANEL_COLOR, height=80)
        spacer_top.pack(fill="x")
        
        controls = tk.Frame(center_frame, bg=PANEL_COLOR)
        controls.pack(expand=True)

        # Model yükleme
        tk.Label(controls, text="Model Yükle (Classifier)", fg=TEXT_COLOR, bg=PANEL_COLOR, font=("Arial", 11, "bold")).pack(anchor="w")
        model_row = tk.Frame(controls, bg=PANEL_COLOR)
        model_row.pack(anchor="w", pady=(2, 12), fill="x")
        tk.Button(model_row, text="Dosya Seç", command=self.on_select_model, bg=BUTTON_BG, fg=BUTTON_FG, width=14, bd=0, relief="ridge").pack(side="left")
        self.model_label = tk.Label(model_row, text="Seçilmedi", fg=TEXT_COLOR, bg=PANEL_COLOR, anchor="w", wraplength=220, justify="left")
        self.model_label.pack(side="left", padx=8)

        # Conf slider
        tk.Label(controls, text="Hassasiyet", fg=TEXT_COLOR, bg=PANEL_COLOR, font=("Arial", 11, "bold")).pack(anchor="w")
        slider = tk.Scale(
            controls,
            from_=0.05,
            to=0.5,
            resolution=0.01,
            orient="horizontal",
            variable=self.conf_value,
            length=180,
            fg=TEXT_COLOR,
            bg=PANEL_COLOR,
            highlightthickness=0,
            troughcolor=BUTTON_BG,
            activebackground=ACCENT_COLOR,
            command=lambda v: self.on_conf_sliding(),  # Kaydırma sırasında sadece label güncelle
        )
        # Slider bırakıldığında maskot değişimi
        slider.bind("<ButtonRelease-1>", lambda e: self.on_conf_released())
        slider.pack(anchor="w", pady=(2, 4))
        self.conf_label = tk.Label(controls, text="0.25", fg=TEXT_COLOR, bg=PANEL_COLOR)
        self.conf_label.pack(anchor="w", pady=(0, 12))

        # Görsel yükleme
        tk.Label(controls, text="Görsel Yükle", fg=TEXT_COLOR, bg=PANEL_COLOR, font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 0))
        image_row = tk.Frame(controls, bg=PANEL_COLOR)
        image_row.pack(anchor="w", pady=(2, 12), fill="x")
        tk.Button(image_row, text="Dosya Seç", command=self.on_select_image, bg=BUTTON_BG, fg=BUTTON_FG, width=14, bd=0, relief="ridge").pack(side="left")
        self.image_label = tk.Label(image_row, text="Seçilmedi", fg=TEXT_COLOR, bg=PANEL_COLOR, anchor="w", wraplength=220, justify="left")
        self.image_label.pack(side="left", padx=8)

        # Çalıştır
        self.run_btn = tk.Button(
            controls,
            text="Çalıştır",
            state="disabled",
            command=self.on_run,
            bg=RUN_BG,
            fg=RUN_FG,
            font=("Arial", 11, "bold"),
            width=20,
            bd=0,
            relief="ridge",
            activebackground="#4a8f5a",
        )
        self.run_btn.pack(anchor="w", pady=(16, 0))
        
        spacer_bottom = tk.Frame(center_frame, bg=PANEL_COLOR, height=60)
        spacer_bottom.pack(fill="x")

    # ---------------- Events ----------------
    def on_select_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt *.pth"), ("All", "*.*")])
        if not path:
            return
        p = Path(path)
        summary_guess = p.parent / "summary.json"
        if not summary_guess.exists():
            messagebox.showerror("Hata", f"summary.json bulunamadı: {summary_guess}")
            return
        self.classifier_path = p
        self.summary_path = summary_guess
        messagebox.showinfo("Model", f"Yüklendi: {p.name}")
        self.model_label.configure(text=p.name)
        self.set_mascot(1)  # model yüklendi -> 2. görsel
        self.update_run_state()

    def on_conf_sliding(self):
        """Slider kaydırılırken sadece label'ı güncelle (maskot değişmez)"""
        self.conf_label.configure(text=f"{self.conf_value.get():.2f}")
    
    def on_conf_released(self):
        """Slider bırakıldığında maskot görselini güncelle"""
        target_mascot_idx = 2  # conf ayarlandı -> 3. görsel
        # Sadece farklı bir görseldeyse değiştir, aynı görseldeyse hiçbir şey yapma
        if self.current_mascot_idx != target_mascot_idx:
            self.set_mascot(target_mascot_idx, instant=False)

    def on_select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.image_path = Path(path)
        messagebox.showinfo("Görsel", f"Seçildi: {self.image_path.name}")
        self.image_label.configure(text=self.image_path.name)
        self.set_mascot(3)  # görsel seçildi -> 4. görsel
        self.update_run_state()

    def update_run_state(self):
        if self.classifier_path and self.image_path:
            self.run_btn.configure(state="normal")
        else:
            self.run_btn.configure(state="disabled")

    def on_run(self):
        if not self.classifier_path or not self.image_path:
            messagebox.showwarning("Eksik", "Önce model ve görsel seçin.")
            return
        if not DEFAULT_DETECTOR.exists():
            messagebox.showerror("Hata", f"Dedektör bulunamadı: {DEFAULT_DETECTOR}")
            return
        self.run_btn.configure(state="disabled")
        self.set_mascot(4)  # inference başlayınca 5. görsel
        threading.Thread(target=self.run_inference_thread, daemon=True).start()

    def run_inference_thread(self):
        try:
            result = detect_and_classify(
                image_path=self.image_path,
                detector_path=DEFAULT_DETECTOR,
                classifier_path=self.classifier_path,
                summary_path=self.summary_path,
                conf_thres=self.conf_value.get(),
            )
            self.root.after(0, self.on_inference_done, result["image"], result["detections"])
        except Exception as e:
            self.root.after(0, self.on_inference_error, str(e))

    def on_inference_done(self, img, dets):
        self.set_mascot(5)  # çıktı açılınca 6. görsel
        show_image_in_window(img, title="Çıktı")
        self.run_btn.configure(state="normal")

    def on_inference_error(self, msg):
        messagebox.showerror("Hata", msg)
        self.run_btn.configure(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

