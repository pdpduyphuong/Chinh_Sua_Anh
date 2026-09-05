# core.py
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from rembg import remove

def get_system_font(font_name: str, font_size: int):
    """
    Tìm và nạp Font chữ TrueType an toàn trên cả Windows, macOS và Linux/Streamlit Cloud.
    """
    # Bảng ánh xạ tên font sang file .ttf chuẩn
    font_map = {
        "Arial": ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"],
        "Segoe UI": ["segoeui.ttf", "SegoeUI.ttf", "DejaVuSans.ttf"],
        "Times New Roman": ["times.ttf", "Times.ttf", "DejaVuSerif.ttf"],
        "Courier New": ["cour.ttf", "Courier.ttf", "DejaVuSansMono.ttf"],
        "Calibri": ["calibri.ttf", "Calibri.ttf", "DejaVuSans.ttf"],
        "Tahoma": ["tahoma.ttf", "Tahoma.ttf", "DejaVuSans.ttf"],
        "Verdana": ["verdana.ttf", "Verdana.ttf", "DejaVuSans.ttf"]
    }

    # Danh sách các thư mục chứa font hệ thống phổ biến
    font_dirs = [
        "",  # Thư mục hiện tại
        "/usr/share/fonts/truetype/dejavu/",  # Linux Streamlit Cloud
        "/usr/share/fonts/truetype/liberation/",
        "/usr/share/fonts/truetype/freefont/",
        "C:\\Windows\\Fonts\\",  # Windows
        "/Library/Fonts/",  # macOS
    ]

    candidate_files = font_map.get(font_name, ["DejaVuSans.ttf", "arial.ttf"])

    for candidate in candidate_files:
        for fdir in font_dirs:
            full_path = os.path.join(fdir, candidate)
            if os.path.exists(full_path):
                try:
                    return ImageFont.truetype(full_path, font_size)
                except Exception:
                    continue

    # Bản sao lưu cuối cùng nếu không tìm thấy font ttf nào trên hệ thống
    try:
        return ImageFont.load_default(size=font_size) # Hỗ trợ size đối với Pillow >= 10.1.0
    except TypeError:
        return ImageFont.load_default()

def add_text_to_image(
    input_path: str,
    output_path: str,
    text: str,
    position: tuple,
    font_name: str,
    font_size: int,
    color: tuple
):
    """
    Chèn chữ vào ảnh với kích thước font được tỉ lệ chính xác và nạp font an toàn.
    """
    img = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Nạp font an toàn
    font = get_system_font(font_name, font_size)

    # Vẽ chữ lên ảnh
    draw.text(position, text, fill=color, font=font)
    img.save(output_path)

def adjust_image_advanced(
    input_path: str,
    output_path: str,
    exposure: float = 0.0,
    contrast: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    saturation: float = 0.0,
    clarity: float = 0.0,
    dehaze: float = 0.0,
    sharpening: float = 0.0
):
    pil_img = Image.open(input_path).convert("RGB")
    img = np.array(pil_img, dtype=np.float32)

    exp, cnt = float(exposure), float(contrast)
    hl, sh = float(highlights), float(shadows)
    sat, clr = float(saturation), float(clarity)
    dhz, shp = float(dehaze), float(sharpening)

    if exp != 0.0:
        img *= (2.0 ** exp)

    if cnt != 0.0:
        factor = (259.0 * (cnt + 255.0)) / (255.0 * (259.0 - cnt))
        img = 128.0 + factor * (img - 128.0)

    if hl != 0.0 or sh != 0.0:
        gray = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        norm_gray = gray / 255.0

        if hl != 0.0:
            img += (hl * 1.5) * np.power(norm_gray, 2)[:, :, np.newaxis]
        if sh != 0.0:
            img += (sh * 1.5) * (1.0 - np.power(norm_gray, 2))[:, :, np.newaxis]

    img = np.clip(img, 0, 255).astype(np.uint8)
    result_pil = Image.fromarray(img)

    if sat != 0.0:
        enhancer = ImageEnhance.Color(result_pil)
        result_pil = enhancer.enhance(max(0.0, 1.0 + (sat / 100.0)))

    if shp > 0.0:
        img_np = np.array(result_pil)
        blurred = cv2.GaussianBlur(img_np, (0, 0), 3)
        sharpened = cv2.addWeighted(img_np, 1.0 + (shp / 50.0), blurred, -(shp / 50.0), 0)
        result_pil = Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

    result_pil.save(output_path)

def apply_filter(input_path: str, output_path: str, filter_type: str):
    img = Image.open(input_path).convert("RGB")
    if filter_type == "Gốc (Original / Không bộ lọc)":
        img.save(output_path)
        return
    elif filter_type == "Trắng Đen (Grayscale)":
        img = img.convert("L").convert("RGB")
    elif filter_type == "Cổ Điển (Sepia)":
        np_img = np.array(img, dtype=np.float32)
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        img = Image.fromarray(np.clip(np_img.dot(sepia_matrix.T), 0, 255).astype(np.uint8))
    elif filter_type == "Rực Rỡ (Vintage/Warm)":
        np_img = np.array(img, dtype=np.float32)
        np_img[:, :, 0] = np.clip(np_img[:, :, 0] * 1.15, 0, 255)
        np_img[:, :, 2] = np.clip(np_img[:, :, 2] * 0.85, 0, 255)
        img = Image.fromarray(np_img.astype(np.uint8))
    elif filter_type == "Lạnh (Cool Tone)":
        np_img = np.array(img, dtype=np.float32)
        np_img[:, :, 0] = np.clip(np_img[:, :, 0] * 0.85, 0, 255)
        np_img[:, :, 2] = np.clip(np_img[:, :, 2] * 1.15, 0, 255)
        img = Image.fromarray(np_img.astype(np.uint8))

    img.save(output_path)

def rotate_or_flip_image(input_path: str, output_path: str, action: str):
    img = Image.open(input_path)
    if action == "rotate_right":
        img = img.rotate(-90, expand=True)
    elif action == "rotate_left":
        img = img.rotate(90, expand=True)
    elif action == "flip_horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif action == "flip_vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(output_path)

def crop_image(input_path: str, output_path: str, box: tuple):
    img = Image.open(input_path)
    img.crop(box).save(output_path)

def resize_standard(input_path: str, output_path: str, width: int, height: int):
    img = Image.open(input_path)
    img.resize((width, height), Image.Resampling.LANCZOS).save(output_path)

def resize_ai_upscale(input_path: str, output_path: str, scale_factor: int):
    img = Image.open(input_path)
    w, h = img.size
    img.resize((w * scale_factor, h * scale_factor), Image.Resampling.LANCZOS).save(output_path)

def remove_background_ai(input_path: str, output_path: str):
    inp = Image.open(input_path)
    remove(inp).save(output_path)