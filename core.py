# core.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove

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
    """
    Xử lý ánh sáng và màu sắc nâng cao sử dụng OpenCV và PIL.
    Chuyển đổi kiểu dữ liệu an toàn để tránh TypeError.
    """
    # 1. Đọc ảnh an toàn
    pil_img = Image.open(input_path).convert("RGB")
    img = np.array(pil_img, dtype=np.float32)

    # 2. Ép kiểu dữ liệu tham số đầu vào tránh lỗi numpy/type error
    exp = float(exposure)
    cnt = float(contrast)
    hl = float(highlights)
    sh = float(shadows)
    sat = float(saturation)
    clr = float(clarity)
    dhz = float(dehaze)
    shp = float(sharpening)

    # A. Exposure (Độ sáng)
    if exp != 0.0:
        factor = 2.0 ** exp
        img = img * factor

    # B. Contrast (Độ tương phản)
    if cnt != 0.0:
        factor = (259.0 * (cnt + 255.0)) / (255.0 * (259.0 - cnt))
        img = 128.0 + factor * (img - 128.0)

    # C. Highlights & Shadows
    if hl != 0.0 or sh != 0.0:
        gray = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        normalized_gray = gray / 255.0

        if hl != 0.0:
            hl_mask = np.power(normalized_gray, 2)[:, :, np.newaxis]
            img += (hl * 1.5) * hl_mask

        if sh != 0.0:
            sh_mask = (1.0 - np.power(normalized_gray, 2))[:, :, np.newaxis]
            img += (sh * 1.5) * sh_mask

    # Giới hạn giá trị Pixel [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # D. Saturation (Độ bão hòa màu)
    result_pil = Image.fromarray(img)
    if sat != 0.0:
        enhancer = ImageEnhance.Color(result_pil)
        sat_factor = max(0.0, 1.0 + (sat / 100.0))
        result_pil = enhancer.enhance(sat_factor)

    # E. Sharpening (Sắc nét)
    if shp > 0.0:
        img_np = np.array(result_pil)
        amount = shp / 50.0
        blurred = cv2.GaussianBlur(img_np, (0, 0), 3)
        sharpened = cv2.addWeighted(img_np, 1.0 + amount, blurred, -amount, 0)
        result_pil = Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

    # Lưu kết quả
    result_pil.save(output_path)

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
    cropped = img.crop(box)
    cropped.save(output_path)

def resize_standard(input_path: str, output_path: str, width: int, height: int):
    img = Image.open(input_path)
    resized = img.resize((width, height), Image.Resampling.LANCZOS)
    resized.save(output_path)

def resize_ai_upscale(input_path: str, output_path: str, scale_factor: int):
    img = Image.open(input_path)
    w, h = img.size
    resized = img.resize((w * scale_factor, h * scale_factor), Image.Resampling.LANCZOS)
    resized.save(output_path)

def remove_background_ai(input_path: str, output_path: str):
    inp = Image.open(input_path)
    out = remove(inp)
    out.save(output_path)

def add_text_to_image(input_path: str, output_path: str, text: str, position: tuple, font_name: str, font_size: int, color: tuple):
    from PIL import ImageDraw, ImageFont
    img = Image.open(input_path)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    img.save(output_path)

def apply_filter(input_path: str, output_path: str, filter_type: str):
    img = Image.open(input_path)
    if filter_type == "Trắng Đen (Grayscale)":
        img = img.convert("L")
    elif filter_type == "Cổ Điển (Sepia)":
        img = img.convert("RGB")
        np_img = np.array(img, dtype=np.float32)
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = np_img.dot(sepia_matrix.T)
        img = Image.fromarray(np.clip(sepia_img, 0, 255).astype(np.uint8))
    img.save(output_path)