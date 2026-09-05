# core.py
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter,ImageDraw, ImageFont


# --- KIỂM TRA & IMPORT NĂNG ĐỘNG DÀNH CHO OPENCV ---
try:
    import cv2
except ImportError:
    # Trường hợp fallback nếu opencv-python bị thiếu trên Linux
    cv2 = None

# -------------------------------------------------------------------
# 1. BỘ CẤU HÌNH FONT CHỮ DÙNG CHO THÊM TEXT
# -------------------------------------------------------------------
FONT_MAP = {
    "Arial": ["arial.ttf", "Arial.ttf", "C:\\Windows\\Fonts\\arial.ttf"],
    "Times New Roman": ["times.ttf", "Times New Roman.ttf", "C:\\Windows\\Fonts\\times.ttf"],
    "Courier New": ["cour.ttf", "Courier New.ttf", "C:\\Windows\\Fonts\\cour.ttf"],
    "Segoe UI": ["segoeui.ttf", "C:\\Windows\\Fonts\\segoeui.ttf"],
    "Calibri": ["calibri.ttf", "C:\\Windows\\Fonts\\calibri.ttf"],
    "Georgia": ["georgia.ttf", "C:\\Windows\\Fonts\\georgia.ttf"],
    "Tahoma": ["tahoma.ttf", "C:\\Windows\\Fonts\\tahoma.ttf"],
    "Verdana": ["verdana.ttf", "C:\\Windows\\Fonts\\verdana.ttf"]
}

def load_selected_font(font_name: str, font_size: int):
    paths = FONT_MAP.get(font_name, FONT_MAP["Arial"])
    for path in paths:
        try:
            return ImageFont.truetype(path, font_size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()

# -------------------------------------------------------------------
# 2. XỬ LÝ ÁNH SÁNG & MÀU SẮC NÂNG CAO
# -------------------------------------------------------------------

def adjust_image_advanced(
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0
) -> Image.Image:
    """
    Điều chỉnh các thông số ảnh nâng cao.

    :param image: Đối tượng PIL Image.
    :param brightness: Hệ số độ sáng (1.0 là mặc định).
    :param contrast: Hệ số độ tương phản (1.0 là mặc định).
    :param saturation: Hệ số độ bão hòa màu (1.0 là mặc định).
    :param sharpness: Hệ số độ sắc nét (1.0 là mặc định).
    :return: PIL Image đã qua xử lý.
    """
    try:
        # 1. Điều chỉnh độ sáng (Brightness)
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)

        # 2. Điều chỉnh độ tương phản (Contrast)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        # 3. Điều chỉnh độ bão hòa màu (Color/Saturation)
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)

        # 4. Điều chỉnh độ sắc nét (Sharpness)
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)

        return image

    except Exception as e:
        raise RuntimeError(f"Lỗi khi điều chỉnh ảnh nâng cao: {str(e)}")

# -------------------------------------------------------------------
# 3. HÀM CẮT ÁNH (CROP IMAGE)
# -------------------------------------------------------------------
def crop_image(input_path: str, output_path: str, crop_box: tuple) -> None:
    """
    crop_box: tuple (x1, y1, x2, y2)
    """
    with Image.open(input_path) as img:
        w, h = img.size
        x1, y1, x2, y2 = crop_box

        left = float(min(x1, x2))
        right = float(max(x1, x2))
        upper = float(min(y1, y2))
        lower = float(max(y1, y2))

        left = int(max(0, min(round(left), w - 1)))
        upper = int(max(0, min(round(upper), h - 1)))
        right = int(max(left + 1, min(round(right), w)))
        lower = int(max(upper + 1, min(round(lower), h)))

        if right > left and lower > upper:
            cropped = img.crop((left, upper, right, lower))
            cropped.save(output_path, format="PNG")
        else:
            img.save(output_path, format="PNG")

# -------------------------------------------------------------------
# 4. CÁC TÍNH NĂNG PHỤ TRỢ KHÁC
# -------------------------------------------------------------------
def add_text_to_image(
    input_path: str,
    output_path: str,
    text: str,
    position: tuple = (50, 50),
    font_name: str = "Arial",
    font_size: int = 40,
    color: tuple = (255, 0, 0)
) -> None:
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        font = load_selected_font(font_name, font_size)
        draw.text(position, text, fill=color, font=font)
        img.save(output_path, format="PNG")

def apply_filter(input_path: str, output_path: str, filter_type: str) -> None:
    if cv2 is None:
        return

    img = cv2.imread(input_path)
    if img is None:
        return

    if filter_type == "Trắng Đen (Grayscale)":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Cổ Điển (Sepia)":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        result = cv2.transform(img, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
    elif filter_type == "Rực Rỡ (Vintage/Warm)":
        b, g, r = cv2.split(img)
        r = cv2.add(r, 20)
        g = cv2.add(g, 10)
        result = cv2.merge([b, g, r])
    elif filter_type == "Lạnh (Cool Tone)":
        b, g, r = cv2.split(img)
        b = cv2.add(b, 25)
        result = cv2.merge([b, g, r])
    else:
        result = img

    cv2.imwrite(output_path, result)

def resize_standard(input_path: str, output_path: str, width: int, height: int) -> None:
    with Image.open(input_path) as img:
        resized = img.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(output_path, format="PNG")

def resize_ai_upscale(input_path: str, output_path: str, scale_factor: int = 2) -> None:
    with Image.open(input_path) as img:
        new_w = int(img.width * scale_factor)
        new_h = int(img.height * scale_factor)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized.save(output_path, format="PNG")

def remove_background_ai(input_path: str, output_path: str) -> None:
    """Lazy Loading cho rembg để tránh lỗi ImportError khi ứng dụng khởi chạy."""
    try:
        from rembg import remove
    except ImportError as err:
        raise ImportError(
            "Thư viện 'rembg' chưa được cài đặt. Vui lòng thêm 'rembg' và 'onnxruntime' vào requirements.txt."
        ) from err

    with open(input_path, "rb") as i:
        input_data = i.read()
        output_data = remove(input_data)
        with open(output_path, "wb") as o:
            o.write(output_data)

def rotate_or_flip_image(input_path: str, output_path: str, action: str) -> None:
    with Image.open(input_path) as img:
        if action == "rotate_right":
            res = img.rotate(-90, expand=True)
        elif action == "rotate_left":
            res = img.rotate(90, expand=True)
        elif action == "flip_horizontal":
            res = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif action == "flip_vertical":
            res = img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            res = img
        res.save(output_path, format="PNG")