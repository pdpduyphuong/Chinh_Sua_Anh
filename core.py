# core.py
import os
import sys
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from u2net_engine import U2NetInference

# Cấu hình Singleton cho U2Net Inference Engine
_U2NET_ENGINE = None


def get_u2net_engine(model_name: str = "u2netp") -> U2NetInference:
    global _U2NET_ENGINE
    if _U2NET_ENGINE is None:
        _U2NET_ENGINE = U2NetInference(model_name=model_name)
    return _U2NET_ENGINE


def remove_background_u2net(input_path: str, output_path: str, model_type: str = "u2netp"):
    """
    Tách nền ảnh tự động bằng AI U2Net / U2NetP.
    """
    img = Image.open(input_path)
    engine = get_u2net_engine(model_name=model_type)
    result_img = engine.remove_background(img)
    result_img.save(output_path, "PNG")


def remove_background_by_color(
        input_path: str,
        output_path: str,
        target_color_hex: str = "#FFFFFF",
        tolerance: int = 30
):
    """
    Tách nền thủ công bằng chọn màu sắc (Chroma Keying).
    """
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img, dtype=np.int16)

    hex_val = target_color_hex.lstrip('#')
    r_target = int(hex_val[0:2], 16)
    g_target = int(hex_val[2:4], 16)
    b_target = int(hex_val[4:6], 16)

    r_diff = data[:, :, 0] - r_target
    g_diff = data[:, :, 1] - g_target
    b_diff = data[:, :, 2] - b_target

    dist = np.sqrt(r_diff ** 2 + g_diff ** 2 + b_diff ** 2)
    mask = dist <= tolerance
    data[mask, 3] = 0

    result_img = Image.fromarray(data.astype(np.uint8), "RGBA")
    result_img.save(output_path, "PNG")


def analyze_image_fast(input_path: str) -> dict:
    """Phân tích các chỉ số ảnh sử dụng NumPy thuần."""
    pil_img = Image.open(input_path).convert("RGB")
    img_np = np.array(pil_img, dtype=np.float32)

    gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    raw_brightness = np.mean(gray)
    brightness_pct = round((raw_brightness / 255.0) * 100, 1)

    std_contrast = round(float(np.std(gray)), 1)
    p10 = np.percentile(gray, 10) / 2.55
    p90 = np.percentile(gray, 90) / 2.55

    target_pct = 58.0
    suggested_exposure = round((target_pct - brightness_pct) / 25.0, 1)
    suggested_exposure = float(np.clip(suggested_exposure, -1.0, 1.5))

    suggested_shadows = int(np.clip((25 - p10) * 1.2, 0, 50)) if p10 < 25 else 0
    suggested_highlights = -15 if p90 > 75 else 0
    suggested_contrast = 15 if std_contrast < 50 else 5

    return {
        "brightness_pct": brightness_pct,
        "std_contrast": std_contrast,
        "exposure": suggested_exposure,
        "contrast": suggested_contrast,
        "highlights": suggested_highlights,
        "shadows": suggested_shadows,
        "saturation": 10,
        "clarity": 15,
        "dehaze": 0,
        "sharpening": 20,
    }


def get_system_font(font_name: str, font_size: int):
    font_map = {
        "Arial": ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"],
        "Segoe UI": ["segoeui.ttf", "SegoeUI.ttf", "DejaVuSans.ttf"],
        "Times New Roman": ["times.ttf", "Times.ttf", "DejaVuSerif.ttf"],
        "Courier New": ["cour.ttf", "Courier.ttf", "DejaVuSansMono.ttf"],
        "Calibri": ["calibri.ttf", "Calibri.ttf", "DejaVuSans.ttf"],
        "Tahoma": ["tahoma.ttf", "Tahoma.ttf", "DejaVuSans.ttf"],
        "Verdana": ["verdana.ttf", "Verdana.ttf", "DejaVuSans.ttf"]
    }

    font_dirs = [
        "",
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/truetype/liberation/",
        "/usr/share/fonts/truetype/freefont/",
        "C:\\Windows\\Fonts\\",
        "/Library/Fonts/",
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

    try:
        return ImageFont.load_default(size=font_size)
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
    img = Image.open(input_path).convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    font = get_system_font(font_name, font_size)

    draw.text(position, text, fill=color + (255,), font=font)
    combined = Image.alpha_composite(img, txt_layer)
    combined.save(output_path, "PNG")


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
    pil_img = Image.open(input_path)
    is_rgba = pil_img.mode == "RGBA"

    if is_rgba:
        alpha = pil_img.split()[3]
        rgb_img = pil_img.convert("RGB")
    else:
        rgb_img = pil_img.convert("RGB")

    img = np.array(rgb_img, dtype=np.float32)

    exp, cnt = float(exposure), float(contrast)
    hl, sh = float(highlights), float(shadows)
    sat, shp = float(saturation), float(sharpening)

    if exp != 0.0:
        img *= (2.0 ** exp)

    if cnt != 0.0:
        factor = (259.0 * (cnt + 255.0)) / (255.0 * (259.0 - cnt))
        img = 128.0 + factor * (img - 128.0)

    if hl != 0.0 or sh != 0.0:
        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 255.0
        if hl != 0.0:
            img += (hl * 1.5) * np.power(gray, 2)[:, :, np.newaxis]
        if sh != 0.0:
            img += (sh * 1.5) * (1.0 - np.power(gray, 2))[:, :, np.newaxis]

    img = np.clip(img, 0, 255).astype(np.uint8)
    result_pil = Image.fromarray(img)

    if sat != 0.0:
        enhancer = ImageEnhance.Color(result_pil)
        result_pil = enhancer.enhance(max(0.0, 1.0 + (sat / 100.0)))

    if shp > 0.0:
        enhancer = ImageEnhance.Sharpness(result_pil)
        result_pil = enhancer.enhance(1.0 + (shp / 25.0))

    if is_rgba:
        result_pil = result_pil.convert("RGBA")
        result_pil.putalpha(alpha)
        result_pil.save(output_path, "PNG")
    else:
        result_pil.save(output_path)


def apply_filter(input_path: str, output_path: str, filter_type: str):
    img = Image.open(input_path)
    has_alpha = img.mode == "RGBA"

    if has_alpha:
        alpha = img.split()[3]
        rgb_img = img.convert("RGB")
    else:
        rgb_img = img.convert("RGB")

    if filter_type == "Gốc (Original / Không bộ lọc)":
        img.save(output_path)
        return
    elif filter_type == "Trắng Đen (Grayscale)":
        rgb_img = rgb_img.convert("L").convert("RGB")
    elif filter_type == "Cổ Điển (Sepia)":
        np_img = np.array(rgb_img, dtype=np.float32)
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        rgb_img = Image.fromarray(np.clip(np_img.dot(sepia_matrix.T), 0, 255).astype(np.uint8))
    elif filter_type == "Rực Rỡ (Vintage/Warm)":
        np_img = np.array(rgb_img, dtype=np.float32)
        np_img[:, :, 0] = np.clip(np_img[:, :, 0] * 1.15, 0, 255)
        np_img[:, :, 2] = np.clip(np_img[:, :, 2] * 0.85, 0, 255)
        rgb_img = Image.fromarray(np_img.astype(np.uint8))
    elif filter_type == "Lạnh (Cool Tone)":
        np_img = np.array(rgb_img, dtype=np.float32)
        np_img[:, :, 0] = np.clip(np_img[:, :, 0] * 0.85, 0, 255)
        np_img[:, :, 2] = np.clip(np_img[:, :, 2] * 1.15, 0, 255)
        rgb_img = Image.fromarray(np_img.astype(np.uint8))

    if has_alpha:
        rgb_img = rgb_img.convert("RGBA")
        rgb_img.putalpha(alpha)
        rgb_img.save(output_path, "PNG")
    else:
        rgb_img.save(output_path)


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