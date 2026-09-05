import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

def load_selected_font(font_name, font_size):
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
    input_path,
    output_path,
    exposure=0.0,
    contrast=0,
    highlights=0,
    shadows=0,
    saturation=0,
    clarity=0,
    dehaze=0,
    sharpening=0
):
    img_pil = Image.open(input_path).convert("RGB")
    img = np.array(img_pil)

    # A. Exposure
    if exposure != 0:
        factor = 2.0 ** exposure
        img = np.clip(img * factor, 0, 255).astype(np.uint8)

    # B. Contrast
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        img = np.clip(128 + f * (img.astype(np.float32) - 128), 0, 255).astype(np.uint8)

    # C. Shadows & Highlights
    if shadows != 0 or highlights != 0:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b_chan = cv2.split(lab)
        l_float = l.astype(np.float32)

        if shadows > 0:
            shadow_mask = np.clip((100.0 - l_float) / 100.0, 0, 1)
            l_float += shadow_mask * (shadows * 0.8)
        if highlights < 0:
            highlight_mask = np.clip(l_float / 255.0, 0, 1)
            l_float += highlight_mask * (highlights * 0.8)

        l_final = np.clip(l_float, 0, 255).astype(np.uint8)
        lab = cv2.merge([l_final, a, b_chan])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # D. Saturation
    if saturation != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_factor = 1.0 + (saturation / 100.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # E. Sharpening & Clarity
    if sharpening > 0:
        strength = sharpening / 100.0
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    res_pil = Image.fromarray(img)
    res_pil.save(output_path, format="PNG")

# -------------------------------------------------------------------
# 3. HÀM CẮT ÁNH (CROP IMAGE)
# -------------------------------------------------------------------
def crop_image(input_path, output_path, crop_box):
    """
    crop_box: tuple (left, upper, right, lower)
    """
    with Image.open(input_path) as img:
        w, h = img.size
        left = max(0, min(crop_box[0], w - 1))
        upper = max(0, min(crop_box[1], h - 1))
        right = max(left + 1, min(crop_box[2], w))
        lower = max(upper + 1, min(crop_box[3], h))

        cropped = img.crop((left, upper, right, lower))
        cropped.save(output_path, format="PNG")

# -------------------------------------------------------------------
# 4. CÁC TÍNH NĂNG PHỤ TRỢ KHÁC
# -------------------------------------------------------------------
def add_text_to_image(input_path, output_path, text, position=(50, 50), font_name="Arial", font_size=40, color=(255, 0, 0)):
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        font = load_selected_font(font_name, font_size)
        draw.text(position, text, fill=color, font=font)
        img.save(output_path, format="PNG")

def apply_filter(input_path, output_path, filter_type):
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

def resize_standard(input_path, output_path, width, height):
    with Image.open(input_path) as img:
        resized = img.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(output_path, format="PNG")

def resize_ai_upscale(input_path, output_path, scale_factor=2):
    with Image.open(input_path) as img:
        new_w = int(img.width * scale_factor)
        new_h = int(img.height * scale_factor)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized.save(output_path, format="PNG")

def remove_background_ai(input_path, output_path):
    from rembg import remove
    with open(input_path, "rb") as i:
        input_data = i.read()
        output_data = remove(input_data)
        with open(output_path, "wb") as o:
            o.write(output_data)

def rotate_or_flip_image(input_path, output_path, action):
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