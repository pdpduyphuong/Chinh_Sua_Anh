import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont

# -----------------------------------------------------------------
# 1. PHÓNG TO / NÂNG CẤP ẢNH BẰNG AI (SUPER RESOLUTION)
# -----------------------------------------------------------------
def resize_ai_upscale(image_path, output_path, model_name="EDSR", scale_factor=2, weights_dir="weights"):
    """
    Phóng to ảnh bằng mô hình AI Super Resolution (OpenCV DNN Super Resolution).
    Hỗ trợ các file .pb nằm trong thư mục weights (EDSR_x2.pb, EDSR_x4.pb, ...).
    """
    model_file = f"{model_name}_x{scale_factor}.pb"
    model_path = os.path.join(weights_dir, model_file)

    # Nếu tìm thấy mô hình AI (.pb) tương ứng trong thư mục weights
    if os.path.exists(model_path):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Không thể đọc file ảnh đầu vào!")

        has_alpha = len(img.shape) == 3 and img.shape[2] == 4
        if has_alpha:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            bgr = img

        sr.readModel(model_path)
        sr.setModel(model_name.lower(), scale_factor)
        upscaled_bgr = sr.upsample(bgr)

        if has_alpha:
            # Resize kênh alpha bằng INTER_CUBIC để khớp với kích thước mới
            h, w = upscaled_bgr.shape[:2]
            upscaled_alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)
            result = cv2.merge([upscaled_bgr[:, :, 0], upscaled_bgr[:, :, 1], upscaled_bgr[:, :, 2], upscaled_alpha])
        else:
            result = upscaled_bgr

        cv2.imwrite(output_path, result)

    else:
        # Fallback về INTER_CUBIC nếu không tìm thấy file weights AI
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Không thể đọc file ảnh!")

        h, w = img.shape[:2]
        new_w = w * scale_factor
        new_h = h * scale_factor

        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_path, upscaled)


# -----------------------------------------------------------------
# 2. XỬ LÝ NÂNG CAO (ÁNH SÁNG, MÀU SẮC, ĐỘ RÕ NÉT)
# -----------------------------------------------------------------
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
    sharpening=0,
    noise_reduction=0
):
    # 1. Đọc ảnh bằng PIL để giữ nguyên RGB
    img_pil = Image.open(input_path).convert("RGB")
    img = np.array(img_pil)

    # 2. Xử lý Exposure & Contrast
    # Exposure
    if exposure != 0:
        factor = 2.0 ** exposure
        img = np.clip(img * factor, 0, 255).astype(np.uint8)

    # Contrast
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        img = np.clip(128 + f * (img.astype(np.float32) - 128), 0, 255).astype(np.uint8)

    # 3. Xử lý Saturation (Độ bão hòa màu) - Dùng HSV để giữ đúng hệ màu
    if saturation != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_factor = 1.0 + (saturation / 100.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 4. Xử lý Sharpening (Làm nét)
    if sharpening > 0:
        kernel_strength = sharpening / 100.0
        gaussian = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.0 + kernel_strength, gaussian, -kernel_strength, 0)

    # 5. Lưu kết quả ra file chuẩn RGB
    result_pil = Image.fromarray(img)
    result_pil.save(output_path, format="PNG")


# -----------------------------------------------------------------
# 3. XOAY, LẬT, CẮT & RESIZE THƯỜNG
# -----------------------------------------------------------------
def rotate_or_flip_image(image_path, output_path, action="rotate_right"):
    with Image.open(image_path) as img:
        if action == "rotate_left":
            img = img.rotate(90, expand=True)
        elif action == "rotate_right":
            img = img.rotate(-90, expand=True)
        elif action == "flip_horizontal":
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif action == "flip_vertical":
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(output_path)


def resize_standard(image_path, output_path, width=None, height=None):
  with Image.open(image_path) as img:
    w, h = img.size

    # Ép kiểu dữ liệu về int để tránh lỗi float/string từ GUI
    width = int(width) if width is not None and str(width).isdigit() else None
    height = (
        int(height) if height is not None and str(height).isdigit() else None
    )

    if width and not height:
      height = int(h * (width / w))
    elif height and not width:
      width = int(w * (height / h))
    elif not width and not height:
      return

    resized = img.resize((width, height), Image.Resampling.LANCZOS)
    if output_path.lower().endswith((".jpg", ".jpeg")) and resized.mode in (
        "RGBA",
        "P",
    ):
      resized = resized.convert("RGB")
    resized.save(output_path)


def resize_ai_upscale(
    image_path,
    output_path,
    model_name="EDSR",
    scale_factor=2,
    weights_dir="weights",
):
  scale_factor = int(scale_factor)
  model_file = f"{model_name}_x{scale_factor}.pb"

  # Đảm bảo đường dẫn tuyệt đối tới thư mục weights
  base_dir = os.path.dirname(os.path.abspath(__file__))
  model_path = os.path.join(base_dir, weights_dir, model_file)

  if os.path.exists(model_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
      raise ValueError("Không thể đọc file ảnh đầu vào!")

    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    bgr = img[:, :, :3] if has_alpha else img

    sr.readModel(model_path)
    sr.setModel(model_name.lower(), scale_factor)
    upscaled_bgr = sr.upsample(bgr)

    if has_alpha:
      h, w = upscaled_bgr.shape[:2]
      upscaled_alpha = cv2.resize(
          img[:, :, 3], (w, h), interpolation=cv2.INTER_CUBIC
      )
      result = cv2.merge([
          upscaled_bgr[:, :, 0],
          upscaled_bgr[:, :, 1],
          upscaled_bgr[:, :, 2],
          upscaled_alpha,
      ])
    else:
      result = upscaled_bgr

    cv2.imwrite(output_path, result)
  else:
    # Dự phòng bằng cv2.resize chuẩn
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
      raise ValueError("Không thể đọc file ảnh!")

    h, w = img.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, upscaled)


def crop_image(image_path, output_path, crop_box):
    """crop_box: tuple (left, top, right, bottom)"""
    with Image.open(image_path) as img:
        cropped_img = img.crop(crop_box)
        if output_path.lower().endswith(('.jpg', '.jpeg')) and cropped_img.mode in ('RGBA', 'LA', 'P'):
            cropped_img = cropped_img.convert('RGB')
        cropped_img.save(output_path)


# -----------------------------------------------------------------
# 4. TÁCH NỀN (REMOVE BACKGROUND)
# -----------------------------------------------------------------
def remove_background_ai(image_path, output_path):
  try:
    from rembg import new_session, remove

    # Sử dụng session cố định giúp tối ưu tốc độ và ổn định hơn
    session = new_session("bria-rmbg")
    with Image.open(image_path) as inp:
      out = remove(inp, session=session)
      # Chuyển đổi định dạng nếu đầu ra là JPG (vì JPG không hỗ trợ kênh Alpha/nền trong suốt)
      if output_path.lower().endswith((".jpg", ".jpeg")):
        out = out.convert("RGB")
      out.save(output_path)
  except ImportError:
    raise ImportError(
        "Thư viện 'rembg' chưa được cài đặt. Vui lòng chạy lệnh: pip install"
        " rembg"
    )
  except Exception as e:
    raise RuntimeError(
        f"Lỗi khi xóa nền (có thể do chưa tải được model bria-rmbg): {str(e)}"
    )

# -----------------------------------------------------------------
# 5. BỘ LỌC ẢNH (FILTERS)
# -----------------------------------------------------------------
def apply_filter(image_path, output_path, filter_type="Grayscale"):
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        if filter_type == "Grayscale":
            result = img.convert("L").convert("RGB")
        elif filter_type == "Sepia":
            sepia_img = img.convert("L")
            result = Image.merge("RGB", (
                sepia_img.point(lambda p: min(255, int(p * 1.2))),
                sepia_img.point(lambda p: min(255, int(p * 1.0))),
                sepia_img.point(lambda p: min(255, int(p * 0.8)))
            ))
        elif filter_type == "Blur":
            result = img.filter(ImageFilter.BLUR)
        elif filter_type == "Sharpen":
            result = img.filter(ImageFilter.SHARPEN)
        elif filter_type == "Contour":
            result = img.filter(ImageFilter.CONTOUR)
        elif filter_type == "Detail":
            result = img.filter(ImageFilter.DETAIL)
        elif filter_type == "Edge Enhance":
            result = img.filter(ImageFilter.EDGE_ENHANCE)
        else:
            result = img.copy()

        result.save(output_path)


# -----------------------------------------------------------------
# 6. CHÈN TEXT VÀ LOGO / WATERMARK
# -----------------------------------------------------------------
def add_text_watermark(image_path, output_path, text, pos_x=50, pos_y=50, font_size=36, text_color=(255, 255, 255)):
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        draw.text((pos_x, pos_y), text, fill=text_color, font=font)

        combined = Image.alpha_composite(img, txt_layer)
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            combined.convert("RGB").save(output_path)
        else:
            combined.save(output_path)


def add_icon_overlay(image_path, output_path, icon_path, pos_x=50, pos_y=50, scale_percent=100):
    with Image.open(image_path) as base_img, Image.open(icon_path) as icon_img:
        base_img = base_img.convert("RGBA")
        icon_img = icon_img.convert("RGBA")

        if scale_percent != 100:
            new_w = int(icon_img.width * scale_percent / 100)
            new_h = int(icon_img.height * scale_percent / 100)
            icon_img = icon_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        overlay.paste(icon_img, (pos_x, pos_y), mask=icon_img)

        combined = Image.alpha_composite(base_img, overlay)
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            combined.convert("RGB").save(output_path)
        else:
            combined.save(output_path)
def ai_analyze_image(image_np):
    """
    Phân tích ảnh và trả về BỘ THÔNG SỐ GỢI Ý (không trực tiếp biến đổi ảnh).
    """
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image_np

    # 1. Tính toán chỉ số cơ bản
    mean_brightness = np.mean(gray) # Độ sáng trung bình
    std_contrast = np.std(gray)     # Độ tương phản

    # 2. Quy đổi ra thông số tương ứng với các Slider của người dùng
    # Slider Độ sáng: -100 đến 100
    suggested_brightness = 0
    if mean_brightness < 110:
        suggested_brightness = int((125 - mean_brightness) * 0.8)
    elif mean_brightness > 160:
        suggested_brightness = -int((mean_brightness - 145) * 0.7)

    # Slider Tương phản: 0.5 đến 2.0
    suggested_contrast = 1.0
    if std_contrast < 45:
        suggested_contrast = round(1.0 + (50 - std_contrast) / 80.0, 2)
    elif std_contrast > 75:
        suggested_contrast = round(1.0 - (std_contrast - 70) / 120.0, 2)

    # Giới hạn giá trị nằm trong khoảng hợp lệ của Slider
    suggested_brightness = max(-100, min(100, suggested_brightness))
    suggested_contrast = max(0.5, min(2.0, suggested_contrast))

    return {
        "mean_brightness": round(mean_brightness, 1),
        "std_contrast": round(std_contrast, 1),
        "suggested_brightness": suggested_brightness,
        "suggested_contrast": suggested_contrast
    }