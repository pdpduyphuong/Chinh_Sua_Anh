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
    image_path,
    output_path,
    exposure=0.0,
    contrast=0.0,
    highlights=0.0,
    shadows=0.0,
    whites=0.0,
    blacks=0.0,
    tint=0.0,
    vibrance=0.0,
    saturation=0.0,
    clarity=0.0,
    dehaze=0.0,
    sharpening=0.0,
    sharpen_radius=1.0,
    noise_reduction=0.0
):
    """
    Xử lý các thông số ánh sáng, màu sắc và độ rõ nét chuyên nghiệp
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Không thể đọc file ảnh từ đường dẫn đã chọn!")

    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3].astype(np.float32) / 255.0
        alpha = img[:, :, 3]
    else:
        bgr = img.astype(np.float32) / 255.0

    # 1. Khử nhiễu (Noise Reduction)
    if noise_reduction > 0:
        h_val = (noise_reduction / 100.0) * 10.0
        bgr_8u = (bgr * 255.0).astype(np.uint8)
        bgr_8u = cv2.fastNlMeansDenoisingColored(bgr_8u, None, h_val, h_val, 7, 21)
        bgr = bgr_8u.astype(np.float32) / 255.0

    # 2. Tint (Sắc thái)
    if tint != 0:
        shift = (tint / 100.0) * 0.1
        bgr[:, :, 1] = np.clip(bgr[:, :, 1] - shift, 0, 1)

    # 3. Tính toán Luminance & Mask
    luminance = 0.299 * bgr[:, :, 2] + 0.587 * bgr[:, :, 1] + 0.114 * bgr[:, :, 0]

    highlight_mask = np.clip((luminance - 0.5) * 2.0, 0, 1)[:, :, np.newaxis]
    shadow_mask = np.clip((0.5 - luminance) * 2.0, 0, 1)[:, :, np.newaxis]
    white_mask = np.clip((luminance - 0.75) * 4.0, 0, 1)[:, :, np.newaxis]
    black_mask = np.clip((0.25 - luminance) * 4.0, 0, 1)[:, :, np.newaxis]

    # 4. Highlights & Shadows
    if highlights != 0:
        factor = 1.0 + (highlights / 100.0) * 0.5
        bgr = bgr * (1 - highlight_mask) + (bgr * factor) * highlight_mask

    if shadows != 0:
        factor = 1.0 + (shadows / 100.0) * 0.6
        bgr = bgr * (1 - shadow_mask) + np.power(np.clip(bgr, 1e-6, 1.0), 1.0 / factor) * shadow_mask

    # 5. Whites & Blacks
    if whites != 0:
        factor = 1.0 + (whites / 100.0) * 0.4
        bgr = bgr * (1 - white_mask) + (bgr * factor) * white_mask

    if blacks != 0:
        offset = (blacks / 100.0) * 0.15
        bgr = bgr * (1 - black_mask) + np.clip(bgr + offset, 0, 1) * black_mask

    # 6. Exposure (Độ sáng)
    if exposure != 0:
        bgr = bgr * (2.0 ** exposure)

    # 7. Contrast (Độ tương phản)
    if contrast != 0:
        factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
        bgr = 0.5 + factor * (bgr - 0.5)

    # 8. Clarity (Midtone Contrast)
    if clarity != 0:
        blur = cv2.GaussianBlur(bgr, (0, 0), 3.0)
        c_factor = (clarity / 100.0) * 0.5
        bgr = bgr + (bgr - blur) * c_factor

    # 9. Dehaze (Khử sương mờ)
    if dehaze > 0:
        d_factor = 1.0 + (dehaze / 100.0) * 0.3
        bgr = (bgr - 0.1 * (dehaze / 100.0)) * d_factor

    # 10. Sharpening (Unsharp Masking)
    if sharpening > 0:
        radius = max(0.5, float(sharpen_radius))
        blurred = cv2.GaussianBlur(bgr, (0, 0), radius)
        s_amount = (sharpening / 100.0) * 1.5
        bgr = cv2.addWeighted(bgr, 1.0 + s_amount, blurred, -s_amount, 0)

    bgr = np.clip(bgr, 0, 1) * 255.0
    result = bgr.astype(np.uint8)

    if has_alpha:
        result = cv2.merge([result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha])

    # Chuyển sang PIL để xử lý Saturation & Vibrance
    pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB if not has_alpha else cv2.COLOR_BGRA2RGBA))

    if saturation != 0:
        enhancer_sat = ImageEnhance.Color(pil_img)
        pil_img = enhancer_sat.enhance(1.0 + (saturation / 100.0))

    if vibrance != 0:
        enhancer_vib = ImageEnhance.Color(pil_img)
        pil_img = enhancer_vib.enhance(1.0 + (vibrance / 100.0))

    pil_img.save(output_path)


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