import cv2
import numpy as np
from PIL import Image
from rembg import remove


def resize_image(image_np, width=None, height=None, keep_aspect_ratio=True):
    """
    Thay đổi kích thước ảnh theo chiều rộng/chiều cao mong muốn.
    """
    h, w = image_np.shape[:2]

    if width is None and height is None:
        return image_np

    if keep_aspect_ratio:
        if width is not None and height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        elif height is not None and width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = min(width / float(w), height / float(h))
            dim = (int(w * r), int(h * r))
    else:
        dim = (width if width else w, height if height else h)

    resized = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)
    return resized


def remove_background(image_pil):
    """
    Tách nền ảnh sử dụng thư viện rembg.
    """
    return remove(image_pil)


def manual_adjust(image_np, brightness=0, contrast=1.0):
    """
    Chỉnh sửa độ sáng và độ tương phản thủ công.
    brightness: -100 đến 100
    contrast: 0.5 đến 2.0
    """
    adjusted = cv2.convertScaleAbs(image_np, alpha=contrast, beta=brightness)
    return adjusted


def auto_white_balance(image_np):
    """
    Tự động cân bằng màu sắc (White Balance) dựa trên thuật toán Gray World Assumption.
    """
    result = image_np.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    avg_all = (avg_b + avg_g + avg_r) / 3.0

    if avg_b > 0 and avg_g > 0 and avg_r > 0:
        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_all / avg_b), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_all / avg_g), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_all / avg_r), 0, 255)

    return result.astype(np.uint8)


def analyze_and_auto_enhance(image_np):
    """
    AI Phân tích các thông số ảnh (Độ sáng, Tương phản, Độ nét)
    và tự động đề xuất + áp dụng bộ thông số tối ưu.
    """
    # Chuyển sang ảnh xám để tính toán các chỉ số độ sáng/tương phản
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    else:
        gray = image_np

    # 1. Phân tích chỉ số
    mean_brightness = np.mean(gray)  # Độ sáng trung bình (0 - 255)
    std_contrast = np.std(gray)  # Độ tương phản (Độ lệch chuẩn)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Độ sắc nét

    # 2. Thuật toán AI tính toán bộ tham số bù trừ
    alpha = 1.0  # Hệ số tương phản
    beta = 0  # Hệ số độ sáng

    # Điều chỉnh độ sáng (Mục tiêu đưa độ sáng trung bình về ngưỡng 125-135)
    if mean_brightness < 100:
        beta = int((125 - mean_brightness) * 0.75)  # Ảnh tối -> Tăng sáng
    elif mean_brightness > 165:
        beta = -int((mean_brightness - 145) * 0.6)  # Ảnh quá sáng -> Giảm sáng

    # Điều chỉnh tương phản (Mục tiêu đưa độ lệch chuẩn về khoảng 50-60)
    if std_contrast < 45:
        alpha = round(1.0 + (50 - std_contrast) / 80.0, 2)
    elif std_contrast > 75:
        alpha = round(1.0 - (std_contrast - 70) / 120.0, 2)

    # 3. Áp dụng thông số tự động
    enhanced = cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta)

    # 4. Áp dụng Cân bằng trắng tự động
    if len(enhanced.shape) == 3:
        final_result = auto_white_balance(enhanced)
    else:
        final_result = enhanced

    # Báo cáo chi tiết để trả về cho giao diện
    analysis_report = {
        "mean_brightness": round(mean_brightness, 1),
        "std_contrast": round(std_contrast, 1),
        "sharpness": round(laplacian_var, 1),
        "applied_alpha": alpha,
        "applied_beta": beta
    }

    return final_result, analysis_report