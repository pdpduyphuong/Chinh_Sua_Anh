# core.py
import logging
from pathlib import Path
from typing import Tuple, Union
from PIL import Image, ImageOps

# Cấu hình logging chuyên nghiệp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s: %(message)s"
)
logger = logging.getLogger(__name__)


def rotate_or_flip_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    action: str
) -> bool:
    """Xoay hoặc lật ảnh theo yêu cầu và ghi đè/lưu file mới.

    Args:
        input_path: Đường dẫn file ảnh đầu vào.
        output_path: Đường dẫn lưu file ảnh kết quả.
        action: Các thao tác ('rotate_right', 'rotate_left', 'flip_horizontal', 'flip_vertical').
    """
    try:
        in_p = Path(input_path)
        out_p = Path(output_path)

        if not in_p.exists():
            logger.error(f"File không tồn tại: {in_p}")
            return False

        with Image.open(in_p) as img:
            # Tự động xoay ảnh theo EXIF orientation nếu có
            img = ImageOps.exif_transpose(img)

            if action == "rotate_right":
                res = img.rotate(-90, expand=True)
            elif action == "rotate_left":
                res = img.rotate(90, expand=True)
            elif action == "flip_horizontal":
                res = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif action == "flip_vertical":
                res = img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                logger.warning(f"Thao tác không hợp lệ: {action}")
                return False

            # Giữ nguyên mode ảnh RGB khi lưu JPEG
            if res.mode in ("RGBA", "P") and out_p.suffix.lower() in [".jpg", ".jpeg"]:
                res = res.convert("RGB")

            res.save(out_p, quality=95)
            logger.info(f"Thực hiện thành công {action} -> {out_p}")
            return True

    except Exception as e:
        logger.exception(f"Lỗi khi xử lý xoay/lật ảnh: {str(e)}")
        return False


def crop_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    crop_box: Tuple[int, int, int, int]
) -> bool:
    """Cắt ảnh theo Bounding Box (left, top, right, bottom).

    Args:
        input_path: Đường dẫn file ảnh nguồn.
        output_path: Đường dẫn file kết quả.
        crop_box: Tọa độ cắt dạng (left, top, right, bottom).
    """
    try:
        in_p = Path(input_path)
        out_p = Path(output_path)

        if not in_p.exists():
            logger.error(f"File không tồn tại: {in_p}")
            return False

        with Image.open(in_p) as img:
            img = ImageOps.exif_transpose(img)
            orig_w, orig_h = img.size

            left, top, right, bottom = crop_box

            # Standardize & Clamp tọa độ nằm trong phạm vi ảnh
            real_left = max(0, min(left, right))
            real_top = max(0, min(top, bottom))
            real_right = min(orig_w, max(left, right))
            real_bottom = min(orig_h, max(top, bottom))

            width = real_right - real_left
            height = real_bottom - real_top

            if width < 2 or height < 2:
                logger.error(f"Kích thước vùng cắt quá nhỏ: {width}x{height}")
                return False

            cropped_img = img.crop((real_left, real_top, real_right, real_bottom))

            if cropped_img.mode in ("RGBA", "P") and out_p.suffix.lower() in [".jpg", ".jpeg"]:
                cropped_img = cropped_img.convert("RGB")

            cropped_img.save(out_p, quality=95)
            logger.info(f"Đã cắt ảnh thành công: ({real_left}, {real_top}, {real_right}, {real_bottom}) -> {out_p}")
            return True

    except Exception as e:
        logger.exception(f"Lỗi khi cắt ảnh: {str(e)}")
        return False
