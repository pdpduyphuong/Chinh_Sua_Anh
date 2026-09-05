# core.py
import logging
from pathlib import Path
from typing import Tuple, Union, Optional
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# Cấu hình logging
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
    """Xoay hoặc lật ảnh theo chiều chỉ định."""
    try:
        in_p = Path(input_path)
        out_p = Path(output_path)

        if not in_p.exists():
            return False

        with Image.open(in_p) as img:
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
                return False

            if res.mode in ("RGBA", "P") and out_p.suffix.lower() in [".jpg", ".jpeg"]:
                res = res.convert("RGB")

            res.save(out_p, quality=95)
            return True
    except Exception as e:
        logger.exception(f"Lỗi rotate_or_flip_image: {e}")
        return False


def crop_image(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        crop_box: Tuple[int, int, int, int]
) -> bool:
    """Cắt ảnh theo bounding box (left, top, right, bottom)."""
    try:
        in_p = Path(input_path)
        out_p = Path(output_path)

        if not in_p.exists():
            return False

        with Image.open(in_p) as img:
            img = ImageOps.exif_transpose(img)
            orig_w, orig_h = img.size

            left, top, right, bottom = crop_box

            real_left = max(0, min(left, right))
            real_top = max(0, min(top, bottom))
            real_right = min(orig_w, max(left, right))
            real_bottom = min(orig_h, max(top, bottom))

            width = real_right - real_left
            height = real_bottom - real_top

            if width < 2 or height < 2:
                return False

            cropped_img = img.crop((real_left, real_top, real_right, real_bottom))

            if cropped_img.mode in ("RGBA", "P") and out_p.suffix.lower() in [".jpg", ".jpeg"]:
                cropped_img = cropped_img.convert("RGB")

            cropped_img.save(out_p, quality=95)
            return True
    except Exception as e:
        logger.exception(f"Lỗi crop_image: {e}")
        return False


def resize_image(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        new_width: int,
        new_height: int,
        keep_aspect_ratio: bool = True
) -> bool:
    """Thay đổi kích thước ảnh (Resize)."""
    try:
        in_p = Path(input_path)
        out_p = Path(output_path)

        with Image.open(in_p) as img:
            img = ImageOps.exif_transpose(img)
            if keep_aspect_ratio:
                img.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                res = img
            else:
                res = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if res.mode in ("RGBA", "P") and out_p.suffix.lower() in [".jpg", ".jpeg"]:
                res = res.convert("RGB")

            res.save(out_p, quality=95)
            return True
    except Exception as e:
        logger.exception(f"Lỗi resize_image: {e}")
        return False


def adjust_color_and_filters(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        brightness: float = 1.0,
        contrast: float = 1.0,
        sharpness: float = 1.0,
        filter_type: str = "Gốc"
) -> bool:
    """Điều chỉnh độ sáng, độ phản quang, độ sắc nét và áp dụng bộ lọc ảnh."""
    try:
        in_p = Path(input_path)
        out_p = Path(output_path)

        with Image.open(in_p) as img:
            img = ImageOps.exif_transpose(img)
            res = img.copy()

            if brightness != 1.0:
                res = ImageEnhance.Brightness(res).enhance(brightness)
            if contrast != 1.0:
                res = ImageEnhance.Contrast(res).enhance(contrast)
            if sharpness != 1.0:
                res = ImageEnhance.Sharpness(res).enhance(sharpness)

            if filter_type == "Trắng Đen (Grayscale)":
                res = ImageOps.grayscale(res)
            elif filter_type == "Làm Mờ (Blur)":
                res = res.filter(ImageFilter.BLUR)
            elif filter_type == "Sắc Nhét (Sharpen)":
                res = res.filter(ImageFilter.SHARPEN)
            elif filter_type == "Nổi Bật Cạnh (Find Edges)":
                res = res.filter(ImageFilter.FIND_EDGES)

            if res.mode in ("RGBA", "P") and out_p.suffix.lower() in [".jpg", ".jpeg"]:
                res = res.convert("RGB")

            res.save(out_p, quality=95)
            return True
    except Exception as e:
        logger.exception(f"Lỗi adjust_color_and_filters: {e}")
        return False


def convert_image_format(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str
) -> bool:
    """Chuyển đổi định dạng file ảnh (PNG, JPG, WEBP, BMP, TIFF)."""
    try:
        in_p = Path(input_path)
        out_p = Path(output_path)

        with Image.open(in_p) as img:
            img = ImageOps.exif_transpose(img)
            fmt = target_format.upper()

            if fmt == "JPG":
                fmt = "JPEG"

            if fmt == "JPEG" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img.save(out_p, format=fmt, quality=95)
            return True
    except Exception as e:
        logger.exception(f"Lỗi convert_image_format: {e}")
        return False