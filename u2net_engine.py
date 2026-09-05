# u2net_engine.py
import os
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image

# URL Model U2Net Portable (~4.7MB) - Siêu nhẹ, chạy cực nhanh trên Streamlit Cloud
U2NETP_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
# URL Model U2Net Standard (~176MB) - Độ chính xác cao hơn
U2NET_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def get_model_path(model_name: str = "u2netp") -> str:
    """Tải file ONNX model về thư mục local nếu chưa tồn tại."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if model_name == "u2net":
        model_path = os.path.join(MODEL_DIR, "u2net.onnx")
        url = U2NET_MODEL_URL
    else:
        model_path = os.path.join(MODEL_DIR, "u2netp.onnx")
        url = U2NETP_MODEL_URL

    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model from {url}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download completed!")

    return model_path


class U2NetInference:
    """Lớp thực thi suy luận tách nền U2Net bằng ONNX Runtime thuần."""

    def __init__(self, model_name: str = "u2netp"):
        model_path = get_model_path(model_name)
        # Khởi tạo ONNX Runtime Session (CPU Execution Provider)
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Tiền xử lý ảnh theo chuẩn đầu vào của U2Net (320x320, Normalize)."""
        img_resized = img.resize((320, 320), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized, dtype=np.float32)

        # Chuẩn hóa giá trị Pixel về [0, 1]
        img_np /= 255.0

        # Standardize theo ImageNet Mean/Std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        # Chuyển layout từ HWC sang CHW và thêm Batch dimension (1, 3, 320, 320)
        img_np = img_np.transpose((2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)
        return img_np.astype(np.float32)

    def postprocess(self, mask_np: np.ndarray, original_size: tuple) -> Image.Image:
        """Hậu xử lý Alpha Mask từ Output thu được của U2Net."""
        # Squeeze batch & channel dimensions
        mask = mask_np[0, 0]

        # Min-Max Normalization mask về dải [0, 255]
        ma = np.max(mask)
        mi = np.min(mask)
        mask = (mask - mi) / (ma - mi + 1e-08) * 255.0
        mask = mask.astype(np.uint8)

        # Resize Alpha Mask về kích thước ảnh ban đầu
        mask_img = Image.fromarray(mask, mode="L")
        mask_img = mask_img.resize(original_size, Image.Resampling.LANCZOS)
        return mask_img

    def remove_background(self, input_image: Image.Image) -> Image.Image:
        """Thực hiện tách nền và trả về ảnh RGBA trong suốt."""
        orig_size = input_image.size
        rgb_img = input_image.convert("RGB")

        # Preprocess
        input_tensor = self.preprocess(rgb_img)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Lấy output mask đầu tiên d0 của U2Net
        mask_tensor = outputs[0]

        # Postprocess Mask
        alpha_mask = self.postprocess(mask_tensor, orig_size)

        # Ghép Alpha Mask vào ảnh RGB ban đầu
        rgba_img = rgb_img.convert("RGBA")
        rgba_img.putalpha(alpha_mask)
        return rgba_img