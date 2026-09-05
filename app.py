import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Import các hàm xử lý từ file core.py
from core import (
    adjust_image_advanced,
    rotate_or_flip_image,
    resize_standard,
    resize_ai_upscale,
    crop_image,
    remove_background_ai,
)

# Cấu hình trang Web
st.set_page_config(page_title="PDP Photo Editor Web", layout="wide")
st.title("🖼️ PDP Chỉnh Sửa Ảnh Trực Tuyến")

# Khởi tạo thư mục tạm để lưu ảnh
TEMP_DIR = tempfile.gettempdir()
INPUT_PATH = os.path.join(TEMP_DIR, "web_input_temp.png")
OUTPUT_PATH = os.path.join(TEMP_DIR, "web_output_temp.png")


# -------------------------------------------------------------------
# HÀM PHÂN TÍCH ẢNH BẰNG AI (Tự động đề xuất tham số)
# -------------------------------------------------------------------
def analyze_image_ai(image_path):
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, _, _ = cv2.split(img_lab)

    # Quy đổi độ sáng gốc ra %
    raw_brightness = np.mean(l_channel)
    brightness_pct = round((raw_brightness / 255.0) * 100, 1)

    p10 = np.percentile(l_channel, 10) / 2.55
    p90 = np.percentile(l_channel, 90) / 2.55
    std_contrast = round(float(np.std(l_channel)), 1)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)

    # --- ĐƯA VỀ THANG QUY CHIẾU CỦA SLIDER ---
    # Exposure: bước 0.1 (-2.0 đến 2.0)
    target_pct = 58.0
    suggested_exposure = round((target_pct - brightness_pct) / 25.0, 1)
    suggested_exposure = float(np.clip(suggested_exposure, -1.0, 1.5))

    # Shadows (-100 đến 100)
    suggested_shadows = int(np.clip((25 - p10) * 1.2, 0, 50)) if p10 < 25 else 0

    # Highlights (-100 đến 100)
    suggested_highlights = -15 if p90 > 75 else 0

    # Contrast (-100 đến 100)
    suggested_contrast = 15 if std_contrast < 50 else 5

    # Clarity (-100 đến 100) & Dehaze (0 đến 100)
    suggested_clarity = 15
    suggested_dehaze = 10

    # Sharpening (0 đến 100)
    if laplacian_var < 100:
        suggested_sharpening = 40
    elif laplacian_var < 300:
        suggested_sharpening = 25
    else:
        suggested_sharpening = 15

    # Saturation (-100 đến 100)
    suggested_saturation = 10

    return {
        "brightness_pct": brightness_pct,
        "std_contrast": std_contrast,
        "sharpness": laplacian_var,
        "exposure": suggested_exposure,
        "contrast": suggested_contrast,
        "highlights": suggested_highlights,
        "shadows": suggested_shadows,
        "saturation": suggested_saturation,
        "clarity": suggested_clarity,
        "dehaze": suggested_dehaze,
        "sharpening": suggested_sharpening,
    }

# -------------------------------------------------------------------
# KHỞI TẠO SESSION STATE CHO CÁC SLIDER
# -------------------------------------------------------------------
slider_keys = [
    "exposure", "contrast", "highlights", "shadows",
    "saturation", "clarity", "dehaze", "sharpening", "noise_reduction"
]

defaults = {
    "exposure": 0.0, "contrast": 0, "highlights": 0, "shadows": 0,
    "saturation": 0, "clarity": 0, "dehaze": 0, "sharpening": 0, "noise_reduction": 0
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "ai_analysis" not in st.session_state:
    st.session_state["ai_analysis"] = None


# Callback áp dụng thông số AI gợi ý vào Slider
def apply_ai_suggestions():
    if st.session_state["ai_analysis"]:
        ai = st.session_state["ai_analysis"]
        st.session_state["exposure"] = ai["exposure"]
        st.session_state["contrast"] = ai["contrast"]
        st.session_state["sharpening"] = ai["sharpening"]


# Sidebar: Tải ảnh lên
st.sidebar.header("📂 Tải Ảnh Lên")
uploaded_file = st.sidebar.file_uploader("Chọn tệp ảnh", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Lưu ảnh tải lên vào tệp tạm
    image = Image.open(uploaded_file)
    image.save(INPUT_PATH)

    # Chia màn hình làm 2 cột: Ảnh gốc & Ảnh sau khi chỉnh
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(image, use_container_width=True)

    # Các Tab tính năng
    tab1, tab2, tab3, tab4 = st.tabs([
        "✨ Chỉnh Sáng & Chi Tiết",
        "📐 Resize & AI Upscale",
        "✂️ Tách Nền AI",
        "🔄 Xoay & Lật"
    ])

    # --- TAB 1: CHỈNH SÁNG & CHI TIẾT ---
    with tab1:
        st.markdown("### Chỉnh sửa thông số ảnh")

        # KHU VỰC PHÂN TÍCH AI
        with st.expander("🤖 **Phân tích ảnh bằng AI & Gợi ý thông số**", expanded=True):
            col_ai1, col_ai2 = st.columns([1, 2])
            with col_ai1:
                if st.button("🔍 Phân tích ảnh bằng AI"):
                    with st.spinner("AI đang phân tích các chỉ số ảnh..."):
                        st.session_state["ai_analysis"] = analyze_image_ai(INPUT_PATH)

            with col_ai2:
                if st.session_state["ai_analysis"]:
                    ai_res = st.session_state["ai_analysis"]
                    st.write(
                        f"📊 **Chỉ số gốc:** Độ sáng `{ai_res['brightness_pct']}%` | "
                        f"Tương phản `{ai_res['std_contrast']}` | "
                        f"Độ nét `{ai_res['sharpness']}`"
                    )
                    # Hiển thị chính xác từng giá trị tương ứng với Slider
                    st.write(
                        f"💡 **AI đề xuất:** "
                        f"Exp `{ai_res['exposure']:+.1f}` | "
                        f"Contrast `{ai_res['contrast']:+d}` | "
                        f"Shadows `{ai_res['shadows']:+d}` | "
                        f"Clarity `{ai_res['clarity']:+d}` | "
                        f"Sharpening `{ai_res['sharpening']}`"
                    )
                    st.button("👉 Áp dụng thông số AI vào Slider", on_click=apply_ai_suggestions)

        st.markdown("---")

        # KHU VỰC THANH SLIDER (Dùng value=st.session_state[...] để giữ giá trị chuẩn)
        c1, c2 = st.columns(2)
        with c1:
            exposure = st.slider("Độ sáng (Exposure)", -2.0, 2.0, value=float(st.session_state.get("exposure", 0.0)),
                                 step=0.1)
            contrast = st.slider("Độ tương phản (Contrast)", -100, 100, value=int(st.session_state.get("contrast", 0)))
            highlights = st.slider("Vùng sáng (Highlights)", -100, 100,
                                   value=int(st.session_state.get("highlights", 0)))
            shadows = st.slider("Vùng tối (Shadows)", -100, 100, value=int(st.session_state.get("shadows", 0)))
            saturation = st.slider("Độ bão hòa (Saturation)", -100, 100,
                                   value=int(st.session_state.get("saturation", 0)))
        with c2:
            clarity = st.slider("Độ rõ nét (Clarity)", -100, 100, value=int(st.session_state.get("clarity", 0)))
            dehaze = st.slider("Khử mờ (Dehaze)", 0, 100, value=int(st.session_state.get("dehaze", 0)))
            sharpening = st.slider("Sắc nét (Sharpening)", 0, 100, value=int(st.session_state.get("sharpening", 0)))
            noise_reduction = st.slider("Khử nhiễu (Noise Reduction)", 0, 100,
                                        value=int(st.session_state.get("noise_reduction", 0)))

        if st.button("Áp dụng ánh sáng & màu sắc"):
            with st.spinner("Đang xử lý ảnh..."):
                adjust_image_advanced(
                    INPUT_PATH,
                    OUTPUT_PATH,
                    exposure=exposure,
                    contrast=contrast,
                    highlights=highlights,
                    shadows=shadows,
                    saturation=saturation,
                    clarity=clarity,
                    dehaze=dehaze,
                    sharpening=sharpening,
                    noise_reduction=noise_reduction
                )
                st.session_state["processed_img"] = OUTPUT_PATH

    # --- TAB 2: RESIZE & AI UPSCALE ---
    with tab2:
        st.markdown("### Phóng to / Thay đổi kích thước")
        resize_type = st.radio("Chọn phương pháp:", ["Resize Chuẩn", "AI Super Resolution (Upscale)"])

        if resize_type == "Resize Chuẩn":
            w = st.number_input("Chiều rộng (px)", value=image.width)
            h = st.number_input("Chiều cao (px)", value=image.height)
            if st.button("Thực hiện Resize"):
                resize_standard(INPUT_PATH, OUTPUT_PATH, width=w, height=h)
                st.session_state["processed_img"] = OUTPUT_PATH
        else:
            scale = st.selectbox("Tỉ lệ phóng to:", [2, 4])
            if st.button("Phóng to bằng AI"):
                with st.spinner("Đang nâng cấp chất lượng bằng AI..."):
                    resize_ai_upscale(INPUT_PATH, OUTPUT_PATH, scale_factor=scale)
                    st.session_state["processed_img"] = OUTPUT_PATH

    # --- TAB 3: TÁCH NỀN AI ---
    with tab3:
        st.markdown("### Tách nền tự động bằng AI (rembg)")
        if st.button("Bắt đầu Tách Nền"):
            with st.spinner("Đang tách nền..."):
                try:
                    remove_background_ai(INPUT_PATH, OUTPUT_PATH)
                    st.session_state["processed_img"] = OUTPUT_PATH
                    st.success("Tách nền thành công!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # --- TAB 4: XOAY & LẬT ---
    with tab4:
        st.markdown("### Xoay và lật hướng ảnh")
        action = st.selectbox("Chọn thao tác:", [
            ("Xoay phải 90°", "rotate_right"),
            ("Xoay trái 90°", "rotate_left"),
            ("Lật ngang", "flip_horizontal"),
            ("Lật dọc", "flip_vertical")
        ], format_func=lambda x: x[0])

        if st.button("Thực hiện"):
            rotate_or_flip_image(INPUT_PATH, OUTPUT_PATH, action=action[1])
            st.session_state["processed_img"] = OUTPUT_PATH

    # Hiển thị kết quả & Nút tải về
    with col2:
        st.subheader("Kết Quả")
        if "processed_img" in st.session_state and os.path.exists(st.session_state["processed_img"]):
            result_img = Image.open(st.session_state["processed_img"])
            st.image(result_img, use_container_width=True)

            # Nút Tải ảnh về máy
            with open(st.session_state["processed_img"], "rb") as file:
                st.download_button(
                    label="💾 Tải Ảnh Kết Quả Về Máy",
                    data=file,
                    file_name="edited_image.png",
                    mime="image/png"
                )
else:
    st.info("Vui lòng tải một tệp ảnh lên từ thanh bên (Sidebar) để bắt đầu chỉnh sửa.")