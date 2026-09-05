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
    """
    Phân tích ảnh màu chuẩn RGB và đề xuất bộ thông số làm sáng, nổi khối, sắc nét.
    """
    # 1. Đọc ảnh bằng PIL để giữ chuẩn màu RGB (tránh lỗi BGR/grayscale của OpenCV)
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    # 2. Chuyển sang không gian màu LAB để đo độ sáng chuẩn (kênh L) mà không làm lệch màu
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    # Tính toán các chỉ số trên kênh độ sáng L
    mean_brightness = np.mean(l_channel)  # Thang độ sáng 0 - 255
    p10 = np.percentile(l_channel, 10)     # Vùng tối (Shadows)
    p90 = np.percentile(l_channel, 90)     # Vùng sáng (Highlights)
    std_contrast = np.std(l_channel)      # Độ tương phản

    # Đo độ nét bằng OpenCV Laplacian
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # -------------------------------------------------------------
    # TÍNH TOÁN BỘ THÔNG SỐ LÀM ĐẸP (SÁNG - NÉT - RỰC RỠ)
    # -------------------------------------------------------------
    # A. Exposure (Mục tiêu đưa độ sáng trung bình L lên khoảng 135 - 145)
    target_l = 140.0
    suggested_exposure = (target_l - mean_brightness) / 80.0
    suggested_exposure = float(np.clip(suggested_exposure, -0.8, 1.2))

    # B. Shadows (Kéo sáng vùng tối bị chìm)
    suggested_shadows = 0
    if p10 < 60:
        suggested_shadows = int((60 - p10) * 1.1)
    suggested_shadows = int(np.clip(suggested_shadows, 0, 50))

    # C. Highlights (Hạ nhẹ vùng sáng để tránh gắt/cháy)
    suggested_highlights = -15 if p90 > 190 else 0

    # D. Contrast (Giữ độ tương phản vừa đủ)
    suggested_contrast = 15 if std_contrast < 50 else 5

    # E. Clarity & Dehaze (Giúp ảnh trong trẻo, xóa mù)
    suggested_clarity = 20
    suggested_dehaze = 15

    # F. Sharpening (Tăng độ nét dựa trên độ nhòe gốc)
    if laplacian_var < 100:
        suggested_sharpening = 45
    elif laplacian_var < 300:
        suggested_sharpening = 30
    else:
        suggested_sharpening = 15

    # G. Saturation (Đảm bảo luôn giữ màu sắc tươi tắn)
    suggested_saturation = 10

    return {
        "mean_brightness": round(mean_brightness, 1),
        "std_contrast": round(std_contrast, 1),
        "sharpness": round(laplacian_var, 1),
        "exposure": round(suggested_exposure, 1),
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
                        f"📊 **Chỉ số gốc:** Độ sáng `{ai_res['mean_brightness']}/255` | "
                        f"Tương phản `{ai_res['std_contrast']}` | "
                        f"Độ nét `{ai_res['sharpness']}`"
                    )
                    st.write(
                        f"💡 **AI gợi ý:** Exposure `{ai_res['exposure']:+.1f}` | "
                        f"Contrast `{ai_res['contrast']:+d}` | "
                        f"Sharpening `{ai_res['sharpening']}`"
                    )
                    st.button("👉 Áp dụng thông số AI vào Slider", on_click=apply_ai_suggestions)

        st.markdown("---")

        # KHU VỰC THANH SLIDER
        c1, c2 = st.columns(2)
        with c1:
            exposure = st.slider("Độ sáng (Exposure)", -2.0, 2.0, key="exposure", step=0.1)
            contrast = st.slider("Độ tương phản (Contrast)", -100, 100, key="contrast")
            highlights = st.slider("Vùng sáng (Highlights)", -100, 100, key="highlights")
            shadows = st.slider("Vùng tối (Shadows)", -100, 100, key="shadows")
            saturation = st.slider("Độ bão hòa (Saturation)", -100, 100, key="saturation")
        with c2:
            clarity = st.slider("Độ rõ nét (Clarity)", -100, 100, key="clarity")
            dehaze = st.slider("Khử mờ (Dehaze)", 0, 100, key="dehaze")
            sharpening = st.slider("Sắc nét (Sharpening)", 0, 100, key="sharpening")
            noise_reduction = st.slider("Khử nhiễu (Noise Reduction)", 0, 100, key="noise_reduction")

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