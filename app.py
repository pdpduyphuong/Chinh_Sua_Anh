import os
import tempfile
import streamlit as st
from PIL import Image

# Import các hàm xử lý từ file core.py sẵn có của bạn
from core import (
    adjust_image_advanced,
    rotate_or_flip_image,
    resize_standard,
    resize_ai_upscale,
    crop_image,
    remove_background_ai,
)
# Import hàm phân tích và chỉnh ảnh từ core.py
from core import (
    manual_adjust,
    ai_analyze_image,
    resize_image,
    remove_background
)
# Cấu hình trang Web
st.set_page_config(page_title="PDP Photo Editor Web", layout="wide")
st.title("🖼️ PDP Chỉnh Sửa Ảnh Trực Tuyến")

# Khởi tạo thư mục tạm để lưu ảnh
TEMP_DIR = tempfile.gettempdir()
INPUT_PATH = os.path.join(TEMP_DIR, "web_input_temp.png")
OUTPUT_PATH = os.path.join(TEMP_DIR, "web_output_temp.png")

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

    # Các Tab tính năng tương tự bản Desktop
    tab1, tab2, tab3, tab4 = st.tabs([
        "✨ Chỉnh Sáng & Chi Tiết",
        "📐 Resize & AI Upscale",
        "✂️ Tách Nền AI",
        "🔄 Xoay & Lật"
    ])

    # --- TAB 1: CHỈNH SÁNG & CHI TIẾT ---
    with tab1:
        st.markdown("### Chỉnh sửa thông số ảnh")
        c1, c2 = st.columns(2)
        with c1:
            exposure = st.slider("Độ sáng (Exposure)", -2.0, 2.0, 0.0, 0.1)
            contrast = st.slider("Độ tương phản (Contrast)", -100, 100, 0)
            highlights = st.slider("Vùng sáng (Highlights)", -100, 100, 0)
            shadows = st.slider("Vùng tối (Shadows)", -100, 100, 0)
            saturation = st.slider("Độ bão hòa (Saturation)", -100, 100, 0)
        with c2:
            clarity = st.slider("Độ rõ nét (Clarity)", -100, 100, 0)
            dehaze = st.slider("Khử mờ (Dehaze)", 0, 100, 0)
            sharpening = st.slider("Sắc nét (Sharpening)", 0, 100, 0)
            noise_reduction = st.slider("Khử nhiễu (Noise Reduction)", 0, 100, 0)

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
st.set_page_config(page_title="AI Photo Editor Pro", layout="wide")
st.title("🖼️ AI Photo Editor - Chỉnh Sửa & Tối Ưu Ảnh")

# 1. Khởi tạo Session State để lưu trữ giá trị của Slider
if "slider_brightness" not in st.session_state:
    st.session_state.slider_brightness = 0

if "slider_contrast" not in st.session_state:
    st.session_state.slider_contrast = 1.0

if "ai_analysis_result" not in st.session_state:
    st.session_state.ai_analysis_result = None


# Hàm callback khi bấm "Áp dụng thông số AI"
def apply_ai_params():
    if st.session_state.ai_analysis_result:
        st.session_state.slider_brightness = st.session_state.ai_analysis_result["suggested_brightness"]
        st.session_state.slider_contrast = st.session_state.ai_analysis_result["suggested_contrast"]
        st.success("✅ Đã áp dụng thông số AI gợi ý vào thanh điều khiển!")


# Upload File
uploaded_file = st.file_uploader("Tải ảnh lên để bắt đầu:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🖼️ Ảnh Gốc")
        st.image(image_pil, use_container_width=True)

    # BẢNG ĐIỀU KHIỂN BÊN THANH SIDEBAR (HOẶC TAB CHỈNH SÁNG)
    st.sidebar.header("🎛️ Chỉnh Sáng & Chi Tiết")

    # ==========================================
    # KHU VỰC PHÂN TÍCH AI
    # ==========================================
    with st.sidebar.expander("🤖 **Phân tích ảnh bằng AI**", expanded=True):
        if st.button("🔍 Bắt đầu phân tích AI"):
            with st.spinner("AI đang đọc thông số ảnh..."):
                st.session_state.ai_analysis_result = ai_analyze_image(image_np)

        # Hiển thị kết quả sau khi phân tích
        if st.session_state.ai_analysis_result:
            res = st.session_state.ai_analysis_result
            st.markdown("---")
            st.write(f"📊 **Chỉ số gốc:** Độ sáng `{res['mean_brightness']}`, Tương phản `{res['std_contrast']}`")
            st.write(
                f"💡 **AI gợi ý:** Độ sáng `{res['suggested_brightness']:+d}`, Tương phản `{res['suggested_contrast']}`")

            # Nút bấm chuyển thông số AI vào Slider
            st.button("👉 Áp dụng thông số AI", on_click=apply_ai_params)

    st.sidebar.markdown("---")

    # ==========================================
    # THANH SLIDER ĐIỀU CHỈNH THỦ CÔNG / ÁP DỤNG TỪ AI
    # ==========================================
    brightness = st.sidebar.slider(
        "Độ sáng (Brightness)",
        -100, 100,
        key="slider_brightness"
    )

    contrast = st.sidebar.slider(
        "Độ tương phản (Contrast)",
        0.5, 2.0,
        key="slider_contrast",
        step=0.05
    )

    # Thực hiện chỉnh sửa ảnh theo giá trị hiện tại của Slider
    processed_np = manual_adjust(image_np, brightness=brightness, contrast=contrast)

    # Hiển thị ảnh kết quả
    with col_right:
        st.subheader("✨ Ảnh Kết Quả")
        st.image(processed_np, use_container_width=True)
