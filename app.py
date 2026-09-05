import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Import các hàm từ core.py
from core import (
    resize_image,
    remove_background,
    manual_adjust,
    analyze_and_auto_enhance
)

st.set_page_config(
    page_title="AI Photo Editor Pro",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Photo Editor - Chỉnh Sửa & Tối Ưu Ảnh Thông Minh")
st.write("Tải ảnh của bạn lên để bắt đầu chỉnh sửa thủ công hoặc sử dụng AI tự động phân tích!")

# Thanh tải file
uploaded_file = st.file_uploader("Chọn hình ảnh (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh và chuyển đổi định dạng
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    orig_h, orig_w = image_np.shape[:2]

    # Hiển thị khu vực xem trước ảnh gốc
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🖼️ Ảnh Gốc")
        st.image(image_pil, use_container_width=True)
        st.caption(f"Kích thước gốc: {orig_w} x {orig_h} pixels")

    # Bảng chọn Chế độ Xử lý
    st.sidebar.header("⚙️ Chế độ xử lý")
    mode = st.sidebar.radio(
        "Lựa chọn phương thức chỉnh sửa:",
        ["🤖 AI Tự động phân tích & Tối ưu", "🛠️ Chỉnh sửa Thủ công (Manual)"]
    )

    # Khởi tạo biến lưu kết quả ảnh
    processed_np = image_np.copy()
    analysis_report = None

    # ==========================================
    # CHẾ ĐỘ 1: AI TỰ ĐỘNG PHÂN TÍCH
    # ==========================================
    if mode == "🤖 AI Tự động phân tích & Tối ưu":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 Cấu hình AI")
        enable_autofix = st.sidebar.checkbox("Kích hoạt AI Auto-Enhance", value=True)

        if enable_autofix:
            processed_np, analysis_report = analyze_and_auto_enhance(image_np)

    # ==========================================
    # CHẾ ĐỘ 2: CHỈNH TAY THỦ CÔNG
    # ==========================================
    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Bảng điều khiển")
        brightness = st.sidebar.slider("Độ sáng (Brightness)", -100, 100, 0, step=5)
        contrast = st.sidebar.slider("Độ tương phản (Contrast)", 0.5, 2.0, 1.0, step=0.05)

        # Áp dụng thông số thủ công
        processed_np = manual_adjust(image_np, brightness=brightness, contrast=contrast)

    # ==========================================
    # CÁC TÍNH NĂNG BỔ SUNG (RESIZE & TÁCH NỀN)
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Thay đổi kích thước (Resize)")
    use_resize = st.sidebar.checkbox("Kích hoạt Resize")

    if use_resize:
        keep_ratio = st.sidebar.checkbox("Giữ tỷ lệ khung hình", value=True)
        new_width = st.sidebar.number_input("Chiều rộng (Width px)", min_value=1, value=orig_w)
        new_height = st.sidebar.number_input("Chiều cao (Height px)", min_value=1, value=orig_h)

        target_w = new_width if new_width > 0 else None
        target_h = new_height if new_height > 0 else None
        processed_np = resize_image(processed_np, width=target_w, height=target_h, keep_aspect_ratio=keep_ratio)

    st.sidebar.markdown("---")
    st.sidebar.subheader("✂️ Tách nền (Remove Background)")
    btn_remove_bg = st.sidebar.button("Thực hiện tách nền bằng AI")

    is_bg_removed = False
    if btn_remove_bg:
        with st.spinner("Đang tách nền ảnh bằng rembg..."):
            proc_pil = Image.fromarray(processed_np)
            processed_pil_nobg = remove_background(proc_pil)
            is_bg_removed = True

    # Convert ngược về PIL Image để hiển thị & xuất file
    if not is_bg_removed:
        final_output_pil = Image.fromarray(processed_np)
    else:
        final_output_pil = processed_pil_nobg

    # Hiển thị kết quả xử lý
    with col_right:
        st.subheader("✨ Ảnh Sau Khi Xử Lý")
        st.image(final_output_pil, use_container_width=True)
        res_w, res_h = final_output_pil.size
        st.caption(f"Kích thước kết quả: {res_w} x {res_h} pixels")

    # ==========================================
    # HIỂN THỊ BÁO CÁO PHÂN TÍCH CỦA AI
    # ==========================================
    if mode == "🤖 AI Tự động phân tích & Tối ưu" and analysis_report:
        st.markdown("---")
        st.info("📊 **Báo cáo Phân tích Thông số từ AI:**")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Độ sáng ban đầu", f"{analysis_report['mean_brightness']}/255")
        m2.metric("Độ tương phản gốc", f"{analysis_report['std_contrast']}")
        m3.metric("Độ sắc nét (Sharpness)", f"{analysis_report['sharpness']}")
        m4.metric("Tăng/Giảm sáng (Beta)", f"{analysis_report['applied_beta']}")
        m5.metric("Hệ số tương phản (Alpha)", f"{analysis_report['applied_alpha']}")

    # ==========================================
    # NÚT TẢI ẢNH VỀ MÁY
    # ==========================================
    st.markdown("---")
    buf = io.BytesIO()
    if is_bg_removed:
        final_output_pil.save(buf, format="PNG")
        mime_type = "image/png"
        file_ext = "png"
    else:
        final_output_pil.save(buf, format="JPEG", quality=95)
        mime_type = "image/jpeg"
        file_ext = "jpg"

    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Tải ảnh kết quả về máy",
        data=byte_im,
        file_name=f"processed_image.{file_ext}",
        mime=mime_type
    )