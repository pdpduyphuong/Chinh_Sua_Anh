import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
# Import đầy đủ các hàm xử lý từ core.py
from core import (
    adjust_image_advanced,
    rotate_or_flip_image,
    resize_standard,
    resize_ai_upscale,
    remove_background_ai,
    add_text_to_image,
    apply_filter,
)

st.set_page_config(page_title="PDP Photo Editor Web", layout="wide")
st.title("🖼️ PDP Chỉnh Sửa Ảnh Trực Tuyến")

TEMP_DIR = tempfile.gettempdir()
INPUT_PATH = os.path.join(TEMP_DIR, "web_input_temp.png")
OUTPUT_PATH = os.path.join(TEMP_DIR, "web_output_temp.png")

# -------------------------------------------------------------------
# HÀM PHÂN TÍCH ẢNH BẰNG AI (Quy về chuẩn hệ quy chiếu với Slider)
# -------------------------------------------------------------------
def analyze_image_ai(image_path):
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, _, _ = cv2.split(img_lab)

    raw_brightness = np.mean(l_channel)
    brightness_pct = round((raw_brightness / 255.0) * 100, 1)

    p10 = np.percentile(l_channel, 10) / 2.55
    p90 = np.percentile(l_channel, 90) / 2.55
    std_contrast = round(float(np.std(l_channel)), 1)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)

    # Tính toán chính xác 8 tham số khớp hệ quy chiếu với Slider
    target_pct = 58.0
    suggested_exposure = round((target_pct - brightness_pct) / 25.0, 1)
    suggested_exposure = float(np.clip(suggested_exposure, -1.0, 1.5))

    suggested_shadows = int(np.clip((25 - p10) * 1.2, 0, 50)) if p10 < 25 else 0
    suggested_highlights = -15 if p90 > 75 else 0
    suggested_contrast = 15 if std_contrast < 50 else 5
    suggested_clarity = 15
    suggested_dehaze = 10
    suggested_saturation = 10

    if laplacian_var < 100:
        suggested_sharpening = 40
    elif laplacian_var < 300:
        suggested_sharpening = 25
    else:
        suggested_sharpening = 15

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
# KHỞI TẠO VÀ ĐỒNG BỘ NÚT BẤM AI VỚI CÁC SLIDER
# -------------------------------------------------------------------
slider_keys = [
    "exposure", "contrast", "highlights", "shadows",
    "saturation", "clarity", "dehaze", "sharpening"
]

defaults = {
    "exposure": 0.0, "contrast": 0, "highlights": 0, "shadows": 0,
    "saturation": 0, "clarity": 0, "dehaze": 0, "sharpening": 0
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "ai_analysis" not in st.session_state:
    st.session_state["ai_analysis"] = None

# Hàm callback áp dụng TRỌN BỘ 8 THAM SỐ AI vào các slider
def apply_ai_suggestions():
    if st.session_state["ai_analysis"]:
        ai = st.session_state["ai_analysis"]
        st.session_state["exposure"] = float(ai["exposure"])
        st.session_state["contrast"] = int(ai["contrast"])
        st.session_state["highlights"] = int(ai["highlights"])
        st.session_state["shadows"] = int(ai["shadows"])
        st.session_state["saturation"] = int(ai["saturation"])
        st.session_state["clarity"] = int(ai["clarity"])
        st.session_state["dehaze"] = int(ai["dehaze"])
        st.session_state["sharpening"] = int(ai["sharpening"])

# Sidebar: Tải ảnh
st.sidebar.header("📂 Tải Ảnh Lên")
uploaded_file = st.sidebar.file_uploader("Chọn tệp ảnh", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save(INPUT_PATH)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(image, use_container_width=True)

    # DANH SÁCH TAB ĐẦY ĐỦ
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "✨ Chỉnh Sáng & AI",
        "🎨 Bộ Lọc Màu",
        "📝 Thêm Chữ (Text)",
        "📐 Resize & Upscale",
        "✂️ Tách Nền AI",
        "🔄 Xoay & Lật"
    ])

    # --- TAB 1: CHỈNH SÁNG VÀ AI ---
    with tab1:
        st.markdown("### Chỉnh sửa thông số ảnh")

        with st.expander("🤖 **Phân tích ảnh bằng AI & Gợi ý thông số**", expanded=True):
            col_ai1, col_ai2 = st.columns([1, 2])
            with col_ai1:
                if st.button("🔍 Phân tích ảnh bằng AI"):
                    with st.spinner("AI đang phân tích..."):
                        st.session_state["ai_analysis"] = analyze_image_ai(INPUT_PATH)

            with col_ai2:
                if st.session_state["ai_analysis"]:
                    ai_res = st.session_state["ai_analysis"]
                    st.write(
                        f"📊 **Chỉ số gốc:** Độ sáng `{ai_res['brightness_pct']}%` | "
                        f"Tương phản `{ai_res['std_contrast']}` | "
                        f"Độ nét `{ai_res['sharpness']}`"
                    )
                    st.write(
                        f"💡 **AI đề xuất:** "
                        f"Exp `{ai_res['exposure']:+.1f}` | "
                        f"Contrast `{ai_res['contrast']:+d}` | "
                        f"Shadows `{ai_res['shadows']:+d}` | "
                        f"Sharpening `{ai_res['sharpening']}`"
                    )
                    st.button("👉 Áp dụng thông số AI vào Slider", on_click=apply_ai_suggestions)

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            exposure = st.slider("Độ sáng (Exposure)", -2.0, 2.0, value=float(st.session_state["exposure"]), step=0.1, key="exposure_slider", on_change=lambda: st.session_state.update({"exposure": st.session_state.exposure_slider}))
            contrast = st.slider("Độ tương phản (Contrast)", -100, 100, value=int(st.session_state["contrast"]), key="contrast_slider", on_change=lambda: st.session_state.update({"contrast": st.session_state.contrast_slider}))
            highlights = st.slider("Vùng sáng (Highlights)", -100, 100, value=int(st.session_state["highlights"]), key="highlights_slider", on_change=lambda: st.session_state.update({"highlights": st.session_state.highlights_slider}))
            shadows = st.slider("Vùng tối (Shadows)", -100, 100, value=int(st.session_state["shadows"]), key="shadows_slider", on_change=lambda: st.session_state.update({"shadows": st.session_state.shadows_slider}))
            saturation = st.slider("Độ bão hòa (Saturation)", -100, 100, value=int(st.session_state["saturation"]), key="saturation_slider", on_change=lambda: st.session_state.update({"saturation": st.session_state.saturation_slider}))
        with c2:
            clarity = st.slider("Độ rõ nét (Clarity)", -100, 100, value=int(st.session_state["clarity"]), key="clarity_slider", on_change=lambda: st.session_state.update({"clarity": st.session_state.clarity_slider}))
            dehaze = st.slider("Khử mờ (Dehaze)", 0, 100, value=int(st.session_state["dehaze"]), key="dehaze_slider", on_change=lambda: st.session_state.update({"dehaze": st.session_state.dehaze_slider}))
            sharpening = st.slider("Sắc nét (Sharpening)", 0, 100, value=int(st.session_state["sharpening"]), key="sharpening_slider", on_change=lambda: st.session_state.update({"sharpening": st.session_state.sharpening_slider}))

        if st.button("Áp dụng ánh sáng & màu sắc"):
            with st.spinner("Đang xử lý ảnh..."):
                adjust_image_advanced(
                    INPUT_PATH,
                    OUTPUT_PATH,
                    exposure=st.session_state["exposure"],
                    contrast=st.session_state["contrast"],
                    highlights=st.session_state["highlights"],
                    shadows=st.session_state["shadows"],
                    saturation=st.session_state["saturation"],
                    clarity=st.session_state["clarity"],
                    dehaze=st.session_state["dehaze"],
                    sharpening=st.session_state["sharpening"]
                )
                st.session_state["processed_img"] = OUTPUT_PATH

    # --- TAB 2: BỘ LỌC MÀU ---
    with tab2:
        st.markdown("### Chọn Bộ Lọc Nghệ Thuật")
        filter_option = st.selectbox(
            "Chọn hiệu ứng:",
            ["Trắng Đen (Grayscale)", "Cổ Điển (Sepia)", "Rực Rỡ (Vintage/Warm)", "Lạnh (Cool Tone)"]
        )
        if st.button("Áp dụng Bộ Lọc"):
            apply_filter(INPUT_PATH, OUTPUT_PATH, filter_option)
            st.session_state["processed_img"] = OUTPUT_PATH

            # --- TAB 3: THÊM TEXT TƯƠNG TÁC (CHỌN FONT CHỮ) ---
            with tab3:
                st.markdown("### 🎯 Chèn Chữ Trực Quan (Chọn Font & Click Vị Trí)")

                # 1. Nhập thông số chữ
                col_txt1, col_txt2 = st.columns(2)
                with col_txt1:
                    input_text = st.text_input("Nội dung chữ:", value="PDP Photo Editor")
                    font_size = st.slider("Kích thước chữ (px)", 12, 150, 40)
                with col_txt2:
                    font_name = st.selectbox(
                        "Kiểu Font chữ:",
                        ["Arial", "Times New Roman", "Courier New", "Segoe UI", "Calibri", "Georgia", "Tahoma",
                         "Verdana"]
                    )
                    text_color = st.color_picker("Màu chữ", "#FF0000")

                # Quy đổi mã Hex sang RGB
                hex_c = text_color.lstrip('#')
                rgb_color = tuple(int(hex_c[i:i + 2], 16) for i in (0, 2, 4))

                st.info(
                    "👉 **Hướng dẫn:** Click chuột trực tiếp vào vị trí trên bức ảnh bên dưới để đặt tâm điểm chèn chữ.")

                # 2. Khung ảnh Canvas cho phép Click chọn vị trí
                canvas_bg = Image.open(INPUT_PATH)
                bg_width, bg_height = canvas_bg.size

                disp_width = min(bg_width, 700)
                disp_height = int(bg_height * (disp_width / bg_width))

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    background_image=canvas_bg,
                    update_streamlit=True,
                    height=disp_height,
                    width=disp_width,
                    drawing_mode="point",
                    key="canvas_text_picker",
                )

                # 3. Tính toán tọa độ từ điểm Click
                pos_x, pos_y = 50, 50
                if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                    last_point = canvas_result.json_data["objects"][-1]
                    click_x = last_point["left"]
                    click_y = last_point["top"]

                    pos_x = int(click_x * (bg_width / disp_width))
                    pos_y = int(click_y * (bg_height / disp_height))

                    st.success(f"📍 Đã chọn vị trí: X = `{pos_x}px`, Y = `{pos_y}px`")

                # 4. Nút thực thi chèn chữ
                if st.button("✨ Áp Dụng Thêm Chữ"):
                    add_text_to_image(
                        INPUT_PATH,
                        OUTPUT_PATH,
                        text=input_text,
                        position=(pos_x, pos_y),
                        font_name=font_name,
                        font_size=font_size,
                        color=rgb_color
                    )
                    st.session_state["processed_img"] = OUTPUT_PATH
                    st.rerun()

    # --- TAB 4: RESIZE & UPSCALE ---
    with tab4:
        st.markdown("### Phóng to / Thay đổi kích thước")
        resize_type = st.radio("Phương pháp:", ["Resize Chuẩn", "AI Super Resolution (Upscale)"])
        if resize_type == "Resize Chuẩn":
            w = st.number_input("Chiều rộng (px)", value=image.width)
            h = st.number_input("Chiều cao (px)", value=image.height)
            if st.button("Thực hiện Resize"):
                resize_standard(INPUT_PATH, OUTPUT_PATH, width=w, height=h)
                st.session_state["processed_img"] = OUTPUT_PATH
        else:
            scale = st.selectbox("Tỉ lệ phóng to:", [2, 4])
            if st.button("Phóng to bằng AI"):
                with st.spinner("Đang nâng cấp chất lượng..."):
                    resize_ai_upscale(INPUT_PATH, OUTPUT_PATH, scale_factor=scale)
                    st.session_state["processed_img"] = OUTPUT_PATH

    # --- TAB 5: TÁCH NỀN AI ---
    with tab5:
        st.markdown("### Tách nền tự động bằng AI (rembg)")
        if st.button("Bắt đầu Tách Nền"):
            with st.spinner("Đang tách nền..."):
                try:
                    remove_background_ai(INPUT_PATH, OUTPUT_PATH)
                    st.session_state["processed_img"] = OUTPUT_PATH
                    st.success("Tách nền thành công!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # --- TAB 6: XOAY & LẬT ---
    with tab6:
        st.markdown("### Xoay và lật hướng ảnh")
        action = st.selectbox("Chọn thao tác:", [
            ("Xoay phải 90°", "rotate_right"),
            ("Xoay trái 90°", "rotate_left"),
            ("Lật ngang", "flip_horizontal"),
            ("Lật dọc", "flip_vertical")
        ], format_func=lambda x: x[0])

        if st.button("Thực hiện Xoay/Lật"):
            rotate_or_flip_image(INPUT_PATH, OUTPUT_PATH, action=action[1])
            st.session_state["processed_img"] = OUTPUT_PATH

    # Hiển thị Kết Quả
    with col2:
        st.subheader("Kết Quả")
        if "processed_img" in st.session_state and os.path.exists(st.session_state["processed_img"]):
            result_img = Image.open(st.session_state["processed_img"])
            st.image(result_img, use_container_width=True)

            with open(st.session_state["processed_img"], "rb") as file:
                st.download_button(
                    label="💾 Tải Ảnh Kết Quả Về Máy",
                    data=file,
                    file_name="edited_image.png",
                    mime="image/png"
                )
else:
    st.info("Vui lòng tải một tệp ảnh lên từ thanh bên (Sidebar) để bắt đầu chỉnh sửa.")