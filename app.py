# app.py
import os
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Import toàn bộ hàm xử lý từ core.py
from core import (
    rotate_or_flip_image,
    crop_image,
    resize_image,
    adjust_color_and_filters,
    convert_image_format,
)

# Cấu hình giao diện trang
st.set_page_config(
    page_title="Ứng Dụng Xử Lý & Chỉnh Sửa Ảnh Chuyên Nghiệp",
    page_icon="🖼️",
    layout="wide"
)

# Thư mục làm việc tạm thời
WORKING_DIR = Path("temp_workspace")
WORKING_DIR.mkdir(exist_ok=True)

INPUT_PATH = WORKING_DIR / "current_input.jpg"
OUTPUT_PATH = WORKING_DIR / "current_output.jpg"


def initialize_session_state():
    if "canvas_key_counter" not in st.session_state:
        st.session_state["canvas_key_counter"] = 0
    if "crop_coords" not in st.session_state:
        st.session_state["crop_coords"] = None


def update_input_image_and_refresh():
    """Cập nhật ảnh đầu ra làm ảnh đầu vào mới và reset canvas."""
    if OUTPUT_PATH.exists():
        shutil.copy(OUTPUT_PATH, INPUT_PATH)
        st.session_state["crop_coords"] = None
        st.session_state["canvas_key_counter"] += 1


initialize_session_state()

st.title("🖼️ Ứng Dụng Xử Lý & Chỉnh Sửa Ảnh Chuyên Nghiệp")

# Sidebar Upload
with st.sidebar:
    st.header("📂 Nguồn Ảnh Đầu Vào")
    uploaded_file = st.file_uploader("Tải ảnh lên từ máy tính:", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_file is not None:
        if st.button("📥 Bắt Đầu Chỉnh Sửa Ảnh Mới", type="primary", use_container_width=True):
            with open(INPUT_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())
            shutil.copy(INPUT_PATH, OUTPUT_PATH)
            st.session_state["crop_coords"] = None
            st.session_state["canvas_key_counter"] += 1
            st.success("Đã nạp ảnh thành công!")
            st.rerun()

    if INPUT_PATH.exists():
        st.markdown("---")
        st.markdown("### 📊 Thông Số Ảnh Hiện Tại")
        with Image.open(INPUT_PATH) as img_info:
            w, h = img_info.size
            st.write(f"- **Kích thước:** `{w} x {h} px`")
            st.write(f"- **Dung lượng:** `{os.path.getsize(INPUT_PATH) / 1024:.1f} KB`")
            st.write(f"- **Định dạng:** `{img_info.format}`")

if not INPUT_PATH.exists():
    st.info("👈 Vui lòng chọn và tải một tấm ảnh lên từ thanh công cụ bên trái để bắt đầu.")
    st.stop()

# TẠO CÁC TABS CÔNG CỤ HOÀN CHỈNH
tab_crop, tab_resize, tab_color, tab_convert, tab_preview = st.tabs([
    "✂️ 1. Xoay, Lật & Cắt Ảnh",
    "📏 2. Đổi Kích Thước (Resize)",
    "🎨 3. Màu Sắc & Bộ Lọc",
    "🔄 4. Đổi Định Dạng File",
    "👁️ 5. Xem Xem So Sánh Ảnh"
])

# ==========================================
# TAB 1: XOAY, LẬT & CẮT ÁNH (TẬP TRUNG EDIT FIX LỖI)
# ==========================================
with tab_crop:
    st.markdown("### 1.1. Xoay và lật hướng ảnh")
    c_rot1, c_rot2 = st.columns([3, 1])
    with c_rot1:
        action = st.selectbox(
            "Chọn thao tác xoay/lật:",
            [
                ("Xoay phải 90°", "rotate_right"),
                ("Xoay trái 90°", "rotate_left"),
                ("Lật ngang (Horizontal)", "flip_horizontal"),
                ("Lật dọc (Vertical)", "flip_vertical"),
            ],
            format_func=lambda x: x[0],
            key="select_rotate_action"
        )
    with c_rot2:
        st.write(" ")
        st.write(" ")
        if st.button("🔄 Thực hiện Xoay/Lật", key="btn_apply_rotate", use_container_width=True):
            if rotate_or_flip_image(INPUT_PATH, OUTPUT_PATH, action=action[1]):
                update_input_image_and_refresh()
                st.success("Đã xoay/lật ảnh thành công!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 1.2. Cắt ảnh (Crop Image)")

    crop_mode = st.radio(
        "Chọn phương thức cắt ảnh:",
        ["Kéo thả trực quan (Canvas Drag)", "Tỉ lệ cố định (Aspect Ratio)", "Nhập Tọa độ Pixel"],
        horizontal=True,
        key="radio_crop_mode"
    )

    curr_img = Image.open(INPUT_PATH)
    orig_w, orig_h = curr_img.size

    # A. CẮT BẰNG KÉO THẢ TRÊN CANVAS (ĐÃ FIX TRIỆT ĐỂ 100%)
    if crop_mode == "Kéo thả trực quan (Canvas Drag)":
        st.info(
            "👉 **Hướng dẫn:** Đè giữ chuột trái và kéo vẽ hình chữ nhật màu đỏ lên ảnh. Tọa độ sẽ **tự động nhận diện ngay lập tức** bên dưới.")

        disp_w = min(orig_w, 700)
        disp_h = int(orig_h * (disp_w / orig_w))

        # Khởi tạo Canvas độc lập hoàn toàn không dùng form
        canvas_key = f"canvas_widget_{st.session_state['canvas_key_counter']}"

        crop_canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.25)",
            stroke_color="#FF0000",
            stroke_width=2,
            background_image=curr_img,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=canvas_key
        )

        # ENGINE TỰ ĐỘNG BÓC TÁCH TỌA ĐỘ REAL-TIME KHÔNG CẦN BẤM NÚT PHỤ
        if crop_canvas.json_data is not None:
            objects = crop_canvas.json_data.get("objects", [])
            if len(objects) > 0:
                # Lấy hình chữ nhật mới nhất
                last_obj = objects[-1]
                scale_x = orig_w / disp_w
                scale_y = orig_h / disp_h

                raw_l = float(last_obj.get("left", 0))
                raw_t = float(last_obj.get("top", 0))
                raw_w = float(last_obj.get("width", 0)) * float(last_obj.get("scaleX", 1.0))
                raw_h = float(last_obj.get("height", 0)) * float(last_obj.get("scaleY", 1.0))

                # Tính toán tọa độ thực tế trên ảnh gốc
                x1 = int(raw_l * scale_x)
                y1 = int(raw_t * scale_y)
                x2 = int((raw_l + raw_w) * scale_x)
                y2 = int((raw_t + raw_h) * scale_y)

                # Chuẩn hóa (Xử lý cả trường hợp kéo ngược chuột từ phải sang trái)
                real_left = max(0, min(x1, x2))
                real_top = max(0, min(y1, y2))
                real_right = min(orig_w, max(x1, x2))
                real_bottom = min(orig_h, max(y1, y2))

                if (real_right - real_left) >= 5 and (real_bottom - real_top) >= 5:
                    st.session_state["crop_coords"] = (real_left, real_top, real_right, real_bottom)

        # HIỂN THỊ KẾT QUẢ VÀ BẤM NÚT CẮT TRỰC TIẾP
        coords = st.session_state.get("crop_coords")
        if coords:
            lx, ty, rx, by = coords
            cw = rx - lx
            ch = by - ty

            st.success(
                f"🎯 **ĐÃ NHẬN DIỆN VÙNG KÉO:** X1=`{lx}`, Y1=`{ty}` | X2=`{rx}`, Y2=`{by}` | **Kích thước vùng cắt:** `{cw} x {ch} px`")

            if st.button("✂️ BẤM VÀO ĐÂY ĐỂ CẮT VÙNG ĐÃ CHỌN", key="btn_execute_crop", type="primary",
                         use_container_width=True):
                if crop_image(INPUT_PATH, OUTPUT_PATH, coords):
                    update_input_image_and_refresh()
                    st.success("🎉 Đã cắt ảnh thành công!")
                    st.rerun()
                else:
                    st.error("Lỗi khi cắt ảnh. Vui lòng thử lại.")
        else:
            st.warning("⚠️ Hãy đè giữ chuột trái và kéo một khung màu đỏ trên ảnh để chọn vùng cắt.")

    # B. CẮT THEO TỈ LỆ CỐ ĐỊNH
    elif crop_mode == "Tỉ lệ cố định (Aspect Ratio)":
        ratio_option = st.selectbox(
            "Chọn tỉ lệ cắt mong muốn:",
            ["1:1 (Ảnh Vuông / Avatar)", "4:3 (Chuẩn Nhiếp Ảnh)", "16:9 (Màn Hình Rộng)", "3:4 (Chân Dung)",
             "9:16 (Story / TikTok)"],
            key="select_crop_ratio"
        )

        ratio_map = {
            "1:1 (Ảnh Vuông / Avatar)": (1, 1),
            "4:3 (Chuẩn Nhiếp Ảnh)": (4, 3),
            "16:9 (Màn Hình Rộng)": (16, 9),
            "3:4 (Chân Dung)": (3, 4),
            "9:16 (Story / TikTok)": (9, 16),
        }
        rw, rh = ratio_map[ratio_option]

        target_aspect = rw / rh
        current_aspect = orig_w / orig_h

        if current_aspect > target_aspect:
            new_w = int(orig_h * target_aspect)
            new_h = orig_h
        else:
            new_w = orig_w
            new_h = int(orig_w / target_aspect)

        left = (orig_w - new_w) // 2
        top = (orig_h - new_h) // 2
        right = left + new_w
        bottom = top + new_h

        st.info(f"📐 Vùng cắt trung tâm tự động: `{new_w} x {new_h} px` (Tỉ lệ {rw}:{rh})")

        if st.button("✂️ Áp Dụng Cắt Theo Tỉ Lệ", key="btn_crop_ratio", type="primary", use_container_width=True):
            if crop_image(INPUT_PATH, OUTPUT_PATH, (left, top, right, bottom)):
                update_input_image_and_refresh()
                st.success("Cắt ảnh theo tỉ lệ thành công!")
                st.rerun()

    # C. CẮT THEO TỌA ĐỘ PIXEL
    else:
        col_cr1, col_cr2 = st.columns(2)
        with col_cr1:
            crop_x = st.number_input("Tọa độ X (Góc trái)", 0, max(0, orig_w - 1), 0, key="num_crop_x")
            crop_y = st.number_input("Tọa độ Y (Góc trên)", 0, max(0, orig_h - 1), 0, key="num_crop_y")
        with col_cr2:
            crop_w = st.number_input("Chiều rộng (Width)", 1, orig_w - crop_x, orig_w, key="num_crop_w")
            crop_h = st.number_input("Chiều cao (Height)", 1, orig_h - crop_y, orig_h, key="num_crop_h")

        if st.button("✂️ Áp Dụng Cắt Theo Tọa Độ Pixel", key="btn_crop_pixel", type="primary",
                     use_container_width=True):
            box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
            if crop_image(INPUT_PATH, OUTPUT_PATH, box):
                update_input_image_and_refresh()
                st.success("Cắt ảnh theo pixel thành công!")
                st.rerun()

# ==========================================
# TAB 2: ĐỔI KÍCH THƯỚC ÁNH (RESIZE)
# ==========================================
with tab_resize:
    st.markdown("### 2. Thay đổi kích thước (Resize Image)")
    curr_img = Image.open(INPUT_PATH)
    ow, oh = curr_img.size

    st.write(f"Kích thước gốc: **{ow} x {oh} px**")

    keep_ratio = st.checkbox("Giữ nguyên tỉ lệ khung hình (Keep Aspect Ratio)", value=True)

    col_rs1, col_rs2 = st.columns(2)
    with col_rs1:
        target_w = st.number_input("Chiều rộng mới (px):", min_value=10, max_value=10000, value=ow)
    with col_rs2:
        if keep_ratio:
            calculated_h = int(oh * (target_w / ow))
            st.number_input("Chiều cao mới (px) [Tự động]:", value=calculated_h, disabled=True)
            target_h = calculated_h
        else:
            target_h = st.number_input("Chiều cao mới (px):", min_value=10, max_value=10000, value=oh)

    if st.button("📏 Thực Hiện Resize Ảnh", key="btn_apply_resize", type="primary", use_container_width=True):
        if resize_image(INPUT_PATH, OUTPUT_PATH, target_w, target_h, keep_aspect_ratio=keep_ratio):
            update_input_image_and_refresh()
            st.success("Đã thay đổi kích thước ảnh thành công!")
            st.rerun()

# ==========================================
# TAB 3: MÀU SẮC VÀ BỘ LỌC (COLOR & FILTERS)
# ==========================================
with tab_color:
    st.markdown("### 3. Tinh chỉnh màu sắc & Bộ lọc nghệ thuật")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        bright = st.slider("☀️ Độ sáng (Brightness):", 0.2, 2.0, 1.0, 0.1)
    with col_c2:
        contra = st.slider("🌓 Độ tương phản (Contrast):", 0.2, 2.0, 1.0, 0.1)
    with col_c3:
        sharp = st.slider("🔪 Độ sắc nét (Sharpness):", 0.0, 3.0, 1.0, 0.1)

    filter_choice = st.selectbox(
        "Chọn hiệu ứng bộ lọc:",
        ["Gốc (Không lọc)", "Trắng Đen (Grayscale)", "Làm Mờ (Blur)", "Sắc Nhét (Sharpen)", "Nổi Bật Cạnh (Find Edges)"]
    )

    if st.button("🎨 Áp Dụng Màu Sắc & Bộ Lọc", key="btn_apply_color", type="primary", use_container_width=True):
        if adjust_color_and_filters(INPUT_PATH, OUTPUT_PATH, bright, contra, sharp, filter_choice):
            update_input_image_and_refresh()
            st.success("Đã áp dụng điều chỉnh màu sắc!")
            st.rerun()

# ==========================================
# TAB 4: ĐỔI ĐỊNH DẠNG FILE (CONVERT FORMAT)
# ==========================================
with tab_convert:
    st.markdown("### 4. Chuyển đổi định dạng file ảnh")

    target_fmt = st.selectbox("Chọn định dạng muốn xuất:", ["PNG", "JPG", "WEBP", "BMP", "TIFF"])

    if st.button("🔄 Thực Hiện Chuyển Đổi Định Dạng", key="btn_apply_convert", type="primary", use_container_width=True):
        out_converted = WORKING_DIR / f"converted_image.{target_fmt.lower()}"
        if convert_image_format(INPUT_PATH, out_converted, target_fmt):
            st.success(f"Đã xuất file thành công dạng `{target_fmt}`!")
            with open(out_converted, "rb") as file_data:
                st.download_button(
                    label=f"💾 Tải Ảnh {target_fmt} Về Máy",
                    data=file_data,
                    file_name=f"output_image.{target_fmt.lower()}",
                    mime=f"image/{target_fmt.lower()}",
                    use_container_width=True
                )

# ==========================================
# TAB 5: XEM & SO SÁNH ÁNH (PREVIEW)
# ==========================================
with tab_preview:
    st.markdown("### 5. Xem ảnh làm việc hiện tại")
    col_pv1, col_pv2 = st.columns(2)
    with col_pv1:
        st.markdown("#### Ảnh làm việc hiện tại")
        st.image(str(INPUT_PATH), use_container_width=True)
    with col_pv2:
        if OUTPUT_PATH.exists():
            st.markdown("#### Ảnh kết quả xử lý mới nhất")
            st.image(str(OUTPUT_PATH), use_container_width=True)
            with open(OUTPUT_PATH, "rb") as f_out:
                st.download_button(
                    label="💾 Tải Ảnh Này Về Máy",
                    data=f_out,
                    file_name="edited_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )