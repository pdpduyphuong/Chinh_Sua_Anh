# app.py
import os
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Import core business logic
from core import rotate_or_flip_image, crop_image

# Thiết lập trang Streamlit
st.set_page_config(
    page_title="Ứng Dụng Xử Lý Ảnh Chuyên Nghiệp",
    page_icon="🖼️",
    layout="wide"
)

# Đường dẫn đệm làm việc
WORKING_DIR = Path("temp_workspace")
WORKING_DIR.mkdir(exist_ok=True)

INPUT_PATH = WORKING_DIR / "current_input.jpg"
OUTPUT_PATH = WORKING_DIR / "current_output.jpg"


def initialize_session_state():
    """Khởi tạo trạng thái bộ nhớ cho ứng dụng."""
    if "canvas_key_id" not in st.session_state:
        st.session_state["canvas_key_id"] = 0
    if "crop_coords" not in st.session_state:
        st.session_state["crop_coords"] = None


def update_input_image_and_refresh():
    """Copy kết quả mới làm ảnh đầu vào tiếp theo."""
    if OUTPUT_PATH.exists():
        shutil.copy(OUTPUT_PATH, INPUT_PATH)
        st.session_state["crop_coords"] = None
        st.session_state["canvas_key_id"] += 1


initialize_session_state()

st.title("🖼️ Ứng Dụng Chỉnh Sửa & Cắt Ảnh")

# Sidebar Upload Ảnh
with st.sidebar:
    st.header("📂 Dữ Liệu Đầu Vào")
    uploaded_file = st.file_uploader("Tải ảnh lên:", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        if st.button("📥 Nạp Ảnh Mới", type="primary"):
            with open(INPUT_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())
            shutil.copy(INPUT_PATH, OUTPUT_PATH)
            st.session_state["crop_coords"] = None
            st.session_state["canvas_key_id"] += 1
            st.success("Đã tải ảnh lên thành công!")
            st.rerun()

if not INPUT_PATH.exists():
    st.info("👈 Vui lòng tải một tấm ảnh lên từ thanh bên trái để bắt đầu.")
    st.stop()

# Khởi tạo Tabs ứng dụng
tab6, tab_preview = st.tabs(["✂️ Xoay, Lật & Cắt Ảnh", "👁️ Xem Ảnh Hiện Tại"])

with tab_preview:
    if INPUT_PATH.exists():
        st.image(str(INPUT_PATH), caption="Ảnh làm việc hiện tại", use_container_width=True)

# --- TAB 6: XOAY, LẬT & CẮT ÁNH ---
with tab6:
    st.markdown("### 1. Xoay và lật hướng ảnh")
    c_rot1, c_rot2 = st.columns([3, 1])
    with c_rot1:
        action = st.selectbox(
            "Chọn thao tác xoay/lật:",
            [
                ("Xoay phải 90°", "rotate_right"),
                ("Xoay trái 90°", "rotate_left"),
                ("Lật ngang", "flip_horizontal"),
                ("Lật dọc", "flip_vertical"),
            ],
            format_func=lambda x: x[0],
            key="select_rotate_action"
        )
    with c_rot2:
        st.write(" ")
        st.write(" ")
        if st.button("🔄 Thực hiện Xoay/Lật", key="btn_apply_rotate"):
            if rotate_or_flip_image(INPUT_PATH, OUTPUT_PATH, action=action[1]):
                update_input_image_and_refresh()
                st.success("Đã thực hiện xoay/lật thành công!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 2. Cắt ảnh (Crop Image)")

    crop_mode = st.radio(
        "Chọn chế độ cắt ảnh:",
        ["Kéo thả trực quan (Canvas Drag)", "Tỉ lệ cố định (Aspect Ratio)", "Tùy chỉnh Tọa độ / Pixel"],
        horizontal=True,
        key="radio_crop_mode"
    )

    curr_img = Image.open(INPUT_PATH)
    orig_w, orig_h = curr_img.size

    # A. CẮT BẰNG KÉO THẢ TRÊN CANVAS (GIẢI PHÁP ĐỒNG BỘ ST.FORM)
    if crop_mode == "Kéo thả trực quan (Canvas Drag)":
        st.info(
            "👉 **Hướng dẫn:** Đè giữ chuột kéo khung hình chữ nhật trên ảnh $\\rightarrow$ Bấm nút **'📍 Xác nhận & Lưu Vùng Chọn'** nằm bên trong khung bên dưới.")

        disp_w = min(orig_w, 700)
        disp_h = int(orig_h * (disp_w / orig_w))

        # SỬ DỤNG ST.FORM ĐỂ ĐÓNG BỎ VẤN ĐỀ TRỄ DỮ LIỆU CANVAS
        with st.form(key=f"crop_canvas_form_{st.session_state['canvas_key_id']}"):
            crop_canvas = st_canvas(
                fill_color="rgba(255, 0, 0, 0.25)",
                stroke_color="#FF0000",
                stroke_width=2,
                background_image=curr_img,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="rect",
                key=f"canvas_widget_{st.session_state['canvas_key_id']}"
            )

            submit_canvas = st.form_submit_button("📍 1. Xác nhận & Lưu Vùng Chọn", type="secondary")

        if submit_canvas:
            if crop_canvas.json_data is not None:
                objects = crop_canvas.json_data.get("objects", [])
                rects = [obj for obj in objects if obj.get("type") == "rect"]

                if rects:
                    last_rect = rects[-1]
                    scale_x = orig_w / disp_w
                    scale_y = orig_h / disp_h

                    raw_l = float(last_rect.get("left", 0))
                    raw_t = float(last_rect.get("top", 0))
                    raw_w = float(last_rect.get("width", 0)) * float(last_rect.get("scaleX", 1.0))
                    raw_h = float(last_rect.get("height", 0)) * float(last_rect.get("scaleY", 1.0))

                    # Bóc tách tọa độ thực tế trên ảnh chuẩn
                    x1 = int(raw_l * scale_x)
                    y1 = int(raw_t * scale_y)
                    x2 = int((raw_l + raw_w) * scale_x)
                    y2 = int((raw_t + raw_h) * scale_y)

                    real_left = max(0, min(x1, x2))
                    real_top = max(0, min(y1, y2))
                    real_right = min(orig_w, max(x1, x2))
                    real_bottom = min(orig_h, max(y1, y2))

                    if (real_right - real_left) >= 5 and (real_bottom - real_top) >= 5:
                        st.session_state["crop_coords"] = (real_left, real_top, real_right, real_bottom)
                        st.rerun()
                    else:
                        st.warning("⚠️ Vùng kéo quá nhỏ. Vui lòng kéo lại khung rộng hơn.")
                else:
                    st.warning("⚠️ Không tìm thấy khung chữ nhật. Vui lòng kéo chuột vẽ lại trên ảnh!")
            else:
                st.error("❌ Không thể kết nối dữ liệu Canvas. Vui lòng thử lại.")

        # HIỂN THỊ KẾT QUẢ BẮT TỌA ĐỘ & NÚT THỰC HIỆN CẮT
        coords = st.session_state.get("crop_coords")
        if coords:
            lx, ty, rx, by = coords
            cw = rx - lx
            ch = by - ty

            st.success(
                f"✅ **ĐÃ BẮT THÀNH CÔNG VÙNG CHỌN:** X1=`{lx}`, Y1=`{ty}` | X2=`{rx}`, Y2=`{by}` | Kích thước: **{cw} x {ch} px**")

            if st.button("✂️ 2. THỰC HIỆN CẮT ẢNH NGAY", key="btn_execute_crop", type="primary"):
                if crop_image(INPUT_PATH, OUTPUT_PATH, coords):
                    update_input_image_and_refresh()
                    st.success("🎉 Đã cắt ảnh thành công!")
                    st.rerun()
                else:
                    st.error("Lỗi khi thực hiện cắt ảnh. Vui lòng thử lại.")
        else:
            st.warning(
                "⚠️ Vui lòng đè giữ chuột kéo khung màu đỏ trên ảnh, sau đó bấm nút '📍 1. Xác nhận & Lưu Vùng Chọn' ở trên.")

    # B. CẮT THEO TỈ LỆ CỐ ĐỊNH
    elif crop_mode == "Tỉ lệ cố định (Aspect Ratio)":
        ratio_option = st.selectbox(
            "Chọn tỉ lệ cắt:",
            ["1:1 (Vuông)", "4:3 (Chuẩn)", "16:9 (Màn hình rộng)", "3:4 (Chân dung)", "9:16 (Story/Reels)"],
            key="select_crop_ratio"
        )

        ratio_map = {
            "1:1 (Vuông)": (1, 1),
            "4:3 (Chuẩn)": (4, 3),
            "16:9 (Màn hình rộng)": (16, 9),
            "3:4 (Chân dung)": (3, 4),
            "9:16 (Story/Reels)": (9, 16),
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

        st.info(f"📐 Kích thước vùng cắt đề xuất: `{new_w} x {new_h} px` (Trung tâm ảnh)")

        if st.button("✂️ Áp dụng Cắt theo tỉ lệ", key="btn_crop_ratio", type="primary"):
            if crop_image(INPUT_PATH, OUTPUT_PATH, (left, top, right, bottom)):
                update_input_image_and_refresh()
                st.success("Cắt ảnh theo tỉ lệ thành công!")
                st.rerun()

    # C. CẮT THEO TỌA ĐỘ PIXEL TRỰC TIẾP
    else:
        st.subheader("Cắt theo thông số Pixel chính xác")
        col_cr1, col_cr2 = st.columns(2)
        with col_cr1:
            crop_x = st.number_input("Tọa độ X (Góc trái)", 0, max(0, orig_w - 1), 0, key="num_crop_x")
            crop_y = st.number_input("Tọa độ Y (Góc trên)", 0, max(0, orig_h - 1), 0, key="num_crop_y")
        with col_cr2:
            crop_w = st.number_input("Chiều rộng vùng cắt (W)", 1, orig_w - crop_x, orig_w, key="num_crop_w")
            crop_h = st.number_input("Chiều cao vùng cắt (H)", 1, orig_h - crop_y, orig_h, key="num_crop_h")

        if st.button("✂️ Áp dụng Cắt Pixel", key="btn_crop_pixel", type="primary"):
            box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
            if crop_image(INPUT_PATH, OUTPUT_PATH, box):
                update_input_image_and_refresh()
                st.success("Cắt ảnh theo pixel thành công!")
                st.rerun()