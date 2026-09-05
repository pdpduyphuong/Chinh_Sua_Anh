# app.py
import os
import sys
import shutil
import tempfile
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

try:
    from core import (
        adjust_image_advanced,
        rotate_or_flip_image,
        crop_image,
        resize_standard,
        resize_ai_upscale,
        remove_background_ai,
        add_text_to_image,
        apply_filter,
        cv2
    )
except ImportError as e:
    st.error(f"❌ Lỗi khởi tạo ứng dụng (ImportError): {e}")
    st.info("💡 Vui lòng kiểm tra lại file core.py và requirements.txt.")
    st.stop()

st.set_page_config(page_title="PDP Photo Editor Web", layout="wide")
st.title("🖼️ PDP Chỉnh Sửa Ảnh Trực Tuyến")

# Cấu hình đường dẫn tệp tạm thời
TEMP_DIR = tempfile.gettempdir()
INPUT_PATH = os.path.join(TEMP_DIR, "web_input_temp.png")
OUTPUT_PATH = os.path.join(TEMP_DIR, "web_output_temp.png")
FILTER_BASE_PATH = os.path.join(TEMP_DIR, "web_filter_base_temp.png")


def update_input_image_and_refresh():
    """Đồng bộ file output sang input và lưu mốc khôi phục cho bộ lọc màu."""
    if os.path.exists(OUTPUT_PATH):
        shutil.copy(OUTPUT_PATH, INPUT_PATH)
        shutil.copy(OUTPUT_PATH, FILTER_BASE_PATH)
        st.session_state["processed_img"] = OUTPUT_PATH


def analyze_image_ai(image_path):
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    if cv2 is not None:
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, _, _ = cv2.split(img_lab)
        raw_brightness = np.mean(l_channel)
        p10 = np.percentile(l_channel, 10) / 2.55
        p90 = np.percentile(l_channel, 90) / 2.55
        std_contrast = round(float(np.std(l_channel)), 1)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
    else:
        raw_brightness = np.mean(img_np)
        p10, p90 = 20.0, 80.0
        std_contrast = 50.0
        laplacian_var = 150.0

    brightness_pct = round((raw_brightness / 255.0) * 100, 1)
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


# Session State
SLIDER_KEYS = {
    "exposure": ("exp_s", 0.0),
    "contrast": ("cnt_s", 0),
    "highlights": ("hl_s", 0),
    "shadows": ("sh_s", 0),
    "saturation": ("sat_s", 0),
    "clarity": ("clr_s", 0),
    "dehaze": ("dhz_s", 0),
    "sharpening": ("shp_s", 0),
}

for param, (sk, default_val) in SLIDER_KEYS.items():
    if param not in st.session_state:
        st.session_state[param] = default_val
    if sk not in st.session_state:
        st.session_state[sk] = default_val

if "ai_analysis" not in st.session_state:
    st.session_state["ai_analysis"] = None

if "crop_coords_store" not in st.session_state:
    st.session_state["crop_coords_store"] = None

if "active_filter" not in st.session_state:
    st.session_state["active_filter"] = "Gốc (Original / Không bộ lọc)"


def apply_ai_suggestions():
    if st.session_state["ai_analysis"]:
        ai = st.session_state["ai_analysis"]
        for param, (sk, _) in SLIDER_KEYS.items():
            val = ai[param]
            st.session_state[param] = val
            st.session_state[sk] = val


st.sidebar.header("📂 Tải Ảnh Lên")
uploaded_file = st.sidebar.file_uploader("Chọn tệp ảnh", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
        image = Image.open(uploaded_file)
        image.save(INPUT_PATH)
        image.save(FILTER_BASE_PATH)
        st.session_state["last_uploaded"] = uploaded_file.name
        st.session_state["processed_img"] = INPUT_PATH
        st.session_state["crop_coords_store"] = None
        st.session_state["active_filter"] = "Gốc (Original / Không bộ lọc)"

    image = Image.open(INPUT_PATH)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh Gốc / Hiện Tại")
        st.image(image, use_container_width=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "✨ Chỉnh Sáng & AI",
        "🎨 Bộ Lọc Màu",
        "📝 Thêm Chữ (Text)",
        "📐 Resize & Upscale",
        "✂️ Tách Nền AI",
        "🔄 Xoay, Lật & Cắt Ảnh"
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
            st.slider(
                "Độ sáng (Exposure)", -2.0, 2.0, step=0.1, key="exp_s",
                on_change=lambda: st.session_state.update({"exposure": st.session_state.exp_s})
            )
            st.slider(
                "Độ tương phản (Contrast)", -100, 100, key="cnt_s",
                on_change=lambda: st.session_state.update({"contrast": st.session_state.cnt_s})
            )
            st.slider(
                "Vùng sáng (Highlights)", -100, 100, key="hl_s",
                on_change=lambda: st.session_state.update({"highlights": st.session_state.hl_s})
            )
            st.slider(
                "Vùng tối (Shadows)", -100, 100, key="sh_s",
                on_change=lambda: st.session_state.update({"shadows": st.session_state.sh_s})
            )
            st.slider(
                "Độ bão hòa (Saturation)", -100, 100, key="sat_s",
                on_change=lambda: st.session_state.update({"saturation": st.session_state.sat_s})
            )
        with c2:
            st.slider(
                "Độ rõ nét (Clarity)", -100, 100, key="clr_s",
                on_change=lambda: st.session_state.update({"clarity": st.session_state.clr_s})
            )
            st.slider(
                "Khử mờ (Dehaze)", 0, 100, key="dhz_s",
                on_change=lambda: st.session_state.update({"dehaze": st.session_state.dhz_s})
            )
            st.slider(
                "Sắc nét (Sharpening)", 0, 100, key="shp_s",
                on_change=lambda: st.session_state.update({"sharpening": st.session_state.shp_s})
            )

        if st.button("Áp dụng ánh sáng & màu sắc"):
            with st.spinner("Đang xử lý ảnh..."):
                try:
                    exp_val = float(st.session_state.get("exposure", 0.0))
                    cnt_val = float(st.session_state.get("contrast", 0))
                    hl_val = float(st.session_state.get("highlights", 0))
                    sh_val = float(st.session_state.get("shadows", 0))
                    sat_val = float(st.session_state.get("saturation", 0))
                    clr_val = float(st.session_state.get("clarity", 0))
                    dhz_val = float(st.session_state.get("dehaze", 0))
                    shp_val = float(st.session_state.get("sharpening", 0))

                    adjust_image_advanced(
                        INPUT_PATH,
                        OUTPUT_PATH,
                        exposure=exp_val,
                        contrast=cnt_val,
                        highlights=hl_val,
                        shadows=sh_val,
                        saturation=sat_val,
                        clarity=clr_val,
                        dehaze=dhz_val,
                        sharpening=shp_val
                    )
                    update_input_image_and_refresh()
                    st.success("Đã áp dụng thông số thành công!")
                    st.rerun()
                except Exception as err:
                    st.error(f"❌ Lỗi khi xử lý ảnh: {err}")

    # --- TAB 2: BỘ LỌC MÀU (ĐÃ FIX TÍNH NĂNG KHÔI PHỤC) ---
    with tab2:
        st.markdown("### Chọn Bộ Lọc Nghệ Thuật")

        filter_options = [
            "Gốc (Original / Không bộ lọc)",
            "Trắng Đen (Grayscale)",
            "Cổ Điển (Sepia)",
            "Rực Rỡ (Vintage/Warm)",
            "Lạnh (Cool Tone)"
        ]

        selected_filter = st.selectbox(
            "Chọn hiệu ứng bộ lọc màu:",
            filter_options,
            index=filter_options.index(st.session_state.get("active_filter", "Gốc (Original / Không bộ lọc)"))
        )

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            if st.button("✨ Áp Dụng / Đổi Bộ Lọc"):
                # Luôn dùng FILTER_BASE_PATH làm gốc để áp dụng filter, tránh bị đè hiệu ứng cũ
                source_path = FILTER_BASE_PATH if os.path.exists(FILTER_BASE_PATH) else INPUT_PATH
                apply_filter(source_path, OUTPUT_PATH, selected_filter)

                # Cập nhật ảnh làm việc
                shutil.copy(OUTPUT_PATH, INPUT_PATH)
                st.session_state["processed_img"] = OUTPUT_PATH
                st.session_state["active_filter"] = selected_filter
                st.success(f"Đã áp dụng bộ lọc: **{selected_filter}**")
                st.rerun()

        with col_f2:
            if st.button("🔄 Khôi Phục Về Mẫu Ảnh Ban Đầu"):
                if os.path.exists(FILTER_BASE_PATH):
                    shutil.copy(FILTER_BASE_PATH, INPUT_PATH)
                    shutil.copy(FILTER_BASE_PATH, OUTPUT_PATH)
                    st.session_state["processed_img"] = OUTPUT_PATH
                    st.session_state["active_filter"] = "Gốc (Original / Không bộ lọc)"
                    st.success("Đã khôi phục về ảnh ban đầu (chưa áp dụng bộ lọc)!")
                    st.rerun()

    # --- TAB 3: THÊM TEXT ---
    with tab3:
        st.markdown("### 🎯 Chèn Chữ Trực Quan")
        col_txt1, col_txt2 = st.columns(2)
        with col_txt1:
            input_text = st.text_input("Nội dung chữ:", value="PDP Photo Editor")
            font_size_ui = st.slider("Kích thước chữ (px)", 12, 200, 86)
        with col_txt2:
            font_name = st.selectbox(
                "Kiểu Font chữ:",
                ["Segoe UI", "Arial", "Times New Roman", "Courier New", "Calibri", "Tahoma", "Verdana"]
            )
            text_color = st.color_picker("Màu chữ", "#FF0000")

        hex_c = text_color.lstrip('#')
        rgb_color = tuple(int(hex_c[i:i + 2], 16) for i in (0, 2, 4))

        canvas_bg = Image.open(INPUT_PATH)
        bg_width, bg_height = canvas_bg.size
        disp_width = min(bg_width, 700)
        disp_height = int(bg_height * (disp_width / bg_width))

        # Tỉ lệ quy đổi giữa Canvas web và Ảnh thực tế
        scale_ratio = bg_width / disp_width
        actual_font_size = int(font_size_ui * scale_ratio)

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

        pos_x, pos_y = 50, 50
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            last_point = canvas_result.json_data["objects"][-1]
            pos_x = int(last_point["left"] * scale_ratio)
            pos_y = int(last_point["top"] * scale_ratio)
            st.success(f"📍 Tọa độ chọn: X = `{pos_x}px`, Y = `{pos_y}px` | Kích thước font thực tế: `{actual_font_size}px`")

        if st.button("✨ Áp Dụng Thêm Chữ"):
            if not input_text.strip():
                st.warning("Vui lòng nhập nội dung chữ!")
            else:
                add_text_to_image(
                    INPUT_PATH,
                    OUTPUT_PATH,
                    text=input_text,
                    position=(pos_x, pos_y),
                    font_name=font_name,
                    font_size=actual_font_size,  # Truyền kích thước chuẩn đã tính tỉ lệ
                    color=rgb_color
                )
                update_input_image_and_refresh()
                st.success("Đã thêm chữ thành công!")
                st.rerun()

    # --- TAB 4: RESIZE & UPSCALE ---
    with tab4:
        st.markdown("### Phóng to / Thay đổi kích thước")
        resize_type = st.radio("Phương pháp:", ["Resize Chuẩn", "AI Super Resolution (Upscale)"])
        if resize_type == "Resize Chuẩn":
            w = st.number_input("Chiều rộng (px)", value=image.width)
            h = st.number_input("Chiều cao (px)", value=image.height)
            if st.button("Thực hiện Resize"):
                resize_standard(INPUT_PATH, OUTPUT_PATH, width=int(w), height=int(h))
                update_input_image_and_refresh()
                st.rerun()
        else:
            scale = st.selectbox("Tỉ lệ phóng to:", [2, 4])
            if st.button("Phóng to bằng AI"):
                with st.spinner("Đang nâng cấp chất lượng..."):
                    resize_ai_upscale(INPUT_PATH, OUTPUT_PATH, scale_factor=scale)
                    update_input_image_and_refresh()
                    st.rerun()

    # --- TAB 5: TÁCH NỀN AI ---
    with tab5:
        st.markdown("### Tách nền tự động bằng AI (rembg)")
        if st.button("Bắt đầu Tách Nền"):
            with st.spinner("Đang tách nền..."):
                try:
                    remove_background_ai(INPUT_PATH, OUTPUT_PATH)
                    update_input_image_and_refresh()
                    st.success("Tách nền thành công!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi tách nền: {e}")

        # --- TAB 6: XOAY, LẬT & CẮT ÁNH (ĐÃ FIX TRIỆT ĐỂ LỖI CANVAS CẮT ÁNH) ---
        with tab6:
            st.markdown("### 1. Xoay và lật hướng ảnh")
            c_rot1, c_rot2 = st.columns([3, 1])
            with c_rot1:
                action = st.selectbox("Chọn thao tác xoay/lật:", [
                    ("Xoay phải 90°", "rotate_right"),
                    ("Xoay trái 90°", "rotate_left"),
                    ("Lật ngang", "flip_horizontal"),
                    ("Lật dọc", "flip_vertical")
                ], format_func=lambda x: x[0])
            with c_rot2:
                st.write(" ")
                st.write(" ")
                if st.button("🔄 Thực hiện Xoay/Lật"):
                    rotate_or_flip_image(INPUT_PATH, OUTPUT_PATH, action=action[1])
                    st.session_state["crop_coords_store"] = None  # Reset khung cắt sau khi xoay
                    update_input_image_and_refresh()
                    st.rerun()

            st.markdown("---")
            st.markdown("### 2. Cắt ảnh (Crop Image)")

            crop_mode = st.radio(
                "Chọn chế độ cắt ảnh:",
                ["Tỉ lệ cố định (Aspect Ratio)", "Kéo thả trực quan (Canvas Drag)", "Tùy chỉnh Tọa độ / Pixel"],
                horizontal=True
            )

            curr_img = Image.open(INPUT_PATH)
            orig_w, orig_h = curr_img.size

            # CHẾ ĐỘ 1: TỈ LỆ CỐ ĐỊNH
            if crop_mode == "Tỉ lệ cố định (Aspect Ratio)":
                ratio_option = st.selectbox(
                    "Chọn tỉ lệ cắt:",
                    ["1:1 (Vuông)", "4:3 (Chuẩn)", "16:9 (Màn hình rộng)", "3:4 (Chân dung)", "9:16 (Story/Reels)"]
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

                st.info(f"📐 Kích thước vùng cắt đề xuất: `{new_w} x {new_h} px`")

                if st.button("✂️ Áp dụng Cắt theo tỉ lệ"):
                    crop_image(INPUT_PATH, OUTPUT_PATH, (left, top, right, bottom))
                    st.session_state["crop_coords_store"] = None
                    update_input_image_and_refresh()
                    st.rerun()

            # CHẾ ĐỘ 2: KÉO THẢ TRỰC QUAN (FIXED)
            elif crop_mode == "Kéo thả trực quan (Canvas Drag)":
                st.info("👉 **Hướng dẫn:** Đè giữ chuột và kéo thành một hình chữ nhật trên ảnh.")
                disp_w = min(orig_w, 700)
                disp_h = int(orig_h * (disp_w / orig_w))

                # Render Canvas Component
                crop_canvas = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.2)",
                    stroke_color="#FF0000",
                    stroke_width=2,
                    background_image=curr_img,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="rect",
                    key="crop_canvas_widget_fixed",
                )

                # Lấy thông tin vùng kéo thả
                if crop_canvas.json_data is not None:
                    objects = crop_canvas.json_data.get("objects", [])

                    # Tìm hình chữ nhật vẽ gần nhất trong danh sách objects
                    rect_objects = [obj for obj in objects if obj.get("type") == "rect"]

                    if rect_objects:
                        last_rect = rect_objects[-1]
                        scale_x = orig_w / disp_w
                        scale_y = orig_h / disp_h

                        raw_left = last_rect["left"]
                        raw_top = last_rect["top"]
                        raw_w = last_rect["width"]
                        raw_h = last_rect["height"]

                        x1 = int(raw_left * scale_x)
                        y1 = int(raw_top * scale_y)
                        x2 = int((raw_left + raw_w) * scale_x)
                        y2 = int((raw_top + raw_h) * scale_y)

                        left_px = max(0, min(x1, x2))
                        right_px = min(orig_w, max(x1, x2))
                        top_px = max(0, min(y1, y2))
                        bottom_px = min(orig_h, max(y1, y2))

                        w_box = right_px - left_px
                        h_box = bottom_px - top_px

                        if w_box > 10 and h_box > 10:
                            # Lưu ngay vào session_state để không bị mất khi rerun
                            st.session_state["crop_coords_store"] = (left_px, top_px, right_px, bottom_px)

                # Kiểm tra xem đã có tọa độ lưu trữ chưa
                saved_coords = st.session_state.get("crop_coords_store")

                if saved_coords:
                    lx, ty, rx, by = saved_coords
                    st.success(
                        f"📍 **Đã ghi nhận khung cắt:** X1=`{lx}px`, Y1=`{ty}px` | Kích thước: `{rx - lx} x {by - ty} px`")
                else:
                    st.warning(
                        "⚠️ Chưa nhận diện được khung chọn. Vui lòng đè giữ chuột và kéo một ô hình chữ nhật trên ảnh.")

                if st.button("✂️ Cắt Vùng Đã Chọn"):
                    if saved_coords:
                        crop_image(INPUT_PATH, OUTPUT_PATH, saved_coords)
                        st.session_state["crop_coords_store"] = None  # Xóa cache tọa độ sau khi cắt
                        update_input_image_and_refresh()
                        st.success("Đã cắt ảnh thành công!")
                        st.rerun()
                    else:
                        st.error("❌ Không thể cắt! Vui lòng thao tác kéo chọn khung hình chữ nhật trên ảnh trước.")

            # CHẾ ĐỘ 3: TÙY CHỈNH PIXEL
            else:
                col_cr1, col_cr2 = st.columns(2)
                with col_cr1:
                    crop_x = st.number_input("Tọa độ X bắt đầu", 0, max(0, orig_w - 1), 0)
                    crop_y = st.number_input("Tọa độ Y bắt đầu", 0, max(0, orig_h - 1), 0)
                with col_cr2:
                    crop_w = st.number_input("Chiều rộng (W)", 1, orig_w - crop_x, orig_w)
                    crop_h = st.number_input("Chiều cao (H)", 1, orig_h - crop_y, orig_h)

                if st.button("✂️ Áp dụng Cắt theo Pixel"):
                    box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
                    crop_image(INPUT_PATH, OUTPUT_PATH, box)
                    st.session_state["crop_coords_store"] = None
                    update_input_image_and_refresh()
                    st.rerun()

    # --- KẾT QUẢ HIỂN THỊ (CỘT 2) ---
    with col2:
        st.subheader("Kết Quả")
        if "processed_img" in st.session_state and os.path.exists(st.session_state["processed_img"]):
            res_img = Image.open(st.session_state["processed_img"])
            st.image(res_img, use_container_width=True)

            with open(st.session_state["processed_img"], "rb") as file:
                st.download_button(
                    label="💾 Tải Ảnh Kết Quả Về Máy",
                    data=file,
                    file_name="edited_image.png",
                    mime="image/png"
                )
else:
    st.info("Vui lòng tải một tệp ảnh lên từ thanh bên (Sidebar) để bắt đầu chỉnh sửa.")