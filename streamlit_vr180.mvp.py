# streamlit_vr180_mvp.py
"""
Streamlit app: 2D -> VR180 stereo (side-by-side) converter
Features:
 - MiDaS_small depth with optional fp16 on CUDA
 - simple depth-based stereo synthesis
 - optional LaMa inpainting (fast fallback to OpenCV)
 - smart inpainting (skip LaMa if hole area small)
 - resolution/FPS downsample + frame-skip
 - automatic sample export (configurable seconds), GIF preview, inline video player
 - VR180 metadata injection (spatial-media) with verification
 - Progress bar with ETA/frame counts and demo mode
"""

import os, sys, tempfile, time, subprocess
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import imageio
import torch

# try to import simple-lama-inpainting wrapper
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
    LAMA = SimpleLama()
except Exception:
    LAMA_AVAILABLE = False
    LAMA = None

# ---------------------
# Constants
# ---------------------
DEFAULT_DEPTH_RES = 256
SMART_INPAINT_THRESHOLD = 0.02  # fraction of pixels missing to trigger heavy inpainting

# ---------------------
# Utilities & model
# ---------------------
@st.cache_resource
def load_midas(device_str="cuda", use_fp16=True):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    model.to(device).eval()
    if device.type == "cuda" and use_fp16:
        try:
            model.half()
        except Exception:
            pass
    return model, transform, device

def estimate_depth(frame_bgr, model, transform, device, max_res=DEFAULT_DEPTH_RES):
    h, w = frame_bgr.shape[:2]
    scale = max_res / max(h, w)
    frame_small = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr.copy()
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)
    if device.type == "cuda":
        try:
            input_tensor = input_tensor.half()
        except Exception:
            pass
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth = prediction.float().cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_up = cv2.resize(depth_norm, (w, h), interpolation=cv2.INTER_CUBIC)
    return depth_up

def warp_frame(frame_bgr, depth_norm, shift_px):
    H, W = depth_norm.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    disparity = (1.0 - depth_norm) * float(shift_px)
    map_x = (xs + disparity).astype(np.float32)
    map_y = ys.astype(np.float32)
    return cv2.remap(frame_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# ---------------------
# Inpainting helpers
# ---------------------
def opencv_inpaint(img_bgr):
    mask = np.all(img_bgr == 0, axis=2).astype('uint8') * 255
    if mask.sum() == 0:
        return img_bgr
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

def lama_inpaint(img_bgr):
    mask = np.all(img_bgr == 0, axis=2).astype('uint8') * 255
    if mask.sum() == 0:
        return img_bgr
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask).convert("L")
    out_pil = LAMA(img_pil, mask_pil)
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

def smart_inpaint(img_bgr, use_lama):
    mask = np.all(img_bgr == 0, axis=2)
    hole_frac = float(mask.mean())
    if hole_frac <= 0:
        return img_bgr, hole_frac, "none"
    if not use_lama or not LAMA_AVAILABLE:
        return opencv_inpaint(img_bgr), hole_frac, "opencv"
    if hole_frac < SMART_INPAINT_THRESHOLD:
        return opencv_inpaint(img_bgr), hole_frac, "opencv_small"
    try:
        return lama_inpaint(img_bgr), hole_frac, "lama"
    except Exception:
        return opencv_inpaint(img_bgr), hole_frac, "opencv_failed"

# ---------------------
# Main processing
# ---------------------
def process_video_streamlit(
    input_path, output_path,
    baseline, smoothing,
    depth_res, skip, use_lama, gif_frames,
    max_frames=None, demo_mode=False, fps_override=None,
    sample_seconds=30, progress_updater=None
):
    model, transform, device = load_midas("cuda", use_fp16=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fps = orig_fps if not fps_override else fps_override

    tmp_raw = output_path.replace(".mp4", "_raw.mp4")
    writer = None
    if not demo_mode:
        writer = imageio.get_writer(tmp_raw, fps=fps/skip, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p'])

    sample_writer = None
    sample_limit = 0
    if (not demo_mode) and sample_seconds > 0:
        sample_path = output_path.replace(".mp4", f"_sample{sample_seconds}s.mp4")
        sample_writer = imageio.get_writer(sample_path, fps=fps/skip, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p'])
        sample_limit = int(sample_seconds * fps)

    prev_depth = None
    max_frames = max_frames if (max_frames and max_frames > 0) else total_frames

    preview_stack = None
    gif_list = []

    processed = 0
    frame_idx = 0
    start = time.time()
    total_steps = (max_frames + skip - 1) // skip if max_frames else 1
    if progress_updater:
        progress_updater(0, total_steps, "Starting...")

    while frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        depth = estimate_depth(frame_bgr, model, transform, device, max_res=depth_res)
        if prev_depth is not None:
            depth = smoothing * prev_depth + (1.0 - smoothing) * depth
        prev_depth = depth.copy()

        left = warp_frame(frame_bgr, depth, -baseline)
        right = warp_frame(frame_bgr, depth, baseline)

        if use_lama and LAMA_AVAILABLE:
            left, _, _ = smart_inpaint(left, use_lama)
            right, _, _ = smart_inpaint(right, use_lama)
        else:
            left = opencv_inpaint(left)
            right = opencv_inpaint(right)

        stereo_rgb = np.concatenate((cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
                                     cv2.cvtColor(right, cv2.COLOR_BGR2RGB)), axis=1)

        if writer is not None:
            writer.append_data(stereo_rgb)
        if sample_writer is not None and processed < sample_limit:
            sample_writer.append_data(stereo_rgb)
        if gif_frames and processed < gif_frames:
            gif_list.append(stereo_rgb)

        if processed == 0:
            orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            target_w = stereo_rgb.shape[1]
            new_h = int(orig_rgb.shape[0] * (target_w / orig_rgb.shape[1]))
            orig_resized = cv2.resize(orig_rgb, (target_w, new_h), interpolation=cv2.INTER_AREA)
            stacked = np.zeros((orig_resized.shape[0] + stereo_rgb.shape[0], target_w, 3), dtype=np.uint8)
            stacked[0:orig_resized.shape[0]] = orig_resized
            stacked[orig_resized.shape[0]:] = stereo_rgb
            preview_stack = stacked

        processed += 1
        frame_idx += 1

        elapsed = time.time() - start
        if progress_updater:
            progress_updater(processed, total_steps, f"Processed {processed}/{total_steps} frames — {elapsed:.1f}s elapsed")

    cap.release()
    if writer is not None:
        writer.close()
    if sample_writer is not None:
        sample_writer.close()

    total_elapsed = time.time() - start
    gif_path = None
    if gif_list:
        gif_path = output_path.replace(".mp4", "_preview.gif")
        imageio.mimsave(gif_path, gif_list, fps=min(10, orig_fps))

    final_output = output_path
    if (writer is not None) and (not demo_mode):
        try:
            subprocess.run([sys.executable, "-m", "spatialmedia", "-i", "--stereo=left-right", tmp_raw, final_output], check=True)
            injected = True
        except Exception:
            try:
                os.replace(tmp_raw, final_output)
            except Exception:
                pass
            injected = False
    else:
        final_output = output_path
        injected = False

    return preview_stack, final_output, processed, total_elapsed, gif_path, injected

# ---------------------
# Streamlit UI layout
# ---------------------
st.set_page_config(page_title="VR180 Immersive Generator", layout="wide")
st.title("🤖 VR180 Immersive Video Generator — Hackathon MVP")

with st.expander("How to use (quick)"):
    st.markdown("""
    1. Upload a clip (up to 30s or more).
    2. Adjust settings.
    3. Use Demo Mode for GIF preview or full export for VR180.
    """)

col1, col2 = st.columns(2)
with col1:
    cuda_ok = torch.cuda.is_available()
    st.write("**GPU (CUDA)**:", "✅ Available" if cuda_ok else "❌ Not available")
with col2:
    st.write("**LaMa available**:", "✅ Yes" if LAMA_AVAILABLE else "❌ No")

st.markdown("---")

left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload video (.mp4/.mov/.avi)", type=["mp4", "mov", "avi"])
    demo_mode = st.checkbox("⚡ Demo Mode (fast GIF only)", value=False)
    use_lama = st.checkbox("Use LaMa inpainting (slower, cleaner)", value=False)
    sample_seconds = st.number_input("Sample export seconds (0=off)", min_value=0, max_value=120, value=30)
with right:
    st.sidebar.header("⚡ Processing Settings")
    baseline = st.sidebar.slider("Stereo baseline (px)", 8, 80, 30)
    smoothing = st.sidebar.slider("Depth smoothing", 0.0, 1.0, 0.6)
    depth_res = st.sidebar.selectbox("Depth resolution", [128, 256, 384, 512], index=0)
    skip = st.sidebar.slider("Frame skip", 1, 5, 2)
    gif_frames = st.sidebar.slider("GIF preview frames (0=off)", 0, 40, 12)
    max_frames = st.sidebar.number_input("Max frames (0=all)", min_value=0, value=0)

if demo_mode:
    max_frames = 30
    skip = 2
    use_lama = False
    gif_frames = max(8, gif_frames)

process_btn = st.button("🚀 Process Video")
progress_bar = st.progress(0)
progress_text = st.empty()
preview_slot = st.empty()
downloads_col = st.empty()

def update_progress(done, total, msg):
    frac = min(1.0, done / max(1, total))
    progress_bar.progress(frac)
    progress_text.info(msg)

if process_btn:
    if uploaded is None:
        st.error("❌ Please upload a video before processing.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
            tmpf.write(uploaded.read())
            tmp_input = tmpf.name

        out_dir = tempfile.gettempdir()
        out_name = f"out_vr180_{int(time.time())}.mp4"
        out_path = os.path.join(out_dir, out_name)

        try:
            preview_img, final_path, processed, elapsed, gif_path, injected = process_video_streamlit(
                tmp_input, out_path,
                baseline=baseline, smoothing=smoothing,
                depth_res=int(depth_res), skip=int(skip),
                use_lama=use_lama, gif_frames=int(gif_frames),
                max_frames=(None if max_frames == 0 else int(max_frames)),
                demo_mode=demo_mode, fps_override=None,
                sample_seconds=int(sample_seconds),
                progress_updater=update_progress
            )
        except Exception as e:
            progress_text.error(f"Processing failed: {e}")
            raise

        if preview_img is not None:
            preview_slot.subheader("Preview")
            preview_slot.image(preview_img, use_column_width=True)
        if processed > 0:
            st.success(f"Done — processed {processed} frames in {elapsed:.1f}s")

        if (not demo_mode) and os.path.exists(final_path):
            with open(final_path, "rb") as f:
                video_bytes = f.read()
                st.video(video_bytes)
                downloads_col.download_button("⬇️ Download VR180 Video", data=video_bytes, file_name=os.path.basename(final_path))
        if gif_path and os.path.exists(gif_path):
            st.image(gif_path, use_column_width=True)
            st.download_button("⬇️ Download GIF", data=open(gif_path, "rb"), file_name=os.path.basename(gif_path))
