# app.py

import streamlit as st
import cv2
import os
import tempfile
from pathlib import Path
import torch # <-- Diperlukan untuk memuat model YOLOv5
import subprocess
from huggingface_hub import hf_hub_download

# --- PENGATURAN HALAMAN & MODEL ---

st.set_page_config(
    page_title="Deteksi APD dengan YOLOv5",
    page_icon="👷",
    layout="wide"
)

# Fungsi untuk mengunduh dan memuat model YOLOv5
@st.cache_resource
def load_model():
    """
    Mengunduh model YOLOv5 dari Hugging Face Hub dan memuatnya menggunakan torch.hub.
    Ini adalah cara yang benar untuk model YOLOv5.
    """
    model_path = "yolov5s_safety.pt"
    # Menggunakan model YOLOv5 yang terbukti
    repo_id = "keremberke/yolov5m-construction-safety"
    filename = "yolov5m.pt"
    
    if not os.path.exists(model_path):
        st.info(f"Mengunduh model '{filename}' dari repo '{repo_id}'...")
        try:
            # Mengunduh bobot model
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir='.',
                local_dir_use_symlinks=False,
                # Mengganti nama file yang diunduh agar sesuai dengan path
                local_dir_use_symlinks=False,
                rename=model_path
            )
            st.success("Model berhasil diunduh.")
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            return None
    
    try:
        # ===== PERUBAHAN UTAMA: MEMUAT MODEL DENGAN TORCH.HUB =====
        # Menggunakan repositori 'ultralytics/yolov5' untuk kerangka kerja
        # dan memuat bobot kustom dari file yang diunduh
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        st.success("Model YOLOv5 berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dengan PyTorch Hub: {e}")
        return None


# Memuat model saat aplikasi dimulai
model = load_model()

# --- ANTARMUKA APLIKASI STREAMLIT ---

st.title("👷 Aplikasi Deteksi Alat Pelindung Diri (APD) - YOLOv5")
st.markdown("""
Aplikasi ini menggunakan model **YOLOv5** untuk mendeteksi peralatan keselamatan kerja (_helmet_, _vest_, _person_) 
dari video yang Anda unggah.
""")

st.sidebar.header("Pengaturan")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file video...", 
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    st.sidebar.info(f"Video yang diunggah: **{uploaded_file.name}**")
    st.video(uploaded_file)

    if st.button("Mulai Deteksi pada Video Ini"):
        if model is None:
            st.error("Model tidak berhasil dimuat. Tidak dapat melanjutkan proses deteksi.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name

            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = Path(temp_dir) / "frames"
                detected_frames_dir = Path(temp_dir) / "frames_detected"
                frames_dir.mkdir()
                detected_frames_dir.mkdir()
                output_video_path = os.path.join(temp_dir, "video_detected.mp4")

                with st.spinner("Sedang memproses video... Harap tunggu."):
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    st.info(f"Langkah 1/3: Mengekstrak {total_frames} frame dari video...")
                    
                    frame_count = 0
                    while cap.isOpened():
                        success, frame = cap.read()
                        if not success: break
                        cv2.imwrite(str(frames_dir / f"frame_{frame_count:04d}.jpg"), frame)
                        frame_count += 1
                    cap.release()
                    st.success("Ekstraksi frame selesai.")

                    st.info("Langkah 2/3: Melakukan deteksi APD...")
                    progress_bar = st.progress(0, text="Progress Deteksi")
                    frame_files = sorted(list(frames_dir.glob("*.jpg")))
                    
                    for i, frame_path in enumerate(frame_files):
                        # ===== PERUBAHAN LOGIKA DETEKSI UNTUK YOLOv5 =====
                        frame_bgr = cv2.imread(str(frame_path))
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        results = model(frame_rgb)
                        
                        # Visualisasi dengan results.render()
                        rendered_image = results.render()[0] 
                        output_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
                        
                        output_path = detected_frames_dir / frame_path.name
                        cv2.imwrite(str(output_path), output_image)
                        
                        progress_bar.progress((i + 1) / len(frame_files), text=f"Memproses frame {i+1}/{len(frame_files)}")

                    st.success("Deteksi pada semua frame selesai.")

                    st.info("Langkah 3/3: Menyusun kembali video...")
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    ffmpeg_command = [
                        "ffmpeg", "-framerate", str(fps), "-i", f"{detected_frames_dir}/frame_%04d.jpg",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", output_video_path
                    ]
                    
                    try:
                        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                        st.success("Video hasil deteksi berhasil dibuat!")
                        st.header("Hasil Deteksi")
                        st.video(output_video_path)
                        with open(output_video_path, "rb") as file:
                            st.download_button(
                                label="Unduh Video Hasil Deteksi", data=file,
                                file_name=f"hasil_deteksi_{uploaded_file.name}", mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"Gagal membuat video dengan FFMPEG. Error: {e}")

            os.unlink(video_path)
else:
    st.info("Silakan unggah file video melalui panel di sebelah kiri untuk memulai.")
