# app.py

import streamlit as st
import cv2
import os
import tempfile
from pathlib import Path
from ultralytics import YOLO
import subprocess
from huggingface_hub import hf_hub_download
import shutil

# --- PENGATURAN HALAMAN & MODEL ---

st.set_page_config(
    page_title="Deteksi APD dengan YOLOv8",
    page_icon="👷",
    layout="wide"
)

# Fungsi untuk mengunduh dan memuat model YOLOv8
@st.cache_resource
def load_model():
    """
    Mengunduh model YOLOv8 dari Hugging Face Hub dan memuatnya.
    """
    model_path = "yolov8_hardhat.pt"
    repo_id = "jancodr/YOLOv8-Hardhat-Detection"
    filename = "best.pt"
    
    if not os.path.exists(model_path):
        st.info(f"Mengunduh model '{filename}' dari repo '{repo_id}'...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir='.',
                local_dir_use_symlinks=False,
                rename=model_path
            )
            st.success("Model berhasil diunduh.")
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            return None
    
    try:
        model = YOLO(model_path)
        st.success("Model YOLOv8 berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLOv8: {e}")
        return None

# Memuat model saat aplikasi dimulai
model = load_model()

# --- ANTARMUKA APLIKASI STREAMLIT ---

st.title("👷 Aplikasi Deteksi Alat Pelindung Diri (APD) - YOLOv8")
st.markdown("""
Aplikasi ini menggunakan model **YOLOv8** untuk mendeteksi peralatan keselamatan kerja dari video yang Anda unggah.
""")

st.sidebar.header("Pengaturan")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file video...", 
    type=["mp4", "mov", "avi", "mkv", "webm"]
)

if uploaded_file is not None:
    # Buat direktori sementara untuk semua operasi
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_file.name)

    # Tulis file yang diunggah ke path sementara
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.info(f"Video '{uploaded_file.name}' berhasil diunggah.")
    
    # Tidak menampilkan video asli dulu untuk menyederhanakan
    # st.video(uploaded_file) 
    
    if st.button("Mulai Deteksi"):
        if model is None:
            st.error("Model tidak berhasil dimuat. Tidak dapat melanjutkan.")
        else:
            with st.spinner("Sedang memproses video..."):
                # Definisikan path untuk output
                frames_dir = Path(temp_dir) / "frames"
                detected_frames_dir = Path(temp_dir) / "frames_detected"
                frames_dir.mkdir()
                detected_frames_dir.mkdir()
                output_video_path = os.path.join(temp_dir, "video_hasil_deteksi.mp4")

                # Langkah 1: Ekstraksi Frame
                st.info("Langkah 1/3: Mengekstrak frame...")
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success: break
                    cv2.imwrite(str(frames_dir / f"frame_{frame_count:04d}.jpg"), frame)
                    frame_count += 1
                cap.release()
                st.success(f"Ekstraksi {frame_count} frame selesai.")

                # Langkah 2: Deteksi Objek
                st.info("Langkah 2/3: Melakukan deteksi objek...")
                progress_bar = st.progress(0)
                frame_files = sorted(list(frames_dir.glob("*.jpg")))
                for i, frame_path in enumerate(frame_files):
                    results = model(str(frame_path), verbose=False)
                    result_plotted = results[0].plot()
                    cv2.imwrite(str(detected_frames_dir / frame_path.name), result_plotted)
                    progress_bar.progress((i + 1) / len(frame_files))
                st.success("Deteksi objek selesai.")

                # Langkah 3: Rekonstruksi Video
                st.info("Langkah 3/3: Membuat video hasil deteksi...")
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                reconstruct_command = [
                    "ffmpeg", "-framerate", str(fps), "-i", f"{detected_frames_dir}/frame_%04d.jpg",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                    "-movflags", "+faststart", # Flag untuk optimasi web
                    "-y", output_video_path
                ]
                
                try:
                    process = subprocess.run(reconstruct_command, check=True, capture_output=True, text=True)
                    st.success("FFMPEG selesai dijalankan.")

                    if os.path.exists(output_video_path):
                        st.info("File video hasil ditemukan. Mencoba menampilkan...")
                        with open(output_video_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.header("Hasil Deteksi")
                        st.video(video_bytes)
                        st.download_button("Unduh Video Hasil", video_bytes, f"deteksi_{uploaded_file.name}")
                    else:
                        st.error("FFMPEG berjalan tanpa error, tetapi file video output tidak ditemukan.")
                        st.code(process.stdout, language='bash')
                        
                except subprocess.CalledProcessError as e:
                    st.error("Gagal menjalankan FFMPEG. Pesan error:")
                    st.code(e.stderr, language='bash')
                except Exception as e:
                    st.error(f"Terjadi kesalahan tak terduga: {e}")

            # Bersihkan direktori sementara setelah selesai
            shutil.rmtree(temp_dir)
else:
    st.info("Silakan unggah file video untuk memulai.")
