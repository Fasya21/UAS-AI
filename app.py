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

# Inisialisasi session state
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

if uploaded_file is not None:
    # Buat direktori sementara untuk semua operasi
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    temp_dir = st.session_state.temp_dir
    video_path = os.path.join(temp_dir, uploaded_file.name)

    # Tulis file yang diunggah ke path sementara
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.info(f"Video '{uploaded_file.name}' berhasil diunggah.")
    
    if st.button("Mulai Deteksi"):
        if model is None:
            st.error("Model tidak berhasil dimuat. Tidak dapat melanjutkan.")
        else:
            with st.spinner("Sedang memproses video..."):
                # Definisikan path untuk output
                frames_dir = Path(temp_dir) / "frames"
                detected_frames_dir = Path(temp_dir) / "frames_detected"
                frames_dir.mkdir(exist_ok=True)
                detected_frames_dir.mkdir(exist_ok=True)
                output_video_path = os.path.join(temp_dir, "video_hasil_deteksi.mp4")

                # Langkah 1: Ekstraksi Frame
                st.info("Langkah 1/3: Mengekstrak frame...")
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                fps = cap.get(cv2.CAP_PROP_FPS)
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success: break
                    cv2.imwrite(str(frames_dir / f"frame_{frame_count:04d}.jpg"), frame)
                    frame_count += 1
                cap.release()
                st.success(f"Ekstraksi {frame_count} frame selesai. FPS: {fps:.2f}")

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

                # Langkah 3: Rekonstruksi Video (Dengan pengaturan kompatibilitas)
                st.info("Langkah 3/3: Membuat video hasil deteksi...")
                
                reconstruct_command = [
                    "ffmpeg",
                    "-y",  # Overwrite output file
                    "-framerate", str(fps),
                    "-i", f"{detected_frames_dir}/frame_%04d.jpg",
                    "-c:v", "libx264",
                    "-profile:v", "baseline",  # Kompatibilitas maksimal
                    "-pix_fmt", "yuv420p",
                    "-crf", "23",  # Kontrol kualitas
                    "-movflags", "+faststart", 
                    output_video_path
                ]
                
                try:
                    process = subprocess.run(
                        reconstruct_command, 
                        check=True, 
                        capture_output=True,
                        text=True
                    )
                    
                    # Verifikasi video hasil
                    if os.path.exists(output_video_path):
                        st.session_state.processed_video = output_video_path
                        st.success("Video hasil berhasil dibuat!")
                    else:
                        st.error("Proses FFmpeg selesai tetapi file video tidak ditemukan")
                        st.code(process.stderr, language='bash')
                        
                except subprocess.CalledProcessError as e:
                    st.error(f"Error FFmpeg: {e.stderr}")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")

# Tampilkan video hasil jika tersedia
if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
    st.header("Hasil Deteksi")
    
    # Tampilkan video dengan kontrol pemutaran
    with open(st.session_state.processed_video, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)
    
    # Tombol unduh
    st.download_button(
        "Unduh Video Hasil",
        video_bytes,
        f"deteksi_{os.path.basename(uploaded_file.name)}",
        mime="video/mp4"
    )
    
    # Tombol reset
    if st.button("Proses Video Baru"):
        # Hapus file sementara
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
        # Reset session state
        st.session_state.processed_video = None
        st.session_state.temp_dir = None
        st.experimental_rerun()
else:
    st.info("Silakan unggah file video dan klik 'Mulai Deteksi'")
