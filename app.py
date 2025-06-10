# app.py

import streamlit as st
import cv2
import os
import tempfile
from pathlib import Path
from ultralytics import YOLO
import subprocess
from huggingface_hub import hf_hub_download

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
Aplikasi ini menggunakan model **YOLOv8** untuk mendeteksi peralatan keselamatan kerja (_Hardhat_, _Person_, dll.) 
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
                    st.info(f"Langkah 1/3: Mengekstrak frame dari video...")
                    
                    frame_count = 0
                    while cap.isOpened():
                        success, frame = cap.read()
                        if not success: break
                        cv2.imwrite(str(frames_dir / f"frame_{frame_count:04d}.jpg"), frame)
                        frame_count += 1
                    cap.release()
                    st.success(f"Ekstraksi {frame_count} frame selesai.")

                    st.info("Langkah 2/3: Melakukan deteksi APD...")
                    progress_bar = st.progress(0, text="Progress Deteksi")
                    frame_files = sorted(list(frames_dir.glob("*.jpg")))
                    
                    for i, frame_path in enumerate(frame_files):
                        results = model(str(frame_path), verbose=False)
                        result_plotted = results[0].plot()
                        output_path = detected_frames_dir / frame_path.name
                        cv2.imwrite(str(output_path), result_plotted)
                        progress_bar.progress((i + 1) / len(frame_files), text=f"Memproses frame {i+1}/{len(frame_files)}")

                    st.success("Deteksi pada semua frame selesai.")

                    st.info("Langkah 3/3: Menyusun kembali video dengan FFMPEG...")
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    ffmpeg_command = [
                        "ffmpeg", "-framerate", str(fps), "-i", f"{detected_frames_dir}/frame_%04d.jpg",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", output_video_path
                    ]
                    
                    try:
                        # ========= PERUBAHAN UTAMA DI SINI: PENANGANAN ERROR LEBIH BAIK =========
                        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                        
                        # Verifikasi file output setelah FFMPEG berjalan
                        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                            st.success("Video hasil deteksi berhasil dibuat!")
                            st.header("Hasil Deteksi")
                            st.video(output_video_path)
                            with open(output_video_path, "rb") as file:
                                st.download_button(
                                    label="Unduh Video Hasil Deteksi", data=file,
                                    file_name=f"hasil_deteksi_{uploaded_file.name}", mime="video/mp4"
                                )
                        else:
                            st.error("FFMPEG berjalan, namun file video gagal dibuat atau kosong. Silakan periksa log di bawah.")
                            # Jika ada output standar dari ffmpeg, tampilkan
                            if process.stdout:
                                st.code(process.stdout, language='bash')
                            # Jika ada error standar dari ffmpeg, tampilkan
                            if process.stderr:
                                st.code(process.stderr, language='bash')

                    except subprocess.CalledProcessError as e:
                        # Jika FFMPEG berhenti dengan error, tampilkan pesan error spesifiknya
                        st.error("Gagal menjalankan FFMPEG. Pesan error:")
                        st.code(e.stderr, language='bash')
                    except FileNotFoundError:
                        st.error("Perintah 'ffmpeg' tidak ditemukan. Pastikan FFMPEG terinstall di lingkungan Anda.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan tak terduga saat membuat video: {e}")

            os.unlink(video_path)
else:
    st.info("Silakan unggah file video melalui panel di sebelah kiri untuk memulai.")
```

Setelah Anda mengganti kodenya, coba jalankan kembali. Jika masih gagal, sekarang aplikasi akan memberikan pesan error yang **jauh lebih spesifik**. Tolong **salin dan kirimkan pesan error baru** yang muncul di dalam kotak abu-abu tersebut. Itu akan menjadi kunci untuk menyelesaikan masalah ini secara tunt
