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
    type=["mp4", "mov", "avi", "mkv", "webm"]
)

if uploaded_file is not None:
    # Simpan file yang diunggah ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tfile:
        tfile.write(uploaded_file.read())
        uploaded_video_path = tfile.name

    # Path untuk video yang dikonversi
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as converted_tfile:
        converted_video_path = converted_tfile.name

    st.sidebar.info(f"Video yang diunggah: **{uploaded_file.name}**")
    
    # ===== LANGKAH BARU: KONVERSI VIDEO UNTUK KOMPATIBILITAS WEB =====
    try:
        with st.spinner("Mengonversi video agar kompatibel dengan web..."):
            ffmpeg_command = [
                "ffmpeg", "-i", uploaded_video_path,
                "-c:v", "libx264", "-crf", "23",
                "-preset", "veryfast", "-c:a", "aac",
                "-pix_fmt", "yuv420p", "-y", converted_video_path
            ]
            process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        st.success("Video berhasil dikonversi.")
        
        # Tampilkan video yang sudah dikonversi
        st.video(converted_video_path)
        
        # Tombol untuk memulai proses deteksi
        if st.button("Mulai Deteksi pada Video Ini"):
            # Proses deteksi sekarang menggunakan video yang sudah dikonversi
            video_path = converted_video_path 
            
            if model is None:
                st.error("Model tidak berhasil dimuat. Tidak dapat melanjutkan proses deteksi.")
            else:
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
                        
                        reconstruct_command = [
                            "ffmpeg", "-framerate", str(fps), "-i", f"{detected_frames_dir}/frame_%04d.jpg",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", output_video_path
                        ]
                        
                        try:
                            subprocess.run(reconstruct_command, check=True, capture_output=True, text=True)
                            st.success("Video hasil deteksi berhasil dibuat!")
                            st.header("Hasil Deteksi")
                            st.video(output_video_path)
                            with open(output_video_path, "rb") as file:
                                st.download_button(
                                    label="Unduh Video Hasil Deteksi", data=file,
                                    file_name=f"hasil_deteksi_{uploaded_file.name}", mime="video/mp4"
                                )
                        except subprocess.CalledProcessError as e:
                            st.error("Gagal membuat video hasil deteksi. Pesan error:")
                            st.code(e.stderr, language='bash')

    except subprocess.CalledProcessError as e:
        st.error("Gagal mengonversi video. Format video mungkin tidak didukung atau file rusak.")
        st.code(e.stderr, language='bash')
    except FileNotFoundError:
        st.error("Perintah 'ffmpeg' tidak ditemukan. Pastikan FFMPEG terinstall di lingkungan Anda.")
    finally:
        # Selalu hapus file sementara
        if os.path.exists(uploaded_video_path):
            os.unlink(uploaded_video_path)
        if os.path.exists(converted_video_path):
            os.unlink(converted_video_path)

else:
    st.info("Silakan unggah file video melalui panel di sebelah kiri untuk memulai.")
