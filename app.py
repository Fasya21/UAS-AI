# app.py

import streamlit as st
import cv2
import os
import tempfile
from pathlib import Path
from ultralytics import YOLO
import subprocess

# --- PENGATURAN HALAMAN & MODEL ---

# Mengatur judul halaman dan ikon yang akan muncul di tab browser
st.set_page_config(
    page_title="Deteksi APD dengan YOLOv8",
    page_icon="👷",
    layout="wide"
)

# Fungsi untuk mengunduh model jika belum ada, dengan caching agar tidak diunduh berulang kali
@st.cache_resource
def load_model():
    """
    Mengunduh model YOLOv8 menggunakan wget dan memuatnya.
    Menggunakan cache Streamlit agar model hanya diunduh dan dimuat sekali per sesi.
    """
    model_path = "best.pt"
    # URL yang digunakan adalah URL yang valid agar aplikasi bisa berjalan
    model_url = "https://huggingface.co/keremberke/yolov8m-hard-hat-detection/resolve/main/best.pt"
    
    if not os.path.exists(model_path):
        st.info(f"Mengunduh model dengan wget dari URL yang valid...")
        try:
            # Membangun dan menjalankan perintah wget melalui subprocess
            command = ["wget", "-O", model_path, model_url]
            subprocess.run(command, check=True, capture_output=True, text=True)
            st.success("Model berhasil diunduh.")
        except FileNotFoundError:
            st.error("Perintah 'wget' tidak ditemukan. Metode ini mungkin tidak berfungsi di semua lingkungan. Pastikan wget terinstall.")
            return None
        except subprocess.CalledProcessError as e:
            st.error(f"Gagal mengunduh model dengan wget. URL mungkin tidak valid atau ada masalah jaringan. Error: {e.stderr}")
            return None
        except Exception as e:
            st.error(f"Terjadi kesalahan tak terduga saat mengunduh: {e}")
            return None
            
    # Memuat model YOLO dari file yang sudah diunduh
    model = YOLO(model_path)
    return model

# Memuat model saat aplikasi dimulai
model = load_model()

# --- ANTARMUKA APLIKASI STREAMLIT ---

st.title("👷 Aplikasi Deteksi Alat Pelindung Diri (APD)")
st.markdown("""
Aplikasi ini menggunakan model **YOLOv8** untuk mendeteksi peralatan keselamatan kerja seperti helm (_Hardhat_) 
dari video yang Anda unggah.
""")

# Sidebar untuk unggah file dan pengaturan
st.sidebar.header("Pengaturan")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file video...", 
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    st.sidebar.info(f"Video yang diunggah: **{uploaded_file.name}**")
    
    # Menampilkan video asli yang diunggah
    st.video(uploaded_file)

    # Tombol untuk memulai proses deteksi
    if st.button("Mulai Deteksi pada Video Ini"):
        if model is None:
            st.error("Model tidak berhasil dimuat. Tidak dapat melanjutkan proses deteksi.")
        else:
            # Menggunakan temporary file untuk menyimpan video yang diunggah
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name

            # Membuat direktori sementara untuk menyimpan frame
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = Path(temp_dir) / "frames"
                detected_frames_dir = Path(temp_dir) / "frames_detected"
                frames_dir.mkdir()
                detected_frames_dir.mkdir()

                output_video_path = os.path.join(temp_dir, "video_detected.mp4")

                with st.spinner("Sedang memproses video... Harap tunggu."):
                    # --- LANGKAH 1: EKSTRAKSI FRAME DARI VIDEO ---
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FPS) * 30) # Proses maks 30 detik pertama
                    frame_count = 0
                    st.info("Langkah 1/3: Mengekstrak frame dari video...")
                    while cap.isOpened() and frame_count < total_frames:
                        success, frame = cap.read()
                        if not success:
                            break
                        frame_path = frames_dir / f"frame_{frame_count:04d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        frame_count += 1
                    cap.release()
                    st.success(f"{frame_count} frame berhasil diekstrak.")

                    # --- LANGKAH 2: DETEKSI OBJEK PADA SETIAP FRAME ---
                    st.info("Langkah 2/3: Melakukan deteksi APD pada setiap frame...")
                    progress_bar = st.progress(0, text="Progress Deteksi")
                    frame_files = sorted(list(frames_dir.glob("*.jpg")))
                    
                    for i, frame_path in enumerate(frame_files):
                        results = model(str(frame_path), verbose=False)
                        result_plotted = results[0].plot()
                        output_path = detected_frames_dir / frame_path.name
                        cv2.imwrite(str(output_path), result_plotted)
                        
                        # Update progress bar
                        progress_bar.progress((i + 1) / len(frame_files), text=f"Memproses frame {i+1}/{len(frame_files)}")

                    st.success("Deteksi pada semua frame selesai.")

                    # --- LANGKAH 3: REKONSTRUKSI VIDEO DENGAN FFMPEG ---
                    st.info("Langkah 3/3: Menyusun kembali frame menjadi video...")
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()

                    # Perintah FFMPEG untuk hasil terbaik
                    ffmpeg_command = [
                        "ffmpeg",
                        "-framerate", str(fps),
                        "-i", f"{detected_frames_dir}/frame_%04d.jpg",
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-y",
                        output_video_path
                    ]
                    
                    try:
                        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                        st.success("Video hasil deteksi berhasil dibuat!")

                        # Menampilkan video hasil deteksi
                        st.header("Hasil Deteksi")
                        st.video(output_video_path)

                        # Tombol untuk mengunduh video hasil
                        with open(output_video_path, "rb") as file:
                            st.download_button(
                                label="Unduh Video Hasil Deteksi",
                                data=file,
                                file_name=f"hasil_deteksi_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                    except subprocess.CalledProcessError as e:
                        st.error(f"Gagal membuat video dengan FFMPEG. Error: {e.stderr}")
                    except FileNotFoundError:
                        st.error("FFMPEG tidak ditemukan. Pastikan FFMPEG terinstall di sistem Anda jika menjalankan secara lokal.")

            # Hapus file video sementara
            os.unlink(video_path)

else:
    st.info("Silakan unggah file video melalui panel di sebelah kiri untuk memulai.")
