import streamlit as st
from detection import split_video, detect_on_frames, combine_frames_to_video
import os

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

st.title("Deteksi Safety Video")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi"])

if uploaded_file:
    video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.video(video_path)

    if st.button("Mulai Deteksi"):
        progress = st.progress(0)

        frames = split_video(video_path)
        progress.progress(30)

        detected_frames = detect_on_frames(frames)
        progress.progress(70)

        output_path = os.path.join(OUTPUT_FOLDER, "hasil_" + uploaded_file.name)
        combine_frames_to_video(detected_frames, output_path)
        progress.progress(100)

        st.success("Deteksi selesai!")
        st.video(output_path)
