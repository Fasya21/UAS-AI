import cv2
from model import load_model

def split_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_on_frames(frames):
    model = load_model()
    detected_frames = []

    # Ambil 30 frame dari tengah video
    total_frames = len(frames)
    if total_frames > 30:
        start = (total_frames - 30) // 2
        frames = frames[start:start+30]

    st.info(f"Proses deteksi dimulai pada {len(frames)} frame...")
    progress = st.progress(0)
    
    for i, frame in enumerate(frames):
        results = model(frame)
        annotated_frame = results[0].plot()
        detected_frames.append(annotated_frame)
        progress.progress((i + 1) / len(frames))

    st.success("Deteksi selesai!")
    return detected_frames

def combine_frames_to_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
