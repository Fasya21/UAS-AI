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
    for frame in frames:
        results = model(frame)
        annotated_frame = results.plot()  # ✅ cukup langsung .plot()
        detected_frames.append(annotated_frame)
    return detected_frames

def combine_frames_to_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
