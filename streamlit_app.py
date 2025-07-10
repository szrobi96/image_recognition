import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
from ultralytics import YOLO

# ---- Simple Page UI ----
st.set_page_config(page_title="Object Detector", page_icon="🤖", layout="centered")
st.title("🤖 Live Object Detection")
st.markdown(
    "<span style='color:green; font-size:1.2em;'>Show your toys to the camera! Let's see what the AI can find.</span><br><br><small>(Works on phone or computer. Allow camera access if asked!)</small>",
    unsafe_allow_html=True
)
st.divider()

# ---- Model Selection Dropdown ----
MODEL_DIR = "."  # Change to your models folder if needed
pt_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
if not pt_files:
    st.error("No .pt files found in the folder!")
    st.stop()
selected_model = st.selectbox("Select YOLO model file:", pt_files)

# ---- YOLO Model Loading ----
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)
model = load_model(os.path.join(MODEL_DIR, selected_model))

# ---- Video Processing Class ----
class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.skip_n = 2  # Run inference every 2nd frame (adjust for more/less skipping)
        self.last_result = None
        self.target_size = (416, 416)  # Resize for faster inference

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = img.shape[:2]
        self.frame_count += 1

        # Only run inference every skip_n frames
        if self.frame_count % self.skip_n == 0 or self.last_result is None:
            # Resize for faster inference
            resized = cv2.resize(img, self.target_size)
            results = model(resized, verbose=False)
            boxes = results[0].boxes
            self.last_result = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Scale boxes back to original image size
                    x1 = int(x1 * orig_w / self.target_size[0])
                    y1 = int(y1 * orig_h / self.target_size[1])
                    x2 = int(x2 * orig_w / self.target_size[0])
                    y2 = int(y2 * orig_h / self.target_size[1])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    self.last_result.append((x1, y1, x2, y2, conf, cls))
        # Draw last detections
        if self.last_result:
            for x1, y1, x2, y2, conf, cls in self.last_result:
                label = model.model.names[cls] if hasattr(model.model, "names") else str(cls)
                color = (
                    int(37 * cls) % 255,
                    int(17 * cls + 100) % 255,
                    int(53 * cls + 200) % 255,
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return frame.from_ndarray(img, format="bgr24")

# ---- Open Camera and Start Detection ----
webrtc_streamer(
    key="yolo-detect",
    video_processor_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

st.divider()
