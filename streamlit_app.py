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

# ---- YOLO Model Loading ----
# Using the fastest pre-trained YOLOv8 nano model.
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')
model = load_model()

# ---- Video Processing Class ----
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # YOLO inference
        results = model(img, verbose=False)
        boxes = results[0].boxes
        if boxes is not None:
            # Draw boxes and labels
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
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
        return img

# ---- STUN and TURN Server Configuration ----
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        # Public STUN Server
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]
})

# ---- Open Camera and Start Detection ----
webrtc_streamer(
    key="yolo-detect",
    video_processor_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],  # Add STUN and TURN configuration
        }
)

st.divider()