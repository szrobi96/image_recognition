# YOLO Object Detection Suite

This repository provides a complete workflow for training, testing, and deploying YOLO object detection models using Ultralytics YOLOv8/YOLO11. It includes:

- A Jupyter/Colab notebook for training custom models (`Train_YOLO_Models.ipynb`)
- A flexible Python script for running YOLO inference on images, videos, folders, or webcams (`yolo_detect.py`)
- A modern Streamlit web app for live webcam detection in your browser (`streamlit_app.py`)

---

## Attribution

The Jupyter notebook and the local inference script (`yolo_detect.py`) were originally written by Evan Juras, [EJ Technology Consultants](https://ejtech.io), and can be found at [EdjeElectronics/Train-and-Deploy-YOLO-Models](https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models).

These scripts have been adapted and extended by Róbert Szabó.

---

## 1. Training a Custom YOLO Model

Use the notebook `Train_YOLO_Models.ipynb` to:
- Download and prepare datasets (e.g., from Roboflow)
- Train YOLOv8/YOLO11 models on your own data
- Export and download trained weights for local use

**Features:**
- Step-by-step instructions for dataset preparation
- GPU training in Google Colab
- Example commands for local deployment

**Quick Start:**
1. Open the notebook in Google Colab or Jupyter
2. Follow the instructions to prepare your dataset and train a model
3. Download the resulting `best.pt` weights

---

## 2. Local Inference Script (`yolo_detect.py`)

A command-line tool to run YOLO detection on images, videos, folders, or live webcam feeds.

**Usage Example:**
```shell
python yolo_detect.py --model yolov8n.pt --source usb0 --resolution 1280x720
```

**Arguments:**
- `--model`: Path to a YOLO model file (e.g., `best.pt`, `yolov8n.pt`, or stock model name)
- `--source`: Image/video file, folder, or camera index (e.g., `usb0`)
- `--resolution`: (Optional) Output resolution, e.g., `1280x720`
- `--thresh`: (Optional) Confidence threshold (default: 0.5)
- `--record`: (Optional) Record video output to `demo1.avi`

**Features:**
- Supports webcam, video, image, and folder sources
- Displays FPS and object count
- Hotkeys: `q` to quit, `s` to pause, `p` to save frame, or close window to exit
- Automatically downloads stock models if not present

---

## 3. Streamlit Web App (`streamlit_app.py`)

A browser-based app for real-time object detection using your webcam.

**Features:**
- Live webcam detection in your browser (desktop or mobile)
- Fast YOLOv8/YOLO11 inference with frame resizing and frame skipping for performance
- Modern UI with Streamlit and streamlit-webrtc

**How to Run:**
```shell
streamlit run streamlit_app.py
```

**Optional:**
- To share your app over the internet, use [ngrok](https://ngrok.com/) or similar tools. For security, consider adding authentication or access control.

---

## Requirements
- Python 3.8+
- [Ultralytics](https://pypi.org/project/ultralytics/)
- OpenCV (`opencv-python`)
- Numpy
- Streamlit (`streamlit`)
- streamlit-webrtc
- (Optional) pyngrok for public sharing

Install all requirements:
```shell
pip install ultralytics opencv-python numpy streamlit streamlit-webrtc pyngrok
```

or use the provided `requirements.txt`
```shell
pip install -r requirements.txt
```


---

## Security Notes
- If exposing the Streamlit app publicly (e.g., via ngrok), add authentication to prevent unauthorized access.
- Do not expose sensitive data or admin interfaces without protection.

---

## References
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Streamlit](https://streamlit.io/)
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
- [EdjeElectronics/Train-and-Deploy-YOLO-Models](https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models)

---

## License
MIT License

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
