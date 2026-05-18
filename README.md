# Face Age Tracker

Real-time face detection, tracking, age estimation, and gender classification.

Detects faces from your webcam, tracks them across frames, and displays a stable age range and gender on each face — with smoothing to prevent flickering.

## Features

- **Face Detection** — OpenCV DNN SSD (Caffe) face detector
- **Face Tracking** — centroid-based tracking with unique IDs per face
- **Age Estimation** — DeepFace (VGGFace-based, alignment-aware)
- **Gender Classification** — DeepFace
- **Emotion Detection** — DeepFace (FER2013 model)
- **Race/Ethnicity Estimation** — DeepFace
- **Stable Display** — predictions smoothed over a rolling window of 5 samples
- **Camera Switching** — press `c` to cycle through available cameras
- **CSV Logging** — timestamps, face IDs, bounding boxes, and age/gender saved to `face_log.csv`

## Requirements

- Python 3.8+
- Webcam

## Installation

```bash
git clone https://github.com/ahmoodio/face-age-tracker
cd face-age-tracker
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python face_tracker_age.py
```

On first run, the following models are auto-downloaded:
- Face detection model (~10 MB)
- Age model (~539 MB)
- Gender model (~20 MB)
- Emotion model (~20 MB)
- Race model (~20 MB)
All models (except face detection) are managed by DeepFace and downloaded to `~/.deepface/weights/`.

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Switch camera (cycles 0→1→2→…) |

The window shows:
- Green bounding box around each detected face
- Face ID, age range (e.g. `12-18yr`), and gender (`M`/`W`)
- Emotion and race below the ID (yellow text)
- Face count and current camera index

All data is logged to `face_log.csv` with columns: `timestamp, face_id, x, y, width, height, age, gender`.

## How It Works

1. **Face detection** runs on every frame using OpenCV's DNN SSD (Caffe `res10_300x300`)
2. **Tracking** assigns persistent IDs via centroid distance matching
3. **Age, Gender, Emotion, Race** are all predicted every 60 frames via a single `DeepFace.analyze()` call per face
4. DeepFace applies face alignment before prediction for better accuracy
5. Results are smoothed over a rolling window of 5 predictions and displayed as an age range (e.g. `17-23yr M`) with emotion and race below

## Files

| File | Description |
|------|-------------|
| `face_tracker_age.py` | Main script |
| `deploy.prototxt` | Face detection network definition (auto-downloaded) |
| `face_log.csv` | Generated CSV log |

## Notes

- All models (age, gender, emotion, race) are managed by DeepFace and auto-download to `~/.deepface/weights/` on first use
- Face detection model (~10 MB) downloads to the project folder
- CPU only (no GPU required, but TensorFlow will use GPU if available)
- All analysis runs every 60 frames (~2s) to keep the video smooth
