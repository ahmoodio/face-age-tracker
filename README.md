# Face Age Tracker

Real-time face detection, tracking, age estimation, and gender classification.

Detects faces from your webcam, tracks them across frames, and displays a stable age range and gender on each face — with smoothing to prevent flickering.

## Features

- **Face Detection** — OpenCV DNN SSD (Caffe) face detector
- **Face Tracking** — centroid-based tracking with unique IDs per face
- **Age Estimation** — VGGFace-based CNN with argmax (more accurate for younger faces)
- **Gender Classification** — OpenCV DNN Caffe model (Adience dataset)
- **Emotion Detection** — DeepFace FER2013 model (happy, sad, angry, etc.)
- **Race/Ethnicity Estimation** — DeepFace model
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
- Gender classification model (~44 MB)
- Age estimation model (~539 MB, downloaded to `~/.deepface/weights/`)
- Emotion model (~20 MB)
- Race model (~20 MB)

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
3. **Age** is predicted every 60 frames using a VGGFace CNN, using **argmax** (most likely age) instead of expected value for better accuracy on younger faces
4. **Gender** is predicted every 60 frames using an AlexNet-based Caffe model trained on the Adience dataset
5. **Emotion + Race** are predicted every 60 frames using DeepFace (FER2013 emotion model + built-in race model)
6. Results are smoothed over a rolling window of 5 predictions and displayed as an age range (e.g. `17-23yr M`) with emotion and race below

## Files

| File | Description |
|------|-------------|
| `face_tracker_age.py` | Main script (age + gender + tracking) |
| `face_age_tracker.py` | Alternate version using Caffe age model |
| `face_tracker.py` | Face tracking only (no age/gender) |
| `face_tracker_simple.py` | Simplified face tracking |
| `deploy.prototxt` | Face detection network definition (auto-downloaded) |
| `deploy_gender.prototxt` | Gender classification network definition |
| `face_log.csv` | Generated CSV log |

## Notes

- Age model is ~539 MB (VGGFace), downloaded once to `~/.deepface/weights/`
- Gender model (~44 MB) and face detection model (~10 MB) download automatically to the project folder
- Emotion and race models (~20 MB each) download automatically via DeepFace on first run
- The age model uses **argmax** instead of the default expected value for better accuracy on younger subjects
- CPU only (no GPU required, but TensorFlow will use GPU if available)
- Emotion + race analysis runs every 60 frames (same as age/gender) to keep the video smooth
