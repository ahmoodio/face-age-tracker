import cv2, csv, numpy as np, os, urllib.request
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['QT_QPA_PLATFORM'] = 'xcb'
from datetime import datetime
from collections import defaultdict, Counter

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Convolution2D, ZeroPadding2D, MaxPooling2D,
    Flatten, Dropout, Activation
)

SMOOTHING_WINDOW = 5
AGE_MARGIN = 3


def download_weights(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5"
    print(f"Downloading age model weights (~539 MB) to {path}...")
    urllib.request.urlretrieve(url, path)
    print("Download complete.")

def build_age_model():
    base = Sequential()
    base.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    base.add(Convolution2D(64, (3, 3), activation="relu"))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(64, (3, 3), activation="relu"))
    base.add(MaxPooling2D((2, 2), strides=(2, 2)))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(128, (3, 3), activation="relu"))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(128, (3, 3), activation="relu"))
    base.add(MaxPooling2D((2, 2), strides=(2, 2)))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(256, (3, 3), activation="relu"))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(256, (3, 3), activation="relu"))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(256, (3, 3), activation="relu"))
    base.add(MaxPooling2D((2, 2), strides=(2, 2)))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(512, (3, 3), activation="relu"))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(512, (3, 3), activation="relu"))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(512, (3, 3), activation="relu"))
    base.add(MaxPooling2D((2, 2), strides=(2, 2)))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(512, (3, 3), activation="relu"))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(512, (3, 3), activation="relu"))
    base.add(ZeroPadding2D((1, 1)))
    base.add(Convolution2D(512, (3, 3), activation="relu"))
    base.add(MaxPooling2D((2, 2), strides=(2, 2)))
    base.add(Convolution2D(4096, (7, 7), activation="relu"))
    base.add(Dropout(0.5))
    base.add(Convolution2D(4096, (1, 1), activation="relu"))
    base.add(Dropout(0.5))
    base.add(Convolution2D(2622, (1, 1)))
    base.add(Flatten())
    base.add(Activation("softmax"))
    out = Convolution2D(101, (1, 1), name="predictions")(base.layers[-4].output)
    out = Flatten()(out)
    out = Activation("softmax")(out)
    model = Model(inputs=base.inputs, outputs=out)
    weights = os.path.expanduser("~/.deepface/weights/age_model_weights.h5")
    if not os.path.exists(weights):
        download_weights(weights)
    model.load_weights(weights)
    return model


def predict_age(face_img, model):
    resized = cv2.resize(face_img, (224, 224))
    batch = np.expand_dims(resized.astype(np.float32), 0)
    probs = model(batch, training=False).numpy()[0, :]
    return int(probs.argmax())


class FaceTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.next_id = 0
        self.faces = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.age_gender_data = {}

    def register(self, centroid):
        self.faces[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.age_gender_data[self.next_id] = {"ages": [], "genders": [], "stable_age": None, "stable_gender": None}
        self.next_id += 1

    def deregister(self, face_id):
        del self.faces[face_id]
        del self.disappeared[face_id]
        if face_id in self.age_gender_data:
            del self.age_gender_data[face_id]

    def update(self, detections):
        if not detections:
            for fid in list(self.faces.keys()):
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    self.deregister(fid)
            return []
        ic = [(x + w // 2, y + h // 2) for (x, y, w, h) in detections]
        if not self.faces:
            for c in ic:
                self.register(c)
            result = []
            for i, det in enumerate(detections):
                result.append((i, det[0], det[1], det[2], det[3]))
            return result
        ei, ec = list(self.faces.keys()), list(self.faces.values())
        D = np.linalg.norm(np.array(ec)[:, None] - np.array(ic), axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        ur, uc, a = set(), set(), []
        for r, c in zip(rows, cols):
            if r in ur or c in uc or D[r, c] > self.max_distance:
                continue
            a.append((ei[r], c))
            ur.add(r)
            uc.add(c)
        for face_id, c in a:
            self.faces[face_id] = ic[c]
            self.disappeared[face_id] = 0
        for r in range(len(ei)):
            if r not in ur:
                self.disappeared[ei[r]] += 1
                if self.disappeared[ei[r]] > self.max_disappeared:
                    self.deregister(ei[r])
        for c in range(len(ic)):
            if c not in uc:
                self.register(ic[c])
        result = []
        for det, cent in zip(detections, ic):
            for face_id, (cx, cy) in self.faces.items():
                if (cx, cy) == cent:
                    result.append((face_id, *det))
                    break
        return result

    def update_age_gender(self, face_id, age, gender):
        if face_id not in self.age_gender_data:
            return
        data = self.age_gender_data[face_id]
        data["ages"].append(int(age))
        data["genders"].append(gender)
        if len(data["ages"]) > SMOOTHING_WINDOW:
            data["ages"].pop(0)
        if len(data["genders"]) > SMOOTHING_WINDOW:
            data["genders"].pop(0)
        data["stable_age"] = int(np.median(data["ages"]))
        if data["genders"]:
            data["stable_gender"] = Counter(data["genders"]).most_common(1)[0][0]

    def get_stable_age_gender(self, face_id):
        if face_id not in self.age_gender_data:
            return None, None
        d = self.age_gender_data[face_id]
        return d.get("stable_age"), d.get("stable_gender")


def get_gender_short(gender):
    return gender[0].upper()


print("Setting up...")
if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
if not os.path.exists("deploy.prototxt"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "deploy.prototxt"
    )
if not os.path.exists("deploy_gender.prototxt"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
        "deploy_gender.prototxt"
    )
if not os.path.exists("gender_net.caffemodel"):
    urllib.request.urlretrieve(
        "https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0",
        "gender_net.caffemodel"
    )

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
print("Face detection + gender loaded. Loading age model (~539 MB)...")
age_model = build_age_model()
print("Age model loaded (argmax). Running every 60 frames.")

tracker = FaceTracker()

csv_file = open("face_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "face_id", "x", "y", "width", "height", "age", "gender"])

cam_idx = 0
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        cam_idx = i
        break
    cap.release()
else:
    cam_idx = 0
    cap = cv2.VideoCapture(0)
frame_count = 0


def switch_camera():
    global cap, cam_idx, tracker
    cap.release()
    cam_idx = (cam_idx + 1) % 10
    tracker = FaceTracker()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        cam_idx = 0
        cap = cv2.VideoCapture(0)
        print(f"Camera {cam_idx} unavailable, fell back to 0")
    else:
        print(f"Switched to camera {cam_idx}")


print("Running! Press 'q' to quit, 'c' to switch camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    dets = []
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            bw, bh = x2 - x1, y2 - y1
            if bw > 30 and bh > 30:
                dets.append((x1, y1, bw, bh))

    tracked = tracker.update(dets)
    ts = datetime.now().isoformat()

    if frame_count % 60 == 0 and len(dets) > 0 and len(tracked) > 0:
        for idx, (x, y, bw, bh) in enumerate(dets):
            # Add 20px padding for context
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + bw + pad)
            y2 = min(frame.shape[0], y + bh + pad)
            face_img = frame[y1:y2, x1:x2]

            fid = None
            for t in tracked:
                if t[1] == x and t[2] == y:
                    fid = t[0]
                    break
            if fid is None:
                continue

            age = predict_age(face_img, age_model)

            blob_gender = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
            gender_net.setInput(blob_gender)
            preds = gender_net.forward()[0]
            gender = "Man" if preds[0] > preds[1] else "Woman"

            tracker.update_age_gender(fid, age, gender)

    for idx, item in enumerate(tracked):
        fid, x, y, bw, bh = item
        label = f"Face #{fid}"
        stable_age, stable_gender = tracker.get_stable_age_gender(fid)
        age_text = ""
        if stable_age is not None:
            lo = max(0, stable_age - AGE_MARGIN)
            hi = stable_age + AGE_MARGIN
            gender_str = get_gender_short(stable_gender) if stable_gender else "?"
            age_text = f"{lo}-{hi}yr {gender_str}"
            label += f" {age_text}"

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        csv_writer.writerow([ts, fid, x, y, bw, bh, stable_age, stable_gender])

    cv2.putText(frame, f"Faces: {len(tracked)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Cam: {cam_idx}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Face Tracker + Age + Gender", frame)
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        switch_camera()

cap.release()
csv_file.close()
cv2.destroyAllWindows()
