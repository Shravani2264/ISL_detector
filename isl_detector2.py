import os

# Reduce TensorFlow noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import pickle
import tensorflow as tf
from skimage.feature import hog
from collections import deque
from io import BytesIO

# ── Load label encoder ──────────────────────────────────────
class _NumpyCompatUnpickler(pickle.Unpickler):
    """Allow pickles created against older/newer NumPy internals to load."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle_compat(path):
    with open(path, "rb") as f:
        data = f.read()

    try:
        return pickle.loads(data)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        return _NumpyCompatUnpickler(BytesIO(data)).load()


le = load_pickle_compat('label_encoder.pkl')

# ── Load normalization ──────────────────────────────────────
mean = np.load('norm_mean.npy').reshape(-1)
std  = np.load('norm_std.npy').reshape(-1)
std[std == 0] = 1.0

print("Mean shape:", mean.shape)
print("Std shape:", std.shape)

# ── Rebuild model (manual fix) ──────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM

NUM_CLASSES = len(le.classes_)

model = Sequential([
    Input(shape=(30, 292)),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

# Load weights only (bypass broken config)
model.load_weights('isl_model.h5')

print("✅ Model loaded via weights")

# ── Constants ───────────────────────────────────────────────
MAX_FRAMES  = 30
FRAME_SIZE  = (64, 64)
BUFFER      = deque(maxlen=MAX_FRAMES)
prev_gray   = None

# ── Feature Extraction ──────────────────────────────────────
def extract_features(frame):
    global prev_gray

    resized = cv2.resize(frame, FRAME_SIZE)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray    = cv2.GaussianBlur(gray, (5, 5), 0)

    hog_feat = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False
    )

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_feat = np.array([
            flow[..., 0].mean(),
            flow[..., 1].mean(),
            flow[..., 0].std(),
            flow[..., 1].std()
        ])
    else:
        flow_feat = np.zeros(4)

    prev_gray = gray
    return np.concatenate([hog_feat, flow_feat])

# ── Prediction ──────────────────────────────────────────────
def predict():
    if len(BUFFER) < MAX_FRAMES:
        return None, None

    sequence = np.array(BUFFER, dtype=np.float32)

    # Normalize
    sequence = (sequence - mean) / std
    sequence = np.clip(sequence, -3, 3)

    sequence = sequence[np.newaxis, ...]

    probs = model.predict(sequence, verbose=0)[0]
    idx   = np.argmax(probs)

    return le.classes_[idx], probs[idx]

# ── Webcam Setup ────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

prediction = "Warming up..."
confidence = 0.0

PREDICT_EVERY = 5
frame_count = 0

print("🟢 ISL Detector running — press Q to quit")

# ── Main Loop ───────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat = extract_features(frame)
    BUFFER.append(feat)
    frame_count += 1

    # Reset optical flow occasionally
    if frame_count % 100 == 0:
        prev_gray = None

    if frame_count % PREDICT_EVERY == 0:
        word, conf = predict()
        if word:
            prediction = word
            confidence = conf

    # ── UI Overlay ─────────────────────────────────────────
    h, w = frame.shape[:2]

    # Buffer bar
    filled = int((len(BUFFER) / MAX_FRAMES) * w)
    cv2.rectangle(frame, (0, h-8), (filled, h), (0, 200, 0), -1)

    # Top panel
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

    cv2.putText(frame, prediction,
                (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2)

    if confidence:
        cv2.putText(frame, f"{confidence*100:.1f}%",
                    (w-120, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2)

    cv2.imshow('ISL Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
