import os
import time
import cv2
import numpy as np
import mediapipe as mp
import pickle
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM

# ==============================
# LOAD MODEL + LABELS
# ==============================
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 292
sequence = []


class _NumpyCompatUnpickler(pickle.Unpickler):
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


label_encoder = load_pickle_compat("label_encoder.pkl")
labels = list(label_encoder.classes_)

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(labels), activation="softmax"),
])
model.load_weights("isl_model.h5")

mean = np.load("norm_mean.npy").reshape(-1)
std = np.load("norm_std.npy").reshape(-1)
std[std == 0] = 1.0

# ==============================
# MEDIAPIPE TASKS SETUP
# ==============================
MODEL_ASSET_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_ASSET_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ==============================
# KEYPOINT EXTRACTION
# ==============================
def extract_keypoints(landmarks):
    keypoints = []
    for lm in landmarks:
        keypoints.extend([lm.x, lm.y, lm.z])
    keypoints = np.array(keypoints, dtype=np.float32)

    if keypoints.shape[0] < FEATURE_SIZE:
        keypoints = np.pad(keypoints, (0, FEATURE_SIZE - keypoints.shape[0]))
    elif keypoints.shape[0] > FEATURE_SIZE:
        keypoints = keypoints[:FEATURE_SIZE]

    return keypoints

# ==============================
# DRAW HAND LANDMARKS
# ==============================
HAND_CONNECTIONS = (
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
)

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = []

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # Draw connections
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (255,200,0), 2)

    # Draw points
    for p in points:
        cv2.circle(frame, p, 4, (0,255,0), -1)

# ==============================
# CAMERA START
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("🚀 ISL Interpreter Running... Press Q to exit")

start_time = time.time()

# ==============================
# MAIN LOOP
# ==============================
with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ✅ FIXED timestamp (monotonic)
        timestamp = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp)

        # ==========================
        # PROCESS HAND
        # ==========================
        if result.hand_landmarks and len(result.hand_landmarks) > 0:

            hand = result.hand_landmarks[0]

            draw_hand(frame, hand)

            keypoints = extract_keypoints(hand)

            # ✅ Ensure consistent shape
            if keypoints.shape[0] == FEATURE_SIZE:
                sequence.append(keypoints)

            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            # ==========================
            # PREDICTION
            # ==========================
            if len(sequence) == SEQUENCE_LENGTH:

                input_data = np.array(sequence, dtype=np.float32)
                input_data = (input_data - mean) / std
                input_data = np.clip(input_data, -3, 3)
                input_data = np.expand_dims(input_data, axis=0)

                try:
                    res = model.predict(input_data, verbose=0)[0]

                    confidence = float(np.max(res))
                    predicted_class = labels[int(np.argmax(res))]

                    if confidence > 0.6:
                        cv2.putText(
                            frame,
                            f"{predicted_class} ({confidence:.2f})",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2
                        )
                except Exception as e:
                    print("Prediction error:", e)

        else:
            # Clear sequence if no hand
            sequence.clear()

        cv2.imshow("ISL Interpreter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
