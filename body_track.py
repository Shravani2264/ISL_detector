import time
import cv2
import numpy as np
import mediapipe as mp
import pickle
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 292
POSE_MODEL_PATH = "pose_landmarker_lite.task"
sequence = []

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS


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


def extract_keypoints(pose_landmarks):
    if pose_landmarks:
        keypoints = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z,
                    getattr(lm, "visibility", 1.0),
                ]
                for lm in pose_landmarks
            ],
            dtype=np.float32,
        ).flatten()
    else:
        keypoints = np.zeros(33 * 4, dtype=np.float32)

    if keypoints.shape[0] < FEATURE_SIZE:
        keypoints = np.pad(keypoints, (0, FEATURE_SIZE - keypoints.shape[0]))
    elif keypoints.shape[0] > FEATURE_SIZE:
        keypoints = keypoints[:FEATURE_SIZE]

    return keypoints


def draw_pose(frame, pose_landmarks):
    if not pose_landmarks:
        return 0

    h, w = frame.shape[:2]
    points = []
    visible_points = 0

    for lm in pose_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        visibility = getattr(lm, "visibility", 1.0)
        points.append((x, y, visibility))
        if visibility > 0.5:
            visible_points += 1

    for connection in POSE_CONNECTIONS:
        start = connection.start
        end = connection.end
        if start < len(points) and end < len(points):
            x1, y1, v1 = points[start]
            x2, y2, v2 = points[end]
            if v1 > 0.3 and v2 > 0.3:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)

    for x, y, visibility in points:
        color = (0, 255, 0) if visibility > 0.5 else (0, 100, 255)
        cv2.circle(frame, (x, y), 4, color, -1)

    return visible_points


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

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("Body tracker running. Press Q to exit.")
start_time = time.time()

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else []

        visible_points = draw_pose(frame, pose_landmarks)

        keypoints = extract_keypoints(pose_landmarks)
        sequence.append(keypoints)
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Body points visible: {visible_points}/33",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Sequence: {len(sequence)}/{SEQUENCE_LENGTH}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

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
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )
            except Exception as exc:
                print("Prediction error:", exc)

        elapsed = int(time.time() - start_time)
        cv2.putText(
            frame,
            f"Time: {elapsed}s",
            (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Body Track", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
