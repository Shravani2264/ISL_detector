import sys

try:
    import cv2
    print("✅ cv2 OK")
except Exception as e:
    print(f"❌ cv2: {e}"); sys.exit()

try:
    import tensorflow as tf
    print("✅ tf OK")
except Exception as e:
    print(f"❌ tf: {e}"); sys.exit()

try:
    model = tf.keras.models.load_model('best_model.keras')
    print(f"✅ Model loaded: {model.input_shape}")
except Exception as e:
    print(f"❌ Model load failed: {e}"); sys.exit()

try:
    import pickle
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print(f"✅ Label encoder: {len(le.classes_)} classes")
except Exception as e:
    print(f"❌ Label encoder: {e}"); sys.exit()

try:
    import numpy as np
    mean = np.load('norm_mean.npy')
    std  = np.load('norm_std.npy')
    print(f"✅ Norm files loaded")
except Exception as e:
    print(f"❌ Norm files: {e}"); sys.exit()

try:
    cap = cv2.VideoCapture(0)
    print(f"✅ Webcam opened: {cap.isOpened()}")
    cap.release()
except Exception as e:
    print(f"❌ Webcam: {e}"); sys.exit()

print("\n🟢 Everything OK!")