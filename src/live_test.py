# src/live_test.py
# Live Anti-Spoofing Test with Multi-cue Liveness Detection

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

# ====================== CONFIG ======================
MODEL_PATH = '../models/fixed_spoofing_model.keras'
NUM_FRAMES = 20
DEEP_THRESHOLD = 0.65          # Adjust between 0.6 - 0.75
MIN_BLINKS = 1
MIN_HEAD_MOVES = 2
MIN_BG_CHANGES = 3
# ===================================================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("MediaPipe FaceMesh initialized.")

def capture_frame(quality=0.8):
    js = Javascript('''
        async function captureFrame(quality) {
            const div = document.createElement('div');
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getTracks().forEach(track => track.stop());
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js('captureFrame({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    frame = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
    return frame

def detect_blink(landmarks):
    if not landmarks:
        return False
    p33, p133, p159, p145 = landmarks[33], landmarks[133], landmarks[159], landmarks[145]
    vertical = np.linalg.norm([p159.x - p145.x, p159.y - p145.y])
    horizontal = np.linalg.norm([p33.x - p133.x, p33.y - p133.y])
    ear = vertical / horizontal if horizontal > 0 else 0
    return ear < 0.24

def detect_head_movement(prev_lm, curr_lm, threshold=0.012):
    if not prev_lm or not curr_lm:
        return False
    dx = abs(curr_lm[1].x - prev_lm[1].x)
    dy = abs(curr_lm[1].y - prev_lm[1].y)
    return (dx + dy) > threshold

def background_changed(prev_frame, curr_frame, threshold=7.5):
    if prev_frame is None:
        return False
    diff = cv2.absdiff(prev_frame, curr_frame)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

def live_test():
    frames = []
    prev_frame_gray = None
    prev_landmarks = None

    blinks = 0
    head_moves = 0
    bg_changes = 0

    print("\n" + "="*60)
    print("          LIVE ANTI-SPOOFING TEST STARTED")
    print("   → Look at camera, blink naturally, move head gently")
    print("   → Click 'Capture' button each time it appears")
    print("="*60 + "\n")

    for i in range(NUM_FRAMES):
        print(f"Capturing frame {i+1}/{NUM_FRAMES} ...")
        frame = capture_frame()
        if frame is None:
            print("Capture failed.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            if detect_blink(landmarks):
                blinks += 1
                print(f"    → Blink detected ({blinks})")

            if prev_landmarks and detect_head_movement(prev_landmarks, landmarks):
                head_moves += 1
                print(f"    → Head movement detected ({head_moves})")

            prev_landmarks = landmarks

        # Background change
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if background_changed(prev_frame_gray, frame_gray):
            bg_changes += 1
            print(f"    → Background change detected ({bg_changes})")

        prev_frame_gray = frame_gray

        # Prepare frame for model
        frame_resized = cv2.resize(frame, (224, 224))
        frames.append(frame_resized / 255.0)

        time.sleep(0.35)

    # Prepare input
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])
    input_data = np.expand_dims(np.array(frames[:NUM_FRAMES]), axis=0)

    # Deep model prediction
    pred = model.predict(input_data, verbose=0)[0][0]
    label = "SPOOF" if pred > DEEP_THRESHOLD else "REAL"
    confidence = max(pred, 1 - pred)

    print("\n" + "-"*50)
    print("LIVENESS SUMMARY:")
    print(f"  Blinks detected       : {blinks}")
    print(f"  Head movements        : {head_moves}")
    print(f"  Background changes    : {bg_changes}")
    print("-"*50)

    if blinks < MIN_BLINKS or head_moves < MIN_HEAD_MOVES or bg_changes < MIN_BG_CHANGES:
        print("RESULT: SPOOF DETECTED")
        print("Reason: Insufficient natural liveness signals")
    else:
        print(f"FINAL RESULT: {label}")
        print(f"Deep model score: {pred:.4f} | Confidence: {confidence:.3f}")

# =============== RUN TEST ===============
if __name__ == "__main__":
    live_test()