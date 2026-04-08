# Replay Attack Spoofing Detection

**Real-time Face Anti-Spoofing System** trained on the **Replay Attack Dataset** using a **MobileNetV2 + LSTM** hybrid model.

### Project Overview
This project detects presentation attacks such as printed photos, mobile screen replays, and high-quality video replays (hand-held & fixed).

### Results
- **Test Accuracy**: **92.08%**

### Key Features
- MobileNetV2 + LSTM for temporal modeling
- Sequence-based video processing
- Live webcam testing with MediaPipe Face Mesh
- Multi-cue liveness detection (blink, head movement, background change)

### How to Use (Live Testing)

1. Clone the repository:
```bash
git clone https://github.com/abeerajaved1/Replay-Attack-Spoofing-Detection.git
cd Replay-Attack-Spoofing-Detection

Install dependencies:

Bashpip install -r requirements.txt

Download Models (Important - files are large):
Go to this Google Drive folder
https://drive.google.com/drive/folders/1VjP9Wqax3Jdj4MrbJFRqWhEfx7Nle79j
Download these two files:
fixed_spoofing_model.keras ← Recommended
best_spoofing_model.h5

Place them inside the models/ folder

Run the live test:

Bashpython src/live_test.py
Repository Structure
textReplay-Attack-Spoofing-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── spoofing_mobilenet_lstm_train.ipynb
├── models/                 # Place downloaded models here
├── src/
│   └── live_test.py
Training Details

Architecture: MobileNetV2 (frozen) + LSTM
Input: 15–20 frames (224×224)
Optimizer: Adam
Loss: Binary Crossentropy

Future Work

Improve liveness cues
Ensemble with other models
Web deployment

Trained by Abeera Javed


