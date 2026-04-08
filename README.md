# Replay Attack Spoofing Detection

**Real-time Face Anti-Spoofing System** trained on the **Replay Attack Dataset** using a **MobileNetV2 + LSTM** hybrid model.

### Project Overview
This project detects **presentation attacks** (spoofing) such as:
- Printed photos
- Mobile screen replays  
- High-quality video replays (hand-held & fixed)

The model processes sequences of frames to distinguish between **real faces** and **spoof attempts**.

### Results
- **Test Accuracy**: **92.08%**
- Strong performance on both hand-held and fixed replay attacks

### Key Features
- MobileNetV2 as efficient feature extractor
- LSTM for temporal modeling across video frames
- Custom VideoDataGenerator for memory-efficient training
- Live webcam testing using MediaPipe Face Mesh
- Multi-cue liveness detection:
  - Blink detection
  - Head movement detection
  - Background consistency check

### Repository Structure


Replay-Attack-Spoofing-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── spoofing_mobilenet_lstm_train.ipynb
├── models/
│   ├── best_spoofing_model.h5
│   └── fixed_spoofing_model.keras
├── src/
│   └── live_test.py
└── processed/          # (contains .npy files - not uploaded)


### How to Use (Live Testing)

1. Clone the repository:
```bash
git clone https://github.com/abeerajaved1/Replay-Attack-Spoofing-Detection.git
cd Replay-Attack-Spoofing-Detection

Install dependencies:
pip install -r requirements.txt

Run live anti-spoofing test:
python src/live_test.py

Models Uploaded

best_spoofing_model.h5 → Best checkpoint during training
fixed_spoofing_model.keras → Final fixed model with correct input shape (recommended for inference)

Training Details

Architecture: MobileNetV2 (frozen) + LSTM
Input: 15–20 frames per video (224×224)
Optimizer: Adam
Loss: Binary Crossentropy
Trained with custom Sequence generator

Future Work

Add more liveness cues (eye movement, lip movement)
Ensemble with other anti-spoofing models
Real-time deployment (Streamlit / Gradio)

Trained by Abeera Javed