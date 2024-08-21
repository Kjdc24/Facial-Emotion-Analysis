# Real-time Emotion Detection

This project performs real-time emotion analysis using a live camera feed. It uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset to detect and classify emotions.

## Directory Structure

```plaintext
emotion_detection/
│
├── data/
│   └── fer2013.csv            # FER-2013 dataset
├── models/
│   └── emotion_detection_model.h5  # Saved trained model
├── notebooks/
│   └── emotion_detection.ipynb  # Jupyter notebook with all code
├── scripts/
│   └── real_time_emotion_analysis.py  # Script for real-time emotion analysis
└── README.md  # Instructions and documentation
