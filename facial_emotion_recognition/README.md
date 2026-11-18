Facial Emotion Recognition – Real‑Time & Batch Analysis

This project performs emotion recognition from facial expressions using a webcam or static images. It detects seven basic emotions: angry, disgust, fear, happy, neutral, sad, surprise.

Features

Face detection – Haar cascades with four tunable configurations (balanced, sensitive, strict, fine‑grained).
Emotion recognition – Based on the FER library (deep CNN trained on FER‑2013).
Visualisation – Coloured bounding box, dominant emotion label, horizontal bar chart for all emotions.
Batch evaluation – Process a labelled dataset, compute accuracy, confusion matrix, and save results.
Real‑time control – Adjust frame skipping, save screenshots, choose detection config.
Installation

Python 3.7+ required. Install dependencies:

bash
pip install opencv-python numpy pandas matplotlib scikit-learn fer
Usage

Run the script from the command line:

bash
python emotion_recognition.py [--mode MODE] [--camera CAM] [--skip N] [--config IDX]
Mode	Command
Webcam (default)	python emotion_recognition.py
Webcam with config 1, skip 2 frames	python emotion_recognition.py --mode webcam --config 1 --skip 2
Batch mode	python emotion_recognition.py --mode batch
Show detection parameters	python emotion_recognition.py --mode params
Arguments

--mode : webcam, batch, or params (default: webcam)
--camera : camera index (default: 0)
--skip : process every N‑th frame (default: 5)
--config : detection configuration 0–3 (default: 0)
Webcam Controls

Press q – quit
Press s – save a screenshot (saved to screenshots/ folder)
Batch Mode Folder Structure

Create a folder dataset/ with subfolders named after emotions:

text
dataset/
  angry/
    img1.jpg
    img2.jpg
  happy/
    ...
  neutral/
    ...
The script will:

Compare predictions against the folder name (ground truth).
Print overall accuracy.
Save results.csv and confusion_matrix.png.