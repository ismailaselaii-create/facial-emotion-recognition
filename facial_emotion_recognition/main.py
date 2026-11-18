import cv2
import time
import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from fer.fer import FER

emotion_detector = FER(mtcnn=False)

# Preprocessing
def preprocess_image(img, size=(224, 224)):
    try:
        resized = cv2.resize(img, size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


# Face detection & tuning
DETECTION_CONFIGS = [
    {
        "name": "Config 1 — Default (balanced)",
        "scaleFactor": 1.05,
        "minNeighbors": 4,
        "minSize": (40, 40),
        "notes": "Good for most cases.",
    },
    {
        "name": "Config 2 — Sensitive",
        "scaleFactor": 1.03,
        "minNeighbors": 3,
        "minSize": (30, 30),
        "notes": "Finds more faces, but may add false positives.",
    },
    {
        "name": "Config 3 — Strict",
        "scaleFactor": 1.1,
        "minNeighbors": 6,
        "minSize": (50, 50),
        "notes": "More strict detection.",
    },
    {
        "name": "Config 4 — Fine-grained",
        "scaleFactor": 1.04,
        "minNeighbors": 5,
        "minSize": (35, 35),
        "notes": "Balanced between strict and sensitive.",
    },
]


def print_detection_param_table():
    print("\n" + "=" * 70)
    print(f"  {'Config':<30} {'scaleFactor':>12} {'minNeighbors':>14} {'minSize':>12}")
    print("=" * 70)
    for cfg in DETECTION_CONFIGS:
        print(
            f"  {cfg['name']:<30} {cfg['scaleFactor']:>12.2f} "
            f"{cfg['minNeighbors']:>14} {str(cfg['minSize']):>12}"
        )
        print(f"    -> {cfg['notes']}")
    print("=" * 70 + "\n")


def detect_faces(img, config_index=0):
    cfg = DETECTION_CONFIGS[config_index]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=cfg["scaleFactor"],
        minNeighbors=cfg["minNeighbors"],
        minSize=cfg["minSize"],
    )

    if len(faces) == 0:
        face_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        faces = face_cascade_alt.detectMultiScale(
            gray,
            scaleFactor=cfg["scaleFactor"],
            minNeighbors=max(3, cfg["minNeighbors"] - 1),
            minSize=cfg["minSize"],
        )

    print("Faces found:", len(faces))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# Emotion analysis with FER
def get_emotion_scores(img_rgb):
    if img_rgb is None or img_rgb.size == 0:
        print("Empty image")
        return None

    h, w = img_rgb.shape[:2]
    if h < 48 or w < 48:
        print(f"Too small: {w}x{h}")
        return None

    try:
        results = emotion_detector.detect_emotions(img_rgb)

        if not results:
            print("No emotion detected")
            return None

        emotions = results[0]["emotions"]
        print("Emotion result:", emotions)
        return emotions

    except Exception as e:
        print("FER error:", e)
        return None


# Visualization
EMOTION_COLORS = {
    "happy": (0, 220, 255),
    "neutral": (200, 200, 200),
    "sad": (255, 120, 50),
    "angry": (0, 0, 255),
    "surprise": (0, 165, 255),
    "fear": (130, 0, 130),
    "disgust": (0, 140, 0),
}


def draw_visualization(frame, x, y, w, h, emotions):
    if not emotions:
        return frame

    dominant_emotion = max(emotions, key=emotions.get)
    max_score = emotions[dominant_emotion] * 100

    color = EMOTION_COLORS.get(dominant_emotion, (255, 255, 255))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{dominant_emotion.upper()} ({max_score:.1f}%)"
    text_y = y - 10 if y - 10 > 20 else y + h + 20
    cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    bar_x = x + w + 10
    bar_y = y
    bar_max_w = 90
    row_h = 18
    panel_w = bar_max_w + 60
    panel_h = len(emotions) * row_h + 10

    if bar_x + panel_w < frame.shape[1] and bar_y + panel_h < frame.shape[0]:
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + panel_w, bar_y + panel_h),
            (30, 30, 30),
            -1
        )

        for idx, (emo, score) in enumerate(emotions.items()):
            score_percent = score * 100
            row_y = bar_y + 5 + idx * row_h
            bar_w = int((score_percent / 100) * bar_max_w)
            bar_color = EMOTION_COLORS.get(emo, (180, 180, 180))

            cv2.rectangle(
                frame,
                (bar_x + 3, row_y),
                (bar_x + 3 + bar_w, row_y + 11),
                bar_color,
                -1
            )

            cv2.putText(
                frame,
                f"{emo[:3]} {score_percent:.0f}%",
                (bar_x + bar_max_w + 6, row_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (220, 220, 220),
                1
            )

    return frame


def show_matplotlib_summary(image, faces_data):
    n = len(faces_data)
    if n == 0:
        print("No faces to display.")
        return

    fig, axes = plt.subplots(1, 1 + n, figsize=(5 * (1 + n), 5))

    if n == 1:
        axes = np.array(axes).reshape(-1)

    display = image.copy()
    for fd in faces_data:
        draw_visualization(display, fd["x"], fd["y"], fd["w"], fd["h"], fd["emotions"])

    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    axes[0].imshow(rgb)
    axes[0].set_title("Detected Faces", fontweight="bold")
    axes[0].axis("off")

    for i, fd in enumerate(faces_data):
        ax = axes[i + 1]
        emotions = fd["emotions"]
        labels = list(emotions.keys())
        values = [v * 100 for v in emotions.values()]

        bars = ax.barh(labels, values)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        ax.set_title(f"Face #{i + 1}", fontweight="bold")
        ax.invert_yaxis()

        for bar, val in zip(bars, values):
            if val > 1:
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=8)

    plt.suptitle("Facial Emotion Recognition — Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Batch analysis
def run_batch_mode(dataset_path="dataset"):
    true_labels = []
    pred_labels = []
    results = []

    print("Starting batch mode analysis...")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found.")
        return

    for true_emotion in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, true_emotion)

        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emotions = get_emotion_scores(rgb)

            if emotions:
                pred_emotion = max(emotions, key=emotions.get)
            else:
                pred_emotion = "unknown"

            true_labels.append(true_emotion)
            pred_labels.append(pred_emotion)
            results.append({
                "filename": filename,
                "true_label": true_emotion,
                "pred_label": pred_emotion
            })

            print(f"Processed {filename}: True={true_emotion}, Pred={pred_emotion}")

    df = pd.DataFrame(results)
    import os
    print("Текущая директория:", os.getcwd())
    df.to_csv("results.csv", index=False)
    print("Saved results to results.csv")

    if true_labels:
        acc = accuracy_score(true_labels, pred_labels)
        print(f"Overall Accuracy: {acc * 100:.2f}%")

        labels = np.unique(true_labels)
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.show()


# Real-time webcam
def run_webcam_mode(skip_frames=5, camera_index=0, config_index=0):
    print_detection_param_table()

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs("screenshots", exist_ok=True)

    frame_counter = 0
    last_faces_data = []

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_counter += 1

        if frame_counter % skip_frames == 0:
            faces = detect_faces(frame, config_index=config_index)
            new_faces_data = []

            for (x, y, w, h) in faces:
                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                if face_crop.shape[0] < 48 or face_crop.shape[1] < 48:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, "Face too small", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    continue

                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                emotions = get_emotion_scores(rgb_face)

                if emotions:
                    new_faces_data.append({
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "emotions": emotions
                    })
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "No emotion", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if new_faces_data:
                last_faces_data = new_faces_data

        for fd in last_faces_data:
            draw_visualization(frame, fd["x"], fd["y"], fd["w"], fd["h"], fd["emotions"])

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Q: quit | S: screenshot",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        cv2.imshow("Emotion Recognition - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"screenshots/screenshot_{ts}.png"
            cv2.imwrite(path, frame)
            print(f"Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition Project")
    parser.add_argument("--mode", type=str, choices=["webcam", "batch", "params"], default="webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera index: usually 0")
    parser.add_argument("--skip", type=int, default=5, help="Process every N frames")
    parser.add_argument("--config", type=int, default=0, help="Detection config index: 0-3")
    args = parser.parse_args()

    if args.mode == "webcam":
        run_webcam_mode(skip_frames=args.skip, camera_index=args.camera, config_index=args.config)
    elif args.mode == "batch":
        run_batch_mode()
    elif args.mode == "params":
        print_detection_param_table()