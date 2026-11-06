"""Detect faces and label their emotion, from a webcam, a video file or an image.

    python detect_emotions.py                  # webcam (press q to quit)
    python detect_emotions.py --source clip.mp4
    python detect_emotions.py --source photo.jpg --save labelled.jpg

Faces are found with OpenCV's Haar cascade, cropped, converted to 48x48
grayscale (the FER-2013 format) and classified by the trained model.
"""

import argparse
from pathlib import Path

import cv2
import keras
import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def label_faces(frame, model):
    """Draw a box and the predicted emotion on every face in a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))
    for x, y, w, h in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        # The model rescales pixels itself, so it takes raw 0-255 values like the training data.
        probs = model.predict(face[None, :, :, None].astype("float32"), verbose=0)[0]
        k = int(np.argmax(probs))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(frame, f"{EMOTIONS[k]} {probs[k]:.0%}", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
    return frame, len(faces)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="emotion_cnn.keras")
    parser.add_argument("--source", default="0", help="webcam index, video file or image file")
    parser.add_argument("--save", help="output path for a labelled image")
    args = parser.parse_args()

    model = keras.models.load_model(args.model)

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f"could not read {args.source}")
        frame, n = label_faces(frame, model)
        print(f"{n} face(s) found")
        if args.save:
            cv2.imwrite(args.save, frame)
        else:
            cv2.imshow("Emotion detection", frame)
            cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, _ = label_faces(frame, model)
        cv2.imshow("Emotion detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
