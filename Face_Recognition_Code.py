import cv2
import os

# -----------------------------
# Load Face Detector
# -----------------------------
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if cascade.empty():
    raise Exception("Cannot load haarcascade_frontalface_default.xml")

# -----------------------------
# Load LBPH Model
# -----------------------------
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("trainer/trainer.yml")

# -----------------------------
# Load Labels
# -----------------------------
labels = {}

with open("trainer/labels.txt", "r") as f:
    for line in f:
        idx, name = line.strip().split(",")
        labels[int(idx)] = name


def recognize(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    results = []

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 95:
            name = labels.get(label, "Unknown")
        else:
            name = "Unknown"

        results.append({
            "name": name,
            "confidence": confidence,
            "box": (x, y, w, h)
        })

    return results
