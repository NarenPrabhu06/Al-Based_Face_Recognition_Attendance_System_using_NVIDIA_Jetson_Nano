import cv2
import os
import time

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if cascade.empty():
    print("ERROR: Haar Cascade XML not found!")
    exit()

user_id = input("Enter User ID: ")
name = input("Enter Name: ")

save_path = os.path.join("dataset", name)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opened!")
    exit()

print("\nLook at the camera...")
print("Collecting 100 face images...\n")

count = 0

while count < 100:

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        count += 1

        face = gray[y:y+h, x:x+w]

        filename = os.path.join(save_path, f"{count}.jpg")

        cv2.imwrite(filename, face)

        print(f"Captured {count}/100")

        time.sleep(0.15)

        if count >= 100:
            break

cap.release()

print("\nDataset Collection Completed!")
