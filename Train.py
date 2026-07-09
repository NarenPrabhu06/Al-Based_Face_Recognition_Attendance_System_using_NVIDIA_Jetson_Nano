import cv2
import os
import numpy as np

# Create LBPH recognizer (OpenCV 3.2)
recognizer = cv2.face.createLBPHFaceRecognizer()

dataset_path = "dataset"

faces = []
labels = []

name_to_id = {}
id_to_name = {}

current_id = 0

print("Loading images...")

for person in sorted(os.listdir(dataset_path)):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    name_to_id[person] = current_id
    id_to_name[current_id] = person

    print("Reading:", person)

    for image in os.listdir(person_path):

        image_path = os.path.join(person_path, image)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_id)

    current_id += 1

print("\nTotal Faces :", len(faces))
print("Total Persons :", len(id_to_name))

if len(faces) == 0:
    print("No training images found!")
    exit()

recognizer.train(faces, np.array(labels))

os.makedirs("trainer", exist_ok=True)

recognizer.save("trainer/trainer.yml")

with open("trainer/labels.txt", "w") as f:
    for idx in id_to_name:
        f.write(f"{idx},{id_to_name[idx]}\n")

print("\nTraining Completed Successfully!")
print("Model saved to trainer/trainer.yml")
