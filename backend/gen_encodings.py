import face_recognition
import cv2
import os
import pickle

known_encodings = []
known_names = []

# Folder containing images of people you want to recognize
dataset_dir = "datasets"  # create this folder and put images inside

for filename in os.listdir(dataset_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(dataset_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            # Use the filename (without extension) as the person's name
            known_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No face found in {filename}")

# Save encodings + names to pkl file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("face_encodings.pkl created successfully!")
