from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2, os, sqlite3, pickle, io
import numpy as np
import face_recognition
from PIL import Image

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load encodings
with open("face_encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

print("Model loaded")

# Directory for matched photos
PHOTO_DIR = "voter_photos"
os.makedirs(PHOTO_DIR, exist_ok=True)

DB_PATH = "voters.db"  # Database file


def get_voter_details(name: str):
    """Fetch voter details from SQLite by name."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, dob, address, voter_id, status FROM voters WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "name": row[0],
            "dob": row[1],
            "address": row[2],
            "voterId": row[3],
            "status": row[4]
        }
    return None


@app.route("/verify-face", methods=["POST"])
def verify_face():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "detail": "No file uploaded"})

        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        frame = np.array(img)

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1], dtype=np.uint8)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if not face_encodings:
            return jsonify({"status": "no_face"})

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                matched_index = matches.index(True)
                matched_name = known_names[matched_index]

                # Save matched photo
                photo_path = os.path.join(PHOTO_DIR, f"{matched_name}.jpg")
                img.save(photo_path)

                # Get voter details
                # voter_data = get_voter_details(matched_name)
                voter_data={"name":matched_name}
                print("voter_data",voter_data)
                if voter_data:
                    # voter_data["photo"] = f"/voter_photos/{matched_name}.jpg"
                    return jsonify({"status": "matched", "profile": matched_name})

                return jsonify({"status": "no_data", "name": matched_name})

        return jsonify({"status": "no_match"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)})


@app.route("/mark-voted/", methods=["POST"])
def mark_voted():
    try:
        data = request.get_json()
        if not data or "name" not in data:
            return jsonify({"status": "error", "detail": "Name is required"})

        name = data["name"]
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE voters SET status='Voted' WHERE name=?", (name,))
        conn.commit()
        conn.close()
        return jsonify({"status": "updated"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)})


# Serve images statically
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(PHOTO_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
