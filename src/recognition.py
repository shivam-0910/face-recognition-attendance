"""
Face Recognition Module (Phase 3)

This module recognizes people in the live webcam feed by comparing
detected face encodings against the registrations stored in
models/encodings.pkl (created by src/face_encoding.py). If a detected
face matches a known encoding closely enough, the person's name is shown
on screen; otherwise the face is labeled "Unknown".

This is Phase 3 of the Face Recognition Attendance System. It only
handles recognition/display — no attendance logging is included.
"""

import os
import pickle

import cv2
import face_recognition

from attendance import mark_attendance

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ENCODINGS_PATH = os.path.join(MODELS_DIR, "encodings.pkl")

# Lower = stricter match. 0.6 is face_recognition's commonly used default.
MATCH_TOLERANCE = 0.6


def load_known_encodings():
    """Load registered face encodings from models/encodings.pkl.

    Returns:
        dict or None: A dictionary with "names" and "encodings" lists, or
        None if no valid registrations could be loaded.
    """
    if not os.path.exists(ENCODINGS_PATH):
        print(f"Error: {ENCODINGS_PATH} not found. "
              "Please register at least one person first by running src/face_encoding.py.")
        return None

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        print(f"Error: {ENCODINGS_PATH} is corrupted or unreadable. "
              "Please re-run src/face_encoding.py to register faces again.")
        return None

    if not isinstance(data, dict) or "names" not in data or "encodings" not in data:
        print(f"Error: {ENCODINGS_PATH} has an unexpected format.")
        return None

    if len(data["names"]) == 0:
        print("Error: No registered faces found. "
              "Please register at least one person first by running src/face_encoding.py.")
        return None

    return data


def identify_face(face_encoding, known_data):
    """Compare one face encoding against all known encodings.

    Args:
        face_encoding (numpy.ndarray): The encoding of the detected face.
        known_data (dict): Dataset with "names" and "encodings" lists.

    Returns:
        str: The matched person's name, or "Unknown" if no close-enough
        match is found.
    """
    known_encodings = known_data["encodings"]
    known_names = known_data["names"]

    distances = face_recognition.face_distance(known_encodings, face_encoding)

    if len(distances) == 0:
        return "Unknown"

    best_match_index = distances.argmin()
    best_distance = distances[best_match_index]

    if best_distance <= MATCH_TOLERANCE:
        return known_names[best_match_index]

    return "Unknown"


def draw_recognition_results(frame, face_locations, names):
    """Draw bounding boxes and recognized names on the frame.

    Args:
        frame (numpy.ndarray): The BGR video frame to draw on.
        face_locations (list): List of (top, right, bottom, left) tuples.
        names (list): List of names corresponding to each face location.
    """
    for (top, right, bottom, left), name in zip(face_locations, names):
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10 if top - 10 > 10 else top + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
        )


def run_recognition():
    """Run the live webcam face recognition loop."""
    known_data = load_known_encodings()
    if known_data is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam. "
              "Please check that it is connected and not in use by another application.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        frame_was_read, frame = cap.read()
        if not frame_was_read:
            print("Error: Failed to read frame from webcam.")
            break

        # Mirror the preview so it behaves naturally, and keep detection
        # consistent with what is displayed.
        frame = cv2.flip(frame, 1)

        # face_recognition expects RGB images; OpenCV frames are BGR.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)

        names = [identify_face(encoding, known_data) for encoding in face_encodings]

        for name in names:
            mark_attendance(name)

        draw_recognition_results(frame, face_locations, names)

        cv2.imshow("Face Recognition - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Entry point for face recognition."""
    run_recognition()


if __name__ == "__main__":
    main()