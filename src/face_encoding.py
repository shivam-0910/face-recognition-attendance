"""
Face Encoding / Registration Module (Phase 2)

This module lets a person register their face for later recognition. It
captures a few valid face samples from the webcam, generates a face
encoding for each using the `face_recognition` library, and stores the
encodings (with the person's name) in models/encodings.pkl.

This is Phase 2 of the Face Recognition Attendance System. It only handles
registration/encoding — no recognition or attendance logic is included.
"""

import os
import pickle

import cv2
import face_recognition

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ENCODINGS_PATH = os.path.join(MODELS_DIR, "encodings.pkl")
SAMPLES_REQUIRED = 5


def load_existing_encodings():
    """Load previously saved registrations, if any.

    Returns:
        dict: A dictionary with "names" and "encodings" lists. If the file
        is missing or corrupted, a fresh empty dataset is returned.
    """
    if not os.path.exists(ENCODINGS_PATH):
        return {"names": [], "encodings": []}

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "names" not in data or "encodings" not in data:
            print("Warning: encodings.pkl has an unexpected format. Starting fresh.")
            return {"names": [], "encodings": []}
        return data
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        print("Warning: encodings.pkl could not be read (corrupted). Starting fresh.")
        return {"names": [], "encodings": []}


def save_encodings(data):
    """Save the registrations dictionary to models/encodings.pkl.

    Creates the models/ directory if it does not already exist.

    Args:
        data (dict): The dataset with "names" and "encodings" lists.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)


def remove_existing_person(data, name):
    """Remove all existing samples for a given name from the dataset.

    Args:
        data (dict): The dataset with "names" and "encodings" lists.
        name (str): The person's name to remove.
    """
    keep_names = []
    keep_encodings = []
    for existing_name, existing_encoding in zip(data["names"], data["encodings"]):
        if existing_name != name:
            keep_names.append(existing_name)
            keep_encodings.append(existing_encoding)
    data["names"] = keep_names
    data["encodings"] = keep_encodings


def get_valid_name(data):
    """Prompt the user for a name and handle the duplicate-name case.

    Args:
        data (dict): The current dataset, used to check for duplicates.

    Returns:
        str or None: The validated name to register, or None if the user
        cancels registration.
    """
    while True:
        name = input("Enter person's name: ").strip()
        if not name:
            print("Name cannot be empty. Please try again.")
            continue

        if name in data["names"]:
            answer = input(
                f"'{name}' is already registered. Overwrite existing registration? (y/n): "
            ).strip().lower()
            if answer == "y":
                remove_existing_person(data, name)
                return name
            else:
                print("Registration cancelled for this name. You can enter a different name.")
                continue

        return name


def capture_face_samples(name, samples_required=SAMPLES_REQUIRED):
    """Capture webcam frames and collect valid face encodings for a person.

    Args:
        name (str): The name of the person being registered.
        samples_required (int): Number of valid samples to collect.

    Returns:
        list or None: A list of face encodings, or None if registration
        was aborted (e.g. webcam unavailable or the user pressed 'q').
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam. "
              "Please check that it is connected and not in use by another application.")
        return None

    print(f"Starting registration for '{name}'. Press 'q' to cancel at any time.")

    collected_encodings = []

    while len(collected_encodings) < samples_required:
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

        status_text = ""

        if len(face_locations) == 0:
            status_text = "No face detected"
        elif len(face_locations) > 1:
            status_text = "Multiple faces detected. Please ensure only one person is in frame."
        else:
            # Exactly one face detected: attempt to encode it.
            encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
            if encodings:
                collected_encodings.append(encodings[0])
                status_text = f"Samples: {len(collected_encodings)}/{samples_required}"
            else:
                status_text = "Could not generate encoding. Try again."

        # Draw the face box(es) for visual feedback.
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Registering: {name}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Face Registration - Press 'q' to cancel", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Registration cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    if len(collected_encodings) < samples_required:
        return None

    return collected_encodings


def register_face():
    """Run the full registration flow: get a name, capture samples, save them."""
    data = load_existing_encodings()

    name = get_valid_name(data)
    if name is None:
        print("Registration cancelled.")
        return

    encodings = capture_face_samples(name)
    if not encodings:
        print(f"Registration for '{name}' was not completed. No data was saved.")
        return

    for encoding in encodings:
        data["names"].append(name)
        data["encodings"].append(encoding)

    save_encodings(data)
    print(f"Registration complete for {name}.")


def main():
    """Entry point for face registration."""
    register_face()


if __name__ == "__main__":
    main()