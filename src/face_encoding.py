"""
Face Encoding / Registration Module (Phase 2)

This module lets a person register their face for later recognition. It
captures a few valid face samples from the webcam, generates a face
encoding for each using the `face_recognition` library, and stores the
encodings (with the person's name) in models/encodings.pkl.

It also maintains a simple human-readable person registry and stores the
actual captured face-crop images on disk, under:

    data/faces/<PersonName>/face_1.jpg ... face_5.jpg
    data/faces/persons.csv   (Name, RegisteredDate, FaceFolder)

This is purely a management/inspection layer — it does NOT replace the
recognition encoding store. The actual face-recognition data used by
recognition.py continues to live entirely in models/encodings.pkl, which
is unaffected by data/faces/ or persons.csv.

This is Phase 2 of the Face Recognition Attendance System. It only handles
registration/encoding (and the accompanying person registry / face image
archive) — no recognition or attendance logic is included.
"""

import argparse
import csv
import os
import pickle
import shutil
from datetime import datetime

import cv2
import face_recognition

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ENCODINGS_PATH = os.path.join(MODELS_DIR, "encodings.pkl")

FACES_DIR = os.path.join(PROJECT_ROOT, "data", "faces")
PERSONS_CSV_PATH = os.path.join(FACES_DIR, "persons.csv")

SAMPLES_REQUIRED = 5
CSV_FIELDNAMES = ["Name", "RegisteredDate", "FaceFolder"]
FACE_IMAGE_QUALITY = 90  # JPEG quality, reasonable and not excessive.


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


def _normalize_name(name):
    """Apply the project's consistent name comparison rule.

    Names are compared exactly after trimming surrounding whitespace.
    Different names are never treated as the same.

    Args:
        name (str): The raw name.

    Returns:
        str: The trimmed name.
    """
    return name.strip() if name else name


def _is_safe_person_name(name):
    """Check that a (trimmed) name is safe to use as a single directory
    component under data/faces/.

    Rejects empty names, path separators, and any name that is or
    resolves outside data/faces/ (e.g. "..", "../something", an absolute
    path, or a name containing a path separator). This keeps registered
    names from ever escaping the data/faces/ directory.

    Args:
        name (str): An already-trimmed name.

    Returns:
        bool: True if the name is safe to use as a folder name.
    """
    if not name:
        return False
    if name in (".", ".."):
        return False
    # Reject any path separator or drive designator outright — a valid
    # person name is a single path component, not a path.
    if "/" in name or "\\" in name or ":" in name:
        return False

    candidate = os.path.abspath(os.path.join(FACES_DIR, name))
    faces_dir_abs = os.path.abspath(FACES_DIR)
    # candidate must be a direct child of FACES_DIR (no traversal).
    return os.path.dirname(candidate) == faces_dir_abs


def get_person_face_folder(name):
    """Get the absolute path to a person's face-image folder.

    Args:
        name (str): The person's (trimmed) name.

    Returns:
        str: Absolute path to data/faces/<name>/.
    """
    return os.path.join(FACES_DIR, name)


def load_persons_registry():
    """Load the person registry from data/faces/persons.csv.

    If the CSV file does not exist, an empty registry is returned (the
    file itself is only created on first write, by save_persons_registry).
    If the CSV is unreadable or corrupted, a fresh empty registry is
    returned and a warning is printed.

    Returns:
        dict: Mapping of {name: {"RegisteredDate": str, "FaceFolder": str}},
        in insertion order.
    """
    registry = {}

    if not os.path.exists(PERSONS_CSV_PATH):
        return registry

    try:
        with open(PERSONS_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                # Empty file.
                return registry
            for row in reader:
                name = _normalize_name(row.get("Name", ""))
                if not name:
                    continue
                registry[name] = {
                    "RegisteredDate": (row.get("RegisteredDate") or "").strip(),
                    "FaceFolder": (row.get("FaceFolder") or "").strip(),
                }
    except (OSError, csv.Error, UnicodeDecodeError):
        print(f"Warning: could not read {PERSONS_CSV_PATH}. Treating registry as empty.")
        return {}

    return registry


def save_persons_registry(registry):
    """Write the person registry dict to data/faces/persons.csv.

    Creates the data/faces/ directory if it does not already exist.

    Args:
        registry (dict): Mapping of {name: {"RegisteredDate": str, "FaceFolder": str}}.
    """
    os.makedirs(FACES_DIR, exist_ok=True)
    with open(PERSONS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for name, info in registry.items():
            writer.writerow({
                "Name": name,
                "RegisteredDate": info.get("RegisteredDate", ""),
                "FaceFolder": info.get("FaceFolder", ""),
            })


def _relative_face_folder(name):
    """Build the FaceFolder value stored in the CSV, as a forward-slash
    relative path like 'data/faces/<name>' (matches the requested format,
    independent of OS path separators).
    """
    return f"data/faces/{name}"


def upsert_person_registry(name, registered_date=None):
    """Add or update a single person's row in data/faces/persons.csv.

    If the person already has an entry, their registration date and
    face-folder path are updated in place (no duplicate row is created).
    Otherwise a new row is added.

    Args:
        name (str): The person's name (will be trimmed).
        registered_date (str, optional): ISO date string. Defaults to today.

    Returns:
        dict: The updated registry.
    """
    name = _normalize_name(name)
    if registered_date is None:
        registered_date = datetime.now().date().isoformat()

    registry = load_persons_registry()
    registry[name] = {
        "RegisteredDate": registered_date,
        "FaceFolder": _relative_face_folder(name),
    }
    save_persons_registry(registry)
    return registry


def remove_person_registry(name):
    """Remove a single person's row from data/faces/persons.csv, if present.

    Args:
        name (str): The person's name (will be trimmed).

    Returns:
        bool: True if a row was removed, False if the person had no entry.
    """
    name = _normalize_name(name)
    registry = load_persons_registry()

    if name not in registry:
        return False

    del registry[name]
    save_persons_registry(registry)
    return True


def list_persons():
    """List all registered person names, in registry order.

    Returns:
        list: Names currently in data/faces/persons.csv.
    """
    registry = load_persons_registry()
    return list(registry.keys())


def save_face_images(name, face_crops):
    """Save cropped face images for a person to data/faces/<name>/.

    If the person's folder already exists, it is removed first, so a
    fresh registration never mixes old and new samples.

    Args:
        name (str): The person's (trimmed, validated) name.
        face_crops (list): List of BGR numpy image arrays (already cropped
            to just the face region, not full frames).

    Returns:
        list: Paths to the saved JPG files.

    Raises:
        ValueError: If the name is not safe to use as a folder name.
    """
    if not _is_safe_person_name(name):
        raise ValueError(f"Refusing to save face images: unsafe person name '{name}'.")

    folder = get_person_face_folder(name)

    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    saved_paths = []
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), FACE_IMAGE_QUALITY]
    for i, crop in enumerate(face_crops, start=1):
        file_path = os.path.join(folder, f"face_{i}.jpg")
        cv2.imwrite(file_path, crop, encode_params)
        saved_paths.append(file_path)

    return saved_paths


def delete_person_face_folder(name):
    """Delete a person's data/faces/<name>/ folder, if present.

    Args:
        name (str): The person's (trimmed) name.

    Returns:
        bool: True if a folder was removed, False if it didn't exist.

    Raises:
        ValueError: If the name is not safe to use as a folder name.
    """
    if not name:
        return False
    if not _is_safe_person_name(name):
        raise ValueError(f"Refusing to delete: unsafe person name '{name}'.")

    folder = get_person_face_folder(name)
    if not os.path.isdir(folder):
        return False

    shutil.rmtree(folder)
    return True


def delete_person(name):
    """Delete a registered person's face data and registry entry.

    This removes:
      - the person's encodings from models/encodings.pkl
      - their data/faces/<name>/ folder of face images
      - their row from data/faces/persons.csv

    Their historical attendance records are never touched by this
    function.

    Args:
        name (str): The person's name (will be trimmed).

    Returns:
        dict: {
            "success": bool,
            "message": str,
            "encodings_removed": bool,
            "faces_removed": bool,
            "registry_removed": bool,
        }
    """
    name = _normalize_name(name)
    if not name:
        return {
            "success": False,
            "message": "Name cannot be empty.",
            "encodings_removed": False,
            "faces_removed": False,
            "registry_removed": False,
        }

    if not _is_safe_person_name(name):
        return {
            "success": False,
            "message": f"'{name}' is not a valid/safe person name.",
            "encodings_removed": False,
            "faces_removed": False,
            "registry_removed": False,
        }

    data = load_existing_encodings()
    had_encodings = name in data["names"]
    if had_encodings:
        remove_existing_person(data, name)
        save_encodings(data)

    faces_removed = delete_person_face_folder(name)
    registry_removed = remove_person_registry(name)

    if not had_encodings and not faces_removed and not registry_removed:
        return {
            "success": False,
            "message": f"'{name}' was not found in encodings.pkl, data/faces/, or persons.csv. Nothing to delete.",
            "encodings_removed": False,
            "faces_removed": False,
            "registry_removed": False,
        }

    return {
        "success": True,
        "message": (
            f"Deleted '{name}' "
            f"(encodings: {had_encodings}, faces: {faces_removed}, registry: {registry_removed})."
        ),
        "encodings_removed": had_encodings,
        "faces_removed": faces_removed,
        "registry_removed": registry_removed,
    }


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

        if not _is_safe_person_name(name):
            print(
                "That name can't be used safely as a folder name "
                "(no path separators, '..', or similar). Please try a different name."
            )
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
    """Capture webcam frames and collect valid face encodings and crops.

    Args:
        name (str): The name of the person being registered.
        samples_required (int): Number of valid samples to collect.

    Returns:
        tuple or None: (encodings, face_crops) as two parallel lists (one
        encoding and one cropped BGR face image per sample), or None if
        registration was aborted (e.g. webcam unavailable or the user
        pressed 'q').
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam. "
              "Please check that it is connected and not in use by another application.")
        return None

    print(f"Starting registration for '{name}'. Press 'q' to cancel at any time.")

    collected_encodings = []
    collected_crops = []

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
                # Save the cropped BGR face region (not the full frame),
                # matching the same sample used for the encoding.
                top, right, bottom, left = face_locations[0]
                top = max(top, 0)
                left = max(left, 0)
                collected_crops.append(frame[top:bottom, left:right].copy())
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

    return collected_encodings, collected_crops


def register_face():
    """Run the full registration flow: get a name, capture samples, save
    the encodings (models/encodings.pkl), the face-crop images
    (data/faces/<name>/), and the registry row (data/faces/persons.csv).
    """
    data = load_existing_encodings()

    name = get_valid_name(data)
    if name is None:
        print("Registration cancelled.")
        return

    captured = capture_face_samples(name)
    if not captured:
        print(f"Registration for '{name}' was not completed. No data was saved.")
        return

    encodings, face_crops = captured

    for encoding in encodings:
        data["names"].append(name)
        data["encodings"].append(encoding)

    save_encodings(data)
    save_face_images(name, face_crops)
    upsert_person_registry(name)
    print(f"Registration complete for {name}.")


def _print_persons_list():
    """Print the registered persons list per the CLI's expected format:

    Registered persons:
    1. hgl
    2. Shivam
    3. Rahul

    Or, if nobody is registered:

    No registered persons found.
    """
    names = list_persons()
    if not names:
        print("No registered persons found.")
        return
    print("Registered persons:")
    for i, name in enumerate(names, start=1):
        print(f"{i}. {name}")


def main():
    """Entry point for the face_encoding CLI.

    Uses argparse with mutually-exclusive --list / --delete flags so that
    supplying either one lists/deletes and returns immediately, WITHOUT
    ever falling through to interactive registration (register_face(),
    which calls input()). Only when neither flag is supplied does it run
    normal webcam registration:

        python src/face_encoding.py            -> register a person
        python src/face_encoding.py --list      -> list registered people
        python src/face_encoding.py --delete X  -> delete person X
    """
    parser = argparse.ArgumentParser(
        description="Register faces, or manage the person registry (data/faces/persons.csv)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list", action="store_true",
        help="List all registered persons (from data/faces/persons.csv) and exit.",
    )
    group.add_argument(
        "--delete", metavar="NAME", default=None,
        help="Delete a registered person's face data and registry entry, then exit.",
    )
    args = parser.parse_args()

    if args.list:
        _print_persons_list()
        return

    if args.delete is not None:
        result = delete_person(args.delete)
        print(result["message"])
        return

    # No CLI flags supplied: fall back to normal interactive registration.
    register_face()


if __name__ == "__main__":
    main()