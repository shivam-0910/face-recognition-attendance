"""
Face Detection Module (Phase 1)

This module provides plain face detection from the webcam using OpenCV's
Haar Cascade classifier. It is the foundation Phase 1 module: it only
detects and draws face bounding boxes on live frames — it does not
perform recognition, registration/encoding, or attendance logging.

src/gui.py's "Detect" mode reuses this module's functions directly, so
its public API (load_face_detector, open_webcam, detect_faces,
draw_faces, draw_face_count) is relied upon elsewhere in the project and
must be kept stable.
"""

import cv2

# OpenCV ships this Haar Cascade file alongside the cv2 package.
FACE_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


def load_face_detector():
    """Load the Haar Cascade face detector.

    Returns:
        cv2.CascadeClassifier: The loaded face detector.

    Raises:
        IOError: If the cascade file could not be loaded.
    """
    cascade_path = cv2.data.haarcascades + FACE_CASCADE_FILENAME
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise IOError(
            f"Could not load Haar Cascade classifier from '{cascade_path}'. "
            "Please check your OpenCV installation."
        )

    return detector


def open_webcam(camera_index=0):
    """Open the default webcam.

    Args:
        camera_index (int): The camera device index to open. Defaults to 0.

    Returns:
        cv2.VideoCapture or None: The opened capture object, or None if
        the webcam could not be accessed.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not access the webcam. "
              "Please check that it is connected and not in use by another application.")
        return None
    return cap


def detect_faces(detector, frame):
    """Detect faces in a single BGR frame.

    Args:
        detector (cv2.CascadeClassifier): The loaded face detector.
        frame (numpy.ndarray): The BGR video frame to search.

    Returns:
        list: A list of (x, y, w, h) bounding boxes, one per detected face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return list(faces)


def draw_faces(frame, faces):
    """Draw bounding boxes around detected faces, in place.

    Args:
        frame (numpy.ndarray): The BGR video frame to draw on.
        faces (list): List of (x, y, w, h) bounding boxes.
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_face_count(frame, count):
    """Draw the current detected-face count in the top-left corner, in place.

    Args:
        frame (numpy.ndarray): The BGR video frame to draw on.
        count (int): The number of faces currently detected.
    """
    cv2.putText(
        frame,
        f"Faces detected: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )


def run_detection():
    """Run the live webcam face detection loop (standalone Phase 1 demo)."""
    detector = load_face_detector()

    cap = open_webcam()
    if cap is None:
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

        faces = detect_faces(detector, frame)
        draw_faces(frame, faces)
        draw_face_count(frame, len(faces))

        cv2.imshow("Face Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Entry point for standalone face detection."""
    run_detection()


if __name__ == "__main__":
    main()