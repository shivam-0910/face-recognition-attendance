"""
Face Detection Module (Phase 1)

This module uses the webcam and OpenCV's Haar Cascade classifier to detect
human faces in real time, draw bounding boxes around them, and display the
live video feed.

This is Phase 1 of the Face Recognition Attendance System. It only performs
face detection — no recognition, encoding, or attendance logic is included.
"""

import cv2


def load_face_detector():
    """Load OpenCV's bundled Haar Cascade face detector.

    Returns:
        cv2.CascadeClassifier: The loaded face detector.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    return face_detector


def open_webcam(camera_index=0):
    """Open the webcam and validate that it was opened successfully.

    Args:
        camera_index (int): Index of the camera to open (default: 0).

    Returns:
        cv2.VideoCapture or None: The video capture object, or None if the
        webcam could not be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    return cap


def detect_faces(face_detector, frame):
    """Detect faces in a video frame using the Haar Cascade detector.

    Args:
        face_detector (cv2.CascadeClassifier): The face detector to use.
        frame (numpy.ndarray): The BGR video frame.

    Returns:
        list: A list of (x, y, w, h) rectangles for each detected face.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return faces


def draw_faces(frame, faces):
    """Draw bounding boxes and labels around detected faces.

    Args:
        frame (numpy.ndarray): The BGR video frame to draw on.
        faces (list): A list of (x, y, w, h) rectangles for each face.
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def draw_face_count(frame, face_count):
    """Display the number of detected faces on the video frame.

    Args:
        frame (numpy.ndarray): The BGR video frame to draw on.
        face_count (int): The number of faces currently detected.
    """
    text = f"Faces: {face_count}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )


def main():
    """Run the live webcam face detection loop."""
    face_detector = load_face_detector()

    cap = open_webcam(0)
    if cap is None:
        print("Error: Could not access the webcam. "
              "Please check that it is connected and not in use by another application.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        frame_was_read, frame = cap.read()
        if not frame_was_read:
            print("Error: Failed to read frame from webcam.")
            break

        # Flip the frame horizontally so the preview behaves like a mirror
        # (moving left on screen matches moving left in real life).
        frame = cv2.flip(frame, 1)

        faces = detect_faces(face_detector, frame)
        draw_faces(frame, faces)
        draw_face_count(frame, len(faces))

        cv2.imshow("Face Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()