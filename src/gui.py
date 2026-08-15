"""
Simple Desktop GUI (Phase 5)

This module provides a simple Tkinter desktop application that embeds the
live webcam feed directly inside the window and lets the user switch
between three modes:

- Detect  : plain face detection (reuses face_detection.py logic)
- Register: capture face samples for a new person (reuses face_encoding.py)
- Recognize: live recognition + automatic attendance marking (reuses
             recognition.py and attendance.py)

This is Phase 5 of the Face Recognition Attendance System. It only wires
the existing, already-tested modules into a simple GUI — it does not
duplicate or rewrite their core logic.
"""

import os
import tkinter as tk
from tkinter import messagebox, simpledialog

import cv2
from PIL import Image, ImageTk

from face_detection import load_face_detector, detect_faces, draw_faces, draw_face_count
from face_encoding import (
    load_existing_encodings,
    save_encodings,
    remove_existing_person,
)
from recognition import load_known_encodings, identify_face, draw_recognition_results
from attendance import mark_attendance

import face_recognition

WINDOW_TITLE = "Face Recognition Attendance System"
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
SAMPLES_REQUIRED = 5


class AttendanceApp:
    """Main Tkinter application window."""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.resizable(False, False)

        self.cap = None
        self.mode = "idle"  # "idle", "detect", "register", "recognize"
        self.face_detector = load_face_detector()

        # Registration state
        self.registration_name = None
        self.registration_encodings = []
        self.registration_data = None

        # Recognition state
        self.known_data = None

        self._build_layout()
        self._update_status("Ready. Choose a mode to begin.")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        video_frame = tk.Frame(self.root, bg="black", width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        video_frame.pack(padx=10, pady=10)
        video_frame.pack_propagate(False)

        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(pady=(0, 10))

        self.detect_button = tk.Button(
            controls_frame, text="Start Detection", width=16, command=self.start_detect_mode
        )
        self.detect_button.grid(row=0, column=0, padx=5)

        self.register_button = tk.Button(
            controls_frame, text="Register Person", width=16, command=self.start_register_mode
        )
        self.register_button.grid(row=0, column=1, padx=5)

        self.recognize_button = tk.Button(
            controls_frame, text="Start Recognition", width=16, command=self.start_recognize_mode
        )
        self.recognize_button.grid(row=0, column=2, padx=5)

        self.stop_button = tk.Button(
            controls_frame, text="Stop", width=16, command=self.stop_camera
        )
        self.stop_button.grid(row=0, column=3, padx=5)

        self.status_label = tk.Label(self.root, text="", anchor="w", fg="blue")
        self.status_label.pack(fill="x", padx=10, pady=(0, 10))

    def _update_status(self, text):
        self.status_label.config(text=text)

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    def _open_camera(self):
        if self.cap is not None:
            return True

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Webcam Error",
                "Could not access the webcam. Please check that it is connected "
                "and not in use by another application.",
            )
            self.cap = None
            return False
        return True

    def stop_camera(self):
        """Stop the current mode and release the camera."""
        self.mode = "idle"
        self.registration_name = None
        self.registration_encodings = []

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_label.config(image="")
        self._update_status("Stopped. Choose a mode to begin.")

    # ------------------------------------------------------------------
    # Mode: Detect
    # ------------------------------------------------------------------
    def start_detect_mode(self):
        if not self._open_camera():
            return
        self.mode = "detect"
        self._update_status("Detection running. Click Stop to end.")
        self._process_frame()

    # ------------------------------------------------------------------
    # Mode: Register
    # ------------------------------------------------------------------
    def start_register_mode(self):
        name = simpledialog.askstring("Register Person", "Enter person's name:", parent=self.root)
        if name is not None:
            name = name.strip()

        if not name:
            messagebox.showwarning("Invalid Name", "Name cannot be empty.")
            return

        self.registration_data = load_existing_encodings()

        if name in self.registration_data["names"]:
            overwrite = messagebox.askyesno(
                "Name Already Registered",
                f"'{name}' is already registered. Overwrite existing registration?",
            )
            if not overwrite:
                self._update_status("Registration cancelled.")
                return
            remove_existing_person(self.registration_data, name)

        if not self._open_camera():
            return

        self.registration_name = name
        self.registration_encodings = []
        self.mode = "register"
        self._update_status(f"Registering '{name}'. Samples: 0/{SAMPLES_REQUIRED}")
        self._process_frame()

    def _finish_registration(self):
        for encoding in self.registration_encodings:
            self.registration_data["names"].append(self.registration_name)
            self.registration_data["encodings"].append(encoding)
        save_encodings(self.registration_data)

        completed_name = self.registration_name
        self.registration_name = None
        self.registration_encodings = []

        self.stop_camera()
        messagebox.showinfo("Registration Complete", f"Registration complete for {completed_name}.")

    # ------------------------------------------------------------------
    # Mode: Recognize
    # ------------------------------------------------------------------
    def start_recognize_mode(self):
        self.known_data = load_known_encodings()
        if self.known_data is None:
            messagebox.showwarning(
                "No Registered Faces",
                "No registered faces found. Please register at least one person first.",
            )
            return

        if not self._open_camera():
            return

        self.mode = "recognize"
        self._update_status("Recognition running. Attendance is marked automatically.")
        self._process_frame()

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------
    def _process_frame(self):
        if self.mode == "idle" or self.cap is None:
            return

        frame_was_read, frame = self.cap.read()
        if not frame_was_read:
            self._update_status("Error: failed to read frame from webcam.")
            self.stop_camera()
            return

        # Mirror the preview so it behaves naturally, consistent with the
        # other Phase 1-4 modules.
        frame = cv2.flip(frame, 1)

        if self.mode == "detect":
            faces = detect_faces(self.face_detector, frame)
            draw_faces(frame, faces)
            draw_face_count(frame, len(faces))

        elif self.mode == "register":
            self._process_register_frame(frame)

        elif self.mode == "recognize":
            self._process_recognize_frame(frame)

        self._display_frame(frame)

        # Schedule the next frame update (~30 FPS).
        self.root.after(30, self._process_frame)

    def _process_register_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        status_text = ""

        if len(face_locations) == 0:
            status_text = "No face detected"
        elif len(face_locations) > 1:
            status_text = "Multiple faces detected. Please ensure only one person is in frame."
        else:
            encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
            if encodings:
                self.registration_encodings.append(encodings[0])
                status_text = f"Samples: {len(self.registration_encodings)}/{SAMPLES_REQUIRED}"
                self._update_status(f"Registering '{self.registration_name}'. {status_text}")
            else:
                status_text = "Could not generate encoding. Try again."

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(
            frame, f"Registering: {self.registration_name}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
        )

        if len(self.registration_encodings) >= SAMPLES_REQUIRED:
            self.root.after(0, self._finish_registration)

    def _process_recognize_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)

        names = [identify_face(encoding, self.known_data) for encoding in face_encodings]

        for name in names:
            if mark_attendance(name):
                self._update_status(f"Attendance marked: {name}")

        draw_recognition_results(frame, face_locations, names)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _display_frame(self, frame):
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_display)
        photo = ImageTk.PhotoImage(image=image)

        # Keep a reference, otherwise Tkinter garbage-collects the image.
        self.video_label.imgtk = photo
        self.video_label.config(image=photo)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()