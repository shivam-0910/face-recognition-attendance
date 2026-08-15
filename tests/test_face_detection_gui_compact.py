"""
Regression test for a compatibility break where src/face_detection.py had
been overwritten with the wrong module's content, causing:

    ImportError: cannot import name 'load_face_detector' from 'face_detection'

when running `python src/gui.py`. This test asserts that face_detection.py
exposes the Phase 1 API gui.py depends on, and that gui.py itself imports
without error.
"""

import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)


class TestFaceDetectionApi(unittest.TestCase):
    def test_face_detection_exposes_required_functions(self):
        import face_detection

        for name in ("load_face_detector", "open_webcam", "detect_faces",
                     "draw_faces", "draw_face_count", "main"):
            self.assertTrue(
                hasattr(face_detection, name) and callable(getattr(face_detection, name)),
                f"face_detection.{name} is missing or not callable",
            )

    def test_load_face_detector_returns_usable_classifier(self):
        import cv2
        import face_detection

        detector = face_detection.load_face_detector()
        self.assertIsInstance(detector, cv2.CascadeClassifier)
        self.assertFalse(detector.empty())

    def test_detect_faces_draw_faces_draw_face_count_run_on_blank_frame(self):
        import numpy as np
        import face_detection

        detector = face_detection.load_face_detector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = face_detection.detect_faces(detector, frame)
        self.assertIsInstance(faces, list)

        # Should not raise, even with zero faces.
        face_detection.draw_faces(frame, faces)
        face_detection.draw_face_count(frame, len(faces))


class TestGuiImportsSuccessfully(unittest.TestCase):
    def test_gui_module_imports_without_error(self):
        # This is the exact failure mode originally reported:
        # `python src/gui.py` raising ImportError on `from face_detection
        # import load_face_detector, ...`. Importing the gui module here
        # exercises that same import line.
        import gui  # noqa: F401

    def test_gui_uses_expected_face_detection_symbols(self):
        import gui
        import face_detection

        # gui.py imports these names directly into its own namespace;
        # confirm they resolve to the same functions as face_detection's.
        self.assertIs(gui.load_face_detector, face_detection.load_face_detector)
        self.assertIs(gui.detect_faces, face_detection.detect_faces)
        self.assertIs(gui.draw_faces, face_detection.draw_faces)
        self.assertIs(gui.draw_face_count, face_detection.draw_face_count)


if __name__ == "__main__":
    unittest.main()
    