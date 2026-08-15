"""
Tests for the data/faces/ person-data storage introduced in
src/face_encoding.py:

- data/faces/<Name>/face_1.jpg ... face_N.jpg (cropped face images)
- data/faces/persons.csv (Name, RegisteredDate, FaceFolder)
- delete_person() removing encodings + face folder + registry row
- name-safety guarding against path traversal
- compatibility: recognition.py still loads models/encodings.pkl,
  and historical attendance CSVs are never touched

These tests use temporary directories/files only. They never read or
write the real project's models/, data/, or database/ folders.
"""

import csv
import os
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

import face_encoding  # noqa: E402
import recognition  # noqa: E402


class PersonRegistryTestCase(unittest.TestCase):
    """Base test case that redirects all storage paths (in both
    face_encoding and recognition) to a temporary directory, so tests
    never read or write real registration/face-image data.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="face_registry_test_")
        self.models_dir = os.path.join(self.temp_dir, "models")
        self.faces_dir = os.path.join(self.temp_dir, "data", "faces")

        # Save originals to restore in tearDown.
        self._orig = {
            "fe_PROJECT_ROOT": face_encoding.PROJECT_ROOT,
            "fe_MODELS_DIR": face_encoding.MODELS_DIR,
            "fe_ENCODINGS_PATH": face_encoding.ENCODINGS_PATH,
            "fe_FACES_DIR": face_encoding.FACES_DIR,
            "fe_PERSONS_CSV_PATH": face_encoding.PERSONS_CSV_PATH,
            "rec_MODELS_DIR": recognition.MODELS_DIR,
            "rec_ENCODINGS_PATH": recognition.ENCODINGS_PATH,
        }

        face_encoding.PROJECT_ROOT = self.temp_dir
        face_encoding.MODELS_DIR = self.models_dir
        face_encoding.ENCODINGS_PATH = os.path.join(self.models_dir, "encodings.pkl")
        face_encoding.FACES_DIR = self.faces_dir
        face_encoding.PERSONS_CSV_PATH = os.path.join(self.faces_dir, "persons.csv")

        recognition.MODELS_DIR = self.models_dir
        recognition.ENCODINGS_PATH = face_encoding.ENCODINGS_PATH

    def tearDown(self):
        face_encoding.PROJECT_ROOT = self._orig["fe_PROJECT_ROOT"]
        face_encoding.MODELS_DIR = self._orig["fe_MODELS_DIR"]
        face_encoding.ENCODINGS_PATH = self._orig["fe_ENCODINGS_PATH"]
        face_encoding.FACES_DIR = self._orig["fe_FACES_DIR"]
        face_encoding.PERSONS_CSV_PATH = self._orig["fe_PERSONS_CSV_PATH"]
        recognition.MODELS_DIR = self._orig["rec_MODELS_DIR"]
        recognition.ENCODINGS_PATH = self._orig["rec_ENCODINGS_PATH"]

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -- helpers ----------------------------------------------------
    def _write_encodings(self, names, encodings=None):
        """Directly seed models/encodings.pkl with given names (and dummy
        128-d encodings, one per name if not provided)."""
        if encodings is None:
            encodings = [[0.0] * 128 for _ in names]
        os.makedirs(self.models_dir, exist_ok=True)
        with open(face_encoding.ENCODINGS_PATH, "wb") as f:
            pickle.dump({"names": names, "encodings": encodings}, f)

    def _read_csv_rows(self):
        if not os.path.exists(face_encoding.PERSONS_CSV_PATH):
            return []
        with open(face_encoding.PERSONS_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _dummy_face_crops(self, count=5):
        """A handful of small fake BGR image arrays, standing in for real
        cropped face regions (cv2.imwrite works fine on any ndarray)."""
        return [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(count)]

    def _register_person(self, name, date="2026-08-15", num_images=5):
        """Simulate a full registration for one person: write encodings,
        save face images, and upsert the registry row — mirroring what
        register_face()/gui.py do end to end."""
        data = face_encoding.load_existing_encodings()
        for _ in range(num_images):
            data["names"].append(name)
            data["encodings"].append([0.0] * 128)
        face_encoding.save_encodings(data)
        face_encoding.save_face_images(name, self._dummy_face_crops(num_images))
        face_encoding.upsert_person_registry(name, date)


class TestFaceImageStorage(PersonRegistryTestCase):
    def test_registration_creates_person_folder(self):
        self._register_person("Shivam")
        folder = face_encoding.get_person_face_folder("Shivam")
        self.assertTrue(os.path.isdir(folder))

    def test_five_face_images_saved(self):
        self._register_person("Shivam", num_images=5)
        folder = face_encoding.get_person_face_folder("Shivam")
        files = sorted(os.listdir(folder))
        self.assertEqual(files, [f"face_{i}.jpg" for i in range(1, 6)])
        for fname in files:
            self.assertGreater(os.path.getsize(os.path.join(folder, fname)), 0)


class TestCsvCreation(PersonRegistryTestCase):
    def test_persons_csv_created_automatically(self):
        self.assertFalse(os.path.exists(face_encoding.PERSONS_CSV_PATH))
        self._register_person("Shivam")
        self.assertTrue(os.path.exists(face_encoding.PERSONS_CSV_PATH))

    def test_empty_csv_handled_gracefully(self):
        os.makedirs(self.faces_dir, exist_ok=True)
        open(face_encoding.PERSONS_CSV_PATH, "w").close()
        registry = face_encoding.load_persons_registry()
        self.assertEqual(registry, {})

    def test_corrupted_csv_handled_gracefully(self):
        os.makedirs(self.faces_dir, exist_ok=True)
        with open(face_encoding.PERSONS_CSV_PATH, "wb") as f:
            f.write(b"\xff\xfe\x00\x01not,valid,csv\x00\x00")
        registry = face_encoding.load_persons_registry()
        self.assertIsInstance(registry, dict)

    def test_csv_has_exactly_one_row_per_person(self):
        self._register_person("Shivam")
        self._register_person("Rahul")
        self._register_person("Asha")
        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 3)
        names = {row["Name"] for row in rows}
        self.assertEqual(names, {"Shivam", "Rahul", "Asha"})

    def test_csv_row_format(self):
        self._register_person("Shivam", date="2026-08-15")
        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Name"], "Shivam")
        self.assertEqual(rows[0]["RegisteredDate"], "2026-08-15")
        self.assertEqual(rows[0]["FaceFolder"], "data/faces/Shivam")


class TestReRegistration(PersonRegistryTestCase):
    def test_reregistering_replaces_face_folder(self):
        self._register_person("Shivam", num_images=5)
        folder = face_encoding.get_person_face_folder("Shivam")

        # Overwrite: remove old encodings first (as register_face() does
        # via remove_existing_person before capturing new samples), then
        # register again with a different image count to prove the old
        # folder was actually replaced, not merged.
        data = face_encoding.load_existing_encodings()
        face_encoding.remove_existing_person(data, "Shivam")
        face_encoding.save_encodings(data)
        self._register_person("Shivam", num_images=3)

        new_files = set(os.listdir(folder))
        self.assertEqual(len(new_files), 3)

    def test_reregistering_does_not_duplicate_csv_row(self):
        self._register_person("Shivam", date="2026-08-15")
        self._register_person("Shivam", date="2026-08-16")

        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["RegisteredDate"], "2026-08-16")


class TestListing(PersonRegistryTestCase):
    def test_list_reads_persons_csv(self):
        self._register_person("Shivam")
        self._register_person("Rahul")
        self._register_person("Asha")

        names = face_encoding.list_persons()
        self.assertEqual(names, ["Shivam", "Rahul", "Asha"])

    def test_list_empty_when_nobody_registered(self):
        self.assertEqual(face_encoding.list_persons(), [])


class TestDeletion(PersonRegistryTestCase):
    def test_delete_removes_face_folder(self):
        self._register_person("Rahul")
        folder = face_encoding.get_person_face_folder("Rahul")
        self.assertTrue(os.path.isdir(folder))

        result = face_encoding.delete_person("Rahul")

        self.assertTrue(result["success"])
        self.assertTrue(result["faces_removed"])
        self.assertFalse(os.path.isdir(folder))

    def test_delete_removes_csv_row(self):
        self._register_person("Shivam")
        self._register_person("Rahul")

        face_encoding.delete_person("Rahul")

        names_left = face_encoding.list_persons()
        self.assertNotIn("Rahul", names_left)
        self.assertIn("Shivam", names_left)

    def test_delete_removes_encodings(self):
        self._register_person("Shivam", num_images=2)
        self._register_person("Rahul", num_images=3)

        face_encoding.delete_person("Rahul")

        data = face_encoding.load_existing_encodings()
        self.assertNotIn("Rahul", data["names"])
        self.assertEqual(data["names"].count("Shivam"), 2)

    def test_deleting_nonexistent_person_does_not_crash(self):
        self._register_person("Shivam")
        result = face_encoding.delete_person("NoSuchPerson")
        self.assertFalse(result["success"])
        self.assertIn("Shivam", face_encoding.list_persons())

    def test_deleting_one_person_does_not_affect_another(self):
        self._register_person("Shivam")
        self._register_person("Rahul")
        self._register_person("Asha")

        shivam_folder = face_encoding.get_person_face_folder("Shivam")
        asha_folder = face_encoding.get_person_face_folder("Asha")

        face_encoding.delete_person("Rahul")

        self.assertTrue(os.path.isdir(shivam_folder))
        self.assertTrue(os.path.isdir(asha_folder))

        data = face_encoding.load_existing_encodings()
        self.assertIn("Shivam", data["names"])
        self.assertIn("Asha", data["names"])
        self.assertNotIn("Rahul", data["names"])

        names_left = face_encoding.list_persons()
        self.assertIn("Shivam", names_left)
        self.assertIn("Asha", names_left)
        self.assertNotIn("Rahul", names_left)


class TestNameSafety(PersonRegistryTestCase):
    def test_rejects_path_traversal_dotdot(self):
        self.assertFalse(face_encoding._is_safe_person_name(".."))

    def test_rejects_traversal_with_subpath(self):
        self.assertFalse(face_encoding._is_safe_person_name("../something"))
        self.assertFalse(face_encoding._is_safe_person_name("..\\something"))

    def test_rejects_embedded_separators(self):
        self.assertFalse(face_encoding._is_safe_person_name("foo/bar"))
        self.assertFalse(face_encoding._is_safe_person_name("foo\\bar"))

    def test_rejects_absolute_path(self):
        self.assertFalse(face_encoding._is_safe_person_name("/etc/passwd"))

    def test_accepts_normal_name(self):
        self.assertTrue(face_encoding._is_safe_person_name("Shivam"))

    def test_save_face_images_raises_on_unsafe_name(self):
        with self.assertRaises(ValueError):
            face_encoding.save_face_images("../evil", self._dummy_face_crops(1))

    def test_delete_person_rejects_unsafe_name_gracefully(self):
        result = face_encoding.delete_person("../evil")
        self.assertFalse(result["success"])


class TestRecognitionCompatibility(PersonRegistryTestCase):
    def test_recognition_can_still_load_encodings_pkl(self):
        self._write_encodings(["Shivam", "Rahul"])
        # persons.csv / data/faces intentionally NOT created, to prove
        # recognition does not depend on either.
        self.assertFalse(os.path.exists(face_encoding.PERSONS_CSV_PATH))

        data = recognition.load_known_encodings()
        self.assertIsNotNone(data)
        self.assertEqual(data["names"], ["Shivam", "Rahul"])


class TestAttendanceUntouched(PersonRegistryTestCase):
    def test_historical_attendance_csv_not_modified_by_deletion(self):
        database_dir = os.path.join(self.temp_dir, "database")
        os.makedirs(database_dir, exist_ok=True)
        attendance_file = os.path.join(database_dir, "attendance_2026-08-10.csv")
        original_content = "Name,Date,Time\nRahul,2026-08-10,09:00:00\n"
        with open(attendance_file, "w", encoding="utf-8") as f:
            f.write(original_content)

        self._register_person("Rahul", date="2026-08-10")

        face_encoding.delete_person("Rahul")

        with open(attendance_file, "r", encoding="utf-8") as f:
            content_after = f.read()

        self.assertEqual(content_after, original_content)


if __name__ == "__main__":
    unittest.main()