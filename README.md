# Face Recognition Attendance System

## Description

This project is an AI-powered attendance application that uses a webcam,
face recognition, and SQLite to identify registered people and record their
attendance automatically. It aims to replace manual attendance tracking with
a simple, camera-based recognition workflow.

## Project Objective

The goal of this project is to build a simple, working face-recognition-based
attendance system as part of the Bright Hub AI Internship, Project 3.

## Planned Features

- Face detection
- Face registration
- Face encoding
- Face recognition
- Unknown-person detection
- Automatic attendance logging
- Duplicate attendance prevention
- SQLite database
- CSV attendance export
- Simple GUI
- Basic performance evaluation

## Technology Stack

- Python
- OpenCV
- face_recognition
- NumPy
- Pandas
- SQLite
- Tkinter

## Project Structure

```text
face-recognition-attendance/
│
├── data/
│   └── faces/
│       └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── database/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── face_detection.py
│   ├── face_encoding.py
│   ├── recognition.py
│   ├── attendance.py
│   ├── database.py
│   ├── csv_export.py
│   └── gui.py
│
├── tests/
│   └── .gitkeep
│
├── screenshots/
│   └── .gitkeep
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate the virtual environment:

   ```powershell
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Development Status

**Status: Initial project setup**

- Phase 1 — Face Detection: Implemented
- Phase 2 — Face Encoding / Registration: Implemented
- Phase 3 — Face Recognition: Implemented
- Phase 4 — Attendance Marking: Implemented
- Phase 5 — Simple GUI: Implemented
  - Stability fix applied: recognition now runs every 5th frame (not every frame) and attendance checks use an in-memory same-day cache, to prevent the GUI from slowing down/freezing during extended use.

Implementation will be developed incrementally, feature by feature.

### Registered People

The project stores each registered person's captured face images and a
human-readable registry under:

```
data/faces/
├── persons.csv
├── Shivam/
│   ├── face_1.jpg
│   ├── face_2.jpg
│   ├── face_3.jpg
│   ├── face_4.jpg
│   └── face_5.jpg
├── Rahul/
│   └── ...
└── Asha/
    └── ...
```

`data/faces/persons.csv` has one row per registered person: `Name,RegisteredDate,FaceFolder`.

- The `.jpg` files under each person's folder are the actual cropped face samples captured during registration (not full webcam frames).
- `persons.csv` is the human-readable registry — use it to see who is registered, at a glance, without loading the pickle file.
- `models/encodings.pkl` still contains the actual face encodings used for recognition; neither `data/faces/` nor `persons.csv` store encoding data, and recognition does not depend on either being present.
- Deleting a person removes their future recognition data (encodings, face images, and registry row).
- Historical attendance records in `database/attendance_YYYY-MM-DD.csv` are never modified by registration or deletion.

You can manage the registry from the command line, without the GUI:

```powershell
python src/face_encoding.py --list
python src/face_encoding.py --delete PersonName
```

Running `python src/face_encoding.py` with no flags starts normal webcam registration, as before.

## Internship Context

**Bright Hub Private Limited — Artificial Intelligence Internship — Project 3**