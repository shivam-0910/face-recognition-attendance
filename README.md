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

Implementation will be developed incrementally, feature by feature.

## Internship Context

**Bright Hub Private Limited — Artificial Intelligence Internship — Project 3**