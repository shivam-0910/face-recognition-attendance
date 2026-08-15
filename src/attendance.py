"""
Attendance Marking Module (Phase 4)

This module records attendance for recognized faces. When a person is
recognized (via src/recognition.py), their name and the current date/time
are logged to a simple daily text/CSV file in the database/ folder.

This is Phase 4 of the Face Recognition Attendance System. It only
handles attendance logging — no GUI, SQLite database, CSV export, or
reporting is included.
"""

import os
from datetime import datetime

DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")


def get_attendance_file_path(date=None):
    """Get the path to today's (or a given date's) attendance file.

    A separate file is used per day, named attendance_YYYY-MM-DD.csv.

    Args:
        date (datetime.date, optional): The date to use. Defaults to today.

    Returns:
        str: Full path to the attendance file for that date.
    """
    if date is None:
        date = datetime.now().date()
    filename = f"attendance_{date.isoformat()}.csv"
    return os.path.join(DATABASE_DIR, filename)


def load_todays_attendance(date=None):
    """Load the set of names already marked present for a given date.

    Args:
        date (datetime.date, optional): The date to check. Defaults to today.

    Returns:
        set: Names already marked present on that date.
    """
    file_path = get_attendance_file_path(date)

    if not os.path.exists(file_path):
        return set()

    names_present = set()
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Name,"):
                    continue
                parts = line.split(",")
                if parts:
                    names_present.add(parts[0])
    except OSError:
        print(f"Warning: could not read {file_path}. Assuming no attendance recorded yet today.")
        return set()

    return names_present


def mark_attendance(name):
    """Record attendance for a recognized person, once per day.

    Creates the database/ folder and today's attendance file if they do
    not already exist. Writes a header row on first creation.

    Args:
        name (str): The recognized person's name. "Unknown" is ignored.

    Returns:
        bool: True if a new attendance record was written, False if the
        person was already marked present today (or the name was invalid).
    """
    if not name or name == "Unknown":
        return False

    os.makedirs(DATABASE_DIR, exist_ok=True)

    already_present = load_todays_attendance()
    if name in already_present:
        return False

    file_path = get_attendance_file_path()
    file_exists = os.path.exists(file_path)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            if not file_exists:
                f.write("Name,Date,Time\n")
            f.write(f"{name},{date_str},{time_str}\n")
    except OSError as e:
        print(f"Error: could not write attendance record for {name}: {e}")
        return False

    print(f"Attendance marked: {name} at {time_str}")
    return True


def main():
    """Simple manual test entry point (not the primary usage path).

    Normally mark_attendance() is called from recognition.py once a face
    is recognized. This main() just demonstrates marking attendance for a
    name typed in manually.
    """
    name = input("Enter recognized name to mark attendance for: ").strip()
    marked = mark_attendance(name)
    if marked:
        print(f"Recorded attendance for {name}.")
    else:
        print(f"No new record written for '{name}' (already marked today, or invalid name).")


if __name__ == "__main__":
    main()