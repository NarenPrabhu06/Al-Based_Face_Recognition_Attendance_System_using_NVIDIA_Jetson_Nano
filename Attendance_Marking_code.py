import csv
import os
from datetime import datetime

CSV_FILE = "attendance.csv"


def create_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def mark_attendance(name):

    create_csv()

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    already_present = False

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)

        next(reader, None)

        for row in reader:
            if len(row) >= 2:
                if row[0] == name and row[1] == today:
                    already_present = True
                    break

    if not already_present:

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, today, current_time])

        print(name, "Attendance Marked")

        return True

    return False
