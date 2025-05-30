"""
Manages attendance logic and record-keeping in a CSV file.
"""

import pandas as pd
from datetime import datetime

ATTENDANCE_FILE = "attendance_records.csv"

def load_attendance_records():
    """
    Loads attendance records from the CSV file.

    Returns:
        pd.DataFrame: The attendance records.
    """
    try:
        # Define column data types to ensure check_out_time is read as string
        dtype_spec = {
            "student_id": str,
            "date": str,
            "check_in_time": str,
            "check_out_time": str
        }
        return pd.read_csv(ATTENDANCE_FILE, dtype=dtype_spec)
    except FileNotFoundError:
        return pd.DataFrame(columns=["student_id", "date", "check_in_time", "check_out_time"])

def save_attendance_records(df):
    """
    Saves attendance records to the CSV file.

    Args:
        df (pd.DataFrame): The attendance records to save.
    """
    df.to_csv(ATTENDANCE_FILE, index=False)

def record_attendance(student_id):
    """
    Records attendance for a student.

    Args:
        student_id (str): The ID of the recognized student.

    Returns:
        str: A message indicating the attendance action (check-in/check-out/already recorded).
    """
    df = load_attendance_records()
    today = datetime.now().date().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    student_records_today = df[(df["student_id"] == student_id) & (df["date"] == today)]

    if student_records_today.empty:
        # First recognition of the day - check in
        new_record = pd.DataFrame([{"student_id": student_id, "date": today, "check_in_time": current_time, "check_out_time": ""}])
        df = pd.concat([df, new_record], ignore_index=True)
        message = f"{student_id} checked in at {current_time}"
    else:
        # Student recognized today
        idx = student_records_today.index[0]
        # Debug print statements to check the value and type of check_out_time
        checkout_value = student_records_today.iloc[0]["check_out_time"]
        print(f"Debug: check_out_time value from CSV: '{checkout_value}'")
        print(f"Debug: Type of check_out_time value: {type(checkout_value)}")

        # Correctly check for missing value (empty cell)
        if pd.isna(checkout_value):
            # Already checked in, not checked out - check out
            df.loc[idx, "check_out_time"] = current_time
            message = f"{student_id} checked out at {current_time}"
        else:
            # Already checked in and checked out
            message = f"{student_id} already checked in and checked out today."

    save_attendance_records(df)
    return message 