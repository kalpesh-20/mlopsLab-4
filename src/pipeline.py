import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "data.csv"
OUTPUT_FILE = BASE_DIR / "output" / "students_processed.csv"


# 1. Data collection
def collect_data():
    df = pd.read_csv(INPUT_FILE)
    print("1. Data collected successfully.")
    return df


# 2. Data preprocessing
def preprocess_data(df):
    # Fill missing values with column means
    df["Study_Hours"] = df["Study_Hours"].fillna(
        df["Study_Hours"].mean()
    )

    df["Marks"] = df["Marks"].fillna(
        df["Marks"].mean()
    )

    # Remove duplicate records
    df = df.drop_duplicates()

    print("2. Data preprocessing completed.")
    return df


# 3. Feature engineering
def feature_engineering(df):

    def performance_level(marks):
        if marks >= 85:
            return "Excellent"
        elif marks >= 70:
            return "Good"
        elif marks >= 50:
            return "Average"
        else:
            return "Poor"

    df["Performance_Level"] = df["Marks"].apply(
        performance_level
    )

    df["Attendance_Category"] = df["Attendance"].apply(
        lambda x: "Good" if x >= 75 else "Low"
    )

    print("3. Feature engineering completed.")
    return df


# 4. Data validation
def validate_data(df):
    assert df["Student_ID"].notna().all(), \
        "Student_ID contains missing values."

    assert df["Marks"].between(0, 100).all(), \
        "Marks must be between 0 and 100."

    assert df["Attendance"].between(0, 100).all(), \
        "Attendance must be between 0 and 100."

    assert not df.isnull().any().any(), \
        "The dataset contains missing values."

    print("4. Data validation completed.")


# 5. Save processed data
def save_data(df):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("5. Data saved successfully.")
    print(f"Output file: {OUTPUT_FILE}")


# Main pipeline
def main():
    df = collect_data()
    df = preprocess_data(df)
    df = feature_engineering(df)

    validate_data(df)
    save_data(df)

    print("\n--- Pipeline completed successfully! ---")


if __name__ == "__main__":
    main()