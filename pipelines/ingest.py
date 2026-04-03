import pandas as pd
import re
import os
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "telecom.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned.csv")

REQUIRED_COLS = ["category", "question", "answer"]

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "pipeline.log")),
        logging.StreamHandler()
    ]
)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into one and strip edges."""
    return re.sub(r"[ \t]+", " ", str(text)).strip()


def main():
    logging.info("📥 Reading raw data...")

    df = pd.read_csv(INPUT_PATH, on_bad_lines="skip")
    logging.info(f"Rows loaded: {len(df)}")

    # Validate required columns exist
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column in raw data: '{col}'")

    # Drop rows missing any required column value
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    logging.info(f"Dropped {before - len(df)} rows with missing required fields")

    # Normalize whitespace in all string columns
    for col in REQUIRED_COLS:
        df[col] = df[col].apply(normalize_whitespace)

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=REQUIRED_COLS)
    logging.info(f"Removed {before - len(df)} duplicate rows")

    logging.info(f"Rows after cleaning: {len(df)}")

    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    logging.info(f"✅ Cleaned data saved at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()