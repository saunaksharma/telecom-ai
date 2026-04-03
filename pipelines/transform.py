import pandas as pd
import os

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "transformed.csv")


def main():
    print("🔄 Starting transformation pipeline...")

    # Check if cleaned data exists
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Cleaned file not found at {INPUT_PATH}. Run pipelines/ingest.py first.")

    df = pd.read_csv(INPUT_PATH)

    print(f"📄 Loaded {len(df)} rows")

    # ===== DATA VALIDATION =====
    required_cols = ["category", "question", "answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ===== TRANSFORMATION =====
    # Normalize whitespace in each field before building the text column
    for col in required_cols:
        df[col] = df[col].astype(str).str.replace(r"[ \t]+", " ", regex=True).str.strip()

    df["text"] = df.apply(
        lambda row: f"Category: {row['category']}. Question: {row['question']}. Answer: {row['answer']}",
        axis=1
    )

    print("🧠 Text column created")

    # Save transformed data
    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Transformation complete!")
    print(f"📁 Saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()