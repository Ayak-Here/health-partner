import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "skin_cancer_dataset"
IMAGES_DIR = DATA_DIR / "images"

df = pd.read_csv(DATA_DIR / "metadata.csv", sep=",", dtype=str)
df = df.fillna("")   

print("Loaded metadata.csv with columns:")
print(df.columns.tolist())

existing_images = {p.name for p in IMAGES_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]}

df = df[df["img_id"].isin(existing_images)].copy()

df["diagnostic"] = df["diagnostic"].str.strip().str.upper()

allowed_classes = ["NEV", "BCC", "ACK", "SEK", "MEL", "SCC"]

df = df[df["diagnostic"].isin(allowed_classes)].copy()

output_path = DATA_DIR / "skin_metadata_filtered.csv"
df.to_csv(output_path, index=False)

print("\n=== CLEANING COMPLETE ===")
print("Valid rows:", len(df))
print("Saved cleaned metadata at:", output_path)
