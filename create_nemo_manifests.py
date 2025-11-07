#!/usr/bin/env python
# create_nemo_manifests.py
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# Copied directly from your make_manifest_and_stats.py for consistency
def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text).lower()
    # Keep letters incl. diacritics, apostrophes, hyphens, and spaces
    t = re.sub(r"[^a-zà-ž' -\u0108\u0109\u011c\u011d\u0124\u0125\u0134\u0135\u015c\u015d\u016c\u016d]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def get_duration(audio_path: Path) -> float:
    try:
        with sf.SoundFile(str(audio_path)) as f:
            return len(f) / f.samplerate
    except Exception:
        print(f"Warning: Could not read {audio_path}")
        return 0.0

def create_manifest(root: Path, split: str, out_path: Path):
    csv_path = root / f"{split}.csv"
    df = pd.read_csv(csv_path)
    base_data_dir = root.parent 

    print(f"Writing manifest for {split}...")
    with open(out_path, 'w', encoding='utf-8') as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            rel_path = row['file']
            abs_path = (base_data_dir / rel_path).resolve()
            
            if not abs_path.exists():
                print(f"Warning: File not found {abs_path}")
                continue
                
            duration = get_duration(abs_path)
            if duration == 0.0:
                continue

            entry = {
                "audio_filepath": str(abs_path),
                "duration": duration
            }
            
            if split in ["train", "dev"]:
                transcript = normalize(row['transcript'])
                if not transcript:
                    continue # Skip entries with no text for training
                entry["text"] = transcript
            
            fout.write(json.dumps(entry) + '\n')
            
    # For the test set, we ALSO need a simple mapping file for the final submission
    if split == "test":
        map_path = out_path.parent / "test_submission_map.csv"
        print(f"Writing test submission map to {map_path}...")
        df_test_map = pd.DataFrame({
            "original_file": df["file"],
            "absolute_path": df["file"].apply(lambda x: str((base_data_dir / x).resolve()))
        })
        df_test_map.to_csv(map_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to the 'geo' dir containing CSVs")
    ap.add_argument("--out_dir", type=str, default="nemo_manifests", help="Dir to save manifest files")
    args = ap.parse_args()
    
    root_path = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (root_path / "train.csv").exists():
        raise FileNotFoundError(f"train.csv not found in {root_path}")
        
    create_manifest(root_path, "train", out_dir / "train_manifest.json")
    create_manifest(root_path, "dev", out_dir / "dev_manifest.json")
    create_manifest(root_path, "test", out_dir / "test_manifest.json")
    
    print("All manifests created.")

if __name__ == "__main__":
    main()