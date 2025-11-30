#!/usr/bin/env python
# predict_test.py
import argparse
from pathlib import Path
import json
import pandas as pd
import nemo.collections.asr as nemo_asr
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned_model", type=str, required=True, help="Path to your .nemo model checkpoint")
    ap.add_argument("--test_manifest", type=str, required=True, help="Path to test_manifest.json")
    ap.add_argument("--submission_map", type=str, required=True, help="Path to test_submission_map.csv")
    ap.add_argument("--output_csv", type=str, default="submission.csv", help="Path for the final submission file")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    args = ap.parse_args()

    print(f"Loading fine-tuned model from {args.finetuned_model}...")
    asr_model = nemo_asr.models.ASRModel.restore_from(args.finetuned_model)
    # asr_model = nemo_asr.models.ASRModel.from_pretrained(args.finetuned_model)
    asr_model = asr_model.to('cuda') # Move model to GPU
    asr_model.eval()

    print(f"Reading test manifest: {args.test_manifest}")
    abs_paths = []
    with open(args.test_manifest, 'r') as f:
        for line in f:
            entry = json.loads(line)
            abs_paths.append(entry['audio_filepath'])

    print(f"Transcribing {len(abs_paths)} test files (this may take a while)...")
    transcriptions = asr_model.transcribe(
            abs_paths, 
            batch_size=args.batch_size
        )
    print("Formatting submission file...")
    # The `transcriptions` variable is a list of Hypothesis objects
    
    if len(abs_paths) != len(transcriptions):
        raise ValueError("Mismatch between number of files and transcriptions!")
        
    path_to_hypothesis_map = dict(zip(abs_paths, transcriptions))

    df_map = pd.read_csv(args.submission_map)
    
    final_transcripts = []
    for abs_path in df_map['absolute_path']:
        hypothesis = path_to_hypothesis_map.get(abs_path)
        
        if hypothesis is None:
            print(f"Warning: No transcript found for {abs_path}. Using empty string.")
            transcript_text = ""
        else:
            transcript_text = hypothesis.text
            
        final_transcripts.append(transcript_text)

    # Create the final submission dataframe
    df_submission = pd.DataFrame({
        "file": df_map["original_file"],
        "transcript": final_transcripts
    })

    # Save to CSV
    df_submission.to_csv(args.output_csv, index=False)
    print(f"Done! Submission file saved to: {args.output_csv}")
    print(df_submission.head())

if __name__ == "__main__":
    main()