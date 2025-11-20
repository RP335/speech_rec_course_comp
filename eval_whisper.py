import torch
import librosa
import argparse
import os
import evaluate
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model (Baseline or PEFT) on a test manifest.")
    
    # Arguments
    parser.add_argument("--manifest", type=str, required=True, help="Path to the test manifest JSON file.")
    parser.add_argument("--model", type=str, default="small", help="Base Whisper model (tiny, small)")
    parser.add_argument("--adapter", type=str, default=None, help="Path to the PEFT/LoRA adapter. If None, runs baseline evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for quick testing.")
    
    args = parser.parse_args()

    MODEL = f"openai/whisper-{args.model}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wer_metric = evaluate.load("wer")

    print(f"--- Starting Evaluation ---")
    print(f"Manifest: {args.manifest}")
    print(f"Base Model: {MODEL}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    else:
        print("Adapter: None (Running Baseline)")

    # 1. Load Model and Processor
    print("Loading model and processor...")
    processor = WhisperProcessor.from_pretrained(MODEL, task="transcribe")
    
    # Load in 8-bit to match training configuration and save VRAM
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, load_in_8bit=True, device_map="auto")

    # Load Adapter if provided
    if args.adapter:
        if not os.path.exists(args.adapter):
            print(f"Error: Adapter path '{args.adapter}' does not exist.")
            return
        
        print("Loading LoRA adapter...")
        try:
            model = PeftModel.from_pretrained(model, args.adapter)
        except Exception as e:
            print(f"Error loading adapter: {e}")
            return

    # 2. Force Language Tokens (Esperanto Hack)
    # Explicitly force the model to use <|eo|> token to avoid language detection errors
    eo_token_id = processor.tokenizer.convert_tokens_to_ids("<|eo|>")
    transcribe_token_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    forced_decoder_ids = None
    if eo_token_id is not None and eo_token_id != processor.tokenizer.unk_token_id:
        forced_decoder_ids = [
            (1, eo_token_id),
            (2, transcribe_token_id),
            (3, notimestamps_token_id)
        ]

    # 3. Load Data
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=args.manifest)["train"]
    
    if args.limit:
        dataset = dataset.select(range(args.limit))
        print(f"Limiting evaluation to first {args.limit} samples.")

    print(f"Total samples to evaluate: {len(dataset)}")

    predictions = []
    references = []

    # 4. Evaluation Loop
    print("Computing predictions...")
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        # Handle potential column name differences
        original_text = item.get("text", item.get("text"))
        path = item["audio_filepath"]
        
        try:
            # Load audio using librosa to ensure 16kHz sampling rate
            audio, _ = librosa.load(path, sr=16000)
            
            # Convert input to half precision (fp16) to match the 8-bit model weights
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device).half()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features, 
                    forced_decoder_ids=forced_decoder_ids, 
                    max_new_tokens=255
                )
            
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Normalize text for fair WER comparison (lowercase, remove punctuation)
            ref = processor.tokenizer.normalize(original_text)
            pred = processor.tokenizer.normalize(transcription)
            
            references.append(ref)
            predictions.append(pred)
            
            # Print first sample for verification
            if i == 0:
                print(f"\nSample 1 Reference: {ref}")
                print(f"Sample 1 Prediction: {pred}\n")

        except Exception as e:
            print(f"Error processing file {path}: {e}")
            continue

    # 5. Final Metrics
    if len(references) > 0:
        wer = wer_metric.compute(predictions=predictions, references=references)
        print(f"Final WER Score: {wer:.2%}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()