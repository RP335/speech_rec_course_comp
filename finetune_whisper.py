import torch
import librosa
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 1. Data Preprocessing ---

def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Preprocesses audio data on-the-fly to optimize RAM usage.
    Loads audio, converts to log-Mel spectrograms, and tokenizes text.
    """
    audio_features = []
    label_ids = []
    
    # Iterate over the batch
    for path, sentence in zip(batch["audio_filepath"], batch["text"]):
        try:
            # Load audio with librosa (ensures 16kHz sampling rate)
            # Truncate to 30s to match Whisper's input size
            audio_array, _ = librosa.load(path, sr=16000, duration=30.0)
            
            # Compute log-Mel input features
            features = feature_extractor(audio_array, sampling_rate=16000).input_features[0]
            
            # Encode target text
            labels = tokenizer(sentence).input_ids
            
            audio_features.append(features)
            label_ids.append(labels)
        except Exception as e:
            # Skip corrupted files without crashing
            continue

    return {"input_features": audio_features, "labels": label_ids}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator to dynamically pad inputs and labels to the maximum length in the batch.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss calculation on padding tokens
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove start of sentence token if present (specific to some Whisper versions)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# --- 2. Main Training Logic ---

def main():
    # Argument parsing for experiment reproducibility
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Esperanto using LoRA.")
    
    # Data arguments
    parser.add_argument("--train_manifest", type=str, required=True, help="Path to training manifest JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    
    # Hyperparameters (Defaults set to the configuration that achieved ~52% WER)
    parser.add_argument("--model", type=str, default="small", help="Whisper model size (tiny, small, medium)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total number of training steps")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Gradient accumulation steps")
    
    args = parser.parse_args()

    model_id = f"openai/whisper-{args.model}"
    print(f"--- Starting training with Model: {model_id} | LR: {args.lr} | Steps: {args.max_steps} ---")

    # Load Processor (Feature Extractor + Tokenizer)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_id, task="transcribe")

    # Load Dataset
    print(f"Loading dataset from {args.train_manifest}...")
    dataset = load_dataset("json", data_files=args.train_manifest)["train"]
    
    # Apply the on-the-fly transformation using lambda to inject processor objects
    dataset.set_transform(lambda x: prepare_dataset(x, feature_extractor, tokenizer))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Load Model with 8-bit quantization for memory efficiency
    model = WhisperForConditionalGeneration.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
    
    # Disable cache during training
    model.config.use_cache = False 
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model = prepare_model_for_kbit_training(model)

    # LoRA Configuration (Parameter-Efficient Fine-Tuning)
    # Targeting attention modules (query and value projections)
    config = LoraConfig(
        r=32, 
        lora_alpha=64, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps, # Accumulate gradients to simulate larger batch size (Stability)
        learning_rate=args.lr,
        warmup_steps=100,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,                 # Use mixed precision
        eval_strategy="no",        # Disable evaluation during training to avoid language validation errors
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,        # Keep only the last 2 checkpoints
        logging_steps=50,
        report_to=["none"],        # Disable external logging (WandB)
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0   # Set to 0 for compatibility with Windows/WSL
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor, # Pass feature extractor to process inputs
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    save_path = f"{args.output_dir}/final"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()