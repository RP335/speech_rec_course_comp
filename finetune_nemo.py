#!/usr/bin/env python
# finetune_nemo.py
import argparse
import lightning.pytorch as pl
from omegaconf import OmegaConf, DictConfig
import nemo.collections.asr as nemo_asr
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", type=str, required=True, help="Path to train_manifest.json")
    ap.add_argument("--dev_manifest", type=str, required=True, help="Path to dev_manifest.json")
    ap.add_argument("--epochs", type=int, default=10, help="Number of epochs to fine-tune")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = ap.parse_args()

    print("Loading pre-trained model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_eo_conformer_transducer_large")

    print("Setting up configuration...")
    
    asr_model.cfg.train_ds.manifest_filepath = args.train_manifest
    asr_model.cfg.train_ds.batch_size = args.batch_size
    asr_model.cfg.train_ds.shuffle = True
    asr_model.cfg.train_ds.num_workers = 4
    
    asr_model.cfg.validation_ds.manifest_filepath = args.dev_manifest
    asr_model.cfg.validation_ds.batch_size = args.batch_size
    asr_model.cfg.validation_ds.shuffle = False
    asr_model.cfg.validation_ds.num_workers = 4

    asr_model.cfg.optim.name = 'adamw'
    asr_model.cfg.optim.lr = args.lr
    asr_model.cfg.optim.sched.name = 'CosineAnnealing'
    asr_model.cfg.optim.sched.warmup_steps = 500
    
    asr_model.setup_training_data(asr_model.cfg.train_ds)
    asr_model.setup_validation_data(asr_model.cfg.validation_ds)
    
    asr_model.setup_optimization(asr_model.cfg.optim)
    
    print("Setting up trainer...")
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='{epoch}-{val_wer:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_wer',
        mode='min'
    )

    trainer = pl.Trainer(
        devices=1, 
        accelerator='gpu', 
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        precision=16,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    # --- Start Fine-Tuning ---
    print("Starting fine-tuning...")
    asr_model._trainer = trainer

    trainer.fit(asr_model)
    
    # Save the final model
    final_model_path = "geo_finetuned.nemo"
    asr_model.save_to(final_model_path)
    print(f"Training complete. Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Final model saved at: {final_model_path}")

if __name__ == "__main__":
    main()