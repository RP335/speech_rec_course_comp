# Geo ASR Challenge 2025

This repository contains scripts and data for the Geo ASR Challenge 2025. The following instructions will guide you through the process of setting up the dataset and running the provided training and prediction scripts.

---

## Setup Instructions

### 1. Download and Organize the Dataset

First, download the geo dataset from the kosh computer location:
```
/work/courses/T/S/89/5150/general/data/geo_ASR_challenge_2025
```

Place the downloaded dataset in the root directory of your project.

#### Verify Root Directory Structure

Your root directory should look like this:
```bash
ls
```

**Expected output:**
```
create_nemo_manifests.py  finetune_nemo.py  geo_data  geo_data.tar.gz  inspect_dataset.py  predict_test.py
```

#### Navigate and Verify Dataset Structure

Navigate to the `geo_data` directory:
```bash
cd geo_data
ls
```

**Expected output:**
```
geo_ASR_challenge_2025
```

Display the directory tree structure:
```bash
tree -d
```

**Expected output:**
```
.
└── geo_ASR_challenge_2025
    └── geo
        └── clips

4 directories
```

Navigate to the `geo_ASR_challenge_2025/geo` directory:
```bash
cd geo_ASR_challenge_2025/geo
ls
```

**Expected output:**
```
clips  dev.csv  test.csv  train.csv
```

---

### 2. Create NeMo Manifests

Run the following command from the project root directory to generate NeMo manifests from the dataset:
```bash
python create_nemo_manifests.py --root geo_data/geo_ASR_challenge_2025/geo --out_dir ./nemo_data
```

---

### 3. Fine-tune the Model

Use the following command to fine-tune the ASR model:
```bash
python finetune_nemo.py \
    --train_manifest ./nemo_data/train_manifest.json \
    --dev_manifest ./nemo_data/dev_manifest.json \
    --epochs 10 \
    --batch_size 16
```

---

### 4. Prediction on Test Data

Run the prediction script using the fine-tuned model:
```bash
python predict_test.py \
    --finetuned_model ./checkpoints/epoch=9-val_wer=0.1234.ckpt \
    --test_manifest ./nemo_data/test_manifest.json \
    --submission_map ./nemo_data/test_submission_map.csv \
    --output_csv submission.csv
```

**Note:** Replace `epoch=9-val_wer=0.1234.ckpt` with your actual checkpoint filename.
