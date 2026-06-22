# Speaker-Conditioned Target KWS — Complete Setup Guide

A step-by-step guide to reproduce the **Speaker-Conditioned Target Keyword Spotting (SC-TKWS)** system from scratch — including dataset preparation, 4-stage training, enrollment, inference, and evaluation.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Clone the Repository](#1-clone-the-repository)
- [2. Environment Setup](#2-environment-setup)
- [3. Dataset Preparation](#3-dataset-preparation)
  - [3A. VoxCeleb1 Dataset](#3a-voxceleb1-dataset)
  - [3B. MUSAN Dataset](#3b-musan-dataset)
  - [3C. TTS Corpus Dataset](#3c-tts-corpus-dataset)
- [4. Stage 1 — ECAPA-TDNN Speaker Embedding Model](#4-stage-1--ecapa-tdnn-speaker-embedding-model)
  - [4.1 Train](#41-train)
  - [4.2 Verify](#42-verify)
- [5. Stage 2 — TC-ResNet Keyword Encoder](#5-stage-2--tc-resnet-keyword-encoder)
  - [5.1 Train](#51-train)
  - [5.2 Verify](#52-verify)
- [6. Stage 3 — FiLM Generator (Speaker Separation)](#6-stage-3--film-generator-speaker-separation)
  - [6.0 Preprocessing](#60-preprocessing)
  - [6.1 Train](#61-train)
  - [6.2 Verify](#62-verify)
- [7. Stage 4 — Neural Comparator](#7-stage-4--neural-comparator)
  - [7.1 Train](#71-train)
  - [7.2 Verify](#72-verify)
- [8. End-to-End Pipeline (Enrollment → Inference)](#8-end-to-end-pipeline-enrollment--inference)
- [9. Full Benchmark Evaluation](#9-full-benchmark-evaluation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement    | Minimum                       |
| -------------- | ----------------------------- |
| **OS**         | Ubuntu 20.04+ / WSL2          |
| **Python**     | 3.9 – 3.11                    |
| **GPU**        | NVIDIA GPU with CUDA 11.8+    |
| **RAM**        | 16 GB+                        |
| **Disk**       | ~50 GB free (datasets + models)|
| **ffmpeg**     | Required for audio processing |

> **Note:** CPU-only training is possible but will be significantly slower. A CUDA-capable GPU is strongly recommended.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Nishchal-29/Speaker-Conditioned-Target-KWS.git
cd Speaker-Conditioned-Target-KWS
```

---

## 2. Environment Setup

### 2.1 Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2.2 Install system dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg sox libsndfile1
```

### 2.3 Install Python dependencies

```bash
pip install --upgrade pip

# Step 1: Install PyTorch + torchaudio with the correct CUDA version
# For CUDA 11.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install all remaining dependencies
pip install -r requirements.txt

# (Optional) For GPU-accelerated ONNX inference
# pip install onnxruntime-gpu
```

> **Note:** PyTorch must be installed separately before `requirements.txt` because the correct CUDA version depends on your system. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for all options.

> **Tip:** If you encounter issues with `speechbrain` on Windows/PyTorch 2.x, the codebase already includes compatibility patches at the top of training scripts.

---

## 3. Dataset Preparation

All datasets should be placed inside the `data/` directory at the project root:

```
Speaker-Conditioned-Target-KWS/
└── data/
    ├── VoxCeleb1/
    │   └── wav/
    │       ├── id10001/
    │       ├── id10002/
    │       └── ...
    ├── musan/
    │   ├── music/
    │   ├── noise/
    │   └── speech/
    └── tts_corpus/
        ├── train/
        │   └── <word>/
        │       ├── spk001_utt1.wav
        │       ├── spk001_aug1_p+1.2_s0.84.wav
        │       └── ...
        └── val/
            └── <word>/
                └── ...
```

### 3A. VoxCeleb1 Dataset

VoxCeleb1 is used to fine-tune the ECAPA-TDNN model on PCEN features. Due to its strict redistribution policy, you must request access manually.

1. **Visit:** [https://mm.kaist.ac.kr/datasets/voxceleb/](https://mm.kaist.ac.kr/datasets/voxceleb/)
2. **Request** the audio dataset (VoxCeleb1)
3. **Download** all the zip files provided after approval
4. **Extract** them into the `data/VoxCeleb1/` directory

```bash
mkdir -p data/VoxCeleb1

# After downloading the zip files, extract them:
# (Adjust file names based on what you receive)
unzip vox1_dev_wav_partaa.zip -d data/VoxCeleb1/
unzip vox1_dev_wav_partab.zip -d data/VoxCeleb1/
# ... repeat for all parts

# The final structure should be:
# data/VoxCeleb1/wav/id10001/1zcIwhmdeo4/00001.wav
```

> **Important:** The training script expects audio files at `data/VoxCeleb1/wav/<speaker_id>/<video_id>/<clip_id>.wav`.

---

### 3B. MUSAN Dataset

MUSAN provides music, speech, and noise samples for data augmentation during training.

```bash
# Download MUSAN (~11 GB)
wget https://www.openslr.org/resources/17/musan.tar.gz -P data/

# Extract
tar -xzf data/musan.tar.gz -C data/

# Verify structure
ls data/musan/
# Expected: music/  noise/  speech/

# Clean up archive (optional)
rm data/musan.tar.gz
```

---

### 3C. TTS Corpus Dataset

The TTS corpus is a custom keyword dataset generated using TTS models (edge-tts). It contains recordings from **37 speakers**, **4 utterances per speaker**, covering ~**3000 unique words** with **148 audio files for each speaker**. You have two options:

#### Option A — Download from Hugging Face (Recommended)

```bash
# Install huggingface CLI if not already installed
pip install huggingface_hub

# Download the dataset
huggingface-cli download Nishchal-29/tts_corpus --repo-type dataset --local-dir data/tts_corpus
```

Or visit: [https://huggingface.co/datasets/Nishchal-29/tts_corpus](https://huggingface.co/datasets/Nishchal-29/tts_corpus)

#### Option B — Generate from scratch using edge-tts

```bash
# Step 1: Build the master word dictionary
cd tc_resnet/build_dataset
python build_dictionary.py
# Creates: ../../data/master_words.txt (3000 words)

# Step 2: Generate TTS audio using edge-tts
python generate_tts_data.py
# Creates: ../../data/tts_corpus/train/ and ../../data/tts_corpus/val/
# NOTE: This can take a long time depending on your internet connection

cd ../..
```

> **Note:** `generate_tts_data.py` reads from `data/master_words.txt` and generates audio files using Microsoft's edge-tts API. It creates augmented versions with pitch shifting and time stretching. The process is resumable — it skips already-generated files.

---

## 4. Stage 1 — ECAPA-TDNN Speaker Embedding Model

**Goal:** Fine-tune the pre-trained SpeechBrain ECAPA-TDNN model with a learnable PCEN frontend to produce **192-D speaker embeddings** (e_s).

**What it does:**
- Downloads the pre-trained `speechbrain/spkrec-ecapa-voxceleb` model
- Attaches a learnable PCEN frontend (replacing mel-spectrogram)
- Fine-tunes on VoxCeleb1 with AAM-Softmax loss
- Exports an ONNX backbone + PCEN JSON sidecar

### 4.1 Train

```bash
cd ecapa_tdnn
python train.py
```

**Default hyperparameters** (configured in `train.py`):
| Parameter         | Value  |
| ----------------- | ------ |
| Dataset           | `../data/VoxCeleb1/wav` |
| Epochs            | 10     |
| Batch Size        | 64     |
| Backbone LR       | 1e-5   |
| PCEN LR           | 1e-4   |
| Weight Decay      | 1e-4   |
| Grad Clip          | 1.0    |
| Early Stop Patience| 5     |

**Outputs** (in `ecapa_tdnn/finetuned_models/`):
- `domain_adapted_epoch_<N>.pth` — Training checkpoints
- `ecapa_backbone.onnx` — Exported ONNX model
- `pcen_params.json` — Learned PCEN parameters
- `export_validation_report.json` — EER validation report

### 4.2 Verify

The verify script supports three modes: **verify**, **eer**, and **noise_test**.

#### Same speaker verification

```bash
python verify.py --mode verify \
  --audio1 ../test_audio/accept1.wav \
  --audio2 ../test_audio/accept2.wav
```

#### Different speaker verification

```bash
python verify.py --mode verify \
  --audio1 ../test_audio/accept1.wav \
  --audio2 ../test_audio/accept_neg.wav
```

#### EER validation on VoxCeleb1

```bash
python verify.py --mode eer \
  --data ../data/VoxCeleb1/wav
```

#### Noise robustness test (uses MUSAN)

```bash
python verify.py --mode noise_test \
  --data ../data/VoxCeleb1/wav \
  --musan ../data/musan
```

```bash
cd ..
```

---

## 5. Stage 2 — TC-ResNet Keyword Encoder

**Goal:** Train the TC-ResNet acoustic encoder to produce **128-D word embeddings** using Supervised Contrastive Learning on the TTS corpus.

**What it does:**
- Takes PCEN features as input
- Learns speaker-independent word representations
- Uses a phonetic contrastive sampling strategy
- Exports an ONNX backbone + PCEN JSON sidecar

### 5.1 Train

```bash
cd tc_resnet
python train_encoder.py
```

**Default hyperparameters** (configured in `train_encoder.py`):
| Parameter         | Value  |
| ----------------- | ------ |
| Dataset           | `../data/tts_corpus` |
| Epochs            | 30     |
| Batch Size        | 64     |
| Backbone LR       | 1e-3   |
| PCEN LR           | 1e-2   |
| Weight Decay      | 1e-4   |
| Grad Clip          | 1.0    |
| Temperature       | 0.1    |

**Outputs** (in `tc_resnet/tc_resnet_output/`):
- `best_checkpoint.pth` — Best model checkpoint
- `tc_resnet_backbone.onnx` — Exported ONNX model
- `pcen_params.json` — Learned PCEN parameters
- `tc_resnet_sha256.txt` — Model hash for integrity verification

### 5.2 Verify

The verify script enrolls a custom keyword from 3 utterances and optionally tests against a query:

#### Enroll and verify a keyword

```bash
python verify.py \
  --audio_files test_audio/utt1.wav test_audio/utt2.wav test_audio/utt3.wav \
  --verify test_audio/neg.wav
```

This will:
1. Generate a word template (`w_c`) from the 3 enrollment utterances
2. Check coherence (must be ≥ 0.75)
3. Save the enrolled profile to `enrolled_template.json`
4. Compare the test audio against the enrolled template

```bash
cd ..
```

---

## 6. Stage 3 — FiLM Generator (Speaker Separation)

**Goal:** Train the FiLM (Feature-wise Linear Modulation) generator that conditions the TC-ResNet encoder on speaker identity, enabling **speaker-dependent** keyword representations.

**What it does:**
- Takes a 192-D speaker embedding (e_s) and generates γ (gamma) and β (beta) parameters
- Modulates the 3rd layer of TC-ResNet via FiLM
- Uses Supervised Contrastive Loss + Adversarial Loss
- Freezes layers 1 & 2 of TC-ResNet, trains layer 3 + FiLM generator

### 6.0 Preprocessing

Before training Stage 3, you need to generate speaker embeddings and preprocess the TTS corpus:

#### Step 1: Generate speaker embeddings

```bash
cd film_neural_comparator

python generate_es.py
```

This script:
- Uses the ECAPA-TDNN ONNX model from Stage 1
- Extracts 192-D speaker embeddings for each speaker in the TTS corpus
- Averages multiple utterances per speaker and L2-normalizes

**Output:** `film_neural_comparator/speaker_embeddings/spk001.pt`, `spk002.pt`, ...

#### Step 2: Preprocess the TTS corpus

```bash
python preprocess_dataset.py
```

This script:
- Resamples all audio to 16 kHz mono
- Trims silence (VAD)
- Pads/crops to 24000 samples (1.5 seconds)
- Peak-normalizes to 0.9

**Output:** `film_neural_comparator/tts_corpus_processed/`

### 6.1 Train

```bash
python FilmTrain.py
```

**Default hyperparameters**:
| Parameter          | Value  |
| ------------------ | ------ |
| Stage 2 Checkpoint | `../tc_resnet/tc_resnet_output/best_checkpoint.pth` |
| Epochs             | 30     |
| Batch Size         | 128    |
| FiLM Generator LR  | 1e-3   |
| Layer 3 / FC LR    | 1e-4   |
| PCEN LR            | 1e-4   |
| Temperature        | 0.1    |
| Adversarial Weight | 0.5    |

**Output:** `film_neural_comparator/stage3_output/stage3_best_checkpoint.pth`

### 6.2 Verify

Test the FiLM-conditioned TC-ResNet with dynamic enrollment:

```bash
python FilmVerify.py \
  --checkpoint ./stage3_output/stage3_best_checkpoint.pth \
  --speaker_emb ./speaker_embeddings/spk001.pt \
  --enroll_audio ../data/tts_corpus/train/isolate/spk001_aug1_p-1.7_s0.91.wav \
                 ../data/tts_corpus/train/isolate/spk001_aug2_p+1.2_s0.84.wav \
                 ../data/tts_corpus/train/isolate/spk001_aug3_p+1.7_s0.86.wav \
  --test_audio ../data/tts_corpus/train/isolate/spk001_utt1.wav
```

**Arguments:**
| Flag             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `--checkpoint`   | Path to Stage 3 checkpoint                       |
| `--speaker_emb`  | Path to the 192-D speaker embedding (.pt)        |
| `--enroll_audio` | 3 enrollment audio files (target speaker + word) |
| `--test_audio`   | Test audio file to verify                        |
| `--threshold`    | Cosine similarity threshold (default: 0.9)       |

---

## 7. Stage 4 — Neural Comparator

**Goal:** Train the neural comparator MLP that replaces cosine similarity with a learnable comparison function, producing a calibrated **accept/reject probability**.

**What it does:**
- Loads all weights from Stage 3 (PCEN, TC-ResNet, FiLM)
- Trains a lightweight MLP comparator
- Uses BCE + Margin Ranking Loss
- Exports all components as ONNX models for deployment

### 7.1 Train

```bash
python NcTrain.py
```

**Default hyperparameters**:
| Parameter           | Value  |
| ------------------- | ------ |
| Stage 3 Checkpoint  | `./stage3_output/stage3_best_checkpoint.pth` |
| Epochs              | 20     |
| Batch Size          | 16     |
| Comparator LR       | 1e-3   |
| FiLM / Layer 3 LR   | 1e-5   |
| PCEN LR             | 1e-6   |
| Margin              | 0.4    |

**Outputs** (in `film_neural_comparator/stage4_output/`):
- `stage4_best_checkpoint.pth` — Best model checkpoint
- `film_generator.onnx` — FiLM generator ONNX
- `conditioned_encoder.onnx` — Conditioned TC-ResNet encoder ONNX
- `neural_comparator.onnx` — Neural comparator ONNX
- `pcen_params.json` — Final learned PCEN parameters

### 7.2 Verify

Test the full ONNX edge-simulation pipeline:

#### Same speaker, same word (should accept):

```bash
python NcVerify.py \
  --speaker_emb ./speaker_embeddings/spk001.pt \
  --enroll_audio ../data/tts_corpus/train/isolate/spk001_aug1_p-1.7_s0.91.wav \
                 ../data/tts_corpus/train/isolate/spk001_aug2_p+1.2_s0.84.wav \
                 ../data/tts_corpus/train/isolate/spk001_aug3_p+1.7_s0.86.wav \
  --test_audio ../data/tts_corpus/train/isolate/spk001_utt1.wav
```

#### Different speaker, same word (should reject — speaker mismatch):

```bash
python NcVerify.py \
  --speaker_emb ./speaker_embeddings/spk001.pt \
  --enroll_audio ../data/tts_corpus/train/isolate/spk001_aug1_p-1.7_s0.91.wav \
                 ../data/tts_corpus/train/isolate/spk001_aug2_p+1.2_s0.84.wav \
                 ../data/tts_corpus/train/isolate/spk001_aug3_p+1.7_s0.86.wav \
  --test_audio ../data/tts_corpus/train/isolate/spk010_utt1.wav
```

> **Tip:** Try all four states to evaluate the model's robustness:
> 1. Same speaker + same word → **Accept**
> 2. Same speaker + different word → **Reject**
> 3. Different speaker + same word → **Reject**
> 4. Different speaker + different word → **Reject**

```bash
cd ..
```

---

## 8. End-to-End Pipeline (Enrollment → Inference)

After completing all 4 training stages, copy the exported ONNX models to the `models/` directory for the unified pipeline:

### 8.1 Prepare the models directory

```bash
# The models/ directory should contain these files:
# (These are generated during training, copy the final outputs)
ls models/
# ecapa_pcen.json           — PCEN params for speaker ECAPA-TDNN
# ecapa_tdnn.onnx           — Speaker embedding ONNX model
# kws_pcen.json             — PCEN params for keyword encoder
# conditioned_encoder.onnx  — FiLM-conditioned TC-ResNet ONNX
# film_generator.onnx       — FiLM generator ONNX
# neural_comparator.onnx    — Neural comparator ONNX
```

### 8.2 Enroll a keyword

Record or provide **exactly 3** audio files of the **same speaker** saying the **same keyword**:

```bash
python enroll.py test_audio/accept1.wav test_audio/accept2.wav test_audio/accept3.wav
```

This will:
1. Extract the speaker's biometric signature (192-D embedding)
2. Generate FiLM weights (γ, β) conditioned on the speaker
3. Create a word template (w_c) from the 3 enrollment utterances
4. Save the device profile to `enrolled_profile.json`

### 8.3 Run inference

Test a query audio against the enrolled profile:

#### Same speaker says the same keyword (should accept):

```bash
python inference.py --audio test_audio/accept1.wav
```

#### Same keyword spoken by a different speaker (should reject):

```bash
python inference.py --audio test_audio/accept_neg.wav
```

**Output format:**
```
SC-TKWS VERIFICATION RESULTS
Match Probability: 85.23%
Threshold: 50.00%
Matched! Access Granted
```

### 8.4 Testing with custom audio

You can test with your own audio files:

1. Record 3 enrollment utterances of you saying a keyword (e.g., "hello")
2. Save them as 16 kHz, mono WAV files
3. Enroll:
   ```bash
   python enroll.py my_hello_1.wav my_hello_2.wav my_hello_3.wav
   ```
4. Test inference:
   ```bash
   # Your voice saying "hello" → should accept
   python inference.py --audio my_hello_test.wav

   # Someone else saying "hello" → should reject
   python inference.py --audio other_person_hello.wav

   # Your voice saying a different word → should reject
   python inference.py --audio my_goodbye.wav
   ```

---

## 9. Full Benchmark Evaluation

Run the complete benchmark suite to reproduce the reported KPIs:

```bash
python evaluate.py
```

**Default configuration:**
| Flag            | Default Value                                        |
| --------------- | ---------------------------------------------------- |
| `--model_dir`   | `./models`                                           |
| `--data_dir`    | `./film_neural_comparator/tts_corpus_processed/train`|
| `--ped_matrix`  | `None` (optional phonetic distance matrix)           |

**Custom data directory:**
```bash
python evaluate.py --data_dir /path/to/your/evaluation/data
```

> **Warning:** The full benchmark evaluation can take **~1 hour or more** depending on your hardware and dataset size. It runs:
> - **Latency benchmark** (100 runs of each pipeline component)
> - **N-shot evaluation** (1-shot, 3-shot, 5-shot EER)
> - **Full retrieval suite** (Top-1, Top-5, MRR, EER, FAR, FRR, TAR@1%FAR)
> - **Hard negative analysis** (phonetic and speaker collision rejection rates)

---

## Project Structure

```
Speaker-Conditioned-Target-KWS/
│
├── data/                                  # Datasets (gitignored)
│   ├── VoxCeleb1/wav/                     # Speaker recognition dataset
│   ├── musan/                             # Noise augmentation dataset
│   ├── tts_corpus/                        # TTS-generated keyword dataset
│   └── master_words.txt                   # Word dictionary (3000 words)
│
├── ecapa_tdnn/                            # Stage 1: Speaker Embedding
│   ├── train.py                           # Fine-tune ECAPA-TDNN + PCEN
│   ├── verify.py                          # Verify / EER / Noise test
│   ├── dataset.py                         # VoxCeleb data loader
│   ├── metrics.py                         # EER computation
│   └── finetuned_models/                  # Outputs (ONNX, checkpoints)
│
├── tc_resnet/                             # Stage 2: Keyword Encoder
│   ├── train_encoder.py                   # Train TC-ResNet with SupCon
│   ├── verify.py                          # Enroll & verify keywords
│   ├── model.py                           # TC-ResNet architecture
│   ├── dataset.py                         # TTS corpus data loader
│   ├── metrics.py                         # EER computation
│   ├── build_dataset/                     # Dataset generation tools
│   │   ├── build_dictionary.py            # Build master word list
│   │   └── generate_tts_data.py           # Generate TTS audio
│   └── tc_resnet_output/                  # Outputs (ONNX, checkpoints)
│
├── film_neural_comparator/                # Stages 3 & 4
│   ├── generate_es.py                     # Generate speaker embeddings
│   ├── preprocess_dataset.py              # Preprocess TTS corpus
│   ├── FilmNcModel.py                     # Model architectures
│   ├── FilmNcDataset.py                   # Dataset classes
│   ├── FilmNcMetrics.py                   # EER computation
│   ├── FilmTrain.py                       # Stage 3: Train FiLM generator
│   ├── FilmVerify.py                      # Stage 3: Verify FiLM
│   ├── NcTrain.py                         # Stage 4: Train neural comparator
│   ├── NcVerify.py                        # Stage 4: Verify (ONNX edge sim)
│   ├── speaker_embeddings/                # Generated speaker embeddings
│   ├── tts_corpus_processed/              # Preprocessed audio
│   ├── stage3_output/                     # Stage 3 outputs
│   └── stage4_output/                     # Stage 4 outputs (ONNX models)
│
├── models/                                # Deployment ONNX models
│   ├── ecapa_tdnn.onnx                    # Speaker embedding model
│   ├── ecapa_pcen.json                    # Speaker PCEN params
│   ├── conditioned_encoder.onnx           # FiLM-conditioned TC-ResNet
│   ├── kws_pcen.json                      # Keyword PCEN params
│   ├── film_generator.onnx                # FiLM weight generator
│   └── neural_comparator.onnx             # Accept/reject MLP
│
├── test_audio/                            # Test audio files
│   ├── accept1.wav                        # Enrollment utterance 1
│   ├── accept2.wav                        # Enrollment utterance 2
│   ├── accept3.wav                        # Enrollment utterance 3
│   └── accept_neg.wav                     # Negative (different speaker)
│
├── pcen.py                                # Shared PCEN module
├── model.py                               # SCTKWSPipeline (ONNX inference)
├── enroll.py                              # User enrollment script
├── inference.py                           # Query inference script
├── evaluate.py                            # Full benchmark evaluation
│
└── docs/
    ├── setup.md                           # This file
    ├── ax.md                              # Hackathon documentation
    └── hf_model_details.md                # HuggingFace model card
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'speechbrain'` | Run `pip install speechbrain` |
| `RuntimeError: CUDA out of memory` | Reduce batch size in the training script |
| `FileNotFoundError: ecapa_backbone.onnx` | Complete Stage 1 training first — `train.py` exports the ONNX model |
| `Enrollment rejected: coherence < 0.75` | Re-record enrollment utterances in a quieter environment with consistent pronunciation |
| `ONNX fidelity check FAILED` | Ensure you're using a compatible PyTorch/ONNX version |
| `edge-tts` TTS generation hangs | Check internet connectivity; the TTS API requires network access |
| `PCEN s-collapse detected` | This is handled automatically — the script reduces PCEN LR |
| `No .wav files found` | Verify the dataset directory structure matches the expected layout above |

### GPU Memory Requirements

| Stage | Approximate VRAM |
|-------|-----------------|
| Stage 1 (ECAPA-TDNN) | ~4 GB |
| Stage 2 (TC-ResNet) | ~2 GB |
| Stage 3 (FiLM) | ~4 GB |
| Stage 4 (Neural Comp.) | ~3 GB |

### Audio File Requirements

All input audio files should be:
- **Format:** WAV (PCM)
- **Sample Rate:** 16 kHz
- **Channels:** Mono
- **Duration:** ~1–3 seconds for keywords, ~3 seconds for speaker embeddings

To convert audio files to the correct format:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

---

## Quick Start Summary

```bash
# 1. Clone & setup
git clone https://github.com/Nishchal-29/Speaker-Conditioned-Target-KWS.git
cd Speaker-Conditioned-Target-KWS
python -m venv venv && source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Prepare datasets (see Section 3 for details)

# 3. Stage 1: ECAPA-TDNN
cd ecapa_tdnn && python train.py && cd ..

# 4. Stage 2: TC-ResNet
cd tc_resnet && python train_encoder.py && cd ..

# 5. Stage 3 & 4: FiLM + Neural Comparator
cd film_neural_comparator
python generate_es.py
python preprocess_dataset.py
python FilmTrain.py
python NcTrain.py
cd ..

# 6. Enroll & Infer
python enroll.py test_audio/accept1.wav test_audio/accept2.wav test_audio/accept3.wav
python inference.py --audio test_audio/accept1.wav

# 7. Benchmark
python evaluate.py
```
