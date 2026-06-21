# Speaker-Conditioned Target Keyword Spotting (SC-TKWS)

A zero-shot, speaker-aware keyword spotting system that triggers **only when the enrolled speaker says the enrolled keyword**.

Unlike conventional wake-word systems that activate when anyone speaks a trigger phrase, SC-TKWS combines biometric speaker verification and phonetic keyword verification into a single edge-deployable architecture.

The system was designed for:

* Personalized wake words
* Voice-controlled IoT devices
* Secure voice authentication
* Offline assistants
* Embedded and edge deployment

All inference components are exported to ONNX and optimized for low-latency execution.

---

# Motivation

Traditional keyword spotting systems solve:

> "Was the keyword spoken?"

SC-TKWS solves:

> "Was the enrolled keyword spoken by the enrolled user?"

This requires jointly modeling:

* Speaker Identity (WHO spoke?)
* Keyword Intent (WHAT was spoken?)

while remaining computationally lightweight enough for edge deployment.

---

# Architecture

```
                    Enrollment
┌──────────────────────────────────────────┐
│                                          │
│  User says custom keyword 3 times        │
│                                          │
└──────────────────────────────────────────┘
                    │
                    ▼
        ECAPA-TDNN Speaker Encoder
                    │
                    ▼
         192-D Speaker Embedding
                    │
                    ▼
             FiLM Generator
                    │
          γ (gamma), β (beta)
                    │
                    ▼
        FiLM-Conditioned TC-ResNet
                    │
                    ▼
          128-D Keyword Template
                    │
                    ▼
              Stored Profile
```

During inference:

```
Incoming Audio
      │
      ▼
KWS PCEN Frontend
      │
      ▼
FiLM-TC-ResNet
      │
      ▼
128-D Query Embedding
      │
      ▼
Neural Comparator
      │
      ▼
Accept / Reject
```

---

# Core Components

## 1. ECAPA-TDNN Speaker Encoder

Extracts a 192-dimensional biometric representation.

Input:

* PCEN Features
* 3-second audio window

Output:

```
e_s ∈ ℝ¹⁹²
```

Used only during enrollment.

---

## 2. FiLM Generator

Transforms the speaker embedding into modulation parameters:

```
γ, β = G(e_s)
```

Architecture:

```
Linear(192 → 128)
ReLU
Linear(128 → 384)
```

Output:

```
192 gamma parameters
192 beta parameters
```

These parameters become a biometric gate controlling the TC-ResNet feature space.

---

## 3. FiLM-Conditioned TC-ResNet

Modified TC-ResNet acoustic encoder.

Structure:

```
Conv Projection
│
├── Layer 1 (Speaker Independent)
├── Layer 2 (Speaker Independent)
└── Layer 3 (FiLM Conditioned)
```

FiLM modulation:

```
F' = γ ⊙ F + β
```

This allows the network to dynamically amplify acoustic patterns consistent with the enrolled speaker while suppressing competing voices.

Output:

```
128-D normalized keyword embedding
```

---

## 4. Neural Comparator

Instead of a fixed cosine threshold, SC-TKWS uses a learned comparator.

Input:

```
[target]
[query]
|target - query|
```

Architecture:

```
384
 ↓
Linear(384,32)
 ↓
ReLU
 ↓
Linear(32,1)
```

Output:

```
P(Accept)
```

This improves robustness against:

* Noise
* Similar sounding words
* Speaker leakage
* Hard phonetic negatives

---

# Model Statistics

## FiLM Generator

Parameters:

```
≈ 74K
```

## FiLM-TC-ResNet Encoder

Parameters:

```
≈ 286K
```

## Neural Comparator

Parameters:

```
≈ 12K
```

## Total Stage-4 KWS Stack

Parameters:

```
≈ 372K
```

Excluding the ECAPA-TDNN speaker encoder.

---

# Training Pipeline

The model was trained in four stages.

---

## Stage 1 — Speaker Representation Learning

Model:

* ECAPA-TDNN

Objective:

* Speaker verification

Output:

```
192-D speaker embeddings
```

Artifacts:

```
ecapa_tdnn.onnx
ecapa_pcen.json
```

---

## Stage 2 — Keyword Representation Learning

Model:

* TC-ResNet

Objective:

* Supervised Contrastive Learning

Dataset:

* 3000-word synthetic keyword corpus
* TTS-generated speakers
* Hard phonetic negatives
* Proper nouns
* Commands
* Technology vocabulary
* Synthetic non-words

Output:

```
128-D keyword embeddings
```

Artifacts:

```
conditioned_encoder.onnx
kws_pcen.json
```

---

## Stage 3 — Speaker Conditioning

Goal:

Learn speaker-aware acoustic filtering.

Training strategy:

* Collision mixing
* Target speaker + interfering speaker
* Biometric negative sampling
* Supervised contrastive loss

Only:

* FiLM Generator
* Layer 3
* Projection Head

were trainable.

---

## Stage 4 — Comparator Fine-Tuning

Goal:

Optimize final verification accuracy.

Training signals:

### Positive Pair

```
Correct Speaker
Correct Word
```

### Negative Pair

```
Wrong Speaker
Correct Word
```

### Hard Negative Pair

```
Correct Speaker
Phonetically Similar Word
```

Loss:

```
BCE Loss
+
Margin Ranking Loss
```

---

# Enrollment

Enroll a user with three recordings of the target keyword.

```bash
python enroll.py \
    accept1.wav \
    accept2.wav \
    accept3.wav \
```

Generated profile:

```json
{
  "gamma": "...",
  "beta": "...",
  "template": "..."
}
```

---

# Verification

```bash
python inference.py \
    --audio query.wav \
    --profile profile.json
```

Output:

```text
Probability: 0.983
Match: True
```

---

# ONNX Deployment

All inference modules are exported to ONNX.

Artifacts:

```
ecapa_tdnn.onnx
film_generator.onnx
conditioned_encoder.onnx
neural_comparator.onnx
```

Advantages:

* CPU inference
* Edge deployment
* Cross-platform compatibility
* No PyTorch dependency during inference

---

# Repository Structure

```
speaker-conditioned-target-kws/
│
├── enroll.py
├── inference.py
├── model.py
├── requirements.txt
│
├── models/
│   ├── ecapa_tdnn.onnx
│   ├── ecapa_pcen.json
│   ├── film_generator.onnx
│   ├── conditioned_encoder.onnx
│   ├── kws_pcen.json
│   └── neural_comparator.onnx
│
└── enrolled_profile.json
```