# Your Project Name

- **Problem Statement Number** - 04
- **Problem Statement Title** - Designing a Robust AI System for Speech Disentanglement
- **Team name** - TechHunters
- **Team members (Names)** - Nishchal Kumar Singh, Pranay Shit
- **Institute/College Name** - IIT (ISM) Dhanbad 
- **Final Presentation Google Drive Link** - https://drive.google.com/file/d/1oLY4p6RZ1qtKqkZiOptaRb6Amcwdd_4C/view?usp=sharing
- **Full Submission Demo Video Link** - https://youtu.be/I9eDN023N2k
- **Setup & Result Reproducibility Video Link** - https://youtu.be/1yZgvMwOBZI

### Project Artefacts

## Models Used

### 1. SpeechBrain ECAPA-TDNN (Speaker Verification)
- **Model:** https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- **Paper:** https://arxiv.org/abs/2005.07143
- **Description:** ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN) is a state-of-the-art speaker embedding architecture for speaker verification. The pretrained SpeechBrain model is trained on VoxCeleb1 and VoxCeleb2 and is widely used for speaker verification, speaker identification, and diarization tasks, But was build with Mel-Spectogram frontend, we finetuned it for our PCEN frontend. Used to determine the speaker embeddings of 192-D.

### 2. TC-ResNet (Temporal Convolutional ResNet)
- **Paper:** https://arxiv.org/abs/1904.03814
- **Reference Implementation:** https://github.com/hyperconnect/TC-ResNet
- **Description:** TC-ResNet is a lightweight residual convolutional architecture designed for speech and audio processing tasks. It employs temporal convolutional residual blocks to efficiently model temporal dependencies in speech signals while maintaining a low computational footprint. Used to get teh 128-D embedding representing a particular word.
## Models Published

### 1. Speaker Conditioned Target KWS
- **Link:** https://huggingface.co/Nishchal-29/speaker-conditioned-target-kws
- **Description:** A zero-shot, speaker-aware keyword spotting system that triggers only when the enrolled speaker says the enrolled keyword.

## Datasets Used

### 1. VoxCeleb1 & VoxCeleb2
- **Source:** https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- **Description:** VoxCeleb is a large-scale audio-visual speaker recognition dataset collected from real-world YouTube videos. It contains over one million utterances from thousands of speakers recorded under diverse acoustic conditions, including background chatter, overlapping speech, room reverberation, and varying recording devices.
- **Usage in Project:** Used to finetune the ECAPA-TDNN model on PCEN frontend.

### 2. MUSAN (Music, Speech, and Noise Corpus)
- **Source:** https://www.openslr.org/17/
- **Description:** MUSAN is a corpus containing music, speech, and a diverse collection of technical and non-technical noise recordings. It was specifically designed for tasks such as voice activity detection, speech/music discrimination, and robustness enhancement through noise augmentation. 
- **Usage in Project:** Used in training all the three Models, for injecting real world noises to match the required KPIs.

## Datasets Published

### TTS Corpus Dataset
- **Link:** https://huggingface.co/datasets/Nishchal-29/tts_corpus
- **Description:** A Text-to-Speech (TTS) corpus created for speech synthesis and speaker modeling experiments. The dataset contains recordings from **20 speakers**, with **4 utterances per speaker**, covering approximately **200 unique words**. It is designed to provide a compact, multi-speaker speech dataset suitable for prototyping, benchmarking, and educational research in TTS and speech processing.
- **Usage in Project:** It is used for our custom keyword detection, This dataset consists of some custom keywords so that our model can generalize patterns for incoming new words.
