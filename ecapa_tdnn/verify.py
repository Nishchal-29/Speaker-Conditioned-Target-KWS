import os
import sys
import types
import json
import numpy as np
import librosa
import onnxruntime as ort
import soundfile as sf
import glob
import argparse
from pathlib import Path

# --- BUG FIX FOR WINDOWS / PYTORCH 2.x ---
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
sys.modules['k2'] = types.ModuleType('k2')
sys.modules['flair'] = types.ModuleType('flair')
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = types.ModuleType('fake_flair_emb')
sys.modules['speechbrain.integrations.nlp'] = types.ModuleType('fake_nlp')
sys.modules['speechbrain.integrations.huggingface.wordemb'] = types.ModuleType('fake_wordemb')
sys.modules['speechbrain.integrations.huggingface'] = types.ModuleType('fake_hf')
# -----------------------------------------

import torch
import torch.nn.functional as F
torch._dynamo.config.disable = True
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pcen import PCENProcessor
from metrics import compute_eer

class OnnxSpeakerVerifier:
    """
    Speaker verification using the exported ONNX backbone + PCEN sidecar.
    This class performs:
      1. Audio loading and preprocessing
      2. PCEN feature extraction (frozen params from sidecar)
      3. ONNX inference → 192-D L2-normalised embedding
      4. Cosine similarity comparison
    """
    def __init__(self, onnx_model_path, pcen_params_path, sample_rate=16000):
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        with open(pcen_params_path, 'r') as f:
            pcen_params = json.load(f)

        self.pcen = PCENProcessor(pcen_params, sample_rate=sample_rate)
        self.sr = sample_rate

        print(f"[OnnxSpeakerVerifier] Loaded ONNX model: {onnx_model_path}")
        print(f"  PCEN params: s={pcen_params['s']:.6f}, "
              f"α={pcen_params['alpha']:.6f}, "
              f"δ={pcen_params['delta']:.6f}, "
              f"r={pcen_params['r']:.6f}")

    def extract_embedding(self, audio_path, max_audio_length=48000):
        wav, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        trimmed, _ = librosa.effects.trim(wav, top_db=30)
        if len(trimmed) > 0:
            wav = trimmed

        if len(wav) < max_audio_length:
            repeats = int(np.ceil(max_audio_length / len(wav)))
            wav = np.tile(wav, repeats)[:max_audio_length]
        else:
            wav = wav[:max_audio_length]

        pcen_features = self.pcen.process(wav) 
        pcen_input = pcen_features[np.newaxis, :, :].astype(np.float32)

        pcen_transposed = np.transpose(pcen_input, (0, 2, 1))  
        mean = np.mean(pcen_transposed, axis=1, keepdims=True)
        std = np.std(pcen_transposed, axis=1, keepdims=True)
        pcen_transposed = (pcen_transposed - mean) / (std + 1e-5)
        pcen_input = np.transpose(pcen_transposed, (0, 2, 1))  

        raw_embedding = self.session.run(
            [self.output_name],
            {self.input_name: pcen_input}
        )[0].squeeze()

        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            embedding = raw_embedding / norm
        else:
            embedding = raw_embedding

        return embedding.astype(np.float32)

    def verify(self, audio_path_1, audio_path_2, threshold=0.45):
        """Compare two audio files and return similarity score + match decision"""
        emb1 = self.extract_embedding(audio_path_1)
        emb2 = self.extract_embedding(audio_path_2)
        similarity = float(np.dot(emb1, emb2))
        is_match = similarity >= threshold

        return similarity, is_match

    def run_eer_validation(self, data_dir, n_same=25, n_diff=25, file_ext="wav", seed=42):
        all_files = sorted(glob.glob(f"{data_dir}/**/*.{file_ext}", recursive=True))
        if len(all_files) == 0:
            print(f"Error: No .{file_ext} files found in {data_dir}")
            return 1.0, 0.0, {}

        speaker_files = {}
        for f in all_files:
            spk = f.split(os.sep)[-3]
            if spk not in speaker_files:
                speaker_files[spk] = []
            speaker_files[spk].append(f)

        speakers = sorted(speaker_files.keys())
        print(f"Found {len(speakers)} speakers, {len(all_files)} files")
        valid_speakers = [s for s in speakers if len(speaker_files[s]) >= 2]
        print(f"Speakers with ≥ 2 files: {len(valid_speakers)}")

        rng = np.random.RandomState(seed)
        same_pairs = []
        attempts = 0
        while len(same_pairs) < n_same and attempts < n_same * 10:
            spk = rng.choice(valid_speakers)
            files = speaker_files[spk]
            idx = rng.choice(len(files), size=2, replace=False)
            pair = (files[idx[0]], files[idx[1]])
            if pair not in same_pairs:
                same_pairs.append(pair)
            attempts += 1

        diff_pairs = []
        attempts = 0
        while len(diff_pairs) < n_diff and attempts < n_diff * 10:
            spk1, spk2 = rng.choice(valid_speakers, size=2, replace=False)
            f1 = rng.choice(speaker_files[spk1])
            f2 = rng.choice(speaker_files[spk2])
            pair = (f1, f2)
            if pair not in diff_pairs:
                diff_pairs.append(pair)
            attempts += 1

        print(f"Generated {len(same_pairs)} same-speaker + {len(diff_pairs)} diff-speaker pairs\n")
        scores = []
        labels = []

        print("Computing similarities...")
        for i, (f1, f2) in enumerate(same_pairs):
            sim, _ = self.verify(f1, f2)
            scores.append(sim)
            labels.append(1)
            print(f"Same {i + 1}/{len(same_pairs)}: {sim:.4f}")

        for i, (f1, f2) in enumerate(diff_pairs):
            sim, _ = self.verify(f1, f2)
            scores.append(sim)
            labels.append(0)
            print(f"Diff {i + 1}/{len(diff_pairs)}: {sim:.4f}")

        eer, threshold = compute_eer(scores, labels)
        report = {
            "eer": float(eer),
            "threshold": float(threshold),
            "n_same_pairs": len(same_pairs),
            "n_diff_pairs": len(diff_pairs),
            "same_scores_mean": float(np.mean([s for s, l in zip(scores, labels) if l == 1])),
            "diff_scores_mean": float(np.mean([s for s, l in zip(scores, labels) if l == 0])),
            "passed": eer < 0.05,
        }

        print(f"\n{'=' * 50}")
        print(f"EER: {eer:.4f} (threshold: {threshold:.4f})")
        print(f"Same-speaker mean: {report['same_scores_mean']:.4f}")
        print(f"Diff-speaker mean: {report['diff_scores_mean']:.4f}")

        if report['passed']:
            print(f"PASSED — EER < 5%")
        else:
            print(f"FAILED — EER ≥ 5%")

        return eer, threshold, report


def run_noisy_onnx_test(onnx_path, pcen_path, data_dir, musan_dir):
    verifier = OnnxSpeakerVerifier(onnx_model_path=onnx_path, pcen_params_path=pcen_path)    
    all_files = sorted(glob.glob(f"{data_dir}/**/*.wav", recursive=True))
    speaker_files = {}
    for f in all_files:
        spk = f.split(os.sep)[-3]
        if spk not in speaker_files:
            speaker_files[spk] = []
        speaker_files[spk].append(f)
        
    valid_speakers = [s for s in speaker_files.keys() if len(speaker_files[s]) >= 2]
    spk_A, spk_B = np.random.choice(valid_speakers, 2, replace=False)
    file_A1, file_A2 = np.random.choice(speaker_files[spk_A], 2, replace=False)
    file_B1 = np.random.choice(speaker_files[spk_B], 1)[0]
    print(f"Selected Speaker A: {spk_A}")
    print(f"Selected Speaker B: {spk_B}")
    noise_files = glob.glob(f"{musan_dir}/noise/**/*.wav", recursive=True) + \
                  glob.glob(f"{musan_dir}/speech/**/*.wav", recursive=True)
                  
    if not noise_files:
        print(f"ERROR: No MUSAN files found in {musan_dir}. Cannot run noise test.")
        sys.exit(1)
        
    random_noise_path = np.random.choice(noise_files)
    print(f"Generating -5dB REALISTIC Noisy Audio using {os.path.basename(random_noise_path)}...")
    wav_A2, sr = librosa.load(file_A2, sr=16000, mono=True)    
    info = sf.info(random_noise_path)
    total_frames = info.frames
    noise_sr = info.samplerate        
    target_frames = int(len(wav_A2) * (noise_sr / sr))

    if total_frames > target_frames:
        start_frame = np.random.randint(0, total_frames - target_frames)
        noise, _ = sf.read(random_noise_path, frames=target_frames, start=start_frame, dtype='float32')
    else:
        noise, _ = sf.read(random_noise_path, dtype='float32')
        repeats = int(np.ceil(target_frames / len(noise)))
        noise = np.tile(noise, repeats)[:target_frames]

    if len(noise.shape) > 1:
        noise = np.mean(noise, axis=1)

    if noise_sr != sr:
        noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)
        if len(noise) > len(wav_A2):
            noise = noise[:len(wav_A2)]
        elif len(noise) < len(wav_A2):
            noise = np.pad(noise, (0, len(wav_A2) - len(noise)))

    target_snr = -5
    signal_power = np.mean(wav_A2 ** 2) + 1e-7
    noise_power = np.mean(noise ** 2) + 1e-7
    k = np.sqrt((signal_power / (10 ** (target_snr / 10))) / noise_power)
    noisy_A2 = wav_A2 + (noise * k)

    max_val = np.max(np.abs(noisy_A2))
    if max_val > 1.0:
        noisy_A2 = noisy_A2 / max_val
            
    temp_noisy_path = "./temp_realistic_noisy_A2.wav"
    sf.write(temp_noisy_path, noisy_A2, sr)

    try:
        print("\n--- TEST 1: SAME SPEAKER (Clean A1 vs MUSAN Noisy A2 @ -5dB) ---")
        sim_same, match_same = verifier.verify(file_A1, temp_noisy_path, threshold=0.25)
        print(f"Cosine Similarity Score: {sim_same:.4f}")
        if match_same:
            print("MATCH! The ONNX model successfully saw through the realistic noise.")
        else:
            print("NO MATCH. The noise caused a false rejection.")

        print("\n--- TEST 2: DIFFERENT SPEAKERS (Clean A1 vs Clean B1) ---")
        sim_diff, match_diff = verifier.verify(file_A1, file_B1, threshold=0.25)
        print(f"Cosine Similarity Score: {sim_diff:.4f}")
        if match_diff:
            print("FALSE POSITIVE MATCH! (Check threshold)")
        else:
            print("CORRECT REJECTION! These are different speakers.")
            
    finally:
        if os.path.exists(temp_noisy_path):
            os.remove(temp_noisy_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Verification / EER Validation")
    parser.add_argument("--mode", choices=["verify", "eer", "noise_test"], default="eer")
    parser.add_argument("--onnx", default="./finetuned_models/ecapa_backbone.onnx",
                       help="Path to ONNX backbone model")
    parser.add_argument("--pcen", default="./finetuned_models/pcen_params.json",
                       help="Path to PCEN parameter sidecar JSON")
    parser.add_argument("--data", default="../data/VoxCeleb1/wav",
                       help="Path to audio data directory (for EER mode)")
    parser.add_argument("--musan", default="../data/musan",
                       help="Path to MUSAN data directory (for noise_test mode)")
    parser.add_argument("--audio1", help="First audio file (for verify mode)")
    parser.add_argument("--audio2", help="Second audio file (for verify mode)")
    parser.add_argument("--threshold", type=float, default=0.45,
                       help="Similarity threshold for match decision")

    args = parser.parse_args()

    if args.mode == "noise_test":
        if not os.path.exists(args.onnx):
            print(f"ONNX model not found: {args.onnx}")
            sys.exit(1)
        run_noisy_onnx_test(args.onnx, args.pcen, args.data, args.musan)
        
    elif args.mode == "eer":
        if not os.path.exists(args.onnx):
            print(f"ONNX model not found: {args.onnx}")
            print("Run train.py first to produce the domain-adapted model.")
            sys.exit(1)

        verifier = OnnxSpeakerVerifier(
            onnx_model_path=args.onnx,
            pcen_params_path=args.pcen,
        )
        eer, threshold, report = verifier.run_eer_validation(args.data, file_ext="wav")

        report_path = os.path.join(os.path.dirname(args.onnx), "eer_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")

    elif args.mode == "verify":
        if not args.audio1 or not args.audio2:
            print("--audio1 and --audio2 are required for verify mode")
            sys.exit(1)

        verifier = OnnxSpeakerVerifier(
            onnx_model_path=args.onnx,
            pcen_params_path=args.pcen,
        )

        sim, is_match = verifier.verify(args.audio1, args.audio2, threshold=args.threshold)

        print(f"\nCosine Similarity: {sim:.4f}")
        if is_match:
            print(f"MATCH (threshold: {args.threshold})")
        else:
            print(f"NO MATCH (threshold: {args.threshold})")