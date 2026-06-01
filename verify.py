import os
import sys
import types
import json
import itertools
import urllib.request

import numpy as np
import librosa
import onnxruntime as ort
import soundfile as sf

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

try:
    torch._dynamo.config.disable = True
except Exception:
    pass

from pcen import PCENProcessor
from metrics import compute_eer


# ============================================================================
# ONNX Speaker Verifier
# ============================================================================

class OnnxSpeakerVerifier:
    """
    Speaker verification using the exported ONNX backbone + PCEN sidecar.

    This class performs:
      1. Audio loading and preprocessing
      2. PCEN feature extraction (frozen params from sidecar)
      3. ONNX inference → 192-D L2-normalised embedding
      4. Cosine similarity comparison

    Also provides run_eer_validation() for the 50-pair EER check
    required by Context A Step 6e.
    """

    def __init__(self, onnx_model_path, pcen_params_path, sample_rate=16000):
        """
        Args:
            onnx_model_path: Path to the exported ECAPA-TDNN ONNX backbone.
            pcen_params_path: Path to the PCEN parameter JSON sidecar.
            sample_rate: Audio sample rate (must be 16kHz).
        """
        # Load ONNX model
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Load PCEN parameters (Invariant #1: byte-for-byte identical to Context A)
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
        """
        Extract a 192-D L2-normalised embedding from an audio file.

        Pipeline: load → trim → tile/truncate → PCEN → normalize → ONNX → L2-norm

        Args:
            audio_path: Path to an audio file.
            max_audio_length: Target length in samples (48000 = 3s @ 16kHz).

        Returns:
            embedding: numpy float32[192], L2-normalised.
        """
        # Load audio
        wav, _ = librosa.load(audio_path, sr=self.sr, mono=True)

        # Strip silence
        trimmed, _ = librosa.effects.trim(wav, top_db=30)
        if len(trimmed) > 0:
            wav = trimmed

        # Tile or truncate
        if len(wav) < max_audio_length:
            repeats = int(np.ceil(max_audio_length / len(wav)))
            wav = np.tile(wav, repeats)[:max_audio_length]
        else:
            wav = wav[:max_audio_length]

        # PCEN
        pcen_features = self.pcen.process(wav)  # [n_mels, T]

        # Prepare for ONNX: [1, n_mels, T]
        pcen_input = pcen_features[np.newaxis, :, :].astype(np.float32)

        # Utterance-level normalization (match training)
        pcen_transposed = np.transpose(pcen_input, (0, 2, 1))  # [1, T, n_mels]
        mean = np.mean(pcen_transposed, axis=1, keepdims=True)
        std = np.std(pcen_transposed, axis=1, keepdims=True)
        pcen_transposed = (pcen_transposed - mean) / (std + 1e-5)
        pcen_input = np.transpose(pcen_transposed, (0, 2, 1))  # [1, n_mels, T]

        # ONNX inference
        raw_embedding = self.session.run(
            [self.output_name],
            {self.input_name: pcen_input}
        )[0].squeeze()

        # L2-normalise (Invariant #3)
        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            embedding = raw_embedding / norm
        else:
            embedding = raw_embedding

        return embedding.astype(np.float32)

    def verify(self, audio_path_1, audio_path_2, threshold=0.45):
        """
        Compare two audio files and return similarity score + match decision.

        Args:
            audio_path_1: Path to first audio file.
            audio_path_2: Path to second audio file.
            threshold: Cosine similarity threshold for positive match.

        Returns:
            similarity: float — cosine similarity score.
            is_match: bool — True if similarity >= threshold.
        """
        emb1 = self.extract_embedding(audio_path_1)
        emb2 = self.extract_embedding(audio_path_2)

        # Cosine similarity (both L2-normalised → dot product)
        similarity = float(np.dot(emb1, emb2))
        is_match = similarity >= threshold

        return similarity, is_match

    def run_eer_validation(self, data_dir, n_same=25, n_diff=25, file_ext="flac",
                           seed=42):
        """
        Run the 50-pair EER validation check required by Context A Step 6e.

        Generates 25 same-speaker pairs and 25 different-speaker pairs from
        the provided data directory, computes cosine similarities, and
        calculates EER.

        Args:
            data_dir: Path to audio directory (LibriSpeech-style structure).
            n_same: Number of same-speaker pairs.
            n_diff: Number of different-speaker pairs.
            file_ext: Audio file extension.
            seed: Random seed for pair selection.

        Returns:
            eer: float — the Equal Error Rate.
            threshold: float — the EER threshold.
            report: dict — detailed validation report.
        """
        import glob

        print(f"\n{'=' * 50}")
        print("EER VALIDATION (50-pair check)")
        print(f"{'=' * 50}\n")

        # Discover speakers and their files
        all_files = sorted(glob.glob(f"{data_dir}/**/*.{file_ext}", recursive=True))
        speaker_files = {}
        for f in all_files:
            spk = f.split(os.sep)[-3]
            if spk not in speaker_files:
                speaker_files[spk] = []
            speaker_files[spk].append(f)

        speakers = sorted(speaker_files.keys())
        print(f"Found {len(speakers)} speakers, {len(all_files)} files")

        # Filter speakers with at least 2 files (needed for same-speaker pairs)
        valid_speakers = [s for s in speakers if len(speaker_files[s]) >= 2]
        print(f"Speakers with ≥ 2 files: {len(valid_speakers)}")

        rng = np.random.RandomState(seed)

        # Generate same-speaker pairs
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

        # Generate different-speaker pairs
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

        # Compute similarities
        scores = []
        labels = []

        print("Computing similarities...")
        for i, (f1, f2) in enumerate(same_pairs):
            sim, _ = self.verify(f1, f2)
            scores.append(sim)
            labels.append(1)
            print(f"  Same {i + 1}/{len(same_pairs)}: {sim:.4f}")

        for i, (f1, f2) in enumerate(diff_pairs):
            sim, _ = self.verify(f1, f2)
            scores.append(sim)
            labels.append(0)
            print(f"  Diff {i + 1}/{len(diff_pairs)}: {sim:.4f}")

        # Compute EER
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
            print(f"✅ PASSED — EER < 5%")
        else:
            print(f"⚠️  FAILED — EER ≥ 5%")

        print(f"{'=' * 50}\n")

        return eer, threshold, report


def run_noisy_onnx_test(onnx_path, pcen_path):
    print("\nInitializing Production ONNX Pipeline...")
    verifier = OnnxSpeakerVerifier(onnx_model_path=onnx_path, pcen_params_path=pcen_path)
    
    # Test files from LibriSpeech dev-clean
    file_A1 = "./data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"  # Speaker A, Utterance 1
    file_A2 = "./data/LibriSpeech/dev-clean/84/121550/84-121550-0000.flac"  # Speaker A, Utterance 2
    file_B1 = "./data/LibriSpeech/dev-clean/174/50561/174-50561-0000.flac" # Speaker B, Utterance 1

    def generate_synthetic_babble(duration_sec, sr=16000):
        """
        Generates non-stationary noise mimicking a crowded room.
        It creates 50 overlapping voices with random human-speech frequencies
        and amplitude modulations (simulating breathing/pauses).
        """
        time = np.linspace(0, duration_sec, int(sr * duration_sec))
        babble = np.zeros_like(time)
        
        # Simulate 50 people talking at once
        for _ in range(50):
            base_freq = np.random.uniform(300, 3000) # Human speech band
            
            # Slow amplitude wobble to simulate pauses in talking
            amp_mod = np.sin(2 * np.pi * np.random.uniform(0.5, 2.5) * time)
            amp_mod = np.maximum(0, amp_mod) # Half-wave rectify to create hard silences
            
            phase = np.random.uniform(0, 2 * np.pi)
            babble += amp_mod * np.sin(2 * np.pi * base_freq * time + phase)
            
        # Normalize to avoid clipping
        return babble / np.max(np.abs(babble))

    print("Generating offline synthetic babble noise...")
    wav_A2, sr = librosa.load(file_A2, sr=16000, mono=True)
    
    # Generate the exact amount of noise we need
    duration = len(wav_A2) / sr
    wav_noise = generate_synthetic_babble(duration_sec=duration, sr=sr)

    def mix_audio_at_snr(clean_audio, noise_audio, target_snr):
        """Mixes clean speech with a real noise file at a specific SNR."""
        # Ensure lengths match
        if len(noise_audio) > len(clean_audio):
            noise_audio = noise_audio[:len(clean_audio)]
        elif len(noise_audio) < len(clean_audio):
            # Tile the noise if it's too short
            repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)[:len(clean_audio)]
            
        clean_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        # Calculate the multiplier required for the noise to hit the target SNR
        if noise_power == 0:
            return clean_audio
            
        k = np.sqrt((clean_power / (10 ** (target_snr / 10))) / noise_power)
        
        # Mix them
        noisy_signal = clean_audio + (noise_audio * k)
        
        # Prevent clipping
        max_val = np.max(np.abs(noisy_signal))
        if max_val > 1.0:
            noisy_signal = noisy_signal / max_val
            
        return noisy_signal
    
    noisy_A2 = mix_audio_at_snr(wav_A2, wav_noise, target_snr=-5)
    
    temp_noisy_path = "./temp_realistic_noisy_A2.wav"
    sf.write(temp_noisy_path, noisy_A2, sr)

    try:
        print("\n--- TEST 1: SAME SPEAKER (Clean A1 vs Babble Noisy A2 @ -5dB) ---")
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
        # Cleanup temp files
        if os.path.exists(temp_noisy_path):
            os.remove(temp_noisy_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speaker Verification / EER Validation")
    parser.add_argument("--mode", choices=["verify", "eer", "noise_test"], default="eer")
    parser.add_argument("--onnx", default="./finetuned_models/ecapa_backbone.onnx",
                       help="Path to ONNX backbone model")
    parser.add_argument("--pcen", default="./finetuned_models/pcen_params.json",
                       help="Path to PCEN parameter sidecar JSON")
    parser.add_argument("--data", default="./data/LibriSpeech/dev-clean",
                       help="Path to audio data directory (for EER mode)")
    parser.add_argument("--audio1", help="First audio file (for verify mode)")
    parser.add_argument("--audio2", help="Second audio file (for verify mode)")
    parser.add_argument("--threshold", type=float, default=0.45,
                       help="Similarity threshold for match decision")

    args = parser.parse_args()

    if args.mode == "noise_test":
        if not os.path.exists(args.onnx):
            print(f"ONNX model not found: {args.onnx}")
            sys.exit(1)
        run_noisy_onnx_test(args.onnx, args.pcen)
        
    elif args.mode == "eer":
        if not os.path.exists(args.onnx):
            print(f"ONNX model not found: {args.onnx}")
            print("Run train.py first to produce the domain-adapted model.")
            sys.exit(1)

        verifier = OnnxSpeakerVerifier(
            onnx_model_path=args.onnx,
            pcen_params_path=args.pcen,
        )
        eer, threshold, report = verifier.run_eer_validation(args.data)

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
            print(f"✅ MATCH (threshold: {args.threshold})")
        else:
            print(f"❌ NO MATCH (threshold: {args.threshold})")