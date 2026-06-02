"""
Produces a single, stable 192-D speaker embedding for a specific end-user,
given 5–10 short enrollment utterances spoken into the edge microphone.
Uses the frozen ONNX backbone + PCEN JSON sidecar from Context A.
"""

import os
import json
import base64
import struct
import logging
import hashlib
from datetime import datetime, timezone
import sys
import glob

import numpy as np
import librosa
import onnxruntime as ort

from pcen import PCENProcessor

logger = logging.getLogger("speaker_enrollment")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)

class AudioQualityGate:
    """
    Enforces audio quality requirements before enrollment processing.

    Quality gates (all must pass):
      1. Duration: 2–8 seconds of voiced speech (after silence trimming)
      2. SNR: ≥ 10 dB (WADA-SNR estimate)
      3. VAD: ≥ 60% voiced frames (energy + ZCR)
      4. Clipping: < 0.1% of samples at ±1.0

    Spec reference: Context B, Step 1.
    """

    def __init__(self, sample_rate=16000, min_duration=2.0, max_duration=8.0,
                 min_snr_db=10.0, min_voiced_ratio=0.60, max_clipping_ratio=0.001,
                 silence_threshold_db=40.0):
        self.sr = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_snr_db = min_snr_db
        self.min_voiced_ratio = min_voiced_ratio
        self.max_clipping_ratio = max_clipping_ratio
        self.silence_threshold_db = silence_threshold_db

    def validate(self, audio_path):
        reasons = []

        # Load audio
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            return False, [f"LOAD_FAILED: {str(e)}"], None

        if len(audio) == 0:
            return False, ["EMPTY_AUDIO: File contains no audio data"], None

        # --- Gate 1: Clipping check (before trimming) ---
        clipping_passed, clipping_reason = self._check_clipping(audio)
        if not clipping_passed:
            reasons.append(clipping_reason)

        # --- Silence trimming (40 dB below peak) ---
        trimmed, _ = librosa.effects.trim(audio, top_db=self.silence_threshold_db)
        if len(trimmed) == 0:
            return False, ["SILENCE: Entire audio is below the silence threshold"], None

        # --- Gate 2: Duration check (after trimming) ---
        duration = len(trimmed) / self.sr
        if duration < self.min_duration:
            reasons.append(f"TOO_SHORT: {duration:.2f}s < {self.min_duration}s minimum")
        elif duration > self.max_duration:
            reasons.append(f"TOO_LONG: {duration:.2f}s > {self.max_duration}s maximum")

        # --- Gate 3: SNR estimate ---
        snr_passed, snr_reason = self._check_snr(trimmed)
        if not snr_passed:
            reasons.append(snr_reason)

        # --- Gate 4: VAD check ---
        vad_passed, vad_reason = self._check_vad(trimmed)
        if not vad_passed:
            reasons.append(vad_reason)

        passed = len(reasons) == 0
        return passed, reasons, trimmed if passed else audio

    def _check_clipping(self, audio):
        """Check that fewer than 0.1% of samples are at ±1.0."""
        n_clipped = np.sum(np.abs(audio) >= 0.999)  # near full-scale
        ratio = n_clipped / len(audio)
        if ratio >= self.max_clipping_ratio:
            return False, f"CLIPPED: {ratio * 100:.3f}% samples at full scale (max {self.max_clipping_ratio * 100:.1f}%)"
        return True, ""

    def _check_snr(self, audio):
        snr_db = self._estimate_wada_snr(audio)
        if snr_db < self.min_snr_db:
            return False, f"LOW_SNR: {snr_db:.1f} dB < {self.min_snr_db} dB minimum"
        return True, ""

    def _estimate_wada_snr(self, audio):
        abs_audio = np.abs(audio)
        abs_audio = abs_audio[abs_audio > 0]  # remove zero samples

        if len(abs_audio) < 100:
            return 0.0

        # Sort amplitudes
        sorted_amp = np.sort(abs_audio)

        # Estimate signal level (90th percentile) and noise floor (10th percentile)
        signal_level = sorted_amp[int(0.9 * len(sorted_amp))]
        noise_level = sorted_amp[int(0.1 * len(sorted_amp))]

        if noise_level < 1e-10:
            return 60.0  # essentially clean

        snr_db = 20 * np.log10(signal_level / noise_level)
        return float(snr_db)

    def _check_vad(self, audio):
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.010 * self.sr)    # 10ms hop

        # Compute frame energy
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.mean(frames ** 2, axis=0)

        # Energy threshold: frames with energy > 1% of max frame energy
        energy_threshold = 0.01 * np.max(energy)
        energy_voiced = energy > energy_threshold

        # ZCR: speech typically has ZCR between 0.01 and 0.3
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Trim ZCR to match energy length
        min_len = min(len(energy_voiced), len(zcr))
        energy_voiced = energy_voiced[:min_len]
        zcr = zcr[:min_len]

        speech_zcr = (zcr > 0.01) & (zcr < 0.30)

        # Voiced = high energy AND speech-like ZCR
        voiced_frames = energy_voiced & speech_zcr
        voiced_ratio = np.mean(voiced_frames)

        if voiced_ratio < self.min_voiced_ratio:
            return False, (f"LOW_VAD: {voiced_ratio * 100:.1f}% voiced frames "
                          f"< {self.min_voiced_ratio * 100:.0f}% minimum")
        return True, ""

class SpeakerEnroller:
    """
    Produces a verified 192-D speaker embedding from enrollment utterances.

    Uses the frozen ONNX backbone + PCEN JSON sidecar from Context A.

    Pipeline:
      1. Validate each utterance (AudioQualityGate)
      2. Apply PCEN (frozen params from sidecar)
      3. Extract embeddings via ONNX backbone
      4. Outlier rejection (μ − 1.5σ)
      5. Mean pooling + L2-normalisation
      6. Coherence gate (≥ 0.75)
      7. Write enrolled profile JSON
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

        # Load PCEN parameters (must be byte-for-byte identical to Context A — Invariant #1)
        with open(pcen_params_path, 'r') as f:
            self.pcen_params = json.load(f)

        self.pcen = PCENProcessor(self.pcen_params, sample_rate=sample_rate)

        # Compute backbone hash for enrolled profile
        self.backbone_hash = self._compute_file_hash(onnx_model_path)

        # Quality gate
        self.quality_gate = AudioQualityGate(sample_rate=sample_rate)

        self.sr = sample_rate

        logger.info(f"SpeakerEnroller initialized")
        logger.info(f"  ONNX model: {onnx_model_path}")
        logger.info(f"  PCEN params: s={self.pcen_params['s']:.6f}, "
                    f"α={self.pcen_params['alpha']:.6f}, "
                    f"δ={self.pcen_params['delta']:.6f}, "
                    f"r={self.pcen_params['r']:.6f}")

    def enroll(self, audio_paths, min_accepted=3):
        metrics = {
            "n_utterances_total": len(audio_paths),
            "n_utterances_accepted": 0,
            "n_utterances_rejected": 0,
            "rejection_reasons": {},  # per-utterance: {filename: [reasons]}
            "coherence_score": None,
        }

        logger.info(f"Starting enrollment with {len(audio_paths)} utterances")

        # --- Step 1: Audio Quality Validation ---
        accepted_audio = []
        accepted_paths = []

        for path in audio_paths:
            filename = os.path.basename(path)
            passed, reasons, audio = self.quality_gate.validate(path)

            if passed:
                accepted_audio.append(audio)
                accepted_paths.append(path)
                metrics["n_utterances_accepted"] += 1
                logger.info(f"  ✓ {filename} — passed all quality gates")
            else:
                metrics["n_utterances_rejected"] += 1
                metrics["rejection_reasons"][filename] = reasons
                logger.warning(f"  ✗ {filename} — rejected: {', '.join(reasons)}")

        if metrics["n_utterances_accepted"] < min_accepted:
            error = (f"Insufficient accepted utterances: {metrics['n_utterances_accepted']} "
                    f"< {min_accepted} minimum. Please re-record in a quieter environment.")
            logger.error(error)
            return {"success": False, "error": error, "metrics": metrics}

        # --- Steps 2–3: PCEN + Embedding Extraction ---
        embeddings = []
        for i, audio in enumerate(accepted_audio):
            emb = self._extract_embedding(audio)
            embeddings.append(emb)
            logger.info(f"  Embedding {i + 1}/{len(accepted_audio)}: "
                       f"norm={np.linalg.norm(emb):.4f}")

        embeddings = np.array(embeddings)  # [K, 192]
        K = len(embeddings)

        # --- Step 4: Outlier Rejection ---
        retained_indices, rejected_indices = self._reject_outliers(embeddings)

        if len(rejected_indices) > 0:
            for idx in rejected_indices:
                filename = os.path.basename(accepted_paths[idx])
                logger.warning(f"  Outlier rejected: {filename}")
                metrics["rejection_reasons"][filename] = ["OUTLIER: Embedding too dissimilar from consensus"]
                metrics["n_utterances_rejected"] += 1
                metrics["n_utterances_accepted"] -= 1

        if len(retained_indices) < min_accepted:
            error = (f"Too few embeddings after outlier rejection: {len(retained_indices)} "
                    f"< {min_accepted}. Please re-record.")
            logger.error(error)
            return {"success": False, "error": error, "metrics": metrics}

        # --- Step 4f–4g: Mean Pooling + L2 Normalisation ---
        retained_embeddings = embeddings[retained_indices]
        mean_embedding = np.mean(retained_embeddings, axis=0)
        speaker_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        # --- Step 5: Coherence Gate ---
        coherence = self._compute_coherence(retained_embeddings)
        metrics["coherence_score"] = float(coherence)

        logger.info(f"  Enrollment coherence: {coherence:.4f}")

        if coherence < 0.75:
            error = (f"Enrollment coherence too low: {coherence:.4f} < 0.75. "
                    f"Possible causes: multiple speakers, severe noise, or microphone mismatch. "
                    f"Please re-record in a quieter environment.")
            logger.error(error)
            return {"success": False, "error": error, "metrics": metrics}

        # --- Step 6: Build Enrolled Profile ---
        profile = {
            "speaker_embedding": self._encode_embedding(speaker_embedding),
            "enrollment_coherence": float(coherence),
            "n_utterances_accepted": int(len(retained_indices)),
            "pcen_params": self.pcen_params.copy(),
            "backbone_version": self.backbone_hash,
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"✅ Enrollment successful — coherence: {coherence:.4f}, "
                    f"accepted: {len(retained_indices)}/{len(audio_paths)}")

        # Log observability metrics (Invariant #4)
        logger.info(f"  Metrics: {json.dumps(metrics)}")

        return {"success": True, "profile": profile, "metrics": metrics}

    def _extract_embedding(self, audio):
        # Step 2: Apply PCEN with frozen sidecar parameters
        pcen_features = self.pcen.process(audio)  # [n_mels, T]

        # Prepare for ONNX: add batch dimension → [1, n_mels, T]
        pcen_input = pcen_features[np.newaxis, :, :].astype(np.float32)

        # Utterance-level normalization (match training)
        # Shape: [1, n_mels, T] → transpose to [1, T, n_mels] for normalization
        pcen_transposed = np.transpose(pcen_input, (0, 2, 1))  # [1, T, n_mels]
        mean = np.mean(pcen_transposed, axis=1, keepdims=True)
        std = np.std(pcen_transposed, axis=1, keepdims=True)
        pcen_transposed = (pcen_transposed - mean) / (std + 1e-5)
        pcen_input = np.transpose(pcen_transposed, (0, 2, 1))  # back to [1, n_mels, T]

        # Step 3a: Run ONNX inference
        raw_embedding = self.session.run(
            [self.output_name],
            {self.input_name: pcen_input}
        )[0]

        # Step 3b: L2-normalise → lives on S^191
        raw_embedding = raw_embedding.squeeze()
        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            embedding = raw_embedding / norm
        else:
            embedding = raw_embedding

        return embedding.astype(np.float32)

    def _reject_outliers(self, embeddings):
        K = len(embeddings)
        if K <= 3:
            # Can't afford to reject any
            return list(range(K)), []

        # Step 4a: Pairwise cosine similarity matrix
        # Since embeddings are L2-normalised, cosine sim = dot product
        C = embeddings @ embeddings.T  # [K, K]

        # Step 4b: Mean similarity per utterance (excluding self)
        scores = np.zeros(K)
        for k in range(K):
            others = [C[k, j] for j in range(K) if j != k]
            scores[k] = np.mean(others)

        # Step 4c: Statistics
        mu_score = np.mean(scores)
        sigma_score = np.std(scores)

        # Step 4d: Reject if score_k < μ − 1.5σ
        threshold = mu_score - 1.5 * sigma_score
        retained = [k for k in range(K) if scores[k] >= threshold]
        rejected = [k for k in range(K) if scores[k] < threshold]

        logger.info(f"  Outlier rejection: μ={mu_score:.4f}, σ={sigma_score:.4f}, "
                   f"threshold={threshold:.4f}")
        logger.info(f"  Retained: {len(retained)}/{K}, Rejected: {len(rejected)}")

        return retained, rejected

    def _compute_coherence(self, embeddings):
        K = len(embeddings)
        if K < 2:
            return 0.0

        C = embeddings @ embeddings.T
        # Extract upper triangle (excluding diagonal)
        upper_indices = np.triu_indices(K, k=1)
        pairwise_sims = C[upper_indices]

        return float(np.mean(pairwise_sims))

    @staticmethod
    def _encode_embedding(embedding):
        """
        Encode a float32[192] embedding as a base64 string.
        Base64 ensures exact float32 byte preservation and compact storage
        suitable for edge microcontrollers.

        Args:
            embedding: numpy float32[192] array.

        Returns:
            str: base64-encoded string of 768 bytes (192 × 4 bytes).
        """
        raw_bytes = embedding.astype(np.float32).tobytes()
        return base64.b64encode(raw_bytes).decode('ascii')

    @staticmethod
    def decode_embedding(b64_string):
        """
        Decode a base64-encoded float32[192] embedding.
        Args:
            b64_string: base64 string from the enrolled profile.

        Returns:
            numpy float32[192] array.
        """
        raw_bytes = base64.b64decode(b64_string)
        return np.frombuffer(raw_bytes, dtype=np.float32).copy()

    @staticmethod
    def _compute_file_hash(filepath):
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(8192), b''):
                sha256.update(block)
        return sha256.hexdigest()

    def save_profile(self, profile, output_path):
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Enrolled profile saved to: {output_path}")

    def load_profile(self, profile_path):
        """Load an enrolled profile from a JSON file."""
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        return profile

    def verify_against_profile(self, profile, audio_path, threshold=0.45):
        """
        Compare a new utterance against an enrolled profile.
        Args:
            profile: dict — loaded enrolled profile.
            audio_path: str — path to the test utterance.
            threshold: float — cosine similarity threshold for match.

        Returns:
            similarity: float — cosine similarity score.
            is_match: bool — True if similarity >= threshold.
        """
        # Validate the test utterance
        passed, reasons, audio = self.quality_gate.validate(audio_path)
        if not passed:
            logger.warning(f"Test utterance rejected: {', '.join(reasons)}")
            return 0.0, False

        # Extract embedding
        test_embedding = self._extract_embedding(audio)

        # Decode enrolled embedding
        enrolled_embedding = self.decode_embedding(profile["speaker_embedding"])

        # Cosine similarity (both are L2-normalised → dot product)
        similarity = float(np.dot(test_embedding, enrolled_embedding))

        is_match = similarity >= threshold
        return similarity, is_match

if __name__ == "__main__":
    # Default paths
    ONNX_PATH = "./finetuned_models/ecapa_backbone.onnx"
    PCEN_PATH = "./finetuned_models/pcen_params.json"

    if not os.path.exists(ONNX_PATH):
        print(f"ONNX model not found at {ONNX_PATH}")
        print("Run train.py first to produce the domain-adapted model.")
        sys.exit(1)

    if not os.path.exists(PCEN_PATH):
        print(f"PCEN params not found at {PCEN_PATH}")
        sys.exit(1)

    # Initialize enroller
    enroller = SpeakerEnroller(
        onnx_model_path=ONNX_PATH,
        pcen_params_path=PCEN_PATH,
    )

    # Example: Enroll speaker 84 from LibriSpeech dev-clean
    speaker_dir = "./data/LibriSpeech/dev-clean/84"
    if os.path.exists(speaker_dir):
        audio_files = sorted(glob.glob(f"{speaker_dir}/**/*.flac", recursive=True))[:7]
        print(f"\nEnrolling speaker 84 with {len(audio_files)} utterances...")

        result = enroller.enroll(audio_files)

        if result["success"]:
            profile_path = "./finetuned_models/enrolled_speaker_84.json"
            enroller.save_profile(result["profile"], profile_path)

            print(f"\nEnrollment successful!")
            print(f"  Coherence: {result['profile']['enrollment_coherence']:.4f}")
            print(f"  Accepted: {result['profile']['n_utterances_accepted']}")
            print(f"  Profile: {profile_path}")

            # Quick verification test
            test_file = audio_files[-1]
            sim, match = enroller.verify_against_profile(result["profile"], test_file)
            print(f"\nVerification test: similarity={sim:.4f}, match={match}")
        else:
            print(f"\nEnrollment failed: {result['error']}")
    else:
        print(f"Speaker directory not found: {speaker_dir}")
        print("Update the path to point to your audio data.")