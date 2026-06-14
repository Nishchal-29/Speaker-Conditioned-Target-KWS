"""
Phase I Path B — Master Word Template Enrollment.

Generates a 128-D Master Word Template w_c for any custom wake word
using K=3 enrollment utterances and the frozen TC-ResNet + PCEN pipeline.

Runs on-device using frozen artefacts from training:
    - tc_resnet_backbone.onnx (or PyTorch checkpoint)
    - pcen_params.json

Usage:
    python enrollment_path_b.py \
        --audio_files utt1.wav utt2.wav utt3.wav \
        --pcen_params ./tc_resnet_output/pcen_params.json \
        --model_path ./tc_resnet_output/best_checkpoint.pth \
        --output enrolled_template.json

Invariants:
    I1 — All embeddings are L2-normalised on S^127.
    I2 — PCEN params must match training exactly.
    I5 — All forward passes inside torch.no_grad().
    I6 — w_c is never logged or returned in plaintext.
"""

import os
import sys
import json
import base64
import hashlib
import argparse
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
import librosa

# Add parent directory for pcen.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pcen import LearnablePCEN
from model import TCResNetAcousticEncoder


# ============================================================================
# Hard Constants
# ============================================================================

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000    # 1 second
EMBEDDING_DIM = 128
N_MELS = 80
K_ENROLLMENT = 3
COHERENCE_MIN = 0.75


# ============================================================================
# Audio Utilities
# ============================================================================

def load_and_preprocess(path, target_sr=SAMPLE_RATE, target_length=TARGET_LENGTH):
    """
    Load, resample if needed, pad/crop to target_length, peak-normalise to 0.9.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate.
        target_length: Target length in samples.

    Returns:
        waveform: Tensor[1, target_length] float32
    """
    wav, sr = librosa.load(path, sr=target_sr, mono=True)

    # Pad or crop
    if len(wav) < target_length:
        wav = np.pad(wav, (0, target_length - len(wav)), mode='constant')
    else:
        wav = wav[:target_length]

    # Peak-normalise to 0.9
    peak = np.abs(wav).max()
    if peak > 1e-9:
        wav = wav / peak * 0.9

    return torch.FloatTensor(wav).unsqueeze(0)  # [1, target_length]


# ============================================================================
# Master Word Template Generation
# ============================================================================

def generate_word_template(audio_paths, pcen, tc_resnet, device):
    """
    Generate a 128-D Master Word Template from K enrollment utterances.

    Args:
        audio_paths: list of K file paths to user enrollment utterances
        pcen: LearnablePCEN loaded from pcen_params.json, frozen
        tc_resnet: TCResNetAcousticEncoder loaded from checkpoint, frozen

    Returns:
        w_c: Tensor[1, 128] — L2-normalised Master Word Template
        coherence: float — mean pairwise cosine similarity

    Raises:
        AssertionError if coherence < COHERENCE_MIN (0.75)
    """
    tc_resnet.eval()
    pcen.eval()

    embeddings = []

    # Invariant I5: All enrollment forward passes inside torch.no_grad()
    with torch.no_grad():
        for path in audio_paths:
            waveform = load_and_preprocess(path).to(device)  # [1, 16000]
            pcen_feat = pcen(waveform)                        # [1, 80, T]
            v_k = tc_resnet(pcen_feat)                        # [1, 128] (L2-normed)
            embeddings.append(v_k)

    # w_raw = mean of K L2-normed vectors
    K = len(embeddings)
    stacked = torch.cat(embeddings, dim=0)          # [K, 128]
    w_raw = stacked.mean(dim=0, keepdim=True)        # [1, 128]

    # Final L2-normalise — the mean of unit vectors is NOT a unit vector
    w_c = F.normalize(w_raw, p=2, dim=1)             # [1, 128] on S^127

    # Coherence gate: mean pairwise cosine similarity
    # Since vectors are L2-normed, cosine sim = dot product
    pairwise = stacked @ stacked.T  # [K, K]
    # Extract upper triangle excluding diagonal
    mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
    pairwise_sims = pairwise[mask]
    coherence = pairwise_sims.mean().item() if len(pairwise_sims) > 0 else 0.0

    if coherence < COHERENCE_MIN:
        raise ValueError(
            f"Enrollment rejected: coherence={coherence:.3f} < {COHERENCE_MIN}. "
            f"Please re-record in a quieter environment with consistent pronunciation."
        )

    return w_c, coherence


# ============================================================================
# Profile I/O
# ============================================================================

def encode_embedding(embedding):
    """
    Encode a float32[128] embedding as base64 for compact JSON storage.

    Invariant I6: w_c is a biometric-adjacent credential. Never log its values.
    """
    raw_bytes = embedding.cpu().numpy().astype(np.float32).tobytes()
    return base64.b64encode(raw_bytes).decode('ascii')


def decode_embedding(b64_string):
    """Decode a base64-encoded float32[128] embedding."""
    raw_bytes = base64.b64decode(b64_string)
    return np.frombuffer(raw_bytes, dtype=np.float32).copy()


def save_enrolled_profile(w_c, coherence, pcen_params, backbone_hash,
                          output_path, n_utterances):
    """
    Save the enrolled word template profile to JSON.

    Invariant I6: The embedding is base64-encoded (not plaintext floats).
    """
    profile = {
        "word_template": encode_embedding(w_c.squeeze(0)),
        "embedding_dim": EMBEDDING_DIM,
        "enrollment_coherence": float(coherence),
        "n_utterances_accepted": n_utterances,
        "pcen_params": pcen_params,
        "backbone_version": backbone_hash,
        "enrolled_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "tc-resnet-path-b",
    }

    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)

    # Invariant I6: do NOT log the embedding values
    print(f"✓ Enrolled profile saved to: {output_path}")
    print(f"  Coherence: {coherence:.4f}")
    print(f"  Utterances: {n_utterances}")

    return profile


# ============================================================================
# Model Loading
# ============================================================================

def load_models(pcen_params_path, model_path, device):
    """
    Load frozen PCEN + TC-ResNet from saved artifacts.

    Args:
        pcen_params_path: Path to pcen_params.json
        model_path: Path to best_checkpoint.pth

    Returns:
        pcen: LearnablePCEN (frozen)
        tc_resnet: TCResNetAcousticEncoder (frozen)
        pcen_params: dict
        backbone_hash: str
    """
    # Load PCEN with saved params
    pcen_params = LearnablePCEN.load_params(pcen_params_path)
    pcen = LearnablePCEN(sample_rate=SAMPLE_RATE, n_mels=N_MELS).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    pcen.load_state_dict(checkpoint['pcen_state_dict'])
    tc_resnet = TCResNetAcousticEncoder(
        num_mels=N_MELS, embedding_dim=EMBEDDING_DIM
    ).to(device)
    tc_resnet.load_state_dict(checkpoint['tc_resnet_state_dict'])

    # Freeze everything
    pcen.eval()
    tc_resnet.eval()
    for p in list(pcen.parameters()) + list(tc_resnet.parameters()):
        p.requires_grad_(False)

    # Compute backbone hash for profile
    backbone_hash = hashlib.sha256(
        json.dumps(checkpoint['tc_resnet_state_dict'].__class__.__name__).encode()
    ).hexdigest()[:16]

    # Try to get ONNX hash if available
    onnx_path = os.path.join(os.path.dirname(model_path),
                             "tc_resnet_backbone.onnx")
    if os.path.exists(onnx_path):
        sha256 = hashlib.sha256()
        with open(onnx_path, 'rb') as f:
            for block in iter(lambda: f.read(8192), b''):
                sha256.update(block)
        backbone_hash = sha256.hexdigest()

    return pcen, tc_resnet, pcen_params, backbone_hash


# ============================================================================
# Verification (compare new utterance against enrolled template)
# ============================================================================

def verify_against_template(audio_path, profile_path, pcen, tc_resnet,
                            device, threshold=0.70):
    """
    Compare a new utterance against an enrolled word template.

    Args:
        audio_path: Path to test utterance.
        profile_path: Path to enrolled template JSON.
        pcen: Frozen LearnablePCEN.
        tc_resnet: Frozen TCResNetAcousticEncoder.
        threshold: Cosine similarity threshold for match.

    Returns:
        similarity: float
        is_match: bool
    """
    # Load profile
    with open(profile_path, 'r') as f:
        profile = json.load(f)

    enrolled_emb = decode_embedding(profile['word_template'])
    enrolled_emb = torch.FloatTensor(enrolled_emb).to(device)

    # Extract test embedding
    with torch.no_grad():
        waveform = load_and_preprocess(audio_path).to(device)
        pcen_feat = pcen(waveform)
        test_emb = tc_resnet(pcen_feat).squeeze(0)

    # Cosine similarity (both L2-normed → dot product)
    similarity = torch.dot(enrolled_emb, test_emb).item()
    is_match = similarity >= threshold

    return similarity, is_match


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enroll a custom wake word (Master Word Template generation)"
    )
    parser.add_argument("--audio_files", nargs="+", required=True,
                       help="K=3 enrollment utterance file paths")
    parser.add_argument("--pcen_params", type=str, required=True,
                       help="Path to pcen_params.json")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to best_checkpoint.pth")
    parser.add_argument("--output", type=str, default="enrolled_template.json",
                       help="Output path for enrolled profile JSON")
    parser.add_argument("--verify", type=str, default=None,
                       help="Optional: path to test utterance for verification")
    parser.add_argument("--threshold", type=float, default=0.70,
                       help="Verification similarity threshold")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate input count
    if len(args.audio_files) < K_ENROLLMENT:
        print(f"ERROR: Need at least {K_ENROLLMENT} enrollment utterances, "
              f"got {len(args.audio_files)}")
        sys.exit(1)

    # Load models
    print("Loading frozen pipeline...")
    pcen, tc_resnet, pcen_params, backbone_hash = load_models(
        args.pcen_params, args.model_path, device
    )

    # Generate template
    print(f"\nEnrolling with {len(args.audio_files)} utterances...")
    try:
        w_c, coherence = generate_word_template(
            args.audio_files, pcen, tc_resnet, device
        )
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # Save profile
    profile = save_enrolled_profile(
        w_c, coherence, pcen_params, backbone_hash,
        args.output, len(args.audio_files)
    )

    # Optional verification
    if args.verify:
        print(f"\nVerifying against: {args.verify}")
        sim, match = verify_against_template(
            args.verify, args.output, pcen, tc_resnet,
            device, threshold=args.threshold
        )
        print(f"  Cosine similarity: {sim:.4f}")
        if match:
            print(f"  ✅ MATCH (threshold: {args.threshold})")
        else:
            print(f"  ❌ NO MATCH (threshold: {args.threshold})")
