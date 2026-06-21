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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pcen import LearnablePCEN
from model import TCResNetAcousticEncoder

SAMPLE_RATE = 16000
TARGET_LENGTH = 24000
EMBEDDING_DIM = 128
N_MELS = 80
K_ENROLLMENT = 3
COHERENCE_MIN = 0.75

def load_and_preprocess(path, target_sr=SAMPLE_RATE, target_length=TARGET_LENGTH):
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    if len(wav) < target_length:
        wav = np.pad(wav, (0, target_length - len(wav)), mode='constant')
    else:
        wav = wav[:target_length]

    peak = np.abs(wav).max()
    if peak > 1e-9:
        wav = wav / peak * 0.9

    return torch.FloatTensor(wav).unsqueeze(0)  

def generate_word_template(audio_paths, pcen, tc_resnet, device):
    tc_resnet.eval()
    pcen.eval()
    embeddings = []
    with torch.no_grad():
        for path in audio_paths:
            waveform = load_and_preprocess(path).to(device) 
            pcen_feat = pcen(waveform)                      
            v_k = tc_resnet(pcen_feat)                        
            embeddings.append(v_k)

    K = len(embeddings)
    stacked = torch.cat(embeddings, dim=0)          
    w_raw = stacked.mean(dim=0, keepdim=True)       
    w_c = F.normalize(w_raw, p=2, dim=1)             
    pairwise = stacked @ stacked.T 
    mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
    pairwise_sims = pairwise[mask]
    coherence = pairwise_sims.mean().item() if len(pairwise_sims) > 0 else 0.0

    if coherence < COHERENCE_MIN:
        raise ValueError(
            f"Enrollment rejected: coherence={coherence:.3f} < {COHERENCE_MIN}. "
            f"Please re-record in a quieter environment with consistent pronunciation."
        )

    return w_c, coherence

def encode_embedding(embedding):
    raw_bytes = embedding.cpu().numpy().astype(np.float32).tobytes()
    return base64.b64encode(raw_bytes).decode('ascii')

def decode_embedding(b64_string):
    raw_bytes = base64.b64decode(b64_string)
    return np.frombuffer(raw_bytes, dtype=np.float32).copy()

def save_enrolled_profile(w_c, coherence, pcen_params, backbone_hash, output_path, n_utterances):
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

    print(f"Enrolled profile saved to: {output_path}")
    print(f"Coherence: {coherence:.4f}")
    print(f"Utterances: {n_utterances}")

    return profile

def load_models(pcen_params_path, model_path, device):
    pcen_params = LearnablePCEN.load_params(pcen_params_path)
    pcen = LearnablePCEN(sample_rate=SAMPLE_RATE, n_mels=N_MELS).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    pcen.load_state_dict(checkpoint['pcen_state_dict'])
    tc_resnet = TCResNetAcousticEncoder(num_mels=N_MELS, embedding_dim=EMBEDDING_DIM).to(device)
    tc_resnet.load_state_dict(checkpoint['tc_resnet_state_dict'])

    pcen.eval()
    tc_resnet.eval()
    for p in list(pcen.parameters()) + list(tc_resnet.parameters()):
        p.requires_grad_(False)

    backbone_hash = hashlib.sha256(json.dumps(checkpoint['tc_resnet_state_dict'].__class__.__name__).encode()).hexdigest()[:16]
    onnx_path = os.path.join(os.path.dirname(model_path), "tc_resnet_backbone.onnx")
    if os.path.exists(onnx_path):
        sha256 = hashlib.sha256()
        with open(onnx_path, 'rb') as f:
            for block in iter(lambda: f.read(8192), b''):
                sha256.update(block)
        backbone_hash = sha256.hexdigest()

    return pcen, tc_resnet, pcen_params, backbone_hash

def verify_against_template(audio_path, profile_path, pcen, tc_resnet, device, threshold=0.70):
    with open(profile_path, 'r') as f:
        profile = json.load(f)

    enrolled_emb = decode_embedding(profile['word_template'])
    enrolled_emb = torch.FloatTensor(enrolled_emb).to(device)
    with torch.no_grad():
        waveform = load_and_preprocess(audio_path).to(device)
        pcen_feat = pcen(waveform)
        test_emb = tc_resnet(pcen_feat).squeeze(0)

    similarity = torch.dot(enrolled_emb, test_emb).item()
    is_match = similarity >= threshold

    return similarity, is_match

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a custom wake word (Master Word Template generation)")
    parser.add_argument("--audio_files", nargs="+", required=True, help="K=3 enrollment utterance file paths")
    parser.add_argument("--verify", type=str, default=None, help="Optional: path to test utterance for verification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcen, tc_resnet, pcen_params, backbone_hash = load_models("tc_resnet_output/pcen_params.json", "tc_resnet_output/best_checkpoint.pth", device)
    try:
        w_c, coherence = generate_word_template(args.audio_files, pcen, tc_resnet, device)
    except ValueError as e:
        print(f"\n{e}")
        sys.exit(1)

    profile = save_enrolled_profile(w_c, coherence, pcen_params, backbone_hash, "enrolled_template.json", len(args.audio_files))
    if args.verify:
        sim, match = verify_against_template(args.verify, "enrolled_template.json", pcen, tc_resnet, device, threshold=0.7)
        print(f"Cosine similarity: {sim:.4f}")
        if match:
            print(f"MATCH)")
        else:
            print(f"NO MATCH")