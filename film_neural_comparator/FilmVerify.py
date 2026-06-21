import os
import argparse
import torch
import torch.nn.functional as F
import librosa
import numpy as np

from FilmNcModel import SpeakerConditionedKWS
SAMPLE_RATE = 16000
TARGET_LENGTH = 24000

def load_and_preprocess(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
        
    wav, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(wav) < TARGET_LENGTH:
        wav = np.pad(wav, (0, TARGET_LENGTH - len(wav)), mode='constant')
    else:
        wav = wav[:TARGET_LENGTH]

    peak = np.abs(wav).max()
    if peak > 1e-9:
        wav = wav / peak * 0.9

    return torch.FloatTensor(wav).unsqueeze(0)

def load_embedding(path: str, expected_dim: int, device: torch.device) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
        
    obj = torch.load(path, map_location=device, weights_only=False)
    
    if isinstance(obj, dict):
        tensor_vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        emb = tensor_vals[0]
    else:
        emb = obj
        
    emb = emb.float().view(1, -1)
    
    if emb.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim}-D embedding, got {emb.shape[1]}-D.")
        
    return F.normalize(emb, p=2, dim=1)

def verify_audio(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeakerConditionedKWS(sample_rate=16000, n_mels=80, embedding_dim=128).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint['tc_resnet_state_dict'])
    model.film_gen.load_state_dict(checkpoint['film_gen_state_dict'])
    model.pcen.load_state_dict(checkpoint['pcen_state_dict'])
    model.eval()

    e_s = load_embedding(args.speaker_emb, 192, device)
    enroll_embs = []
    with torch.no_grad():
        for path in args.enroll_audio:
            wav = load_and_preprocess(path).to(device)
            emb = model(wav, e_s)
            enroll_embs.append(emb)
            
    w_c = F.normalize(torch.cat(enroll_embs, dim=0).mean(dim=0, keepdim=True), p=2, dim=1)
    test_wav = load_and_preprocess(args.test_audio).to(device)
    with torch.no_grad():
        test_emb = model(test_wav, e_s) 
    
    similarity = torch.sum(test_emb * w_c, dim=1).item()    
    print("VERIFICATION RESULTS")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Threshold: {args.threshold:.4f}")
    
    if similarity >= args.threshold:
        print("Matched! Access Granted")
    else:
        print("Not Matched! Access Denied")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Stage 3 FiLM-TCResNet with Dynamic Enrollment.")
    parser.add_argument("--checkpoint", type=str, default="./stage3_output/stage3_best_checkpoint.pth", help="Path to stage3_best_checkpoint.pth")
    parser.add_argument("--speaker_emb", type=str, required=True, help="Path to the 192-D speaker embedding (.pt)")
    parser.add_argument("--enroll_audio", nargs='+', required=True, help="Paths to 3 enrollment audio files (target speaker saying target word)")
    parser.add_argument("--test_audio", type=str, required=True, help="Path to the test .wav file")
    parser.add_argument("--threshold", type=float, default=0.70, help="Cosine similarity threshold")
    args = parser.parse_args()
    verify_audio(args)