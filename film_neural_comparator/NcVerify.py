import os
import argparse
import numpy as np
import torch
import librosa
import onnxruntime as ort

from FilmNcModel import TargetKWS
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

def load_speaker_key(path: str) -> np.ndarray:
    emb = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(emb, dict):
        tensor_vals = [v for v in emb.values() if isinstance(v, torch.Tensor)]
        emb = tensor_vals[0]
        
    emb = emb.float().view(1, -1)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.numpy()

def sigmoid(x: np.ndarray) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def run_edge_simulation(args):    
    providers = ['CPUExecutionProvider']
    film_session = ort.InferenceSession(os.path.join(args.onnx_dir, "film_generator.onnx"), providers=providers)
    encoder_session = ort.InferenceSession(os.path.join(args.onnx_dir, "conditioned_encoder.onnx"), providers=providers)
    comparator_session = ort.InferenceSession(os.path.join(args.onnx_dir, "neural_comparator.onnx"), providers=providers)

    frontend = TargetKWS(sample_rate=16000, n_mels=80, embedding_dim=128)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    frontend.pcen.load_state_dict(checkpoint['pcen_state_dict'])
    frontend.eval()

    e_s_np = load_speaker_key(args.speaker_emb)
    gamma_np, beta_np = film_session.run(["gamma", "beta"], {"speaker_embedding": e_s_np})
    
    enroll_embs = []
    with torch.no_grad():
        for i, path in enumerate(args.enroll_audio):
            wav = load_and_preprocess(path)
            pcen_features = frontend.pcen(wav).numpy() 
            emb_np = encoder_session.run(
                ["word_embedding"], 
                {"pcen_features": pcen_features, "gamma": gamma_np, "beta": beta_np}
            )[0]
            enroll_embs.append(emb_np)

    w_c_np = np.mean(np.concatenate(enroll_embs, axis=0), axis=0, keepdims=True)
    w_c_np = w_c_np / np.linalg.norm(w_c_np, ord=2, axis=1, keepdims=True)
    
    with torch.no_grad():
        test_wav = load_and_preprocess(args.test_audio)
        test_pcen = frontend.pcen(test_wav).numpy()
        
    test_gamma, test_beta = film_session.run(["gamma", "beta"], {"speaker_embedding": e_s_np})    
    q_emb_np = encoder_session.run(["word_embedding"], {"pcen_features": test_pcen, "gamma": test_gamma, "beta": test_beta})[0]
    raw_logit = comparator_session.run(["p_accept"], {"target_template": w_c_np, "query_embedding": q_emb_np})[0]
    p_accept = sigmoid(raw_logit.item())

    print("NEURAL COMPARATOR VERIFICATION RESULTS")
    print(f"Acceptance Probability: {p_accept:.2%}")
    print(f"Strict Threshold: {args.threshold:.2%}")
    if p_accept >= args.threshold:
        print("Matched! Access Granted")
    else:
        print("Not Matched! Access Denied")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Edge ONNX Inference for Target KWS.")
    parser.add_argument("--onnx_dir", type=str, default="./stage4_output", help="Directory containing the .onnx files")
    parser.add_argument("--checkpoint", type=str, default="./stage4_output/stage4_best_checkpoint.pth", help="Path to .pth file (used ONLY to load PCEN DSP params)")
    parser.add_argument("--speaker_emb", type=str, required=True, help="Path to the 192-D target speaker embedding (.pt)")
    parser.add_argument("--enroll_audio", nargs='+', required=True, help="Paths to 3 enrollment audio files")
    parser.add_argument("--test_audio", type=str, required=True, help="Path to the test .wav file")
    parser.add_argument("--threshold", type=float, default=0.50, help="MLP Acceptance probability threshold (Default: 0.50)")
    args = parser.parse_args()
    run_edge_simulation(args)