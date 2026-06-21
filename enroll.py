import argparse
import json
import numpy as np
from model import SCTKWSPipeline

def run_enrollment(args):
    print(f"\nInitializing ONNX Pipeline...")
    pipeline = SCTKWSPipeline(args.model_dir, mode="enroll")
    print(f"\nExtracting Biometric Signature via custom ECAPA-TDNN...")
    embeddings = []
    pcen_features = []
    for path in args.audio_files:
        speaker_wav = pipeline.preprocess_speaker_audio(path)
        keyword_wav = pipeline.preprocess_keyword_audio(path)
        speaker_feat = pipeline.extract_speaker_pcen(speaker_wav)
        emb = pipeline.get_speaker_embedding(speaker_feat)
        embeddings.append(emb)
        keyword_feat = pipeline.extract_keyword_pcen(keyword_wav)
        pcen_features.append(keyword_feat)
        
    e_s = np.mean(np.concatenate(embeddings, axis=0), axis=0, keepdims=True)
    e_s = e_s / np.linalg.norm(e_s, ord=2, axis=1, keepdims=True)
    
    gamma, beta = pipeline.generate_film_weights(e_s)    
    print(f"Generating Phonetic Template (w_c)...")
    word_embs = []
    for pcen in pcen_features:
        word_embs.append(pipeline.encode_word(pcen, gamma, beta))
        
    w_c = np.mean(np.concatenate(word_embs, axis=0), axis=0, keepdims=True)
    w_c = w_c / np.linalg.norm(w_c, ord=2, axis=1, keepdims=True)
    
    profile = {
        "gamma": gamma.tolist(),
        "beta": beta.tolist(),
        "w_c": w_c.tolist()
    }
    
    with open(args.output, 'w') as f:
        json.dump(profile, f, indent=4)
        
    print(f"\nEnrollment Complete! Device profile secured at: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a user and keyword into SC-TKWS.")
    parser.add_argument("audio_files", nargs=3, help="Paths to exactly 3 enrollment .wav files")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--output", type=str, default="enrolled_profile.json")
    args = parser.parse_args()
    run_enrollment(args)