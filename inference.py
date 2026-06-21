import argparse
import json
import numpy as np
from model import SCTKWSPipeline

def run_inference(args):
    with open(args.profile, 'r') as f:
        profile = json.load(f)
        
    gamma = np.array(profile["gamma"], dtype=np.float32)
    beta = np.array(profile["beta"], dtype=np.float32)
    w_c = np.array(profile["w_c"], dtype=np.float32)    
    pipeline = SCTKWSPipeline(args.model_dir, mode="inference")    
    wav = pipeline.preprocess_keyword_audio(args.audio)
    pcen_feat = pipeline.extract_keyword_pcen(wav)    
    q_emb = pipeline.encode_word(pcen_feat, gamma, beta)
    p_accept = pipeline.compare(w_c, q_emb)
    print("SC-TKWS VERIFICATION RESULTS")
    print(f"Match Probability: {p_accept:.2%}")
    print(f"Threshold: {args.threshold:.2%}")
    
    if p_accept >= args.threshold:
        print("Matched! Access Granted")
    else:
        print("Not Matched! Access Denied")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SC-TKWS against a saved profile.")
    parser.add_argument("--profile", type=str, default="enrolled_profile.json")
    parser.add_argument("--audio", type=str, required=True, help="Path to the query .wav file")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    run_inference(args)