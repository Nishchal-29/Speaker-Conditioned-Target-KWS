import os
import sys
import glob
import torch
import numpy as np
import librosa
import onnxruntime as ort
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pcen import PCENProcessor 

TTS_DIR = "../data/tts_corpus"
OUTPUT_DIR = "./speaker_embeddings"
ONNX_PATH = "../ecapa_tdnn/finetuned_models/ecapa_backbone.onnx"
PCEN_PATH = "../ecapa_tdnn/finetuned_models/pcen_params.json"
SAMPLE_RATE = 16000
MAX_FILES_PER_SPEAKER = 10

def load_and_trim_audio(path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    trimmed, _ = librosa.effects.trim(audio, top_db=40.0) #
    return trimmed if len(trimmed) > 0 else audio

def extract_ecapa_embedding(audio, pcen_processor, session, input_name, output_name):
    pcen_features = pcen_processor.process(audio)
    pcen_input = pcen_features[np.newaxis, :, :].astype(np.float32)

    pcen_transposed = np.transpose(pcen_input, (0, 2, 1))
    mean = np.mean(pcen_transposed, axis=1, keepdims=True)
    std = np.std(pcen_transposed, axis=1, keepdims=True)
    pcen_transposed = (pcen_transposed - mean) / (std + 1e-5)
    pcen_input = np.transpose(pcen_transposed, (0, 2, 1))

    raw_embedding = session.run([output_name], {input_name: pcen_input})[0]
    return raw_embedding.squeeze()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)    
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    with open(PCEN_PATH, 'r') as f:
        pcen_params = json.load(f)
    pcen_processor = PCENProcessor(pcen_params, sample_rate=SAMPLE_RATE)    
    print("Indexing TTS directory for speakers...")
    all_files = glob.glob(f"{TTS_DIR}/*/*/*.wav")
    speaker_dict = {}
    
    for path in all_files:
        filename = os.path.basename(path)
        spk_id = filename.split('_')[0]
        speaker_dict.setdefault(spk_id, []).append(path)
        
    print(f"Found {len(speaker_dict)} unique speakers. Generating embeddings...")
    
    for spk_id, files in tqdm(speaker_dict.items()):
        selected_files = files[:MAX_FILES_PER_SPEAKER]
        embeddings = []
        
        for fpath in selected_files:
            audio = load_and_trim_audio(fpath)
            if len(audio) == 0:
                continue
            emb = extract_ecapa_embedding(audio, pcen_processor, session, input_name, output_name)
            embeddings.append(emb)
            
        if not embeddings:
            continue
            
        embeddings_matrix = np.array(embeddings)
        mean_emb = np.mean(embeddings_matrix, axis=0)
        norm = np.linalg.norm(mean_emb)
        final_emb = mean_emb / norm if norm > 0 else mean_emb        
        tensor_emb = torch.from_numpy(final_emb.astype(np.float32))
        torch.save(tensor_emb, os.path.join(OUTPUT_DIR, f"{spk_id}.pt"))
        
    print(f"Complete. Saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()