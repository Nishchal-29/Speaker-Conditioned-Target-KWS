import os
import glob
import torch
import torchaudio
import torchaudio.functional as F_audio
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

SOURCE_DIR = "../data/tts_corpus"
DEST_DIR = "./tts_corpus_processed"
TARGET_SR = 16000
TARGET_LENGTH = 24000  

def process_audio_file(file_path):
    rel_path = os.path.relpath(file_path, SOURCE_DIR)
    dest_path = os.path.join(DEST_DIR, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return

    try:
        wav, sr = torchaudio.load(file_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            wav = resampler(wav)

        try:
            y_trimmed = F_audio.vad(wav, sample_rate=TARGET_SR)
            if y_trimmed.numel() > 1600: 
                wav = y_trimmed
        except Exception:
            pass

        current_len = wav.shape[1]
        if current_len > TARGET_LENGTH:
            start = (current_len - TARGET_LENGTH) // 2
            wav = wav[:, start:start + TARGET_LENGTH]
        elif current_len < TARGET_LENGTH:
            pad_left = (TARGET_LENGTH - current_len) // 2
            pad_right = TARGET_LENGTH - current_len - pad_left
            wav = F.pad(wav, (pad_left, pad_right))

        peak = wav.abs().max()
        if peak > 1e-8:
            wav = wav / peak * 0.9

        torchaudio.save(dest_path, wav, TARGET_SR)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    print(f"Scanning {SOURCE_DIR} for audio files...")
    all_files = glob.glob(os.path.join(SOURCE_DIR, "**", "*.wav"), recursive=True)
    if not all_files:
        print("No .wav files found! Please check your SOURCE_DIR.")
        return

    print(f"Found {len(all_files)} files. Starting multi-core processing...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_audio_file, all_files), total=len(all_files)))

    print(f"\nAll files processed and saved to {DEST_DIR}")

if __name__ == "__main__":
    main()