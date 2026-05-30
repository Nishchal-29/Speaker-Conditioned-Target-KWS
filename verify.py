# import os
# import sys
# import types
# import soundfile as sf
# import numpy as np
# import librosa
# import torch
# import torch.nn.functional as F
# import torchaudio
# from speechbrain.inference.speaker import EncoderClassifier

# # --- BUG FIX FOR WINDOWS / PYTORCH 2.x ---
# os.environ["TORCH_DYNAMO_DISABLE"] = "1"
# sys.modules['k2'] = types.ModuleType('k2')
# sys.modules['flair'] = types.ModuleType('flair')
# sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = types.ModuleType('fake_flair_emb')
# sys.modules['speechbrain.integrations.nlp'] = types.ModuleType('fake_nlp')
# sys.modules['speechbrain.integrations.huggingface.wordemb'] = types.ModuleType('fake_wordemb')
# sys.modules['speechbrain.integrations.huggingface'] = types.ModuleType('fake_hf')
# # -----------------------------------------

# torch._dynamo.config.disable = True 

# class SpeakerVerifier:
#     def __init__(self, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu", 
#                  target_snr=-5, n_mels=80, max_frames=400):
#         self.device = torch.device(device)
#         print(f"Loading verifier on {self.device}...")
        
#         # --- FIX: Define the hyperparameters needed for PCEN and Noise ---
#         self.target_snr = target_snr
#         self.n_mels = n_mels
#         self.max_frames = max_frames
        
#         # 1. Load the base ECAPA-TDNN architecture
#         self.classifier = EncoderClassifier.from_hparams(
#             source="speechbrain/spkrec-ecapa-voxceleb", 
#             savedir="./models/speechbrain_cache",
#             run_opts={"device": str(self.device)}
#         )
#         self.encoder = self.classifier.mods.embedding_model.to(self.device)
        
#         # 2. Load your fine-tuned weights into the encoder
#         if os.path.exists(checkpoint_path):
#             checkpoint = torch.load(checkpoint_path, map_location=self.device)
#             self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
#             print(f"Successfully loaded fine-tuned weights from Epoch {checkpoint['epoch']} (Train Acc: {checkpoint['accuracy']:.2f}%)")
#         else:
#             raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")
            
#         self.encoder.eval() # Set to evaluation mode
    
#     def inject_noise(self, y):
#         """Calculates audio power and injects noise to hit the exact target SNR."""
#         signal_power = np.mean(y ** 2)
#         if signal_power == 0:
#             return y  
            
#         noise_power = signal_power / (10 ** (self.target_snr / 10))
#         noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        
#         return y + noise

#     def extract_embedding(self, audio_path):
#         """Processes the audio and extracts the 192-D voice print."""
        
#         # 1. Read audio using SoundFile (Bypassing Windows TorchCodec bug)
#         audio_array, sr = sf.read(audio_path)
        
#         # 2. Ensure it's a 1D mono numpy array for Librosa
#         if len(audio_array.shape) > 1:
#             audio_array = np.mean(audio_array, axis=1)
            
#         # 3. Resample to 16kHz if necessary
#         if sr != 16000:
#             audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
#             sr = 16000
            
#         # 4. Inject Noise
#         # y_noisy = self.inject_noise(audio_array)
#         y_noisy = audio_array
        
#         # 5. Compute Mel and apply PCEN
#         mel = librosa.feature.melspectrogram(y=y_noisy, sr=sr, n_mels=self.n_mels, hop_length=256)
#         pcen = librosa.pcen(S=mel * (2**31), sr=sr, hop_length=256).T 
        
#         # 6. Pad or truncate for uniform sizing (Shape: [frames, n_mels])
#         if pcen.shape[0] < self.max_frames:
#             pad_width = self.max_frames - pcen.shape[0]
#             pcen = np.pad(pcen, ((0, pad_width), (0, 0)), mode='constant')
#         else:
#             pcen = pcen[:self.max_frames, :]
            
#         # --- FIX: Convert Numpy array to PyTorch Tensor, add batch dim, and send to device ---
#         # Shape goes from [frames, n_mels] -> [1, frames, n_mels]
#         # --- FIX: Convert Numpy array to PyTorch Tensor, add batch dim, and send to device ---
#         # Shape goes from [frames, n_mels] -> [1, frames, n_mels]
#         pcen_tensor = torch.FloatTensor(pcen).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             # --- CRITICAL FIX: Apply the exact same normalization used in training ---
#             mean = pcen_tensor.mean(dim=1, keepdim=True)
#             std = pcen_tensor.std(dim=1, keepdim=True)
#             pcen_tensor = (pcen_tensor - mean) / (std + 1e-5)
#             # -------------------------------------------------------------------------
            
#             embedding = self.encoder(pcen_tensor)
#             embedding = embedding.squeeze(1) 
            
#         return embedding

#     def verify(self, audio_path_1, audio_path_2, threshold=0.45):
#         """Compares two audio files and returns the similarity score and a boolean match."""
#         print(f"\nComparing:\n1. {audio_path_1}\n2. {audio_path_2}")
        
#         emb1 = self.extract_embedding(audio_path_1)
#         emb2 = self.extract_embedding(audio_path_2)
        
#         # Calculate Cosine Similarity
#         similarity = F.cosine_similarity(emb1, emb2).item()
        
#         is_match = similarity >= threshold
        
#         print(f"Cosine Similarity Score: {similarity:.4f}")
#         if is_match:
#             print(f"✅ MATCH! These voices likely belong to the SAME speaker.")
#         else:
#             print(f"❌ NO MATCH! These voices likely belong to DIFFERENT speakers.")
            
#         return similarity, is_match

# if __name__ == "__main__":
#     # Point this to your latest saved model
#     CHECKPOINT_FILE = "./finetuned_models/ecapa_pcen_epoch_10.pth" 
    
#     # You can tweak target_snr, n_mels, and max_frames here to match your training data!
#     # FIX: Change max_frames to 300 to match dataset.py
#     verifier = SpeakerVerifier(checkpoint_path=CHECKPOINT_FILE, target_snr=-5, n_mels=80, max_frames=300)
#     # Test 1: Should be the same speaker
#     speaker_A_file1 = "./data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"
#     speaker_A_file2 = "./data/LibriSpeech/dev-clean/84/121123/84-121123-0020.flac"
    
#     # Test 2: Different speaker
#     speaker_B_file1 = "./data/LibriSpeech/dev-clean/174/50561/174-50561-0002.flac"
    
#     if os.path.exists(speaker_A_file1) and os.path.exists(speaker_A_file2):
#         print("\n--- TEST: SAME SPEAKER ---")
#         verifier.verify(speaker_A_file1, speaker_A_file2)
        
#         if os.path.exists(speaker_B_file1):
#             print("\n--- TEST: DIFFERENT SPEAKERS ---")
#             verifier.verify(speaker_A_file1, speaker_B_file1)
#     else:
#         print("Please update the audio paths in the script to point to actual .wav or .flac files on your machine.")

import os
import sys
import types
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

# --- BUG FIX FOR WINDOWS / PYTORCH 2.x ---
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
sys.modules['k2'] = types.ModuleType('k2')
sys.modules['flair'] = types.ModuleType('flair')
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = types.ModuleType('fake_flair_emb')
sys.modules['speechbrain.integrations.nlp'] = types.ModuleType('fake_nlp')
sys.modules['speechbrain.integrations.huggingface.wordemb'] = types.ModuleType('fake_wordemb')
sys.modules['speechbrain.integrations.huggingface'] = types.ModuleType('fake_hf')
torch._dynamo.config.disable = True 
# -----------------------------------------

def load_and_prep_audio(file_path, target_sr=16000, max_audio_length=48000):
    """
    Loads any audio format, strips dead silence, and loops (tiles) short audio 
    to perfectly mirror the training data distribution.
    """
    # 1. Load audio natively via Librosa (handles decoding and downmixing automatically)
    wav, _ = librosa.load(file_path, sr=target_sr, mono=True)
    
    # 2. Strip dead silence (Matches dataset.py top_db=30)
    wav_trimmed, _ = librosa.effects.trim(wav, top_db=30)
    if len(wav_trimmed) > 0:
        wav = wav_trimmed
        
    # 3. Tile (loop) short audio to match the exact 3-second window
    if len(wav) < max_audio_length:
        repeats = int(np.ceil(max_audio_length / len(wav)))
        wav = np.tile(wav, repeats)[:max_audio_length]
    else:
        wav = wav[:max_audio_length]
        
    # Convert to PyTorch tensor expected by SpeechBrain
    return torch.FloatTensor(wav)

def inject_noise(signal, target_snr=-5):
    """Injects Gaussian noise into a 1D PyTorch audio tensor."""
    signal_power = torch.mean(signal ** 2)
    if signal_power == 0:
        return signal
        
    noise_power = signal_power / (10 ** (target_snr / 10))
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    
    return signal + noise

def verify_speakers():
    print("Loading Base SpeechBrain Model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="./finetuned_models/speechbrain_cache"
    )

    # 1. Inject your fine-tuned weights
    checkpoint_path = "./finetuned_models/ecapa_raw_epoch_15.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")
        
    print(f"Injecting fine-tuned weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    classifier.mods.embedding_model.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.mods.embedding_model.eval()

    # 2. Define your test files
    speaker_A_file1 = "./data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"
    speaker_A_file2 = "./data/LibriSpeech/dev-clean/84/121550/84-121550-0000.flac"
    speaker_B_file1 = "./data/LibriSpeech/dev-clean/174/50561/174-50561-0000.flac"

    print("\nLoading Audio and Injecting Noise...")
    
    # 3. Load with the corrected pipeline
    raw_A1 = load_and_prep_audio(speaker_A_file1)
    raw_A2 = load_and_prep_audio(speaker_A_file2)
    raw_B1 = load_and_prep_audio(speaker_B_file1)

    # Inject -5dB noise across all options to establish an even baseline evaluation environment
    noisy_A1 = raw_A1
    noisy_A2 = raw_A2
    noisy_B1 = raw_B1
    noisy_A1 = inject_noise(raw_A1, target_snr=-5)
    # noisy_A2 = inject_noise(raw_A2, target_snr=-5)
    noisy_B1 = inject_noise(raw_B1, target_snr=-5)

    print("Extracting Embeddings natively...")
    with torch.no_grad():
        # encode_batch expects shape [batch, time]
        emb_A1 = classifier.encode_batch(noisy_A1.unsqueeze(0)).squeeze()
        emb_A2 = classifier.encode_batch(noisy_A2.unsqueeze(0)).squeeze()
        emb_B1 = classifier.encode_batch(noisy_B1.unsqueeze(0)).squeeze()

    # 4. Calculate Cosine Similarities
    sim_same = F.cosine_similarity(emb_A1.unsqueeze(0), emb_A2.unsqueeze(0)).item()
    sim_diff = F.cosine_similarity(emb_A1.unsqueeze(0), emb_B1.unsqueeze(0)).item()

    print("\n--- TEST: SAME SPEAKER ---")
    print(f"Cosine Similarity Score: {sim_same:.4f}")
    if sim_same > 0.15:
        print("✅ MATCH! These voices likely belong to the SAME speaker.")
    else:
        print("❌ NO MATCH.")

    print("\n--- TEST: DIFFERENT SPEAKERS ---")
    print(f"Cosine Similarity Score: {sim_diff:.4f}")
    if sim_diff > 0.15:
        print("❌ FALSE POSITIVE MATCH! (Check data)")
    else:
        print("✅ CORRECT REJECTION! These are different speakers.")

if __name__ == "__main__":
    verify_speakers()