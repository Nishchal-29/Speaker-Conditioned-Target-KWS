import torchaudio
import os

# Create the data directory
os.makedirs("data", exist_ok=True)

print("Downloading LibriSpeech 'dev-clean' subset (approx 330 MB)...")
# This automatically downloads and extracts the dataset
dataset = torchaudio.datasets.LIBRISPEECH(
    root="data/", 
    url="dev-clean", 
    download=True
)

print("Download complete!")
print("Dataset located at: data/LibriSpeech/dev-clean/")