import os
import glob
import math
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, Sampler

class DomainAdaptationDataset(Dataset):
    def __init__(self, data_dir, max_audio_length=48000, file_ext="flac"):
        """
        Args:
            data_dir: Path to audio directory (LibriSpeech-style structure).
            max_audio_length: Target length in samples (48000 = 3s @ 16kHz).
            file_ext: Audio file extension to search for.
        """
        self.files = sorted(glob.glob(f"{data_dir}/**/*.{file_ext}", recursive=True))
        if len(self.files) == 0:
            raise ValueError(f"No .{file_ext} files found in {data_dir}")

        # Build speaker → ID mapping
        self.speakers = sorted(list(set([
            path.split(os.sep)[-3] for path in self.files
        ])))
        self.spk_to_id = {spk: i for i, spk in enumerate(self.speakers)}
        self.n_speakers = len(self.speakers)

        # Build per-speaker file index for balanced sampling
        self.speaker_files = {spk: [] for spk in self.speakers}
        for idx, path in enumerate(self.files):
            spk = path.split(os.sep)[-3]
            self.speaker_files[spk].append(idx)

        self.max_audio_length = max_audio_length

        print(f"[DomainAdaptationDataset] Initialized with {len(self.files)} files, "
              f"{self.n_speakers} speakers.")
        print(f"  Audio length: {max_audio_length} samples ({max_audio_length / 16000:.1f}s @ 16kHz)")
        print(f"  Using dynamic SNR augmentation, silence trimming, and audio tiling.")

    def inject_noise(self, y, sr=16000):
        """
        Injects realistic, non-stationary crowd babble for data augmentation.
        Noise profile: 50% clean, 50% noisy (SNR between -5dB and +10dB).
        """
        # 50% chance to return clean audio
        if np.random.rand() < 0.5:
            return y

        # Generate Non-Stationary Synthetic Babble (20 voices)
        duration_sec = len(y) / sr
        time = np.linspace(0, duration_sec, len(y))
        babble = np.zeros_like(time)
        
        for _ in range(20):
            base_freq = np.random.uniform(300, 3000)
            amp_mod = np.sin(2 * np.pi * np.random.uniform(0.5, 2.5) * time)
            amp_mod = np.maximum(0, amp_mod)
            phase = np.random.uniform(0, 2 * np.pi)
            babble += amp_mod * np.sin(2 * np.pi * base_freq * time + phase)
            
        noise = babble / (np.max(np.abs(babble)) + 1e-7)
        
        # Pick a random SNR
        target_snr = np.random.uniform(-5, 10)
        
        signal_power = np.mean(y ** 2) + 1e-7
        noise_power = np.mean(noise ** 2) + 1e-7
        
        k = np.sqrt((signal_power / (10 ** (target_snr / 10))) / noise_power)
        y_noisy = y + (noise * k)

        # Prevent clipping
        max_val = np.max(np.abs(y_noisy))
        if max_val > 1.0:
            y_noisy = y_noisy / max_val
            
        return y_noisy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, mode='train'):
        file_path = self.files[idx]
        spk_name = file_path.split(os.sep)[-3]
        label = self.spk_to_id[spk_name]

        # 1. Load audio at 16kHz mono
        y, _ = librosa.load(file_path, sr=16000)

        # 2. Strip dead silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0:
            y = y_trimmed

        # 3. Inject random noise
        if mode == 'train':
            y_processed = self.inject_noise(y)
        else:
            y_processed = y

        # 4. Tile (loop) or truncate for uniform batching
        if len(y_processed) < self.max_audio_length:
            repeats = int(np.ceil(self.max_audio_length / len(y_processed)))
            y_processed = np.tile(y_processed, repeats)[:self.max_audio_length]
        else:
            y_processed = y_processed[:self.max_audio_length]

        return torch.FloatTensor(y_processed), torch.tensor(label, dtype=torch.long)

    def get_val_split(self, ratio=0.1, seed=42):
        """
        Splits the dataset into train and validation by SPEAKER identity.

        The held-out speakers are entirely unseen during training, which is
        required for a meaningful EER evaluation.

        Args:
            ratio: Fraction of speakers to hold out (default 10%).
            seed: Random seed for reproducibility.

        Returns:
            train_dataset: DomainAdaptationSubset for training speakers.
            val_dataset: DomainAdaptationSubset for held-out speakers.
            val_speakers: list of held-out speaker IDs.
        """
        rng = np.random.RandomState(seed)
        n_val = max(1, int(self.n_speakers * ratio))

        shuffled_speakers = list(self.speakers)
        rng.shuffle(shuffled_speakers)

        val_speakers = set(shuffled_speakers[:n_val])
        train_speakers = set(shuffled_speakers[n_val:])

        train_indices = []
        val_indices = []

        for idx, path in enumerate(self.files):
            spk = path.split(os.sep)[-3]
            if spk in val_speakers:
                val_indices.append(idx)
            else:
                train_indices.append(idx)

        print(f"[Split] Train: {len(train_speakers)} speakers ({len(train_indices)} files), "
              f"Val: {len(val_speakers)} speakers ({len(val_indices)} files)")

        train_subset = DomainAdaptationSubset(self, train_indices, train_speakers, mode='train')
        val_subset = DomainAdaptationSubset(self, val_indices, val_speakers, mode='val')

        return train_subset, val_subset, list(val_speakers)


class DomainAdaptationSubset:
    """
    A subset view of DomainAdaptationDataset that contains only specific indices.
    Maintains its own speaker_files index for balanced sampling.
    """

    def __init__(self, parent_dataset, indices, speakers, mode='train'):
        self.parent = parent_dataset
        self.indices = indices
        self.speakers = sorted(list(speakers))
        self.n_speakers = len(self.speakers)
        self.spk_to_id = parent_dataset.spk_to_id
        self.mode = mode

        # Rebuild per-speaker file index for this subset
        self.speaker_files = {spk: [] for spk in self.speakers}
        for local_idx, global_idx in enumerate(indices):
            path = parent_dataset.files[global_idx]
            spk = path.split(os.sep)[-3]
            if spk in self.speaker_files:
                self.speaker_files[spk].append(local_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        return self.parent.__getitem__(global_idx, mode=self.mode)

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=64):
        """
        Args:
            dataset: Must have .speakers (list), .speaker_files (dict: spk → [indices]),
                     and .n_speakers (int).
            batch_size: Target batch size.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.speakers = dataset.speakers
        self.n_speakers = dataset.n_speakers
        self.speaker_files = dataset.speaker_files

        self.samples_per_speaker = batch_size // self.n_speakers
        if self.samples_per_speaker < 1:
            self.samples_per_speaker = 1

        self.effective_batch_size = self.samples_per_speaker * self.n_speakers

        # Number of batches per epoch: enough to see most of the data
        total_samples = len(dataset)
        self.n_batches = max(1, total_samples // self.effective_batch_size)

        print(f"[BalancedBatchSampler] {self.n_speakers} speakers, "
              f"{self.samples_per_speaker} samples/speaker/batch, "
              f"effective batch size: {self.effective_batch_size}, "
              f"batches/epoch: {self.n_batches}")

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for spk in self.speakers:
                spk_indices = self.speaker_files[spk]
                if len(spk_indices) == 0:
                    continue

                # Sample with replacement if needed
                if len(spk_indices) >= self.samples_per_speaker:
                    selected = np.random.choice(
                        spk_indices, size=self.samples_per_speaker, replace=False
                    ).tolist()
                else:
                    selected = np.random.choice(
                        spk_indices, size=self.samples_per_speaker, replace=True
                    ).tolist()

                batch.extend(selected)

            # Shuffle within the batch to avoid speaker ordering bias
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches