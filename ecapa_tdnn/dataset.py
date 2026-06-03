import os
import glob
import math
import torch
import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, Sampler

class DomainAdaptationDataset(Dataset):
    def __init__(self, data_dir, max_audio_length=48000, musan_dir="../data/musan", file_ext="wav"):
        self.files = sorted(glob.glob(f"{data_dir}/**/*.{file_ext}", recursive=True))
        if len(self.files) == 0:
            raise ValueError(f"No .{file_ext} files found in {data_dir}")

        self.speakers = sorted(list(set([
            path.split(os.sep)[-3] for path in self.files
        ])))
        self.spk_to_id = {spk: i for i, spk in enumerate(self.speakers)}
        self.n_speakers = len(self.speakers)

        self.speaker_files = {spk: [] for spk in self.speakers}
        for idx, path in enumerate(self.files):
            spk = path.split(os.sep)[-3]
            self.speaker_files[spk].append(idx)

        self.noise_files = glob.glob(f"{musan_dir}/noise/**/*.wav", recursive=True) + \
                           glob.glob(f"{musan_dir}/speech/**/*.wav", recursive=True)
        
        if len(self.noise_files) == 0:
            print(f"WARNING: No MUSAN files found in {musan_dir}. Noise injection will fail.")
        else:
            print(f"Loaded {len(self.noise_files)} real-world noise files from MUSAN.")

        self.max_audio_length = max_audio_length

        print(f"[DomainAdaptationDataset] Initialized with {len(self.files)} files, "
              f"{self.n_speakers} speakers.")
        print(f"  Audio length: {max_audio_length} samples ({max_audio_length / 16000:.1f}s @ 16kHz)")

    def inject_noise(self, y, sr=16000):
        """
        Injects real-world noise from the MUSAN dataset.
        Noise profile: 50% clean, 50% noisy (SNR between -5dB and +10dB).
        """
        if np.random.rand() < 0.5 or len(self.noise_files) == 0:
            return y

        random_noise_path = np.random.choice(self.noise_files)
        info = sf.info(random_noise_path)
        total_frames = info.frames
        noise_sr = info.samplerate        
        target_frames = int(len(y) * (noise_sr / sr))

        if total_frames > target_frames:
            start_frame = np.random.randint(0, total_frames - target_frames)
            noise, _ = sf.read(random_noise_path, frames=target_frames, start=start_frame, dtype='float32')
        else:
            noise, _ = sf.read(random_noise_path, dtype='float32')
            repeats = int(np.ceil(target_frames / len(noise)))
            noise = np.tile(noise, repeats)[:target_frames]

        if len(noise.shape) > 1:
            noise = np.mean(noise, axis=1)

        if noise_sr != sr:
            noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)
            if len(noise) > len(y):
                noise = noise[:len(y)]
            elif len(noise) < len(y):
                noise = np.pad(noise, (0, len(y) - len(noise)))

        target_snr = np.random.uniform(-5, 10)        
        signal_power = np.mean(y ** 2) + 1e-7
        noise_power = np.mean(noise ** 2) + 1e-7
        k = np.sqrt((signal_power / (10 ** (target_snr / 10))) / noise_power)
        y_noisy = y + (noise * k)

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

        y, _ = librosa.load(file_path, sr=16000)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0:
            y = y_trimmed

        if mode == 'train':
            y_processed = self.inject_noise(y)
        else:
            y_processed = y  

        if len(y_processed) < self.max_audio_length:
            repeats = int(np.ceil(self.max_audio_length / len(y_processed)))
            y_processed = np.tile(y_processed, repeats)[:self.max_audio_length]
        else:
            y_processed = y_processed[:self.max_audio_length]

        return torch.FloatTensor(y_processed), torch.tensor(label, dtype=torch.long)

    def get_val_split(self, ratio=0.1, seed=42):
        """
        Splits the dataset into train and validation by SPEAKER identity.
        Passes the exact 'mode' down to the subsets to govern noise injection.
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
    A subset view of DomainAdaptationDataset.
    Maintains its own speaker_files index and passes the 'mode' back to the parent.
    """
    def __init__(self, parent_dataset, indices, speakers, mode='train'):
        self.parent = parent_dataset
        self.indices = indices
        self.speakers = sorted(list(speakers))
        self.n_speakers = len(self.speakers)
        self.spk_to_id = parent_dataset.spk_to_id
        self.mode = mode

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
    def __init__(self, dataset, speakers_per_batch=16, utterances_per_speaker=4):
        self.dataset = dataset
        self.speakers = list(dataset.speakers)
        self.speaker_files = dataset.speaker_files
        self.P = speakers_per_batch
        self.K = utterances_per_speaker
        self.batch_size = self.P * self.K

        total_samples = len(dataset)
        self.n_batches = max(1, total_samples // self.batch_size)

        print(
            f"[BalancedBatchSampler] "
            f"{self.P} speakers/batch, "
            f"{self.K} utts/speaker, "
            f"batch size: {self.batch_size}, "
            f"batches/epoch: {self.n_batches}"
        )

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []

            chosen_speakers = np.random.choice(
                self.speakers,
                size=min(self.P, len(self.speakers)),
                replace=False,
            )

            for spk in chosen_speakers:
                indices = self.speaker_files[spk]
                if len(indices) >= self.K:
                    selected = np.random.choice(indices, size=self.K, replace=False)
                else:
                    selected = np.random.choice(indices, size=self.K, replace=True)

                batch.extend(selected.tolist())
            np.random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.n_batches