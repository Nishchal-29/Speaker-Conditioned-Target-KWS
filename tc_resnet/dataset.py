import os
import glob
import json
import random
import difflib
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from g2p_en import G2p

SAMPLE_RATE = 16000
TARGET_LENGTH = 24000 
HARD_NEG_FRAC = 0.50
N_MELS = 80

def compute_ped_matrix(words, cache_path=None):
    """Compute normalised Phoneme Edit Distance (PED) between all word pairs using difflib."""
    if cache_path and os.path.exists(cache_path):
        print(f"[PED] Loading cached phoneme data from {cache_path}")
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        return cached['phoneme_seqs'], cached['hard_neg_pool']

    g2p = G2p()
    print(f"[PED] Computing phoneme sequences for {len(words)} words...")

    phoneme_seqs = {}
    for w in words:
        raw = g2p(w)
        phonemes = [p for p in raw if p.strip() and p.isalpha()]
        phoneme_seqs[w] = phonemes

    print(f"[PED] Computing pairwise PED for {len(words)} words...")
    hard_neg_pool = {w: [] for w in words}

    for i, w1 in enumerate(words):
        p1 = phoneme_seqs[w1]
        candidates = []

        for j, w2 in enumerate(words):
            if i == j:
                continue

            p2 = phoneme_seqs[w2]
            if not p1 and not p2:
                continue

            sm = difflib.SequenceMatcher(None, p1, p2)
            ped = 1.0 - sm.ratio() # distance = 1 - similarity

            if ped < 0.35:
                candidates.append((w2, ped))

        candidates.sort(key=lambda x: x[1])
        hard_neg_pool[w1] = [c[0] for c in candidates[:20]]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({
                'phoneme_seqs': phoneme_seqs,
                'hard_neg_pool': hard_neg_pool,
            }, f)
        print(f"[PED] Cached to {cache_path}")

    return phoneme_seqs, hard_neg_pool


class TripletDataset(Dataset):
    def __init__(self, data_root, musan_dir="../data/musan", sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH, val_words_path=None):
        self.sr = sample_rate
        self.target_length = target_length

        val_words = set()
        if val_words_path and os.path.exists(val_words_path):
            with open(val_words_path, 'r') as f:
                val_words = set(line.strip().lower() for line in f if line.strip())
            print(f"[TripletDataset] Excluding {len(val_words)} validation words")

        self.word_files = {}
        if not os.path.isdir(data_root):
            raise ValueError(f"Data root not found: {data_root}")

        for word_dir in sorted(os.listdir(data_root)):
            word_path = os.path.join(data_root, word_dir)
            if not os.path.isdir(word_path):
                continue

            word = word_dir.lower().strip()

            # Exclude validation words
            if word in val_words:
                continue

            files = sorted(glob.glob(os.path.join(word_path, "*.wav")))
            if len(files) >= 2:  # Need at least 2 files per word for anchor + positive
                self.word_files[word] = files

        self.words = sorted(self.word_files.keys())
        if len(self.words) == 0:
            raise ValueError(f"No valid word directories found in {data_root}")

        self.all_entries = []  # list of (word, file_idx_within_word)
        for word in self.words:
            for i in range(len(self.word_files[word])):
                self.all_entries.append((word, i))

        cache_path = os.path.join(data_root, ".ped_cache.json")
        self.phoneme_seqs, self.hard_neg_pool = compute_ped_matrix(self.words, cache_path=cache_path)
        self.easy_neg_pool = {}
        for w1 in self.words:
            p1 = self.phoneme_seqs.get(w1, [])
            easy = []
            for w2 in self.words:
                if w1 == w2:
                    continue
                p2 = self.phoneme_seqs.get(w2, [])
                if not p1 and not p2:
                    continue
                
                sm = difflib.SequenceMatcher(None, p1, p2)
                ped = 1.0 - sm.ratio()
                
                if ped > 0.60:
                    easy.append(w2)
            self.easy_neg_pool[w1] = easy if easy else self.words

        self.musan_files = []
        if musan_dir and os.path.exists(musan_dir):
            for category in ['noise', 'speech', 'music']:
                self.musan_files.extend(glob.glob(os.path.join(musan_dir, category, "**/*.wav"), recursive=True))
            print(f"[TripletDataset] Loaded {len(self.musan_files)} real MUSAN tracks for background noise.")
        else:
            print(f"[TripletDataset] WARNING: MUSAN dir not found at {musan_dir}. Audio will be clean.")

        n_utts = sum(len(f) for f in self.word_files.values())
        avg_hard = np.mean([len(self.hard_neg_pool.get(w, [])) for w in self.words])
        print(f"\n[TripletDataset] Corpus statistics:")
        print(f"  Words: {len(self.words)}")
        print(f"  Utterances: {n_utts}")
        print(f"  Avg hard negatives/word: {avg_hard:.1f}")
        print(f"  Avg files/word: {n_utts / max(len(self.words), 1):.1f}")

    def __len__(self):
        return len(self.all_entries)

    def __getitem__(self, idx):
        """Returns (anchor, positive, negative) triplet"""
        anchor_word, anchor_file_idx = self.all_entries[idx]
        triplet_type_idx = idx % 10

        anchor_path = self.word_files[anchor_word][anchor_file_idx]
        anchor_wav = self._load_audio(anchor_path)

        pos_files = self.word_files[anchor_word]
        pos_candidates = [i for i in range(len(pos_files)) if i != anchor_file_idx]
        if not pos_candidates:
            pos_candidates = list(range(len(pos_files)))
        pos_idx = random.choice(pos_candidates)
        pos_wav = self._load_audio(pos_files[pos_idx])

        if triplet_type_idx < 3:
            neg_word = self._sample_easy_negative(anchor_word)
            snr_lo, snr_hi = 5.0, 20.0
        elif triplet_type_idx < 8:
            neg_word = self._sample_hard_negative(anchor_word)
            snr_lo, snr_hi = -5.0, 15.0
        else:
            neg_word = self._sample_hard_negative(anchor_word)
            snr_lo, snr_hi = -5.0, -5.0

        neg_files = self.word_files[neg_word]
        neg_wav = self._load_audio(random.choice(neg_files))
        pos_wav = self._inject_musan(pos_wav, snr_lo, snr_hi)
        neg_wav = self._inject_musan(neg_wav, snr_lo, snr_hi)

        return anchor_wav, pos_wav, neg_wav

    def _load_audio(self, path):
        try:
            wav, sr = librosa.load(path, sr=self.sr, mono=True)
        except Exception:
            return torch.zeros(self.target_length, dtype=torch.float32)

        if len(wav) < self.target_length:
            pad_len = self.target_length - len(wav)
            wav = np.pad(wav, (0, pad_len), mode='constant')
        else:
            wav = wav[:self.target_length]

        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak * 0.9

        return torch.FloatTensor(wav)

    def _inject_musan(self, waveform, snr_lo, snr_hi):
        if not self.musan_files:
            return waveform

        wav_np = waveform.numpy()
        snr_db = random.uniform(snr_lo, snr_hi)

        # 1. Load a random MUSAN file
        noise_path = random.choice(self.musan_files)
        try:
            # Fast load without resampling if possible, but librosa handles it safely
            noise, _ = librosa.load(noise_path, sr=self.sr, mono=True)
        except Exception:
            return waveform

        # 2. Slice exactly 1.5 seconds of noise (or wrap-pad if too short)
        if len(noise) > self.target_length:
            start = random.randint(0, len(noise) - self.target_length)
            noise = noise[start:start + self.target_length]
        else:
            # Wrap padding loops the noise naturally if it's shorter than 1.5s
            noise = np.pad(noise, (0, self.target_length - len(noise)), mode='wrap')

        # 3. Calculate RMS and mix based on target SNR
        speech_rms = np.sqrt(np.mean(wav_np ** 2))
        if speech_rms < 1e-10:
            return waveform

        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms < 1e-10:
            return waveform

        noise_rms_target = speech_rms / (10 ** (snr_db / 20))
        noisy = wav_np + noise * (noise_rms_target / noise_rms)

        # 4. Prevent clipping saturation
        peak = np.abs(noisy).max()
        if peak > 1e-6:
            noisy = np.clip(noisy / peak, -1.0, 1.0)

        return torch.FloatTensor(noisy)

    def _sample_hard_negative(self, anchor_word):
        """Sample a word from the hard negative pool (PED < 0.35)."""
        pool = self.hard_neg_pool.get(anchor_word, [])
        valid = [w for w in pool if w in self.word_files]
        if valid:
            return random.choice(valid)
        candidates = [w for w in self.words if w != anchor_word]
        return random.choice(candidates) if candidates else anchor_word

    def _sample_easy_negative(self, anchor_word):
        """Sample a random word with PED > 0.60."""
        pool = self.easy_neg_pool.get(anchor_word, [])
        valid = [w for w in pool if w in self.word_files]
        if valid:
            return random.choice(valid)
        candidates = [w for w in self.words if w != anchor_word]
        return random.choice(candidates) if candidates else anchor_word


class ValidationDataset:
    def __init__(self, data_root, val_words_path=None, sample_rate=SAMPLE_RATE,
                 target_length=TARGET_LENGTH, k_enroll=3, n_test=5):
        self.sr = sample_rate
        self.target_length = target_length
        self.k_enroll = k_enroll
        self.n_test = n_test
        val_words = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

        self.word_files = {}
        for word in val_words:
            word_dir = os.path.join(data_root, word)
            files = sorted(glob.glob(os.path.join(word_dir, "*.wav")))
            if len(files) >= k_enroll + n_test:
                self.word_files[word] = files

        self.words = sorted(self.word_files.keys())
        print(f"[ValidationDataset] {len(self.words)} words with sufficient files "
              f"(need ≥ {k_enroll + n_test})")

    def get_enrollment_and_test(self, word):
        files = self.word_files[word]
        random.shuffle(files)
        enroll = files[:self.k_enroll]
        test = files[self.k_enroll:self.k_enroll + self.n_test]
        return enroll, test

    def load_audio(self, path):
        """Load and preprocess a single audio file."""
        wav, _ = librosa.load(path, sr=self.sr, mono=True)
        if len(wav) < self.target_length:
            wav = np.pad(wav, (0, self.target_length - len(wav)), mode='constant')
        else:
            wav = wav[:self.target_length]

        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak * 0.9

        return torch.FloatTensor(wav)