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
from collections import defaultdict
from torch.utils.data.sampler import Sampler

SAMPLE_RATE = 16000
TARGET_LENGTH = 24000 
HARD_NEG_FRAC = 0.50
N_MELS = 80

def compute_ped_matrix(words, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"[PED] Loading cached phoneme data from {cache_path}")
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        return cached['phoneme_seqs'], cached['hard_neg_pool']

    g2p = G2p()
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
            ped = 1.0 - sm.ratio() 

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

class PhoneticContrastiveSampler(Sampler):
    def __init__(self, dataset, m_per_class=4, batch_size=64):
        self.dataset = dataset
        self.m_per_class = m_per_class
        self.batch_size = batch_size
        self.classes_per_batch = batch_size // m_per_class
        self.word_to_id = {word: idx for idx, word in enumerate(dataset.words)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.class_to_indices = defaultdict(list)
        for idx, (word, _) in enumerate(dataset.all_entries):
            label = self.word_to_id[word]
            self.class_to_indices[label].append(idx)

        self.hard_neg_pool = dataset.hard_neg_pool
        self.all_classes = list(self.class_to_indices.keys())
        self.num_batches = len(dataset.all_entries) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            anchor_class = random.choice(self.all_classes)
            anchor_word = self.id_to_word[anchor_class]
            hard_neg_words = self.hard_neg_pool.get(anchor_word, [])
            hard_neg_classes = [self.word_to_id[w] for w in hard_neg_words if w in self.word_to_id]
            selected_classes = [anchor_class]
            n_hard_negs = (self.classes_per_batch // 2) - 1
            if len(hard_neg_classes) >= n_hard_negs:
                selected_classes.extend(random.sample(hard_neg_classes, n_hard_negs))
            else:
                selected_classes.extend(hard_neg_classes)

            while len(selected_classes) < self.classes_per_batch:
                rand_class = random.choice(self.all_classes)
                if rand_class not in selected_classes:
                    selected_classes.append(rand_class)

            batch_indices = []
            for cls in selected_classes:
                indices = self.class_to_indices[cls]
                if len(indices) >= self.m_per_class:
                    batch_indices.extend(random.sample(indices, self.m_per_class))
                else:
                    batch_indices.extend(random.choices(indices, k=self.m_per_class))

            random.shuffle(batch_indices)            
            yield batch_indices

    def __len__(self):
        return self.num_batches

class SupConDataset(Dataset):
    def __init__(self, data_root, musan_dir="../data/musan", sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH):
        self.sr = sample_rate
        self.target_length = target_length
        self.word_files = {}
        if not os.path.isdir(data_root):
            raise ValueError(f"Data root not found: {data_root}")

        for word_dir in sorted(os.listdir(data_root)):
            word_path = os.path.join(data_root, word_dir)
            if not os.path.isdir(word_path):
                continue

            word = word_dir.lower().strip()
            files = sorted(glob.glob(os.path.join(word_path, "*.wav")))
            if len(files) >= 2:  
                self.word_files[word] = files

        self.words = sorted(self.word_files.keys())
        if len(self.words) == 0:
            raise ValueError(f"No valid word directories found in {data_root}")

        self.word_to_id = {word: idx for idx, word in enumerate(self.words)}
        self.all_entries = [] 
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
        word, file_idx = self.all_entries[idx]
        audio_path = self.word_files[word][file_idx]        
        label = self.word_to_id[word]
        wav = self._load_audio(audio_path)
        snr_lo, snr_hi = -5.0, 15.0
        wav = self._inject_musan(wav, snr_lo, snr_hi)

        return wav, torch.tensor(label, dtype=torch.long)

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
        noise_path = random.choice(self.musan_files)
        try:
            noise, _ = librosa.load(noise_path, sr=self.sr, mono=True)
        except Exception:
            return waveform

        if len(noise) > self.target_length:
            start = random.randint(0, len(noise) - self.target_length)
            noise = noise[start:start + self.target_length]
        else:
            noise = np.pad(noise, (0, self.target_length - len(noise)), mode='wrap')

        speech_rms = np.sqrt(np.mean(wav_np ** 2))
        if speech_rms < 1e-10:
            return waveform
        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms < 1e-10:
            return waveform

        noise_rms_target = speech_rms / (10 ** (snr_db / 20))
        noisy = wav_np + noise * (noise_rms_target / noise_rms)
        peak = np.abs(noisy).max()
        if peak > 1e-6:
            noisy = np.clip(noisy / peak, -1.0, 1.0)

        return torch.FloatTensor(noisy)

class ValidationDataset:
    def __init__(self, data_root, sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH, k_enroll=3, n_test=5):
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
        wav, _ = librosa.load(path, sr=self.sr, mono=True)
        if len(wav) < self.target_length:
            wav = np.pad(wav, (0, self.target_length - len(wav)), mode='constant')
        else:
            wav = wav[:self.target_length]

        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak * 0.9

        return torch.FloatTensor(wav)