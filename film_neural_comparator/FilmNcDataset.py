import os
import math
import random
import glob
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Sampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tc_resnet.dataset import compute_ped_matrix

SAMPLE_RATE = 16000
TARGET_LENGTH = 24000  
NOISE_DATASET = "../data/musan"
ES_PATH = "./speaker_embeddings"

def load_tensor_or_state(path: str, map_location: str = "cpu") -> torch.Tensor:
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, torch.Tensor): return obj
    if isinstance(obj, dict):
        for key in ("embedding", "feat", "features", "tensor", "e_s", "vector"):
            if key in obj and isinstance(obj[key], torch.Tensor): return obj[key]
        tensor_vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensor_vals) == 1: return tensor_vals[0]
    raise ValueError(f"Could not extract tensor from file: {path}")

class BaseKWSDataset(Dataset):
    def __init__(self, data_root: str, target_sr: int = SAMPLE_RATE, max_audio_length: int = TARGET_LENGTH):
        super().__init__()
        self.target_sr = target_sr
        self.max_audio_length = max_audio_length
        self.noise_files = []
        self.noise_cache = []
        if os.path.exists(NOISE_DATASET):
            raw_noise_paths = glob.glob(os.path.join(NOISE_DATASET, "**/*.wav"), recursive=True)            
            sampled_paths = random.sample(raw_noise_paths, min(len(raw_noise_paths), 250))
            for p in sampled_paths:
                try:
                    wav, _ = torchaudio.load(p, num_frames=self.target_sr * 3)
                    if wav.shape[1] > 0:
                        self.noise_cache.append(wav)
                except:
                    pass

        self.speakers = sorted([f[:-3] for f in os.listdir(ES_PATH) if f.endswith(".pt")]) if os.path.exists(ES_PATH) else []        
        all_wavs = glob.glob(os.path.join(data_root, "**", "*.wav"), recursive=True)
        self.index: Dict[str, Dict[str, List[str]]] = {}
        self.all_valid_samples: List[Dict[str, str]] = []
        unique_words = set()
        for path in all_wavs:
            filename = os.path.basename(path)
            word = os.path.basename(os.path.dirname(path)) 
            speaker_id = filename.split("_")[0]
            if speaker_id not in self.speakers:
                continue

            unique_words.add(word)
            self.index.setdefault(speaker_id, {}).setdefault(word, []).append(path)

        self.words = sorted(list(unique_words))
        self.word_to_id = {w: i for i, w in enumerate(self.words)}
        self.spk_to_id = {s: i for i, s in enumerate(self.speakers)}
        for spk, words_dict in self.index.items():
            if len(words_dict) >= 2: 
                for word, files in words_dict.items():
                    for path in files:
                        self.all_valid_samples.append({"speaker": spk, "word": word, "path": path})

        if len(self.all_valid_samples) == 0:
            raise ValueError("CRITICAL: Zero valid samples found. Verify paths.")

    def _load_and_resample(self, file_path: str) -> torch.Tensor:
        wav, _ = torchaudio.load(file_path)
        return wav

    def _inject_musan(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.noise_cache or random.random() > 0.8: return wav        
        noise_wav = random.choice(self.noise_cache) 
        try:
            if noise_wav.shape[1] > self.max_audio_length:
                start = random.randint(0, noise_wav.shape[1] - self.max_audio_length)
                noise_wav = noise_wav[:, start:start + self.max_audio_length]
            else:
                repeats = math.ceil(self.max_audio_length / noise_wav.shape[1])
                noise_wav = noise_wav.repeat(1, repeats)[:, :self.max_audio_length]

            speech_power = torch.mean(wav ** 2) + 1e-8
            noise_power = torch.mean(noise_wav ** 2) + 1e-8
            target_snr = random.uniform(-5.0, 15.0)
            scale_factor = torch.sqrt((speech_power / (10 ** (target_snr / 10.0))) / noise_power)
            
            return torch.clamp(wav + noise_wav * scale_factor, -1.0, 1.0)
        except Exception: 
            return wav

    def _load_embedding(self, path: str, expected_dim: int) -> torch.Tensor:
        emb = load_tensor_or_state(path, map_location="cpu").float().view(-1)
        if emb.numel() != expected_dim: raise ValueError("Dimension mismatch.")
        return F.normalize(emb, p=2, dim=0)

class PKBatchSampler(Sampler):
    def __init__(self, dataset, p_classes: int, k_samples: int):
        self.dataset = dataset
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.batch_size = p_classes * k_samples
        self.label_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.all_valid_samples):
            word = sample["word"]
            word_id = dataset.word_to_id[word]
            self.label_to_indices[word_id].append(idx)

        self.valid_labels = list(self.label_to_indices.keys())        
        self.num_batches = len(dataset.all_valid_samples) // self.batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            sampled_classes = random.sample(self.valid_labels, self.p_classes)
            for cls_id in sampled_classes:
                available_indices = self.label_to_indices[cls_id]                
                if len(available_indices) >= self.k_samples:
                    sampled_indices = random.sample(available_indices, self.k_samples)
                else:
                    sampled_indices = random.choices(available_indices, k=self.k_samples)

                batch_indices.extend(sampled_indices)

            random.shuffle(batch_indices)            
            yield batch_indices

    def __len__(self):
        return self.num_batches

class CollisionDataset(BaseKWSDataset):
    def __init__(self, data_root="./tts_corpus_processed/train", **kwargs):
        super().__init__(data_root=data_root, **kwargs)
        self.virtual_length = len(self.all_valid_samples)

    def __len__(self) -> int:
        return self.virtual_length

    def _create_collision(self, target_wav: torch.Tensor, competing_wav: torch.Tensor) -> torch.Tensor:
        target_power = torch.mean(target_wav ** 2) + 1e-8
        comp_power = torch.mean(competing_wav ** 2) + 1e-8        
        sir = random.uniform(-5.0, 10.0)
        target_comp_power = target_power / (10 ** (sir / 10.0))
        scale = torch.sqrt(target_comp_power / comp_power)
        
        mixture = target_wav + scale * competing_wav
        return torch.clamp(mixture, -1.0, 1.0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor = self.all_valid_samples[idx % len(self.all_valid_samples)]
        owner_spk = anchor["speaker"]
        target_word = anchor["word"]
        e_s_owner = self._load_embedding(os.path.join(ES_PATH, f"{owner_spk}.pt"), 192)
        is_owner_speaking = random.random() > 0.5

        if is_owner_speaking:
            primary_path = anchor["path"]
            label_mask = 1.0
        else:
            imposters = [s for s in self.speakers if s != owner_spk and target_word in self.index.get(s, {})]
            if imposters:
                imp_spk = random.choice(imposters)
                primary_path = random.choice(self.index[imp_spk][target_word])
                label_mask = 0.0
            else:
                primary_path = anchor["path"] 
                label_mask = 1.0

        primary_wav = self._load_and_resample(primary_path)
        primary_wav = primary_wav / (primary_wav.abs().max() + 1e-8)
        comp_spk = random.choice([s for s in self.speakers if s != owner_spk])
        comp_word = random.choice(list(self.index[comp_spk].keys()))
        comp_path = random.choice(self.index[comp_spk][comp_word])
        comp_wav = self._load_and_resample(comp_path)
        comp_wav = comp_wav / (comp_wav.abs().max() + 1e-8)

        mixture = self._create_collision(primary_wav, comp_wav)
        mixture = self._inject_musan(mixture).squeeze(0)
        clean_target_path = random.choice(self.index[owner_spk][target_word])
        clean_target = self._load_and_resample(clean_target_path)
        clean_target = clean_target / (clean_target.abs().max() + 1e-8)
        clean_target = self._inject_musan(clean_target).squeeze(0)

        return {
            "audio_mix": mixture,
            "clean_target": clean_target,
            "e_s_correct": e_s_owner, 
            "e_s": e_s_owner, 
            "word_label": torch.tensor(self.word_to_id[target_word], dtype=torch.long),
            "is_correct_speaker": torch.tensor(label_mask, dtype=torch.float32)
        }

class QuadStateDataset(BaseKWSDataset):
    def __init__(self, data_root="./tts_corpus_processed/train", k_enroll=3, **kwargs):
        super().__init__(data_root=data_root, **kwargs)
        self.k_enroll = k_enroll
        self.virtual_length = min(len(self.all_valid_samples), 80000)
        cache_path = os.path.join(data_root, ".ped_cache.json")
        print("[QuadStateDataset] Computing/Loading PED matrix for Hard Negatives...")
        _, self.hard_neg_pool = compute_ped_matrix(self.words, cache_path=cache_path)

    def __len__(self) -> int:
        return self.virtual_length

    def _prepare_audio(self, path: str) -> torch.Tensor:
        wav = self._load_and_resample(path)
        wav = wav / (wav.abs().max() + 1e-8)
        return self._inject_musan(wav).squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor = random.choice(self.all_valid_samples)
        pos_speaker = anchor["speaker"]
        pos_word = anchor["word"]
        available_files = self.index[pos_speaker][pos_word].copy()
        enroll_paths = random.choices(available_files, k=self.k_enroll) if len(available_files) < self.k_enroll else random.sample(available_files, self.k_enroll)           
        test_pool = list(set(available_files) - set(enroll_paths))
        tp_path = random.choice(test_pool) if test_pool else random.choice(available_files)

        competing_speakers = [s for s in self.speakers if s != pos_speaker and pos_word in self.index[s]]
        in_spk = random.choice(competing_speakers)
        in_path = random.choice(self.index[in_spk][pos_word])
        hard_candidates = self.hard_neg_pool.get(pos_word, [])
        valid_hard_candidates = [w for w in hard_candidates if w in self.index.get(pos_speaker, {})]

        if valid_hard_candidates:
            pd_word = random.choice(valid_hard_candidates)
        else:
            other_words = [w for w in self.index[pos_speaker].keys() if w != pos_word]
            pd_word = random.choice(other_words)

        pd_path = random.choice(self.index[pos_speaker][pd_word])
        easy_speakers = [s for s in self.speakers if s != pos_speaker and pd_word in self.index[s]]
        en_spk = random.choice(easy_speakers)
        en_path = random.choice(self.index[en_spk][pd_word])
        enroll_audio = torch.stack([self._prepare_audio(p) for p in enroll_paths], dim=0)
        query_audio = torch.stack([
            self._prepare_audio(tp_path),
            self._prepare_audio(in_path),
            self._prepare_audio(pd_path),
            self._prepare_audio(en_path)
        ], dim=0)

        e_s = self._load_embedding(os.path.join(ES_PATH, f"{pos_speaker}.pt"), 192)        
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        return {
            "enroll_audio": enroll_audio,  
            "query_audio": query_audio,  
            "e_s": e_s,
            "labels": labels
        }

class ValidationDataset(BaseKWSDataset):
    def __init__(self, data_root: str, k_enroll=3, n_test=15, **kwargs):
        super().__init__(data_root=data_root, **kwargs) 
        self.k_enroll = k_enroll
        self.n_test = n_test
        self.word_files = {}
            
        for word_dir in sorted(os.listdir(data_root)):
            word_path = os.path.join(data_root, word_dir)
            if not os.path.isdir(word_path):
                continue
            
            word = word_dir.lower().strip()
            files = sorted(glob.glob(os.path.join(word_path, "*.wav")))
            if len(files) >= 2:
                self.word_files[word] = files
                
        self.words = sorted(self.word_files.keys())
        if not self.words:
            print(f"WARNING: No valid validation words found in {data_root}")

        self.cached_splits = {}
        self.cached_owners = {}
        for word in self.words:
            enroll, test = self._generate_enrollment_and_test(word)
            owner_id = os.path.basename(enroll[0]).split("_")[0]
            self.cached_splits[word] = (enroll, test)
            self.cached_owners[word] = owner_id

    def _prepare_audio(self, file_path: str) -> torch.Tensor:
        wav = self._load_and_resample(file_path)
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak
            
        wav = self._inject_musan(wav)
        return wav.squeeze(0)

    def _generate_enrollment_and_test(self, word: str):
        speaker_dict = {}
        for path in self.word_files[word]:
            filename = os.path.basename(path)
            spk_id = filename.split("_")[0]
            speaker_dict.setdefault(spk_id, []).append(path)

        speakers = list(speaker_dict.keys())
        random.shuffle(speakers)
        if len(speakers) >= 2:
            n_enroll_speakers = min(3, max(1, len(speakers) // 4))
            enroll_speakers = speakers[:n_enroll_speakers]
            test_speakers = speakers[n_enroll_speakers:]
            enroll_files = []
            test_files = []
            per_spk = max(1, self.k_enroll // len(enroll_speakers))
            for spk in enroll_speakers:
                files = speaker_dict[spk].copy()
                random.shuffle(files)
                enroll_files.extend(files[:per_spk])

            remaining = []
            for spk in enroll_speakers:
                remaining.extend(speaker_dict[spk])

            available_remaining = list(set(remaining) - set(enroll_files))
            random.shuffle(available_remaining)
            needed = self.k_enroll - len(enroll_files)
            if needed > 0 and available_remaining:
                enroll_files.extend(available_remaining[:needed])

            for spk in test_speakers:
                test_files.extend(speaker_dict[spk])

            random.shuffle(test_files)
            test_files = test_files[:self.n_test]
            if len(enroll_files) >= self.k_enroll and len(test_files) >= self.n_test:
                return enroll_files, test_files

        all_files = self.word_files[word].copy()
        random.shuffle(all_files)        
        enroll = all_files[:self.k_enroll]
        test = all_files[self.k_enroll:self.k_enroll + self.n_test]
        
        return enroll, test

    def get_enrollment_and_test(self, word: str):
        return self.cached_splits[word]

    def get_owner_id(self, word):
        return self.cached_owners[word]