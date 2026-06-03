import os
import sys
import types
import math
import json
import hashlib
import itertools
from pathlib import Path
import glob

# --- BUG FIX FOR WINDOWS / PYTORCH 2.x ---
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
sys.modules['k2'] = types.ModuleType('k2')
sys.modules['flair'] = types.ModuleType('flair')
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = types.ModuleType('fake_flair_emb')
sys.modules['speechbrain.integrations.nlp'] = types.ModuleType('fake_nlp')
sys.modules['speechbrain.integrations.huggingface.wordemb'] = types.ModuleType('fake_wordemb')
sys.modules['speechbrain.integrations.huggingface'] = types.ModuleType('fake_hf')
# -----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

torch._dynamo.config.disable = True
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pcen import LearnablePCEN
from dataset import DomainAdaptationDataset, BalancedBatchSampler
from metrics import compute_eer

class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (ArcFace) loss module."""

    def __init__(self, in_features, out_features, s=32.0, m=0.2):
        super(AAMSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output, cosine

class ECAPABackbone(nn.Module):
    """
    Wraps the SpeechBrain ECAPA-TDNN embedding model into a clean nn.Module
    that takes PCEN features as input and outputs 192-D embeddings.
    """

    def __init__(self, speechbrain_encoder):
        super().__init__()
        self.encoder = speechbrain_encoder

    def forward(self, pcen_features):
        features = pcen_features.transpose(1, 2)
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True)
        features = (features - mean) / (std + 1e-5)

        embeddings = self.encoder(features)
        embeddings = embeddings.squeeze(1)

        return embeddings

class DomainAdaptationTrainer:
    """
    Fine-tunes ECAPA-TDNN + learnable PCEN for a specific edge microphone.

    Spec hyperparameters (Context A, Step 5):
        - Optimizer: AdamW
        - Backbone LR: 1e-5
        - PCEN LR: 1e-4 (10× higher, learned from scratch)
        - Weight decay: 1e-4
        - Batch size: 64 (balanced per speaker)
        - Epochs: 5–10 (early stop on validation EER)
        - Gradient clipping: max_norm=1.0
    """

    def __init__(self, data_dir, output_dir="./finetuned_models",
                 num_epochs=10, batch_size=64,
                 lr_backbone=1e-5, lr_pcen=1e-4,
                 weight_decay=1e-4, max_grad_norm=1.0,
                 val_split_ratio=0.1, val_split_seed=42,
                 early_stop_patience=2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr_backbone = lr_backbone
        self.lr_pcen = lr_pcen
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.early_stop_patience = early_stop_patience

        os.makedirs(output_dir, exist_ok=True)
        print("\n[1/5] Initializing Domain Adaptation Dataset...")
        full_dataset = DomainAdaptationDataset(data_dir=data_dir, max_audio_length=48000)

        self.train_dataset, self.val_dataset, self.val_speakers = \
            full_dataset.get_val_split(ratio=val_split_ratio, seed=val_split_seed)

        self.num_classes = full_dataset.n_speakers
        self.spk_to_id = full_dataset.spk_to_id
        self.train_sampler = BalancedBatchSampler(self.train_dataset, speakers_per_batch=16, utterances_per_speaker=4)
        self.train_loader = DataLoader(self.train_dataset, batch_sampler=self.train_sampler, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self._setup_model()

    def _setup_model(self):
        """Load pre-trained ECAPA-TDNN backbone, attach learnable PCEN and AAM head."""

        print("\n[2/5] Initializing Learnable PCEN Frontend...")
        self.pcen = LearnablePCEN().to(self.device)
        print("[3/5] Loading pre-trained ECAPA-TDNN from SpeechBrain...")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(self.output_dir, "speechbrain_cache"),
            run_opts={"device": str(self.device)}
        )

        self.backbone = ECAPABackbone(
            classifier.mods.embedding_model.to(self.device)
        )

        print(f"[4/5] Initializing AAM-Softmax head for {self.num_classes} speakers "
              f"(s=32, m=0.2)...")
        self.classification_head = AAMSoftmax(
            in_features=192,
            out_features=self.num_classes,
            s=32.0,   # Spec: s=32
            m=0.2,    # Spec: m=0.2
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        print("[5/5] Configuring AdamW optimizer with dual learning rates...")
        self.optimizer = AdamW([
            {'params': self.pcen.parameters(), 'lr': self.lr_pcen},
            {'params': self.backbone.parameters(), 'lr': self.lr_backbone},
            {'params': self.classification_head.parameters(), 'lr': self.lr_backbone},
        ], weight_decay=self.weight_decay, foreach=False)

        print(f"  PCEN LR: {self.lr_pcen}, Backbone LR: {self.lr_backbone}, "
              f"Weight Decay: {self.weight_decay}")

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"Starting Domain Adaptation Fine-Tuning on {self.device}")
        print(f"  Epochs: {self.num_epochs}, Batch: {self.batch_size}, "
              f"Grad clip: {self.max_grad_norm}")
        print(f"{'=' * 60}\n")

        best_eer = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            val_eer, val_threshold = self._validate_epoch(epoch)
            pcen_params = self.pcen.export_params()
            print(f"  PCEN params: s={pcen_params['s']:.4f}, α={pcen_params['alpha']:.4f}, "
                  f"δ={pcen_params['delta']:.4f}, r={pcen_params['r']:.4f}")

            if val_eer < best_eer:
                best_eer = val_eer
                patience_counter = 0
                self._save_checkpoint(epoch + 1, train_acc, val_eer)
                print(f"New best EER: {val_eer:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{self.early_stop_patience}")

                if patience_counter >= self.early_stop_patience:
                    print(f"\n[EARLY STOP] No EER improvement for {self.early_stop_patience} epochs.")
                    break

        print(f"\nFine-tuning complete. Best validation EER: {best_eer:.4f}")
        return best_eer

    def _train_epoch(self, epoch):
        """Run one training epoch."""
        self.pcen.train()
        self.backbone.train()
        self.classification_head.train()

        total_loss = 0.0
        correct_preds = 0
        total_preds = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]")

        for raw_waveforms, labels in progress:
            raw_waveforms = raw_waveforms.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward: waveform → PCEN → backbone → AAM-Softmax
            pcen_features = self.pcen(raw_waveforms)
            embeddings = self.backbone(pcen_features)
            outputs, raw_cosines = self.classification_head(embeddings, labels)

            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.pcen.parameters(),
                    self.backbone.parameters(),
                    self.classification_head.parameters()
                ),
                max_norm=self.max_grad_norm
            )

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(raw_cosines.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            running_acc = 100 * correct_preds / total_preds
            progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{running_acc:.2f}%"
            })

        avg_loss = total_loss / max(len(self.train_loader), 1)
        accuracy = 100 * correct_preds / max(total_preds, 1)
        print(f"  Train — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self, epoch):
        """Compute validation EER on the held-out speaker split."""
        self.pcen.eval()
        self.backbone.eval()
        all_embeddings = []
        all_labels = []

        for raw_waveforms, labels in self.val_loader:
            raw_waveforms = raw_waveforms.to(self.device)
            pcen_features = self.pcen(raw_waveforms)
            embeddings = self.backbone(pcen_features)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        spk_to_indices = {}
        for idx, label in enumerate(all_labels.numpy()):
            if label not in spk_to_indices:
                spk_to_indices[label] = []
            spk_to_indices[label].append(idx)
            
        speakers = list(spk_to_indices.keys())

        scores = []
        pair_labels = []
        n_target_pairs = 10000
        
        same_pairs = []
        for spk, indices in spk_to_indices.items():
            if len(indices) >= 2:
                for _ in range(200): 
                    idx1, idx2 = np.random.choice(indices, 2, replace=False)
                    same_pairs.append((idx1, idx2))
                    
        np.random.shuffle(same_pairs)
        same_pairs = same_pairs[:n_target_pairs]

        diff_pairs = []
        for _ in range(n_target_pairs):
            spk1, spk2 = np.random.choice(speakers, 2, replace=False)
            idx1 = np.random.choice(spk_to_indices[spk1])
            idx2 = np.random.choice(spk_to_indices[spk2])
            diff_pairs.append((idx1, idx2))

        for idx1, idx2 in same_pairs:
            sim = F.cosine_similarity(all_embeddings[idx1].unsqueeze(0), all_embeddings[idx2].unsqueeze(0)).item()
            scores.append(sim)
            pair_labels.append(1)

        for idx1, idx2 in diff_pairs:
            sim = F.cosine_similarity(all_embeddings[idx1].unsqueeze(0), all_embeddings[idx2].unsqueeze(0)).item()
            scores.append(sim)
            pair_labels.append(0)

        if len(scores) == 0:
            print(f"Val — No pairs available for EER computation.")
            return 1.0, 0.0

        eer, threshold = compute_eer(scores, pair_labels)
        print(f"  Val — EER: {eer:.4f} (threshold: {threshold:.4f}), "
              f"Pairs: {len(scores)} ({sum(pair_labels)} same, {len(pair_labels) - sum(pair_labels)} diff)")

        self.pcen.train()
        self.backbone.train()

        return eer, threshold

    def _save_checkpoint(self, epoch, accuracy, val_eer):
        """Save training checkpoint (includes all weights for resumption)."""
        checkpoint_path = os.path.join(self.output_dir, f"domain_adapted_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'pcen_state_dict': self.pcen.state_dict(),
            'backbone_state_dict': self.backbone.state_dict(),
            'head_state_dict': self.classification_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'val_eer': val_eer,
            'spk_to_id': self.spk_to_id,
            'pcen_params': self.pcen.export_params(),
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def export(self, checkpoint_path=None):
        """
        Produces:
          1. ONNX backbone (opset 14, dynamic time axis) — no classification head
          2. PCEN parameter JSON sidecar
          3. Validation report (50-pair EER check, must be < 5%)
        """
        print(f"\n{'=' * 60}")
        print("EXPORT — Preparing deployment artifacts")
        print(f"{'=' * 60}\n")

        if checkpoint_path is not None:
            print(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.pcen.load_state_dict(ckpt['pcen_state_dict'])
            self.backbone.load_state_dict(ckpt['backbone_state_dict'])
        print("[1/4] Freezing backbone parameters...")
        self.pcen.eval()
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        pcen_json_path = os.path.join(self.output_dir, "pcen_params.json")
        print(f"[2/4] Exporting PCEN parameters to {pcen_json_path}")
        pcen_params = self.pcen.save_params(pcen_json_path)
        print(f"  s={pcen_params['s']:.6f}, α={pcen_params['alpha']:.6f}, "
              f"δ={pcen_params['delta']:.6f}, r={pcen_params['r']:.6f}")
        onnx_path = os.path.join(self.output_dir, "ecapa_backbone.onnx")
        print(f"[3/4] Exporting backbone to ONNX: {onnx_path}")
        dummy_input = torch.randn(1, 80, 300).to(self.device)

        torch.onnx.export(
            self.backbone,
            dummy_input,
            onnx_path,
            opset_version=14,
            input_names=['pcen_features'],
            output_names=['embedding'],
            dynamic_axes={
                'pcen_features': {0: 'batch', 2: 'time'},
                'embedding': {0: 'batch'},
            },
            do_constant_folding=True,
        )

        onnx_hash = self._compute_file_hash(onnx_path)
        print(f"  ONNX hash: {onnx_hash[:16]}...")
        print("[4/4] Running 50-pair EER validation...")
        eer = self._validate_export(n_same=25, n_diff=25)

        report = {
            "onnx_path": onnx_path,
            "pcen_params_path": pcen_json_path,
            "onnx_hash": onnx_hash,
            "validation_eer": eer,
            "eer_threshold_passed": eer < 0.05,
            "pcen_params": pcen_params,
        }

        report_path = os.path.join(self.output_dir, "export_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        if eer < 0.05:
            print(f"\nEXPORT VALIDATED — EER: {eer:.4f} (< 5% threshold)")
        else:
            print(f"\nWARNING — EER: {eer:.4f} (>= 5% threshold)")
            print("The model may not meet deployment quality requirements.")

        print(f"\nDeployment artifacts saved to: {self.output_dir}/")
        print(f"  - ecapa_backbone.onnx")
        print(f"  - pcen_params.json")
        print(f"  - export_validation_report.json")

        return report

    @torch.no_grad()
    def _validate_export(self, n_same=25, n_diff=25):
        """
        Run 50-pair cosine similarity EER check on validation data.
        Spec: 25 same-speaker pairs + 25 different-speaker pairs.
        """
        self.pcen.eval()
        self.backbone.eval()

        # Collect validation embeddings
        all_embeddings = []
        all_labels = []

        for raw_waveforms, labels in self.val_loader:
            raw_waveforms = raw_waveforms.to(self.device)
            pcen_features = self.pcen(raw_waveforms)
            embeddings = self.backbone(pcen_features)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        spk_to_indices = {}
        for idx, label in enumerate(all_labels.numpy()):
            if label not in spk_to_indices:
                spk_to_indices[label] = []
            spk_to_indices[label].append(idx)
            
        speakers = list(spk_to_indices.keys())

        same_pairs = []
        diff_pairs = []
        
        valid_same_spks = [spk for spk, idxs in spk_to_indices.items() if len(idxs) >= 2]
        for _ in range(n_same):
            spk = np.random.choice(valid_same_spks)
            idx1, idx2 = np.random.choice(spk_to_indices[spk], 2, replace=False)
            same_pairs.append((idx1, idx2))

        for _ in range(n_diff):
            spk1, spk2 = np.random.choice(speakers, 2, replace=False)
            idx1 = np.random.choice(spk_to_indices[spk1])
            idx2 = np.random.choice(spk_to_indices[spk2])
            diff_pairs.append((idx1, idx2))

        scores = []
        labels = []

        for i, j in same_pairs:
            sim = F.cosine_similarity(
                all_embeddings[i].unsqueeze(0),
                all_embeddings[j].unsqueeze(0)
            ).item()
            scores.append(sim)
            labels.append(1)

        for i, j in diff_pairs:
            sim = F.cosine_similarity(
                all_embeddings[i].unsqueeze(0),
                all_embeddings[j].unsqueeze(0)
            ).item()
            scores.append(sim)
            labels.append(0)

        eer, threshold = compute_eer(scores, labels)
        print(f"  Export EER: {eer:.4f} (threshold: {threshold:.4f})")
        print(f"  Pairs: {len(same_pairs)} same + {len(diff_pairs)} diff = {len(scores)} total")

        return eer

    @staticmethod
    def _compute_file_hash(filepath):
        """Compute SHA-256 hash of a file for integrity verification."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(8192), b''):
                sha256.update(block)
        return sha256.hexdigest()

if __name__ == "__main__":
    DATASET_DIR = "../data/VoxCeleb1/wav"

    trainer = DomainAdaptationTrainer(
        data_dir=DATASET_DIR,
        output_dir="./finetuned_models",
        num_epochs=10,
        batch_size=64,
        lr_backbone=1e-5, 
        lr_pcen=1e-4,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        val_split_ratio=0.1,
        val_split_seed=42,
        early_stop_patience=5, 
    )

    best_eer = trainer.train()
    checkpoints = sorted(glob.glob(os.path.join(trainer.output_dir, "domain_adapted_epoch_*.pth")))
    if checkpoints:
        best_ckpt = checkpoints[-1]  
        report = trainer.export(checkpoint_path=best_ckpt)
    else:
        print("No checkpoints found. Exporting current model state...")
        report = trainer.export()