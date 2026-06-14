import os
import sys
import argparse
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pcen import LearnablePCEN
from model import TCResNetAcousticEncoder
from dataset import SupConDataset, PhoneticContrastiveSampler, ValidationDataset

SAMPLE_RATE = 16000
N_MELS = 80
EMBEDDING_DIM = 128
TARGET_LENGTH = 24000
TRIPLET_MARGIN = 0.2
BATCH_SIZE = 64
LR_BACKBONE = 1e-3
LR_PCEN = 1e-2
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

@torch.no_grad()
def validate(pcen, tc_resnet, val_dataset, device, n_inter_samples=10):
    pcen.eval()
    tc_resnet.eval()
    if len(val_dataset.words) == 0:
        print("No validation words with sufficient files. Skipping")
        return 0.0, 1.0

    templates = {}   
    intra_sims = [] 
    for word in val_dataset.words:
        enroll_files, test_files = val_dataset.get_enrollment_and_test(word)
        enroll_embs = []
        for path in enroll_files:
            wav = val_dataset.load_audio(path).unsqueeze(0).to(device)
            pcen_feat = pcen(wav)
            emb = tc_resnet(pcen_feat)  
            enroll_embs.append(emb)

        stacked = torch.cat(enroll_embs, dim=0)         
        w_raw = stacked.mean(dim=0, keepdim=True)        
        w_c = F.normalize(w_raw, p=2, dim=1).squeeze(0)  
        templates[word] = w_c
        for path in test_files:
            wav = val_dataset.load_audio(path).unsqueeze(0).to(device)
            pcen_feat = pcen(wav)
            test_emb = tc_resnet(pcen_feat).squeeze(0)  
            sim = torch.dot(w_c, test_emb).item()  
            intra_sims.append(sim)

    inter_sims = []
    template_words = list(templates.keys())
    for i, w1 in enumerate(template_words):
        others = [w for w in template_words if w != w1]
        if len(others) > n_inter_samples:
            others = np.random.choice(others, size=n_inter_samples, replace=False).tolist()

        for w2 in others:
            sim = torch.dot(templates[w1], templates[w2]).item()
            inter_sims.append(sim)

    val_intra = np.mean(intra_sims) if intra_sims else 0.0
    val_inter = np.mean(inter_sims) if inter_sims else 1.0
    pcen.train()
    tc_resnet.train()

    return float(val_intra), float(val_inter)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("\nInitialising models...")
    pcen = LearnablePCEN(sample_rate=SAMPLE_RATE, n_mels=N_MELS).to(device)
    tc_resnet = TCResNetAcousticEncoder(num_mels=N_MELS, embedding_dim=EMBEDDING_DIM).to(device)
    print("\nLoading dataset...")
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")

    dataset = SupConDataset(data_root=train_dir, sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH)
    sampler = PhoneticContrastiveSampler(dataset, m_per_class=4, batch_size=BATCH_SIZE)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    val_dataset = None
    if os.path.exists(val_dir):
        val_dataset = ValidationDataset(data_root=val_dir, sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH)

    optimiser = AdamW([
        {"params": tc_resnet.parameters(), "lr": LR_BACKBONE,
         "weight_decay": WEIGHT_DECAY},
        {"params": pcen.parameters(), "lr": LR_PCEN,
         "weight_decay": 0.0},  
    ])

    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)
    print(f"Backbone LR: {LR_BACKBONE}, PCEN LR: {LR_PCEN}")
    print(f"Weight decay: {WEIGHT_DECAY} (PCEN: 0.0)")
    print(f"Scheduler: CosineAnnealing → {1e-6}")
    triplet_loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2, reduction='mean')
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Grad clip: {GRAD_CLIP}")
    print(f"Triplet margin: {TRIPLET_MARGIN}\n")

    best_val_score = -float('inf')
    patience_counter = 0
    patience = 5
    pcen_lr_reduced = False
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        pcen.train()
        tc_resnet.train()
        total_loss = 0.0
        n_batches = 0
        n_active_triplets = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for waveform_A, waveform_P, waveform_N in progress:
            wA = waveform_A.to(device) 
            wP = waveform_P.to(device)
            wN = waveform_N.to(device)
            emb_A = tc_resnet(pcen(wA))  
            emb_P = tc_resnet(pcen(wP))  
            emb_N = tc_resnet(pcen(wN))  
            loss = triplet_loss_fn(emb_A, emb_P, emb_N)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(tc_resnet.parameters()) + list(pcen.parameters()), max_norm=GRAD_CLIP)

            optimiser.step()
            total_loss += loss.item()
            n_batches += 1
            with torch.no_grad():
                d_ap = torch.norm(emb_A - emb_P, p=2, dim=1)
                d_an = torch.norm(emb_A - emb_N, p=2, dim=1)
                active = (d_ap - d_an + TRIPLET_MARGIN > 0).sum().item()
                n_active_triplets += active

            progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Active": f"{active}/{len(wA)}",
            })

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        active_ratio = n_active_triplets / max(n_batches * BATCH_SIZE, 1)
        pcen_s = pcen.s.item()
        pcen_alpha = pcen.alpha.item()
        pcen_delta = pcen.delta.item()
        pcen_r = pcen.r.item()

        print(f"Epoch {epoch + 1} | loss={avg_loss:.4f} | "
              f"active={active_ratio:.2%} | "
              f"pcen.s={pcen_s:.4f} alpha={pcen_alpha:.4f} "
              f"delta={pcen_delta:.4f} r={pcen_r:.4f}")

        if epoch >= 10 and pcen_s < 0.01 and not pcen_lr_reduced:
            print(f"\nPCEN s-collapse detected (s={pcen_s:.6f} < 0.01)")
            print("  Reducing PCEN learning rate from 1e-2 to 1e-3")
            for param_group in optimiser.param_groups:
                if param_group.get('weight_decay', -1) == 0.0:
                    param_group['lr'] = 1e-3
            pcen_lr_reduced = True

        if val_dataset and len(val_dataset.words) > 0:
            val_intra, val_inter = validate(pcen, tc_resnet, val_dataset, device)            
            val_score = val_intra - val_inter
            print(f"  Val intra-sim: {val_intra:.4f} (target >= 0.85) | "
                  f"Val inter-sim: {val_inter:.4f} (target <= 0.30)")
            print(f"  Val Score (Intra - Inter): {val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                save_checkpoint(pcen, tc_resnet, optimiser, epoch + 1, avg_loss, val_intra, val_inter, args.output_dir)
                print(f"New best validation score: {val_score:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n[EARLY STOP] No improvement for {patience} epochs.")
                    break
        else:
            save_checkpoint(pcen, tc_resnet, optimiser, epoch + 1, avg_loss, 0.0, 0.0, args.output_dir)

    print(f"\nTraining complete. Best validation score: {best_val_score:.4f}")
    export(pcen, tc_resnet, args.output_dir, device)

def save_checkpoint(pcen, tc_resnet, optimiser, epoch, loss, val_intra, val_inter, output_dir):
    path = os.path.join(output_dir, "best_checkpoint.pth")
    torch.save({
        'epoch': epoch,
        'pcen_state_dict': pcen.state_dict(),
        'tc_resnet_state_dict': tc_resnet.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'loss': loss,
        'val_intra_sim': val_intra,
        'val_inter_sim': val_inter,
        'pcen_params': pcen.export_params(),
    }, path)

def export(pcen, tc_resnet, output_dir, device):
    pcen.eval()
    tc_resnet.eval()

    for p in list(pcen.parameters()) + list(tc_resnet.parameters()):
        p.requires_grad_(False)

    pcen_path = os.path.join(output_dir, "pcen_params.json")
    params = pcen.save_params(pcen_path)
    print(f"[1/4] PCEN sidecar → {pcen_path}")
    print(f"  s={params['s']:.6f}, α={params['alpha']:.6f}, "
          f"δ={params['delta']:.6f}, r={params['r']:.6f}")

    onnx_path = os.path.join(output_dir, "tc_resnet_backbone.onnx")
    print(f"[2/4] ONNX export → {onnx_path}")

    dummy = torch.randn(1, N_MELS, 151).to(device)
    torch.onnx.export(
        tc_resnet, dummy, onnx_path,
        opset_version=14,
        input_names=["pcen_features"],
        output_names=["embedding"],
        dynamic_axes={
            "pcen_features": {0: "batch", 2: "time"},
            "embedding": {0: "batch"},
        },
        do_constant_folding=True,
    )

    print("[3/4] Validating ONNX fidelity (100 samples)...")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    max_diff = 0.0
    tc_resnet_cpu = tc_resnet.cpu()
    for i in range(100):
        x = torch.randn(1, N_MELS, 151)
        pt_out = tc_resnet_cpu(x).detach().numpy()
        ort_out = sess.run(None, {"pcen_features": x.numpy()})[0]
        diff = abs(pt_out - ort_out).max()
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-4, (f"ONNX fidelity check FAILED: max diff = {max_diff:.8f}")
    print(f"ONNX fidelity validated (max diff: {max_diff:.8f})")
    tc_resnet.to(device if str(device) != 'cpu' else 'cpu')

    sha256 = hashlib.sha256()
    with open(onnx_path, 'rb') as f:
        for block in iter(lambda: f.read(8192), b''):
            sha256.update(block)
    backbone_hash = sha256.hexdigest()

    hash_path = os.path.join(output_dir, "tc_resnet_sha256.txt")
    with open(hash_path, 'w') as f:
        f.write(backbone_hash + '\n')

    print(f"[4/4] Backbone SHA-256: {backbone_hash[:32]}...")
    print(f"\nAll export artifacts saved to: {output_dir}/")
    print(f"  - pcen_params.json")
    print(f"  - tc_resnet_backbone.onnx")
    print(f"  - tc_resnet_sha256.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TC-ResNet Acoustic Encoder with Phonetic Explosion")
    parser.add_argument("--data_root", type=str, default="../data/tts_corpus", help="Path to TTS word corpus directory")
    parser.add_argument("--output_dir", type=str, default="./tc_resnet_output", help="Directory for checkpoints and exports")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    args = parser.parse_args()

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.abspath(args.data_root)

    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, args.output_dir)

    train(args)