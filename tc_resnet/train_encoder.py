import warnings
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
from metrics import compute_eer
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
BATCH_SIZE = 64
LR_BACKBONE = 1e-3
LR_PCEN = 1e-2
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = - mean_log_prob_pos.mean()
        return loss

@torch.no_grad()
def validate(pcen, tc_resnet, val_dataset, device):
    pcen.eval()
    tc_resnet.eval()    
    if len(val_dataset.words) < 2:
        print("Not enough validation words. Skipping.")
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "eer": 1.0, "intra": 0.0, "inter": 1.0}

    words = val_dataset.words
    rng = np.random.RandomState(42)
    shuffled = words.copy()
    rng.shuffle(shuffled)
    split_idx = len(shuffled) // 2
    enrolled_words = shuffled[:split_idx]
    unseen_words = shuffled[split_idx:]

    templates = {}
    positive_queries = [] 
    negative_queries = [] 
    for word in enrolled_words:
        enroll_files, test_files = val_dataset.get_enrollment_and_test(word)        
        enroll_embs = []
        for path in enroll_files:
            wav = val_dataset.load_audio(path).unsqueeze(0).to(device)
            emb = tc_resnet(pcen(wav))  
            enroll_embs.append(emb)

        stacked = torch.cat(enroll_embs, dim=0)         
        w_raw = stacked.mean(dim=0, keepdim=True)        
        w_c = F.normalize(w_raw, p=2, dim=1).squeeze(0)  
        templates[word] = w_c
        
        for path in test_files:
            wav = val_dataset.load_audio(path).unsqueeze(0).to(device)
            query_emb = tc_resnet(pcen(wav)).squeeze(0)  
            positive_queries.append((word, query_emb))

    for word in unseen_words:
        _, test_files = val_dataset.get_enrollment_and_test(word)
        for path in test_files:
            wav = val_dataset.load_audio(path).unsqueeze(0).to(device)
            query_emb = tc_resnet(pcen(wav)).squeeze(0)
            negative_queries.append(query_emb)

    top1_correct = 0
    top5_correct = 0
    reciprocal_ranks = []
    open_set_scores = []
    open_set_labels = []
    intra_sims = []
    inter_sims = []
    for gt_word, query_emb in positive_queries:
        scores = []
        for template_word, template_emb in templates.items():
            sim = torch.dot(query_emb, template_emb).item()
            scores.append((template_word, sim))            
            if template_word == gt_word:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_words = [w for w, _ in scores]
        rank = ranked_words.index(gt_word) + 1
        if rank == 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
            
        reciprocal_ranks.append(1.0 / rank)
        max_sim = scores[0][1]
        open_set_scores.append(max_sim)
        open_set_labels.append(1) 

    for query_emb in negative_queries:
        max_sim = -1.0
        for template_emb in templates.values():
            sim = torch.dot(query_emb, template_emb).item()
            if sim > max_sim:
                max_sim = sim
                
        open_set_scores.append(max_sim)
        open_set_labels.append(0) 

    try:
        eer, _ = compute_eer(open_set_scores, open_set_labels)
    except Exception:
        eer = 1.0

    n_queries = max(len(positive_queries), 1)
    metrics = {
        "top1": top1_correct / n_queries,
        "top5": top5_correct / n_queries,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "eer": float(eer),
        "intra": float(np.mean(intra_sims)) if intra_sims else 0.0,
        "inter": float(np.mean(inter_sims)) if inter_sims else 1.0,
    }

    pcen.train()
    tc_resnet.train()

    return metrics

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    pcen = LearnablePCEN(sample_rate=SAMPLE_RATE, n_mels=N_MELS).to(device)
    tc_resnet = TCResNetAcousticEncoder(num_mels=N_MELS, embedding_dim=EMBEDDING_DIM).to(device)
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
    supcon_loss_fn = SupConLoss(temperature=0.1).to(device)
    print(f"Backbone LR: {LR_BACKBONE}, PCEN LR: {LR_PCEN}")
    print(f"Weight decay: {WEIGHT_DECAY} (PCEN: 0.0)")
    print(f"Scheduler: CosineAnnealing → {1e-6}")
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Grad clip: {GRAD_CLIP}")

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
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for waveforms, labels in progress:
            waveforms = waveforms.to(device) 
            labels = labels.to(device)
            embeddings = tc_resnet(pcen(waveforms))
            loss = supcon_loss_fn(embeddings, labels)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(tc_resnet.parameters()) + list(pcen.parameters()), max_norm=GRAD_CLIP)
            optimiser.step()
            total_loss += loss.item()
            n_batches += 1

            progress.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        pcen_s = pcen.s.item()
        pcen_alpha = pcen.alpha.item()
        pcen_delta = pcen.delta.item()
        pcen_r = pcen.r.item()

        print(f"Epoch {epoch + 1} | loss={avg_loss:.4f} | "
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
            metrics = validate(pcen, tc_resnet, val_dataset, device)                        
            val_score = (0.5 * metrics["top1"] + 0.2 * metrics["top5"] + 0.1 * metrics["mrr"] + 0.2 * (1.0 - metrics["eer"]))
            print(f"Val Retrieval -> Top-1: {metrics['top1']:.1%} | Top-5: {metrics['top5']:.1%} | MRR: {metrics['mrr']:.4f}")
            print(f"Val Open-Set  -> EER: {metrics['eer']:.2%} | Intra: {metrics['intra']:.4f} | Inter: {metrics['inter']:.4f}")
            print(f"Val Score: {val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                save_checkpoint(pcen, tc_resnet, optimiser, epoch + 1, avg_loss, metrics['intra'], metrics['inter'], args.output_dir)
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