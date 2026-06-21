import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from FilmNcDataset import CollisionDataset, ValidationDataset, PKBatchSampler
from FilmNcModel import SpeakerConditionedKWS 
from FilmNcMetrics import compute_eer

ES_PATH = "./speaker_embeddings"

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
def validate(model, val_dataset, device):
    model.eval()
    if len(val_dataset.words) < 2:
        return {"top1": 0.0, "mrr": 0.0, "eer": 1.0} 
    
    words = val_dataset.words
    split_idx = len(words) // 2
    rng = np.random.default_rng(42)
    shuffled = words.copy()
    rng.shuffle(shuffled)
    enrolled_words = shuffled[:split_idx]
    unseen_words = shuffled[split_idx:]
    
    devices = {}     
    positive_scores = []
    negative_scores = []
    top1_correct = 0
    reciprocal_ranks = []
    for word in enrolled_words:
        enroll_files, test_files = val_dataset.get_enrollment_and_test(word)        
        owner_id = val_dataset.get_owner_id(word)
        e_s_owner = val_dataset._load_embedding(os.path.join(ES_PATH, f"{owner_id}.pt"), 192).unsqueeze(0).to(device)        
        gamma_owner, beta_owner = model.film_gen(e_s_owner)
        
        enroll_embs = []
        for path in enroll_files:
            wav = val_dataset._prepare_audio(path).unsqueeze(0).to(device)
            emb = model.encoder(model.pcen(wav), gamma_owner, beta_owner)
            enroll_embs.append(emb)

        w_c = F.normalize(torch.cat(enroll_embs, dim=0).mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
        devices[word] = {
            "w_c": w_c,
            "gamma": gamma_owner,
            "beta": beta_owner,
            "test_files": test_files
        }

    device_words = list(devices.keys())
    num_devices = len(device_words)    
    if num_devices == 0:
        model.train()
        return {"top1": 0.0, "mrr": 0.0, "eer": 1.0}

    gammas = torch.cat([devices[w]["gamma"] for w in device_words], dim=0)
    betas = torch.cat([devices[w]["beta"] for w in device_words], dim=0)
    w_cs = torch.stack([devices[w]["w_c"] for w in device_words], dim=0)
    for true_word, device_state in devices.items():
        for path in device_state["test_files"]:
            wav = val_dataset._prepare_audio(path).unsqueeze(0).to(device)
            pcen_feat = model.pcen(wav) 
            pcen_batched = pcen_feat.repeat(num_devices, 1, 1) 

            q_embs = model.encoder(pcen_batched, gammas, betas) 
            q_embs = F.normalize(q_embs, p=2, dim=1)            
            sims = torch.sum(q_embs * w_cs, dim=1)           
            scores = [(device_words[i], sims[i].item()) for i in range(num_devices)]
            scores.sort(key=lambda x: x[1], reverse=True)

            ranked_words = [w for w, _ in scores]
            rank = ranked_words.index(true_word) + 1
            if rank == 1: top1_correct += 1
            reciprocal_ranks.append(1.0 / rank)            
            true_device_score = next(s for w, s in scores if w == true_word)
            positive_scores.append(true_device_score)

    for imposter_word in unseen_words:
        _, test_files = val_dataset.get_enrollment_and_test(imposter_word)
        for path in test_files:
            wav = val_dataset._prepare_audio(path).unsqueeze(0).to(device)
            pcen_feat = model.pcen(wav)            
            pcen_batched = pcen_feat.repeat(num_devices, 1, 1)
            q_embs = model.encoder(pcen_batched, gammas, betas)
            q_embs = F.normalize(q_embs, p=2, dim=1)
            sims = torch.sum(q_embs * w_cs, dim=1)            
            max_accept_prob = sims.max().item()
            negative_scores.append(max_accept_prob)

    open_set_scores = positive_scores + negative_scores
    open_set_labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    eer, _ = compute_eer(open_set_scores, open_set_labels)
    n_queries = max(len(positive_scores), 1)
    
    model.train()
    return {
        "top1": top1_correct / n_queries,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "eer": float(eer)
    }

def load_weights(model: SpeakerConditionedKWS, stage2_checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(stage2_checkpoint_path, map_location=device, weights_only=False)    
    pcen_state = checkpoint.get('pcen_state_dict', checkpoint)
    model.pcen.load_state_dict(pcen_state)
    print("PCEN weights fully loaded")
    resnet_state = checkpoint.get('tc_resnet_state_dict', checkpoint)
    missing_keys, unexpected_keys = model.encoder.load_state_dict(resnet_state, strict=False)
    print("TC-ResNet backbone weights loaded")
    
    if missing_keys:
        print(f"Warning: Missing keys in encoder backbone:\n{missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in Stage 2 checkpoint:\n{unexpected_keys}")
            
    return model

def configure_trainable_parameters(model):
    for param in model.encoder.conv1.parameters():
        param.requires_grad = False
    for param in model.encoder.bn1.parameters():
        param.requires_grad = False
    for param in model.encoder.layer1.parameters():
        param.requires_grad = False
    for param in model.encoder.layer2.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen (Layers 1 & 2): {frozen_params:,}")
    print(f"Trainable (Layer 3, FiLM, FC, PCEN): {trainable_params:,}")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = SpeakerConditionedKWS(sample_rate=16000, n_mels=80, embedding_dim=128).to(device)
    model = load_weights(model, args.stage2_ckpt, device)
    configure_trainable_parameters(model)

    dataset = CollisionDataset()
    k_samples = 4
    p_classes = args.batch_size // k_samples
    pk_sampler = PKBatchSampler(dataset, p_classes=p_classes, k_samples=k_samples)
    loader = DataLoader(
        dataset, 
        batch_sampler=pk_sampler,   
        num_workers=4,               
        prefetch_factor=2,          
        persistent_workers=True,    
        pin_memory=True
    ) 
    val_dataset = ValidationDataset("./tts_corpus_processed/val") 

    optimizer = AdamW([
        {"params": model.pcen.parameters(), "lr": 1e-4},
        {"params": model.encoder.layer3.parameters(), "lr": 1e-4},
        {"params": model.encoder.fc.parameters(), "lr": 1e-4},
        {"params": model.film_gen.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    supcon_loss_fn = SupConLoss(temperature=0.1).to(device)
    patience = 5
    patience_counter = 0
    best_val_score = -float('inf')
    os.makedirs(args.output_dir, exist_ok=True)

    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_supcon, total_adversarial = 0.0, 0.0, 0.0
        n_batches = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch in progress:
            audio_mix = batch["audio_mix"].to(device)
            clean_target = batch["clean_target"].to(device)   
            e_s_correct = batch["e_s_correct"].to(device)     
            e_s = batch["e_s"].to(device)
            word_labels = batch["word_label"].to(device)
            mask_correct = batch["is_correct_speaker"].to(device).bool()            
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                embeddings = model(audio_mix, e_s)
                with torch.no_grad():
                    dynamic_w_c = model(clean_target, e_s_correct)
                
                valid_embs = embeddings[mask_correct]
                valid_labels = word_labels[mask_correct]
                loss_supcon = torch.tensor(0.0).to(device)
                if valid_embs.shape[0] > 1:
                    loss_supcon = supcon_loss_fn(valid_embs, valid_labels)
                
                wrong_embs = embeddings[~mask_correct]
                wrong_w_c = dynamic_w_c[~mask_correct] 
                loss_adversarial = torch.tensor(0.0).to(device)
                if wrong_embs.shape[0] > 0:
                    sims = torch.sum(wrong_embs * wrong_w_c, dim=1)    
                    loss_adversarial = torch.relu(sims - 0.5).mean()
                    
                loss = loss_supcon + (0.5 * loss_adversarial)
            
            if loss.requires_grad:
                scaler.scale(loss).backward()                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            total_loss += loss.item()
            total_supcon += loss_supcon.item()
            total_adversarial += loss_adversarial.item()
            n_batches += 1
            progress.set_postfix({"L_Tot": f"{loss.item():.3f}", "L_Adv": f"{loss_adversarial.item():.3f}"})

        scheduler.step()
        metrics = validate(model, val_dataset, device)        
        val_score = (0.7 * metrics["top1"]) + (0.3 * metrics["mrr"]) - (0.5 * metrics["eer"])
        print(f"\nEpoch {epoch + 1} | Avg Train Loss: {total_loss/n_batches:.4f}")
        print(f"Validation -> Top-1: {metrics['top1']:.1%} | MRR: {metrics['mrr']:.4f} | EER: {metrics['eer']:.2%}")
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            ckpt_path = os.path.join(args.output_dir, "stage3_best_checkpoint.pth")
            torch.save({
                'epoch': epoch + 1,
                'tc_resnet_state_dict': model.encoder.state_dict(),
                'film_gen_state_dict': model.film_gen.state_dict(),
                'pcen_state_dict': model.pcen.state_dict(),
                'val_score': best_val_score,
            }, ckpt_path)
            print(f"Checkpoint saved! New best score: {best_val_score:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nNo improvement for {patience} epochs.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Train FiLM Modulated TC-ResNet")
    parser.add_argument("--stage2_ckpt", type=str, default="../tc_resnet/tc_resnet_output/best_checkpoint.pth", help="Path to best_checkpoint.pth")
    parser.add_argument("--output_dir", type=str, default="./stage3_output", help="Directory for Stage 3 checkpoints")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    args = parser.parse_args()
    train(args)