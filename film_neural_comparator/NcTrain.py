import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from FilmNcDataset import QuadStateDataset, ValidationDataset
from FilmNcModel import TargetKWS
from FilmNcMetrics import compute_eer

ES_PATH = "./speaker_embeddings"

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
            
            raw_logits = model.comparator(w_cs, q_embs).squeeze(-1)
            probs = torch.sigmoid(raw_logits)             
            scores = [(device_words[i], probs[i].item()) for i in range(num_devices)]
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
            raw_logits = model.comparator(w_cs, q_embs).squeeze(-1)
            probs = torch.sigmoid(raw_logits)
            max_accept_prob = probs.max().item()
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

def load_weights(model: TargetKWS, stage3_ckpt_path: str, device: torch.device):
    checkpoint = torch.load(stage3_ckpt_path, map_location=device, weights_only=False)    
    model.pcen.load_state_dict(checkpoint['pcen_state_dict'])
    model.encoder.load_state_dict(checkpoint['tc_resnet_state_dict'])
    model.film_gen.load_state_dict(checkpoint['film_gen_state_dict'])
    return model

def export(model: TargetKWS, output_dir: str, device: torch.device):
    model.eval()
    pcen_params = {
        "alpha": model.pcen.alpha.detach().cpu().numpy().tolist(),
        "delta": model.pcen.delta.detach().cpu().numpy().tolist(),
        "r": model.pcen.r.detach().cpu().numpy().tolist(),
        "s": model.pcen.s.detach().cpu().numpy().tolist(),
        "eps": model.pcen.eps
    }
    
    pcen_path = os.path.join(output_dir, "pcen_params.json")
    with open(pcen_path, "w") as f:
        json.dump(pcen_params, f, indent=4)
    print(f"PCEN Parameters saved -> {pcen_path}")
    dummy_es = torch.randn(1, 192).to(device)
    film_onnx_path = os.path.join(output_dir, "film_generator.onnx")
    torch.onnx.export(
        model.film_gen,
        dummy_es,
        film_onnx_path,
        input_names=["speaker_embedding"],
        output_names=["gamma", "beta"],
        opset_version=14,
        do_constant_folding=True
    )
    print(f"FiLM Generator exported -> {film_onnx_path}")

    dummy_pcen = torch.randn(1, 80, 151).to(device)     
    dummy_gamma = torch.randn(1, 192).to(device)
    dummy_beta = torch.randn(1, 192).to(device)
    encoder_onnx_path = os.path.join(output_dir, "conditioned_encoder.onnx")
    torch.onnx.export(
        model.encoder,
        (dummy_pcen, dummy_gamma, dummy_beta),
        encoder_onnx_path,
        input_names=["pcen_features", "gamma", "beta"],
        output_names=["word_embedding"],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes={
            "pcen_features": {2: "time_frames"} 
        }
    )
    print(f"Conditioned Encoder exported -> {encoder_onnx_path}")

    dummy_wc = torch.randn(1, 128).to(device)
    dummy_query = torch.randn(1, 128).to(device)    
    comparator_onnx_path = os.path.join(output_dir, "neural_comparator.onnx")
    torch.onnx.export(
        model.comparator,
        (dummy_wc, dummy_query),
        comparator_onnx_path,
        input_names=["target_template", "query_embedding"],
        output_names=["p_accept"],
        opset_version=14,
        do_constant_folding=True
    )
    print(f"Neural Comparator exported -> {comparator_onnx_path}")
    print("All deployment artifacts generated successfully!\n")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = TargetKWS().to(device)
    model = load_weights(model, args.stage3_ckpt, device)
    for param in model.encoder.conv1.parameters(): param.requires_grad = False
    for param in model.encoder.bn1.parameters(): param.requires_grad = False
    for param in model.encoder.layer1.parameters(): param.requires_grad = False
    for param in model.encoder.layer2.parameters(): param.requires_grad = False

    dataset = QuadStateDataset(k_enroll=3)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,               
        prefetch_factor=2,          
        persistent_workers=True,    
        pin_memory=True,             
        drop_last=True
    )
    val_dataset = ValidationDataset("./tts_corpus_processed/val")

    optimizer = AdamW([
        {"params": model.comparator.parameters(), "lr": 1e-3},
        {"params": model.film_gen.parameters(), "lr": 1e-5},
        {"params": model.encoder.layer3.parameters(), "lr": 1e-5},
        {"params": model.encoder.fc.parameters(), "lr": 1e-5},
        {"params": model.pcen.parameters(), "lr": 1e-6},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    margin_loss_fn = nn.MarginRankingLoss(margin=0.4) 
    patience = 5
    patience_counter = 0
    best_val_score = -float('inf')
    os.makedirs(args.output_dir, exist_ok=True)

    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_bce, total_metric = 0.0, 0.0, 0.0
        n_batches = 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            enroll_audio = batch["enroll_audio"].to(device) 
            query_audio = batch["query_audio"].to(device)   
            e_s = batch["e_s"].to(device)                  
            labels = batch["labels"].to(device)         
        
            B, K, T = enroll_audio.shape
            _, Q, _ = query_audio.shape
            optimizer.zero_grad()     

            with torch.amp.autocast('cuda'):
                gamma, beta = model.film_gen(e_s)
                flat_enroll = enroll_audio.view(B * K, T)
                enroll_gamma = gamma.repeat_interleave(K, dim=0)
                enroll_beta = beta.repeat_interleave(K, dim=0)
                enroll_embs = model.encoder(model.pcen(flat_enroll), enroll_gamma, enroll_beta)
                enroll_embs = enroll_embs.view(B, K, -1)
                w_c = F.normalize(enroll_embs.mean(dim=1), p=2, dim=1) 
                
                flat_query = query_audio.view(B * Q, T)
                query_gamma = gamma.repeat_interleave(Q, dim=0)
                query_beta = beta.repeat_interleave(Q, dim=0)
                query_embs = model.encoder(model.pcen(flat_query), query_gamma, query_beta)
                query_embs = query_embs.view(B, Q, -1) 
                
                w_c_expanded = w_c.unsqueeze(1).repeat(1, Q, 1) 
                p_accept = model.comparator(w_c_expanded, query_embs).squeeze(-1) 
                loss_bce = bce_loss_fn(p_accept.view(-1), labels.view(-1))
                sims = torch.sum(query_embs * w_c_expanded, dim=-1).view(-1)
                margin_targets = torch.where(labels.view(-1) == 1.0, 1.0, -1.0)
                loss_metric = margin_loss_fn(sims, torch.zeros_like(sims), margin_targets)            
                loss = loss_bce + (0.2 * loss_metric)
            
            scaler.scale(loss).backward()            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_metric += loss_metric.item()
            n_batches += 1
            
            progress.set_postfix({"BCE": f"{loss_bce.item():.3f}", "Met": f"{loss_metric.item():.3f}"})

        scheduler.step()
        metrics = validate(model, val_dataset, device)
        val_score = (0.7 * metrics["top1"]) + (0.3 * metrics["mrr"]) - (0.5 * metrics["eer"])
        print(f"\nEpoch {epoch + 1} | BCE: {total_bce/n_batches:.4f} | Metric: {total_metric/n_batches:.4f}")
        print(f"Validation -> Top-1: {metrics['top1']:.1%} | MRR: {metrics['mrr']:.4f} | EER: {metrics['eer']:.2%}")
        
        if val_score > best_val_score:
            best_val_score = val_score
            ckpt_path = os.path.join(args.output_dir, "stage4_best_checkpoint.pth")
            torch.save({
                'epoch': epoch + 1,
                'tc_resnet_state_dict': model.encoder.state_dict(),
                'film_gen_state_dict': model.film_gen.state_dict(),
                'pcen_state_dict': model.pcen.state_dict(),
                'comparator_state_dict': model.comparator.state_dict(),
                'val_score': best_val_score,
            }, ckpt_path)
            print(f"DEPLOYMENT ARTIFACT SAVED: New best score {best_val_score:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nNo improvement for {patience} epochs.")
                break

    best_ckpt_path = os.path.join(args.output_dir, "stage4_best_checkpoint.pth")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint['tc_resnet_state_dict'])
    model.film_gen.load_state_dict(checkpoint['film_gen_state_dict'])
    model.pcen.load_state_dict(checkpoint['pcen_state_dict'])
    model.comparator.load_state_dict(checkpoint['comparator_state_dict'])
    export(model, args.output_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: End-to-End Fine Tuning")
    parser.add_argument("--stage3_ckpt", type=str, default="./stage3_output/stage3_best_checkpoint.pth")
    parser.add_argument("--output_dir", type=str, default="./stage4_output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args)