import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pcen import LearnablePCEN
from TCResNet import TCResNetAcousticEncoder
from FilmNcDataset import QuadStateDataset

SAMPLE_RATE = 16000
N_MELS = 80
EMBEDDING_DIM = 128


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class FiLM(nn.Module):
    def __init__(self, speaker_dim: int = 192, target_channels: int = 128):
        super().__init__()
        self.target_channels = target_channels
        self.fc = nn.Sequential(
            nn.Linear(speaker_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, target_channels * 2),
        )

    def forward(self, e_s: torch.Tensor):
        out = self.fc(e_s)
        gamma = out[:, :self.target_channels] + 1.0 
        beta = out[:, self.target_channels:]
        return gamma, beta


class NeuralComparator(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, w_c, w_live):
        diff = torch.abs(w_c - w_live)
        mul = w_c * w_live
        vec = torch.cat([w_c, w_live, diff, mul], dim=1)
        return self.net(vec).squeeze(-1)


class Validator(nn.Module):
    def __init__(self, pcen, tc_resnet_backbone):
        super().__init__()
        
        self.pcen = pcen
        for p in self.pcen.parameters():
            p.requires_grad = False
        
        self.film = FiLM(speaker_dim=192, target_channels=128)
        self.projector = nn.Linear(128, 128)
        
        torch.nn.init.eye_(self.projector.weight)
        torch.nn.init.zeros_(self.projector.bias)
        
        self.n_comparator = NeuralComparator(input_dim=512)
        self.backbone = tc_resnet_backbone

        for p in self.backbone.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        self.pcen.eval()
        self.backbone.eval()
        return self

    def forward(self, audio, w_c, e_s):
        with torch.no_grad():
            pcen_features = self.pcen(audio)
            
        wc_raw = self.backbone(pcen_features)
        
        wc_raw = self.projector(wc_raw)
        wc_raw = F.normalize(wc_raw, p=2, dim=-1)
        
        w_c_norm = F.normalize(w_c, p=2, dim=-1)
        
        gamma, beta = self.film(e_s)
        w_live = gamma * wc_raw + beta
        w_live = F.normalize(w_live, p=2, dim=-1)

        scores = self.n_comparator(w_c_norm, w_live)
        return scores


def train_validator():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Quad-State Training on {device}...")

    print("Loading PCEN and TC-ResNet Backbone from checkpoint...")
    model_path = "tc_resnet_output/best_checkpoint.pth"
    pcen_params_path = "tc_resnet_output/pcen_params.json"
    
    if not os.path.exists(model_path) or not os.path.exists(pcen_params_path):
        raise FileNotFoundError("Missing best_checkpoint.pth or pcen_params.json in tc_resnet_output/")
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    pcen_params = LearnablePCEN.load_params(pcen_params_path) 
    
    pcen = LearnablePCEN(sample_rate=SAMPLE_RATE, n_mels=N_MELS).to(device)
    pcen.load_state_dict(checkpoint['pcen_state_dict'])

    tc_resnet = TCResNetAcousticEncoder(num_mels=N_MELS, embedding_dim=EMBEDDING_DIM).to(device)
    tc_resnet.load_state_dict(checkpoint['tc_resnet_state_dict'])

    model = Validator(pcen=pcen, tc_resnet_backbone=tc_resnet).to(device)
    
    head_params = [p for p in model.film.parameters() if p.requires_grad] + \
                  [p for p in model.n_comparator.parameters() if p.requires_grad] + \
                  [p for p in model.projector.parameters() if p.requires_grad]

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    optimizer = optim.AdamW([
        {'params': head_params, 'lr': 5e-4},
        {'params': backbone_params, 'lr': 1e-5} 
    ], weight_decay=1e-5)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    print("Loading Quad-State Dataset...")
    dataset = QuadStateDataset(virtual_length=10000)
    num_workers = 4 if os.name != "nt" else 0
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    epochs = 30
    os.makedirs("./Film_checkpoints", exist_ok=True)
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    best_loss = float("inf")
    start_epoch = 0
    
    checkpoint_path = "./Film_checkpoints/best_validator.pth"
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}. Attempting to load...")
        checkpoint_val = torch.load(checkpoint_path, map_location=device)
        
        try:
            model.load_state_dict(checkpoint_val["validator_state_dict"], strict=True)
            start_epoch = checkpoint_val["epoch"]
            best_loss = checkpoint_val.get("avg_loss", float("inf"))
            print(f"Successfully resumed training from Epoch {start_epoch} | Best Loss: {best_loss:.4f}")
            
        except RuntimeError as e:
            print(f"\n[WARNING] Old architecture detected in checkpoint. It cannot be used with the new code.")
            print("Starting training fresh from Epoch 0 instead.\n")
            start_epoch = 0
            best_loss = float("inf")
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    state_weights = torch.tensor([1.0, 1.2, 1.2, 0.6], device=device)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            audio = batch["audio"].to(device, non_blocking=True)
            e_s = batch["e_s"].to(device, non_blocking=True)
            w_c = batch["w_c"].to(device, non_blocking=True)

            B, Q, T = audio.shape

            flat_audio = audio.reshape(B * Q, T)
            flat_w_c = w_c.unsqueeze(1).expand(-1, Q, -1).reshape(B * Q, -1)
            flat_e_s = e_s.unsqueeze(1).expand(-1, Q, -1).reshape(B * Q, -1)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                scores = model(flat_audio, flat_w_c, flat_e_s)
                scores = scores.view(B, Q)
                
                targets = torch.zeros_like(scores)
                targets[:, 0] = 1.0 
                
                raw_losses = criterion(scores, targets)
                weighted_losses = raw_losses * state_weights
                loss = weighted_losses.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head_params + backbone_params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Total Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"\n====> Epoch {epoch+1} Completed | Avg Total Loss: {avg_loss:.4f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "validator_state_dict": model.state_dict(),
                    "film_state_dict": model.film.state_dict(),
                    "comparator_state_dict": model.n_comparator.state_dict(),
                    "projector_state_dict": model.projector.state_dict(),
                    "tuned_backbone_state_dict": model.backbone.state_dict(),
                    "pcen_state_dict": model.pcen.state_dict()
                },
                "./Film_checkpoints/best_validator.pth"
            )
            print(f"Saved best model (loss={avg_loss:.4f})")

if __name__ == "__main__":
    train_validator()
