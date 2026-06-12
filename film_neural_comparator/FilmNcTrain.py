import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from TCResNet import KWSTrainer
from FilmNcDataset import QuadStateDataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_state_dict_flexible(model: nn.Module, path: str, map_location: str) -> None:
    obj = torch.load(path, map_location=map_location)

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint at {path} does not look like a state dict.")

    cleaned = {}
    for k, v in obj.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)


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
        gamma = out[:, :self.target_channels]
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
        vec = torch.cat([w_c, w_live, diff, mul],dim=1)
        return self.net(vec).squeeze(-1)

class Validator(nn.Module):
    def __init__(self, tc_resnet_backbone):
        super().__init__()

        self.film = FiLM(speaker_dim=192,target_channels=128)
        self.n_comparator = NeuralComparator(input_dim=512)
        self.backbone = tc_resnet_backbone

        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_audio(self, audio):
        self.backbone.eval()
        wc_raw = self.backbone(audio)
        wc_raw = F.normalize(wc_raw, p=2,dim=-1)

        return wc_raw

    def forward(self, audio, w_c, e_s):
        wc_raw = self.encode_audio(audio)
        gamma, beta = self.film(e_s)
        w_live = gamma * wc_raw + beta
        w_live = F.normalize(w_live,p=2,dim=-1)

        scores = self.n_comparator(w_c,w_live)

        return scores


def train_validator():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing Quad-State Training on {device}...")

    print("Loading TC-ResNet Backbone...")

    tc_resnet = KWSTrainer()

    load_state_dict_flexible(tc_resnet,"./checkpoints/tc_resnet_weights_ep50.pth",map_location=device)
    for param in tc_resnet.parameters():
        param.requires_grad = False
    tc_resnet.eval()
    model = Validator(tc_resnet_backbone=tc_resnet).to(device)
    trainable_params = [p for p in model.parameters()if p.requires_grad]
    print(f"Trainable parameters: "f"{sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.AdamW(trainable_params,lr=1e-4,weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    print("Loading Quad-State Dataset...")
    dataset = QuadStateDataset(virtual_length=10000)
    num_workers = 4 if os.name != "nt" else 0
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True,num_workers=num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=(num_workers > 0),)
    epochs = 30
    os.makedirs("./Film_checkpoints",exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        model.backbone.eval()
        total_loss = 0.0
        total_acc = 0.0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            audio = batch["audio"].to(device,non_blocking=True)
            e_s = batch["e_s"].to(device,non_blocking=True)
            w_c = batch["w_c"].to(device,non_blocking=True)

            B, Q, T = audio.shape
            flat_audio = audio.reshape(B * Q,T)

            flat_w_c = (w_c.unsqueeze(1).expand(-1, Q, -1).reshape(B * Q, -1))
            flat_e_s = (e_s.unsqueeze(1).expand(-1, Q, -1).reshape(B * Q, -1))

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                scores = model(flat_audio,flat_w_c,flat_e_s)
                scores = scores.view(B, Q)
                target = torch.zeros(B,dtype=torch.long,device=device)
                ce_loss = criterion(scores,target)
                positive_scores = scores[:, 0]
                negative_scores = scores[:, 1:]
                margin = 0.30
                margin_loss = F.relu(margin- positive_scores.unsqueeze(1)+ negative_scores).mean()
                loss = (ce_loss+ 0.5 * margin_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                pred_idx = scores.argmax(dim=1)
                acc = (pred_idx == target).float().mean()
                tp_score = (scores[:, 0].mean().item())
                neg_score = (scores[:, 1:].mean().item())
            total_loss += loss.item()
            total_acc += acc.item()
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "f"Batch [{batch_idx}/{len(dataloader)}] "f"Loss={loss.item():.4f} "f"CE={ce_loss.item():.4f} "f"Margin={margin_loss.item():.4f} "f"Acc={acc.item():.4f} "f"TP={tp_score:.3f} "f"NEG={neg_score:.3f}")

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        print(f"\n====> Epoch {epoch+1} Completed "f"| Avg Loss: {avg_loss:.4f} "f"| Avg Acc: {avg_acc:.4f}\n")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "avg_acc": avg_acc,
                    "validator_state_dict": model.state_dict(),
                    "film_state_dict": model.film.state_dict(),
                    "comparator_state_dict": model.n_comparator.state_dict(),
                },
                "./Film_checkpoints/best_validator.pth"
            )
            print(f"Saved best model "f"(loss={avg_loss:.4f})")
        if (epoch + 1) % 5 == 0:
            torch.save(model.film.state_dict(),f"./Film_checkpoints/film_ep{epoch+1}.pth")
            torch.save(model.n_comparator.state_dict(),f"./Film_checkpoints/comparator_ep{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "avg_acc": avg_acc,
                    "validator_state_dict": model.state_dict(),
                },
                f"./Film_checkpoints/validator_ep{epoch+1}.pth"
            )
            print(f"Checkpoint saved "f"(epoch {epoch+1})")

if __name__ == "__main__":
    train_validator()