import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from TCResNet import KWSTrainer
from TCDataset import TTS_Triplet_Dataset
import torch.nn.functional as F

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()

def train(data_dir, noise_dir, epochs=50, batch_size=32, learning_rate=1e-3, device='cuda'):
    print(f"Loading Triplet Dataset from: {data_dir}")
    print(f"Using Noise Dataset from: {noise_dir}")
    
    dataset = TTS_Triplet_Dataset(
        data_dir=data_dir, 
        noise_dir=noise_dir,
        target_sr=16000,
        max_seconds=1.0,
        virtual_length=5000 
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    print("Initializing Fused PCEN + TC-ResNet Model...")
    model = KWSTrainer().to(device)
    model.train()
    criterion = CosineTripletLoss(margin=0.6)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Starting Training on {device.upper()} for {epochs} Epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (anchor, pos, neg) in enumerate(dataloader):
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            
            optimizer.zero_grad()
            embed_anchor = model(anchor)
            embed_pos = model(pos)
            embed_neg = model(neg)
            
            # Calculate distance loss
            loss = criterion(embed_anchor, embed_pos, embed_neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        # Epoch Summary
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")
        
        # Checkpointing
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join("checkpoints", f"tc_resnet_weights_ep{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved Checkpoint: {save_path}")

    # Save final model
    final_path = os.path.join("checkpoints", "tc_resnet_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete. Final weights saved to: {final_path}")

if __name__ == "__main__":
    CLEAN_TTS_DIR = "./tts_data"
    NOISE_DIR = "./noise_dataset"
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train(
        data_dir=CLEAN_TTS_DIR, 
        noise_dir=NOISE_DIR, 
        epochs=50, 
        batch_size=32, 
        device=target_device
    )