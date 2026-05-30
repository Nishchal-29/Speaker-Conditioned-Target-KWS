# import os
# import sys
# import types
# import math

# # --- BUG FIX FOR WINDOWS / PYTORCH 2.x ---
# os.environ["TORCH_DYNAMO_DISABLE"] = "1"

# # Fake modules to bypass SpeechBrain's NLP/Text lazy-loaders
# sys.modules['k2'] = types.ModuleType('k2')
# sys.modules['flair'] = types.ModuleType('flair')
# sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = types.ModuleType('fake_flair_emb')
# sys.modules['speechbrain.integrations.nlp'] = types.ModuleType('fake_nlp')
# # -----------------------------------------

# import torch
# # The ultimate kill-switch to stop PyTorch from deep-scanning
# torch._dynamo.config.disable = True 

# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from speechbrain.inference.speaker import EncoderClassifier

# # ---> IMPORT YOUR CUSTOM DATASET HERE <---
# from dataset import NoisySpeakerPCENDataset


# # ==========================================
# # AAM-SOFTMAX (ArcFace) IMPLEMENTATION
# # ==========================================
# class AAMSoftmax(nn.Module):
#     def __init__(self, in_features, out_features, s=30.0, m=0.2):
#         """
#         s: Hypersphere radius scaling factor
#         m: Angular margin penalty
#         """
#         super(AAMSoftmax, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
        
#         # The weight matrix represents the "center" of each speaker's cluster
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
        
#         # Pre-calculate trigonometric constants for the forward pass
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, embeddings, labels):
#         # 1. L2 Normalize the embeddings (e_s) and the weights
#         cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        
#         # 2. Calculate sine to apply the margin addition using trig identities
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
#         # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
#         phi = cosine * self.cos_m - sine * self.sin_m
        
#         # Numerical stability check
#         phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
#         # 3. Create one-hot labels to apply the margin ONLY to the target class
#         one_hot = torch.zeros(cosine.size(), device=embeddings.device)
#         one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
#         # 4. Apply the margin to the target class, keep standard cosine for others
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
#         # 5. Scale by the radius (s)
#         output *= self.s
        
#         return output
# # ==========================================


# class ECAPAFineTuner:
#     def __init__(self, data_dir, output_dir="./models", num_epochs=10, batch_size=16, lr_encoder=1e-5, lr_head=1e-3):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.output_dir = output_dir
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.lr_encoder = lr_encoder
#         self.lr_head = lr_head
        
#         os.makedirs(output_dir, exist_ok=True)
        
#         print("Initializing Noisy PCEN Dataset...")
#         self.train_dataset = NoisySpeakerPCENDataset(data_dir=data_dir, target_snr=-5)
#         self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
#         self.num_classes = len(self.train_dataset.speakers)
        
#         self._setup_model()
        
#     def _setup_model(self):
#         print("\n[1/2] Downloading / Loading pre-trained ECAPA-TDNN from SpeechBrain...")
#         classifier = EncoderClassifier.from_hparams(
#             source="speechbrain/spkrec-ecapa-voxceleb", 
#             savedir=os.path.join(self.output_dir, "speechbrain_cache"),
#             run_opts={"device": str(self.device)}
#         )
        
#         self.encoder = classifier.mods.embedding_model.to(self.device)
#         print(f"[2/2] Initializing AAM-Softmax head for {self.num_classes} unique speakers.")
        
#         # REPLACE standard Linear with AAM-Softmax (192-D vector from ECAPA)
#         self.classification_head = AAMSoftmax(in_features=192, out_features=self.num_classes, s=30.0, m=0.2).to(self.device)
        
#         self.criterion = nn.CrossEntropyLoss()
        
#         self.optimizer = Adam([
#             {'params': self.encoder.parameters(), 'lr': self.lr_encoder},
#             {'params': self.classification_head.parameters(), 'lr': self.lr_head}
#         ], foreach=False) # Keep foreach=False to prevent PyTorch 2.x dynamo optimizer crash

#     def train(self):
#         print(f"\nStarting Fine-Tuning with AAM-Softmax on {self.device} for {self.num_epochs} epochs...")
        
#         for epoch in range(self.num_epochs):
#             self.encoder.train()
#             self.classification_head.train()
            
#             total_loss = 0
#             correct_preds = 0
#             total_preds = 0
            
#             progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
#             for pcen_features, labels in progress_bar:
#                 pcen_features = pcen_features.to(self.device)
#                 labels = labels.to(self.device)
                
#                 self.optimizer.zero_grad()
                
#                 # --- CRITICAL FIX: Utterance-Level Normalization ---
#                 # pcen_features shape is (Batch, Time, Channels)
#                 # We normalize across the Time dimension (dim=1) so mean=0, std=1
#                 mean = pcen_features.mean(dim=1, keepdim=True)
#                 std = pcen_features.std(dim=1, keepdim=True)
#                 pcen_features = (pcen_features - mean) / (std + 1e-5)
#                 # ---------------------------------------------------
                
#                 # 1. Extract Biometric Embeddings (e_s)
#                 embeddings = self.encoder(pcen_features)
#                 embeddings = embeddings.squeeze(1) 
                
#                 # 2. Pass embeddings AND labels to the AAM-Softmax head
#                 # (AAM needs the labels to know which angle to apply the margin to)
#                 outputs = self.classification_head(embeddings, labels)
                
#                 # 3. Calculate Loss (L_AAM) and Backpropagate
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_preds += labels.size(0)
#                 correct_preds += (predicted == labels).sum().item()
                
#                 running_acc = 100 * correct_preds / total_preds
#                 progress_bar.set_postfix({
#                     "Loss": f"{loss.item():.4f}", 
#                     "Train_Acc": f"{running_acc:.2f}%"
#                 })
            
#             self.save_checkpoint(epoch + 1, running_acc)
            
#         print("\nFine-tuning session successfully completed!")

#     def save_checkpoint(self, epoch, accuracy):
#         checkpoint_path = os.path.join(self.output_dir, f"ecapa_pcen_epoch_{epoch}.pth")
#         torch.save({
#             'epoch': epoch,
#             'encoder_state_dict': self.encoder.state_dict(),
#             'head_state_dict': self.classification_head.state_dict(),
#             'accuracy': accuracy,
#             'spk_to_id': self.train_dataset.spk_to_id
#         }, checkpoint_path)
#         print(f" -> Checkpoint saved to: {checkpoint_path}")


# if __name__ == "__main__":
#     # Point this to your local LibriSpeech directory
#     DATASET_DIR = "./data/Librispeech/dev-clean"
    
#     # Initialize the trainer
#     trainer = ECAPAFineTuner(
#         data_dir=DATASET_DIR,
#         output_dir="./finetuned_models",
#         num_epochs=10,
#         batch_size=16,      
#         lr_encoder=1e-5, 
#         lr_head=1e-3     
#     )
    
#     trainer.train()

import os
import sys
import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier

# --- BUG FIX FOR WINDOWS / PYTORCH 2.x ---
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
sys.modules['k2'] = types.ModuleType('k2')
sys.modules['flair'] = types.ModuleType('flair')
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = types.ModuleType('fake_flair_emb')
sys.modules['speechbrain.integrations.nlp'] = types.ModuleType('fake_nlp')
sys.modules['speechbrain.integrations.huggingface.wordemb'] = types.ModuleType('fake_wordemb')
sys.modules['speechbrain.integrations.huggingface'] = types.ModuleType('fake_hf')
torch._dynamo.config.disable = True 
# -----------------------------------------

# Import the updated raw waveform dataset
from dataset import NoisySpeakerRawDataset


# ==========================================
# AAM-SOFTMAX (ArcFace) IMPLEMENTATION
# ==========================================
class AAMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.2):
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
        # Replace your sine calculation with this safe version:
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        # Return BOTH the penalized output for Loss, and raw cosine for Accuracy
        return output, cosine
# ==========================================


class ECAPAFineTuner:
    def __init__(self, data_dir, output_dir="./models", num_epochs=15, batch_size=16, lr_encoder=1e-5, lr_head=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr_encoder = lr_encoder
        self.lr_head = lr_head
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing Raw Audio Dataset...")
        self.train_dataset = NoisySpeakerRawDataset(data_dir=data_dir, max_audio_length=48000)
        
        # --- DATALOADER UPDATE: Multi-threading for faster CPU prep ---
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=4,      # Speeds up batch creation (Adjust if needed)
            pin_memory=True     # Speeds up CPU-to-GPU memory transfer
        )
        self.num_classes = len(self.train_dataset.speakers)
        
        self._setup_model()
        
    def _setup_model(self):
        print("\n[1/2] Loading pre-trained ECAPA-TDNN from SpeechBrain...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir=os.path.join(self.output_dir, "speechbrain_cache"),
            run_opts={"device": str(self.device)}
        )
        
        self.compute_features = self.classifier.mods.compute_features.to(self.device)
        self.mean_var_norm = self.classifier.mods.mean_var_norm.to(self.device)
        self.encoder = self.classifier.mods.embedding_model.to(self.device)

        print(f"[2/2] Initializing AAM-Softmax head for {self.num_classes} unique speakers.")
        
        # --- FIX: Calibrated s and m for Noisy Environments ---
        self.classification_head = AAMSoftmax(
            in_features=192, 
            out_features=self.num_classes, 
            s=40.0,  # Increased to prevent gradient vanishing
            m=0.30   # Increased to force harder separation boundaries
        ).to(self.device)
        # ------------------------------------------------------
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = Adam([
            {'params': self.encoder.parameters(), 'lr': self.lr_encoder},
            {'params': self.classification_head.parameters(), 'lr': self.lr_head}
        ], foreach=False)

    def train(self):
        print(f"\nStarting Fine-Tuning with AAM-Softmax on {self.device}...")
        
        # --- FIX: Define the Freezing Schedule ---
        FREEZE_EPOCHS = 3 
        
        for epoch in range(self.num_epochs):
            
            # --- DYNAMIC FREEZING LOGIC ---
            if epoch < FREEZE_EPOCHS:
                # Stage 1: Lock the backbone. Train ONLY the AAM-Softmax head.
                if epoch == 0: print("\n[INFO] Backbone is LOCKED. Orienting AAM-Softmax head.")
                for param in self.encoder.parameters():
                    param.requires_grad = False
            elif epoch == FREEZE_EPOCHS:
                # Stage 2: Unlock the backbone. Begin joint fine-tuning.
                print("\n[INFO] Backbone UNLOCKED. Joint training initiated.")
                for param in self.encoder.parameters():
                    param.requires_grad = True
            # ------------------------------

            self.encoder.train()
            self.classification_head.train()
            
            total_loss = 0
            correct_preds = 0
            total_preds = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for raw_waveforms, labels in progress_bar:
                raw_waveforms = raw_waveforms.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 1. Extract features safely WITHOUT tracking gradients
                with torch.no_grad():
                    wav_lens = torch.ones(raw_waveforms.shape[0]).to(self.device) 
                    features = self.compute_features(raw_waveforms)
                    features = self.mean_var_norm(features, wav_lens)

                # 2. Pass to encoder WITH gradients enabled
                embeddings = self.encoder(features)
                embeddings = embeddings.squeeze(1)
                
                outputs, raw_cosines = self.classification_head(embeddings, labels)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(raw_cosines.data, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()
                
                running_acc = 100 * correct_preds / total_preds
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "Train_Acc": f"{running_acc:.2f}%"
                })
            
            self.save_checkpoint(epoch + 1, running_acc)
            
        print("\nFine-tuning session successfully completed!")

    def save_checkpoint(self, epoch, accuracy):
        checkpoint_path = os.path.join(self.output_dir, f"ecapa_raw_epoch_{epoch}.pth")
        
        # --- CHECKPOINT UPDATE: Restored head weights and speaker mappings ---
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'head_state_dict': self.classification_head.state_dict(),
            'accuracy': accuracy,
            'spk_to_id': self.train_dataset.spk_to_id 
        }, checkpoint_path)
        print(f" -> Checkpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    DATASET_DIR = "./data/LibriSpeech/dev-clean"
    
    trainer = ECAPAFineTuner(
        data_dir=DATASET_DIR,
        output_dir="./finetuned_models",
        num_epochs=10,         
        batch_size=16,      
        lr_encoder=1e-5, 
        lr_head=1e-3     
    )
    
    trainer.train()