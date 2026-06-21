import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pcen import LearnablePCEN

class StandardTCResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        padding = 4 * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=9, 
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=9, 
            stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return self.relu(out + identity)


class DeepFiLMTCResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        padding = 4 * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=9, 
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=9, 
            stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        identity = self.downsample(x) if self.downsample is not None else x        
        out = self.conv1(x)
        out = self.bn1(out)
        out = (gamma1 * out) + beta1 
        out = self.relu(out)        
        out = self.conv2(out)
        out = self.bn2(out)
        out = (gamma2 * out) + beta2
        
        return self.relu(out + identity)

class FiLMGenerator(nn.Module):
    def __init__(self, speaker_dim=192, layer3_channels=96):
        super().__init__()
        self.params_per_pool = layer3_channels * 2 
        self.mlp = nn.Sequential(
            nn.Linear(speaker_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.params_per_pool * 2) 
        )

    def forward(self, speaker_embedding):
        raw_params = self.mlp(speaker_embedding)
        gamma_pool, beta_pool = torch.chunk(raw_params, 2, dim=1)        
        gamma_pool = 1.0 + 0.1 * torch.tanh(gamma_pool)
        beta_pool  = 0.1 * torch.tanh(beta_pool)
        
        return gamma_pool, beta_pool

class DeepConditionedTCResNetEncoder(nn.Module):
    def __init__(self, num_mels=80, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(num_mels, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)        
        self.layer1 = StandardTCResNetBlock(32, 48, stride=2, dilation=1)
        self.layer2 = StandardTCResNetBlock(48, 64, stride=2, dilation=2)        
        self.layer3 = DeepFiLMTCResNetBlock(64, 96, stride=2, dilation=4)        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, embedding_dim)

    def forward(self, pcen_features, gamma_pool, beta_pool):
        x = self.relu(self.bn1(self.conv1(pcen_features)))         
        x = self.layer1(x)
        x = self.layer2(x)
        g3_1 = gamma_pool[:, :96].unsqueeze(-1)
        g3_2 = gamma_pool[:, 96:].unsqueeze(-1)
        b3_1 = beta_pool[:, :96].unsqueeze(-1)
        b3_2 = beta_pool[:, 96:].unsqueeze(-1)        
        x = self.layer3(x, g3_1, b3_1, g3_2, b3_2)       
        x = self.gap(x).squeeze(2) 
        x = self.fc(x)     
        
        return F.normalize(x, p=2, dim=1)

class NeuralComparator(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, target_template, query_embedding):
        diff = torch.abs(target_template - query_embedding)
        combined = torch.cat([target_template, query_embedding, diff], dim=-1)
        return self.net(combined)

class SpeakerConditionedKWS(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, embedding_dim=128):
        super().__init__()         
        self.pcen = LearnablePCEN(sample_rate=sample_rate, n_mels=n_mels)
        self.film_gen = FiLMGenerator(speaker_dim=192, layer3_channels=96)
        self.encoder = DeepConditionedTCResNetEncoder(num_mels=n_mels, embedding_dim=embedding_dim)

    def forward(self, audio_mix, speaker_embedding):
        mel_features = self.pcen(audio_mix)        
        gamma, beta = self.film_gen(speaker_embedding)        
        final_embedding = self.encoder(mel_features, gamma, beta)
        
        return final_embedding

class TargetKWS(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, embedding_dim=128):
        super().__init__()        
        self.pcen = LearnablePCEN(sample_rate=sample_rate, n_mels=n_mels)
        self.film_gen = FiLMGenerator(speaker_dim=192, layer3_channels=96)
        self.encoder = DeepConditionedTCResNetEncoder(num_mels=n_mels, embedding_dim=embedding_dim)
        self.comparator = NeuralComparator(embedding_dim=embedding_dim)