import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pcen import LearnablePCEN

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=15, time_mask_param=20):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, x):
        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        return x

class TCResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        padding = 4 * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
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

class TCResNetAcousticEncoder(nn.Module):
    def __init__(self, num_mels=80, embedding_dim=128):
        super().__init__()        
        self.spec_augment = SpecAugment()
        self.conv1 = nn.Conv1d(num_mels, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = TCResNetBlock(32, 48, stride=2, dilation=1)
        self.layer2 = TCResNetBlock(48, 64, stride=2, dilation=2)
        self.layer3 = TCResNetBlock(64, 96, stride=2, dilation=4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, embedding_dim)

    def forward(self, pcen_features):
        x = self.spec_augment(pcen_features)
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 32, T]
        x = self.layer1(x)   # [B, 48, T/2]
        x = self.layer2(x)   # [B, 64, T/4]
        x = self.layer3(x)   # [B, 96, T/8]
        x = self.gap(x).squeeze(2)   # [B, 96]
        x = self.fc(x)   # [B, 128]
        return F.normalize(x, p=2, dim=1)

class KWSTrainer(nn.Module):
    def __init__(self, num_mels=80, embedding_dim=128):
        super().__init__()
        self.pcen = LearnablePCEN(n_mels=num_mels, sample_rate=16000)
        self.resnet = TCResNetAcousticEncoder(
            num_mels=num_mels, 
            embedding_dim=embedding_dim, 
        )

    def forward(self, raw_audio):
        features = self.pcen(raw_audio)
        embeddings = self.resnet(features)
        return embeddings