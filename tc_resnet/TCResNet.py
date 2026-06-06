import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

class SpecAugment(nn.Module):
    """
    Applies Frequency and Time masking to the spectrogram.
    Acts as a regularizer to prevent overfitting to specific TTS artifacts.
    """
    def __init__(self, freq_mask_param=15, time_mask_param=20):
        super(SpecAugment, self).__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, x):
        # Only apply masking during the training phase.
        # Bypassed automatically during model.eval() inference.
        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        return x

class learnablePCEN(nn.Module):
    def __init__(self, n_mels=80, sample_rate=16000, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.n_mels = n_mels
        win_length = 480
        hop_length = 160
        n_fft = 512
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=n_fft, 
            win_length=win_length,
            hop_length=hop_length, 
            n_mels=n_mels, 
            power=2.0
        )
        self.alpha_logit = nn.Parameter(torch.empty(1, n_mels, 1).fill_(math.log(0.98/(1-0.98))))
        self.log_delta = nn.Parameter(torch.empty(1, n_mels, 1).fill_(math.log(2.0)))
        self.r_logit = nn.Parameter(torch.empty(1, n_mels, 1).fill_(math.log(0.5/(1-0.5))))
        self.s_logit = nn.Parameter(torch.empty(1, n_mels, 1).fill_(math.log(0.25/(1-0.25))))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spectogram(waveforms)
        alpha = torch.sigmoid(self.alpha_logit)
        delta = torch.exp(self.log_delta)
        r = torch.sigmoid(self.r_logit)
        s = torch.sigmoid(self.s_logit)
        batch_size, n_mels, time_steps = mel_spec.shape
        s_2d = s.squeeze(-1) 
        M = [mel_spec[:, :, 0]]
        for i in range(1, time_steps):
            m_t = (1 - s_2d) * M[-1] + s_2d * mel_spec[:, :, i]
            M.append(m_t)
        M = torch.stack(M, dim=-1)
        smoth_denom = (self.eps + M) ** alpha
        pcen = ((mel_spec + self.eps) / (smoth_denom + delta)) ** r - delta ** r
        return pcen

class TCResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TCResNetBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        return F.relu(out)

class TCResNet8(nn.Module):
    """
    TC-ResNet8 architecture treating frequency bins as channels.
    Outputs a 128-D L2-normalized embedding for Triplet Loss.
    """
    def __init__(self, input_channels=80, embedding_dim=128, width_multiplier=1.0):
        super(TCResNet8, self).__init__()
        c_init = int(16 * width_multiplier)
        c_b1   = int(24 * width_multiplier)
        c_b2   = int(32 * width_multiplier)
        c_b3   = int(48 * width_multiplier)
        self.conv_init = nn.Conv1d(input_channels, c_init, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_init = nn.BatchNorm1d(c_init)
        self.block1 = TCResNetBlock(c_init, c_b1, kernel_size=9, stride=2)
        self.block2 = TCResNetBlock(c_b1, c_b2, kernel_size=9, stride=2)
        self.block3 = TCResNetBlock(c_b2, c_b3, kernel_size=9, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_b3, embedding_dim, bias=False)
        
    def forward(self, x):
        out = F.relu(self.bn_init(self.conv_init(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.gap(out).squeeze(-1)
        out = self.fc(out)
        return F.normalize(out, p=2, dim=1)

class KWSTrainer(nn.Module):
    def __init__(self, width_multiplier=1.0):
        super(KWSTrainer, self).__init__()
        self.pcen = learnablePCEN(n_mels=80)
        self.spec_augment = SpecAugment(freq_mask_param=15, time_mask_param=20)
        self.resnet = TCResNet8(input_channels=80, embedding_dim=128, width_multiplier=width_multiplier)

    def forward(self, raw_audio):
        features = self.pcen(raw_audio)
        features = self.spec_augment(features)
        return self.resnet(features)