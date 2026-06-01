import json
import math
import torch
import torch.nn as nn
import torchaudio

class LearnablePCEN(nn.Module):
    """
    Differentiable PCEN frontend for domain adaptation.

    Pipeline:
        waveform → STFT → mel filterbank → IIR smoother → PCEN formula

    STFT params (fixed, not learnable):
        window_size = 25ms → win_length = 400 samples @ 16kHz
        hop_length  = 10ms → hop_length = 160 samples @ 16kHz
        n_fft       = 512

    Mel filterbank (fixed):
        n_mels = 80, fmin = 80Hz, fmax = 7600Hz

    Learnable PCEN parameters (initialised per spec):
        s     = 0.04   (IIR smoothing coefficient)
        alpha = 0.98   (AGC strength)
        delta = 2.0    (stabilizing bias)
        r     = 0.5    (dynamic range compression exponent)
    """

    def __init__(self, sample_rate=16000, n_mels=80, n_fft=512,
                 win_length=400, hop_length=160, fmin=80.0, fmax=7600.0,
                 s_init=0.04, alpha_init=0.98, delta_init=2.0, r_init=0.5,
                 eps=1e-6):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.eps = eps

        # --- Fixed mel filterbank (not learnable) ---
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=2.0,          # energy (magnitude squared)
            center=True,
            pad_mode='reflect',
            norm='slaney',
            mel_scale='slaney',
        )

        # --- Learnable PCEN parameters ---
        # These are optimised during Context A fine-tuning and frozen for Context B.
        # We use log-space initialization for unbounded optimization.
        self.log_s = nn.Parameter(torch.log(torch.tensor(s_init, dtype=torch.float32)))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init, dtype=torch.float32)))
        self.log_delta = nn.Parameter(torch.log(torch.tensor(delta_init, dtype=torch.float32)))
        self.log_r = nn.Parameter(torch.log(torch.tensor(r_init, dtype=torch.float32)))

    @property
    def s(self):
        """IIR smoothing coefficient, constrained to (0, 1) via sigmoid."""
        return torch.sigmoid(self.log_s)

    @property
    def alpha(self):
        """AGC strength, constrained to (0, +inf) via exp."""
        return torch.exp(self.log_alpha)

    @property
    def delta(self):
        """Stabilizing bias, constrained to (0, +inf) via exp."""
        return torch.exp(self.log_delta)

    @property
    def r(self):
        """Compression exponent, constrained to (0, +inf) via exp."""
        return torch.exp(self.log_r)

    def forward(self, waveform):
        """
        Full PCEN pipeline: waveform → PCEN features.

        Args:
            waveform: torch.Tensor of shape [batch, time] (raw 16kHz audio)

        Returns:
            pcen_out: torch.Tensor of shape [batch, n_mels, time_frames]
        """
        # Step 1a–1b: STFT + mel filterbank → E[batch, n_mels, T]
        E = self.mel_spectrogram(waveform)

        # Step 1c: IIR smoother per mel channel
        # M[m, t] = (1 - s) * M[m, t-1] + s * E[m, t]
        # M[m, 0] = E[m, 0]
        s = self.s
        
        # Pre-allocate M to avoid massive graph fragmentation from python list appending.
        # We must clone E to avoid in-place modification errors during backprop.
        M = E.clone() 
        
        # Sequential temporal loop (differentiable)
        for t in range(1, E.shape[2]):
            M[:, :, t] = (1.0 - s) * M[:, :, t - 1].clone() + s * E[:, :, t]

        # Step 1d: PCEN formula
        # PCEN[m, t] = (E[m,t] / (ε + M[m,t])^α + δ)^r − δ^r
        alpha = self.alpha
        delta = self.delta
        r = self.r

        agc = E / (self.eps + M).pow(alpha)
        pcen_out = (agc + delta).pow(r) - delta.pow(r)

        return pcen_out

    def export_params(self):
        """
        Exports learned PCEN parameters as a plain dict for JSON serialisation.
        """
        return {
            "s": float(self.s.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "delta": float(self.delta.detach().cpu()),
            "r": float(self.r.detach().cpu()),
        }

    def save_params(self, path):
        """Saves PCEN parameters to a JSON sidecar file."""
        params = self.export_params()
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        return params

    @staticmethod
    def load_params(path):
        """Loads PCEN parameters from a JSON sidecar file."""
        with open(path, 'r') as f:
            return json.load(f)


# NumPy PCEN for Context B (enrollment / inference on edge)
# Uses frozen parameters from the JSON sidecar — NOT differentiable.

import numpy as np
import librosa


class PCENProcessor:
    """
    NumPy PCEN processor for Context B enrollment and edge inference.

    Applies the identical PCEN algorithm as LearnablePCEN but using frozen
    parameters from the JSON sidecar produced during Context A training.
    """

    def __init__(self, pcen_params, sample_rate=16000, n_mels=80, n_fft=512,
                 hop_length=160, win_length=400, fmin=80.0, fmax=7600.0, eps=1e-6):
        """
        Args:
            pcen_params: dict with keys {s, alpha, delta, r} from the JSON sidecar.
        """
        self.s = pcen_params['s']
        self.alpha = pcen_params['alpha']
        self.delta = pcen_params['delta']
        self.r = pcen_params['r']

        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def process(self, audio_waveform):
        """
        Full PCEN pipeline: waveform → PCEN features.
        """
        # Step 1a-1b: STFT + mel filterbank
        # NOTE: Using htk=False to match torchaudio's mel_scale='slaney'
        E = librosa.feature.melspectrogram(
            y=audio_waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
            center=True,
            htk=False, 
            norm='slaney'
        )

        # Step 1c: IIR smoother
        M = np.zeros_like(E)
        M[:, 0] = E[:, 0]
        for t in range(1, E.shape[1]):
            M[:, t] = (1.0 - self.s) * M[:, t - 1] + self.s * E[:, t]

        # Step 1d: PCEN formula
        agc = E / (self.eps + M) ** self.alpha
        pcen_out = (agc + self.delta) ** self.r - self.delta ** self.r

        return pcen_out