import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import onnxruntime as ort

class PCENFrontend(nn.Module):
    def __init__(self, pcen_json_path, n_mels=80):
        super().__init__()
        with open(pcen_json_path, 'r') as f:
            params = json.load(f)
            
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels, n_fft=400, hop_length=160)
        self.s = float(np.asarray(params["s"]).item())
        self.alpha = float(np.asarray(params["alpha"]).item())
        self.delta = float(np.asarray(params["delta"]).item())
        self.r = float(np.asarray(params["r"]).item())
        self.eps = float(params.get("eps", 1e-6))

    def forward(self, wav):
        mel = self.mel_transform(wav)
        M = torch.zeros_like(mel)
        M[:, :, 0] = mel[:, :, 0]
        for t in range(1, mel.shape[2]):
            M[:, :, t] = (1 - self.s) * M[:, :, t - 1] + self.s * mel[:, :, t]
        return (mel / (self.eps + M)**self.alpha + self.delta)**self.r - self.delta**self.r

class SCTKWSPipeline:
    def __init__(self, model_dir="./models", mode="inference"):
        self.speaker_pcen = PCENFrontend(os.path.join(model_dir, "ecapa_pcen.json"))
        self.keyword_pcen = PCENFrontend(os.path.join(model_dir, "kws_pcen.json"))
        providers = ['CPUExecutionProvider']
        
        self.encoder_session = ort.InferenceSession(os.path.join(model_dir, "conditioned_encoder.onnx"), providers=providers)
        self.comparator_session = ort.InferenceSession(os.path.join(model_dir, "neural_comparator.onnx"), providers=providers)
        
        if mode == "enroll":
            self.ecapa_session = ort.InferenceSession(os.path.join(model_dir, "ecapa_tdnn.onnx"), providers=providers)
            self.film_session = ort.InferenceSession(os.path.join(model_dir, "film_generator.onnx"), providers=providers)

    def preprocess_audio(self, wav_path, target_len):
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.shape[1] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        else:
            wav = wav[:, :target_len]

        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.9

        return wav
    
    def preprocess_speaker_audio(self, wav_path):
        return self.preprocess_audio(wav_path, target_len=48000)

    def preprocess_keyword_audio(self, wav_path):
        return self.preprocess_audio(wav_path, target_len=24000)

    def extract_speaker_pcen(self, wav):
        with torch.no_grad():
            feat = self.speaker_pcen(wav)
            if feat.ndim == 4:
                feat = feat.squeeze(1)

            return feat.cpu().numpy().astype(np.float32)

    def extract_keyword_pcen(self, wav):
        with torch.no_grad():
            feat = self.keyword_pcen(wav)
            if feat.ndim == 4:
                feat = feat.squeeze(1)

            return feat.cpu().numpy().astype(np.float32)

    def get_speaker_embedding(self, pcen_features):
        e_s = self.ecapa_session.run(["embedding"], {"pcen_features": pcen_features})[0]
        return e_s / np.linalg.norm(e_s, ord=2, axis=1, keepdims=True)

    def generate_film_weights(self, speaker_embedding):
        gamma, beta = self.film_session.run(["gamma", "beta"], {"speaker_embedding": speaker_embedding})
        return gamma, beta

    def encode_word(self, pcen_features, gamma, beta):
        emb = self.encoder_session.run(["word_embedding"], {"pcen_features": pcen_features, "gamma": gamma, "beta": beta})[0]
        return emb / np.linalg.norm(emb, ord=2, axis=1, keepdims=True)

    def compare(self, w_c, query_emb):
        raw_logit = self.comparator_session.run(["p_accept"], {"target_template": w_c, "query_embedding": query_emb})[0]
        return 1.0 / (1.0 + np.exp(-raw_logit.item()))