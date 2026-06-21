import os
import time
import json
import glob
import random
import argparse
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model import SCTKWSPipeline

def compute_security_metrics(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    far_target = 0.01
    idx = np.argmin(np.abs(fpr - far_target))
    tar_at_1_far = tpr[idx]
    
    threshold = 0.50
    pred = (np.array(scores) >= threshold).astype(int)
    tn = np.sum((pred == 0) & (np.array(labels) == 0))
    fp = np.sum((pred == 1) & (np.array(labels) == 0))
    fn = np.sum((pred == 0) & (np.array(labels) == 1))
    tp = np.sum((pred == 1) & (np.array(labels) == 1))
    
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return eer, far, frr, tar_at_1_far

class SC_TKWS_Benchmark:
    def __init__(self, model_dir, data_dir, ped_matrix_path=None):
        self.pipeline = SCTKWSPipeline(model_dir, mode="enroll")
        self.data_dir = data_dir
        
        self.word_speaker_files = defaultdict(lambda: defaultdict(list))
        for path in glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True):
            word = os.path.basename(os.path.dirname(path)).lower()
            spk = os.path.basename(path).split("_")[0]
            self.word_speaker_files[word][spk].append(path)
            
        self.words = list(self.word_speaker_files.keys())
        self.speakers = list({spk for w in self.words for spk in self.word_speaker_files[w].keys()})
        
        self.hard_negatives = {}
        if ped_matrix_path and os.path.exists(ped_matrix_path):
            with open(ped_matrix_path, 'r') as f:
                self.hard_negatives = json.load(f)

    def run_latency_benchmark(self, num_runs=100):
        dummy_spk_wav = torch.randn(1, 48000)
        dummy_kws_wav = torch.randn(1, 24000)
        times = {"spk_pcen": [], "ecapa": [], "film": [], "kws_pcen": [], "encoder": [], "comparator": [], "e2e": []}
        
        for _ in range(num_runs):
            t0 = time.perf_counter()
            spk_feat = self.pipeline.extract_speaker_pcen(dummy_spk_wav)
            times["spk_pcen"].append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            e_s = self.pipeline.get_speaker_embedding(spk_feat)
            times["ecapa"].append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            gamma, beta = self.pipeline.generate_film_weights(e_s)
            times["film"].append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            kws_feat = self.pipeline.extract_keyword_pcen(dummy_kws_wav)
            times["kws_pcen"].append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            q_emb = self.pipeline.encode_word(kws_feat, gamma, beta)
            times["encoder"].append(time.perf_counter() - t0)
            
            dummy_wc = np.random.randn(1, 128).astype(np.float32)
            t0 = time.perf_counter()
            self.pipeline.compare(dummy_wc, q_emb)
            times["comparator"].append(time.perf_counter() - t0)
                  
            t0 = time.perf_counter()
            feat = self.pipeline.extract_keyword_pcen(dummy_kws_wav)
            emb = self.pipeline.encode_word(feat, gamma, beta)
            self.pipeline.compare(dummy_wc, emb)
            times["e2e"].append(time.perf_counter() - t0)

        self.latency_metrics = {k: np.array(v) * 1000 for k, v in times.items()}
        self.throughput = 1000.0 / np.mean(self.latency_metrics["e2e"])

    def evaluate_n_shot(self, shots=[1, 3, 5]):
        self.n_shot_results = {}
        for k in shots:
            scores, labels = [], []
            for word in self.words[:10]:
                spks = list(self.word_speaker_files[word].keys())
                if not spks: continue
                owner = spks[0]
                files = self.word_speaker_files[word][owner]
                if len(files) < k + 1: continue
                
                enroll_files = random.sample(files, k)
                embeddings, pcen_features = [], []
                for f in enroll_files:
                    spk_wav = self.pipeline.preprocess_speaker_audio(f)
                    kws_wav = self.pipeline.preprocess_keyword_audio(f)
                    spk_feat = self.pipeline.extract_speaker_pcen(spk_wav)
                    embeddings.append(self.pipeline.get_speaker_embedding(spk_feat))
                    pcen_features.append(self.pipeline.extract_keyword_pcen(kws_wav))
                    
                e_s = np.mean(np.concatenate(embeddings, axis=0), axis=0, keepdims=True)
                e_s = e_s / np.linalg.norm(e_s, ord=2, axis=1, keepdims=True)
                gamma, beta = self.pipeline.generate_film_weights(e_s)
                
                word_embs = [self.pipeline.encode_word(feat, gamma, beta) for feat in pcen_features]
                w_c = np.mean(np.concatenate(word_embs, axis=0), axis=0, keepdims=True)
                w_c = w_c / np.linalg.norm(w_c, ord=2, axis=1, keepdims=True)
                
                test_pos_file = list(set(files) - set(enroll_files))[0]
                test_pos = self.pipeline.preprocess_keyword_audio(test_pos_file)
                emb_pos = self.pipeline.encode_word(self.pipeline.extract_keyword_pcen(test_pos), gamma, beta)
                scores.append(self.pipeline.compare(w_c, emb_pos))
                labels.append(1)
                
                imposter = spks[1] if len(spks) > 1 else self.speakers[-1]
                if imposter in self.word_speaker_files[word]:
                    test_neg_file = self.word_speaker_files[word][imposter][0]
                    test_neg = self.pipeline.preprocess_keyword_audio(test_neg_file)
                    emb_neg = self.pipeline.encode_word(self.pipeline.extract_keyword_pcen(test_neg), gamma, beta)
                    scores.append(self.pipeline.compare(w_c, emb_neg))
                    labels.append(0)
            
            if len(set(labels)) > 1:
                eer, _, _, _ = compute_security_metrics(labels, scores)
                self.n_shot_results[k] = eer
            else:
                self.n_shot_results[k] = float('nan')

    def run_full_suite(self):
        self.scores, self.labels = [], []
        self.reciprocal_ranks = []
        self.top1, self.top5, self.total_queries = 0, 0, 0
        self.phonetic_collisions = {"total": 0, "rejected": 0}
        self.speaker_collisions = {"total": 0, "rejected": 0}        
        vaults = {}
        for word in self.words:
            spks = list(self.word_speaker_files[word].keys())
            if not spks: continue
            owner = spks[0]
            files = self.word_speaker_files[word][owner]
            if len(files) < 4: continue
            
            enroll_files = files[:3]
            test_files = files[3:]           
            embeddings, pcen_features = [], []
            for f in enroll_files:
                spk_wav = self.pipeline.preprocess_speaker_audio(f)
                kws_wav = self.pipeline.preprocess_keyword_audio(f)
                embeddings.append(self.pipeline.get_speaker_embedding(self.pipeline.extract_speaker_pcen(spk_wav)))
                pcen_features.append(self.pipeline.extract_keyword_pcen(kws_wav))
            
            e_s = np.mean(np.concatenate(embeddings, axis=0), axis=0, keepdims=True)
            e_s = e_s / np.linalg.norm(e_s, ord=2, axis=1, keepdims=True)
            gamma, beta = self.pipeline.generate_film_weights(e_s)
            
            word_embs = [self.pipeline.encode_word(feat, gamma, beta) for feat in pcen_features]
            w_c = np.mean(np.concatenate(word_embs, axis=0), axis=0, keepdims=True)
            w_c = w_c / np.linalg.norm(w_c, ord=2, axis=1, keepdims=True)         
            vaults[word] = {"w_c": w_c, "gamma": gamma, "beta": beta, "owner": owner, "test_files": test_files}

        vault_words = list(vaults.keys())
        for true_word, vault in vaults.items():
            for test_file in vault["test_files"]:
                wav = self.pipeline.preprocess_keyword_audio(test_file)
                pcen = self.pipeline.extract_keyword_pcen(wav)                
                rank_scores = []
                for target_word, target_vault in vaults.items():
                    emb = self.pipeline.encode_word(pcen, target_vault["gamma"], target_vault["beta"])
                    p_accept = self.pipeline.compare(target_vault["w_c"], emb)
                    rank_scores.append((target_word, p_accept))
                    
                rank_scores.sort(key=lambda x: x[1], reverse=True)
                ranked_words = [w for w, _ in rank_scores]
                
                rank = ranked_words.index(true_word) + 1
                if rank == 1: self.top1 += 1
                if rank <= 5: self.top5 += 1
                self.reciprocal_ranks.append(1.0 / rank)
                self.total_queries += 1                
                true_score = next(s for w, s in rank_scores if w == true_word)
                self.scores.append(true_score)
                self.labels.append(1)
                imposters = [s for s in self.word_speaker_files[true_word].keys() if s != vault["owner"]]
                if imposters:
                    imp_file = self.word_speaker_files[true_word][imposters[0]][0]
                    imp_wav = self.pipeline.preprocess_keyword_audio(imp_file)
                    imp_emb = self.pipeline.encode_word(self.pipeline.extract_keyword_pcen(imp_wav), vault["gamma"], vault["beta"])
                    imp_score = self.pipeline.compare(vault["w_c"], imp_emb)
                    
                    self.scores.append(imp_score)
                    self.labels.append(0)                  
                    self.speaker_collisions["total"] += 1
                    if imp_score < 0.50: self.speaker_collisions["rejected"] += 1

                hard_words = self.hard_negatives.get(true_word, [])
                valid_hard = [w for w in hard_words if vault["owner"] in self.word_speaker_files.get(w, {})]
                if valid_hard:
                    pd_file = self.word_speaker_files[valid_hard[0]][vault["owner"]][0]
                    pd_wav = self.pipeline.preprocess_keyword_audio(pd_file)
                    pd_emb = self.pipeline.encode_word(self.pipeline.extract_keyword_pcen(pd_wav), vault["gamma"], vault["beta"])
                    pd_score = self.pipeline.compare(vault["w_c"], pd_emb)
                    
                    self.scores.append(pd_score)
                    self.labels.append(0)              
                    self.phonetic_collisions["total"] += 1
                    if pd_score < 0.50: self.phonetic_collisions["rejected"] += 1

    def print_report(self):
        top1_acc = self.top1 / max(1, self.total_queries)
        top5_acc = self.top5 / max(1, self.total_queries)
        mrr = np.mean(self.reciprocal_ranks) if self.reciprocal_ranks else 0.0
        eer, far, frr, tar_1 = compute_security_metrics(self.labels, self.scores)
        spk_rej = self.speaker_collisions["rejected"] / max(1, self.speaker_collisions["total"])
        phon_rej = self.phonetic_collisions["rejected"] / max(1, self.phonetic_collisions["total"])
        
        print("\n" + "="*50)
        print(" SC-TKWS EVALUATION REPORT".center(50))
        print("="*50)
        
        print("\nRetrieval")
        print("-" * 10)
        print(f"Top1 Accuracy: {top1_acc:.1%}")
        print(f"Top5 Accuracy: {top5_acc:.1%}")
        print(f"MRR:           {mrr:.3f}")
        
        print("\nSecurity")
        print("-" * 10)
        print(f"EER:           {eer:.2%}")
        print(f"TAR@1%FAR:     {tar_1:.1%}")
        print(f"FAR (Th=0.5):  {far:.2%}")
        print(f"FRR (Th=0.5):  {frr:.2%}")
        
        print("\nHard Negatives")
        print("-" * 14)
        print(f"Phonetic Reject Rate: {phon_rej:.1%}")
        print(f"Speaker Reject Rate:  {spk_rej:.1%}")
        
        print("\nEnrollment (EER)")
        print("-" * 16)
        for k, e in self.n_shot_results.items():
            print(f"{k}-shot EER:  {e:.2%}")
            
        print("\nDeployment (ONNX Execution)")
        print("-" * 27)
        print(f"Mean Latency: {np.mean(self.latency_metrics['e2e']):.2f} ms")
        print(f"P95 Latency:  {np.percentile(self.latency_metrics['e2e'], 95):.2f} ms")
        print(f"Throughput:   {self.throughput:.0f} queries/sec")
        
        print("\nComponent Breakdown (Mean):")
        print(f"  Spk PCEN:   {np.mean(self.latency_metrics['spk_pcen']):.2f} ms  (Enrollment)")
        print(f"  ECAPA:      {np.mean(self.latency_metrics['ecapa']):.2f} ms  (Enrollment)")
        print(f"  FiLM:       {np.mean(self.latency_metrics['film']):.2f} ms  (Enrollment)")
        print(f"  KWS PCEN:   {np.mean(self.latency_metrics['kws_pcen']):.2f} ms  (Inference)")
        print(f"  TCResNet:   {np.mean(self.latency_metrics['encoder']):.2f} ms  (Inference)")
        print(f"  Comparator: {np.mean(self.latency_metrics['comparator']):.2f} ms  (Inference)")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Benchmark Suite for SC-TKWS")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory containing ONNX files")
    parser.add_argument("--data_dir", type=str, default="./film_neural_comparator/tts_corpus_processed/train", help="Evaluation dataset")
    parser.add_argument("--ped_matrix", type=str, default=None, help="Path to cached .ped_cache.json for hard negatives")
    args = parser.parse_args()
    benchmark = SC_TKWS_Benchmark(args.model_dir, args.data_dir, args.ped_matrix)
    benchmark.run_latency_benchmark()
    benchmark.evaluate_n_shot([1, 3, 5])
    benchmark.run_full_suite()
    benchmark.print_report()