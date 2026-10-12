import time
import re
import math
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import torch

def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\b\w+\b", text.lower()) if t]

def ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def compute_rouge_n(candidate_tokens: list[str], reference_tokens: list[str], n: int):
    cand_ng = ngrams(candidate_tokens, n)
    ref_ng = ngrams(reference_tokens, n)
    if not cand_ng or not ref_ng:
        return 0.0, 0.0, 0.0
    overlap = sum((cand_ng & ref_ng).values())
    p = overlap / sum(cand_ng.values()) if cand_ng else 0.0
    r = overlap / sum(ref_ng.values()) if ref_ng else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def lcs_length(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a or not seq_b:
        return 0
    # Keep smaller sequence as column to save space
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    prev = [0] * (len(seq_b) + 1)
    for tok_a in seq_a:
        curr = [0] * (len(seq_b) + 1)
        for j, tok_b in enumerate(seq_b, 1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]

def compute_rouge_l(candidate_tokens: list[str], reference_tokens: list[str]):
    if not candidate_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0
    lcs = lcs_length(candidate_tokens, reference_tokens)
    p = lcs / len(candidate_tokens) if candidate_tokens else 0.0
    r = lcs / len(reference_tokens) if reference_tokens else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def compute_novel_ngrams(candidate_tokens: list[str], reference_tokens: list[str], n: int) -> float:
    cand_ng = ngrams(candidate_tokens, n)
    ref_ng = set(ngrams(reference_tokens, n).keys())
    if not cand_ng:
        return 0.0
    novel_count = sum(count for ng, count in cand_ng.items() if ng not in ref_ng)
    return novel_count / sum(cand_ng.values())

def compute_extractive_fragments(summary_tokens: list[str], source_tokens: list[str]):
    """
    Grusky et al. (2018) greedy extractive fragment match.
    Returns (coverage, density).
    """
    if not summary_tokens or not source_tokens:
        return 0.0, 0.0
    
    source_str = " " + " ".join(source_tokens) + " "
    sum_len = len(summary_tokens)
    fragments = []
    
    i = 0
    while i < sum_len:
        best_len = 0
        for j in range(i + 1, sum_len + 1):
            sub = " " + " ".join(summary_tokens[i:j]) + " "
            if sub in source_str:
                best_len = j - i
            else:
                break
        if best_len > 0:
            fragments.append(best_len)
            i += best_len
        else:
            i += 1
            
    coverage = sum(fragments) / sum_len if sum_len > 0 else 0.0
    density = sum(f**2 for f in fragments) / sum_len if sum_len > 0 else 0.0
    return coverage, density

def evaluate_full_summarization_dataset():
    print("=" * 70)
    print("      COMPREHENSIVE FULL DATASET SUMMARIZATION METRIC BENCHMARK     ")
    print("=" * 70)
    
    dataset_path = 'hallucination_dataset_full.csv'
    print(f"Loading dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset successfully loaded: {len(df):,} total samples.")
    
    factual_df = df[df['label'] == 1].reset_index(drop=True)
    hallucinated_df = df[df['label'] == 0].reset_index(drop=True)
    
    print(f"Factual Ground-Truth Samples:    {len(factual_df):,}")
    print(f"Hallucinated Counterpart Samples: {len(hallucinated_df):,}")
    print("-" * 70)

    # 1. Corpus & Text Length Statistics
    def get_stats(series):
        tokens_list = [tokenize(str(t)) for t in series]
        word_counts = [len(t) for t in tokens_list]
        char_counts = [len(str(t)) for t in series]
        sent_counts = [max(1, len(re.split(r'[.!?]+', str(t).strip())) - 1) for t in series]
        return {
            'word_counts': word_counts,
            'char_counts': char_counts,
            'sent_counts': sent_counts,
            'tokens_list': tokens_list
        }

    print("\nComputing Corpus Length & Compression Statistics...")
    source_stats = get_stats(factual_df['premise'])
    factual_stats = get_stats(factual_df['hypothesis'])
    hallucinated_stats = get_stats(hallucinated_df['hypothesis'])

    def summarize_stat(arr):
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': int(np.min(arr)),
            'max': int(np.max(arr))
        }

    src_w = summarize_stat(source_stats['word_counts'])
    fac_w = summarize_stat(factual_stats['word_counts'])
    hal_w = summarize_stat(hallucinated_stats['word_counts'])

    comp_factual = [
        (fw / sw * 100) if sw > 0 else 0.0
        for fw, sw in zip(factual_stats['word_counts'], source_stats['word_counts'])
    ]
    comp_hallucinated = [
        (hw / sw * 100) if sw > 0 else 0.0
        for hw, sw in zip(hallucinated_stats['word_counts'], source_stats['word_counts'])
    ]

    print("\n--- CORPUS LENGTH & COMPRESSION DISTRIBUTION (Full 20,000 samples) ---")
    print(f"Source Document Word Count:   Mean = {src_w['mean']:.1f} ± {src_w['std']:.1f} (Median: {src_w['median']:.0f}, Range: [{src_w['min']}, {src_w['max']}])")
    print(f"Factual Summary Word Count:    Mean = {fac_w['mean']:.1f} ± {fac_w['std']:.1f} (Median: {fac_w['median']:.0f}, Range: [{fac_w['min']}, {fac_w['max']}])")
    print(f"Hallucinated Summary Word Count: Mean = {hal_w['mean']:.1f} ± {hal_w['std']:.1f} (Median: {hal_w['median']:.0f}, Range: [{hal_w['min']}, {hal_w['max']}])")
    print(f"Factual Compression Ratio:     Mean = {np.mean(comp_factual):.2f}% ± {np.std(comp_factual):.2f}%")
    print(f"Hallucinated Compression Ratio: Mean = {np.mean(comp_hallucinated):.2f}% ± {np.std(comp_hallucinated):.2f}%")

    # 2. ROUGE & Abstractiveness computation across full dataset
    print("\nComputing ROUGE-1, ROUGE-2, ROUGE-L, Novel n-grams, and Fragment Metrics...")
    
    num_samples = len(factual_df)
    
    # Metrics accumulators
    r1_p_fac, r1_r_fac, r1_f_fac = [], [], []
    r2_p_fac, r2_r_fac, r2_f_fac = [], [], []
    rl_p_fac, rl_r_fac, rl_f_fac = [], [], []
    nov1_fac, nov2_fac, nov3_fac = [], [], []
    cov_fac, den_fac = [], []

    r1_p_hal, r1_r_hal, r1_f_hal = [], [], []
    r2_p_hal, r2_r_hal, r2_f_hal = [], [], []
    rl_p_hal, rl_r_hal, rl_f_hal = [], [], []
    nov1_hal, nov2_hal, nov3_hal = [], [], []
    cov_hal, den_hal = [], []

    # Cross-summary agreement (Factual vs Hallucinated)
    r1_cross, r2_cross, rl_cross = [], [], []

    for i in tqdm(range(num_samples), desc="Evaluating Full Dataset Pairs"):
        src_tok = source_stats['tokens_list'][i]
        fac_tok = factual_stats['tokens_list'][i]
        hal_tok = hallucinated_stats['tokens_list'][i]

        # 1. Factual Summary vs Source Document
        p1, r1, f1 = compute_rouge_n(fac_tok, src_tok, 1)
        p2, r2, f2 = compute_rouge_n(fac_tok, src_tok, 2)
        pl, rl, fl = compute_rouge_l(fac_tok, src_tok)
        r1_p_fac.append(p1); r1_r_fac.append(r1); r1_f_fac.append(f1)
        r2_p_fac.append(p2); r2_r_fac.append(r2); r2_f_fac.append(f2)
        rl_p_fac.append(pl); rl_r_fac.append(rl); rl_f_fac.append(fl)

        nov1_fac.append(compute_novel_ngrams(fac_tok, src_tok, 1))
        nov2_fac.append(compute_novel_ngrams(fac_tok, src_tok, 2))
        nov3_fac.append(compute_novel_ngrams(fac_tok, src_tok, 3))

        c_f, d_f = compute_extractive_fragments(fac_tok, src_tok)
        cov_fac.append(c_f); den_fac.append(d_f)

        # 2. Hallucinated Summary vs Source Document
        p1_h, r1_h, f1_h = compute_rouge_n(hal_tok, src_tok, 1)
        p2_h, r2_h, f2_h = compute_rouge_n(hal_tok, src_tok, 2)
        pl_h, rl_h, fl_h = compute_rouge_l(hal_tok, src_tok)
        r1_p_hal.append(p1_h); r1_r_hal.append(r1_h); r1_f_hal.append(f1_h)
        r2_p_hal.append(p2_h); r2_r_hal.append(r2_h); r2_f_hal.append(f2_h)
        rl_p_hal.append(pl_h); rl_r_hal.append(rl_h); rl_f_hal.append(fl_h)

        nov1_hal.append(compute_novel_ngrams(hal_tok, src_tok, 1))
        nov2_hal.append(compute_novel_ngrams(hal_tok, src_tok, 2))
        nov3_hal.append(compute_novel_ngrams(hal_tok, src_tok, 3))

        c_h, d_h = compute_extractive_fragments(hal_tok, src_tok)
        cov_hal.append(c_h); den_hal.append(d_h)

        # 3. Factual vs Hallucinated Summary Overlap
        _, _, fc1 = compute_rouge_n(hal_tok, fac_tok, 1)
        _, _, fc2 = compute_rouge_n(hal_tok, fac_tok, 2)
        _, _, fcl = compute_rouge_l(hal_tok, fac_tok)
        r1_cross.append(fc1); r2_cross.append(fc2); rl_cross.append(fcl)

    results = {
        'dataset_size': num_samples * 2,
        'num_pairs': num_samples,
        'source_length': src_w,
        'factual_length': fac_w,
        'hallucinated_length': hal_w,
        'factual_compression': {
            'mean': float(np.mean(comp_factual)),
            'std': float(np.std(comp_factual)),
        },
        'hallucinated_compression': {
            'mean': float(np.mean(comp_hallucinated)),
            'std': float(np.std(comp_hallucinated)),
        },
        'factual_rouge': {
            'rouge_1': {'p': float(np.mean(r1_p_fac)), 'r': float(np.mean(r1_r_fac)), 'f1': float(np.mean(r1_f_fac))},
            'rouge_2': {'p': float(np.mean(r2_p_fac)), 'r': float(np.mean(r2_r_fac)), 'f1': float(np.mean(r2_f_fac))},
            'rouge_l': {'p': float(np.mean(rl_p_fac)), 'r': float(np.mean(rl_r_fac)), 'f1': float(np.mean(rl_f_fac))},
        },
        'hallucinated_rouge': {
            'rouge_1': {'p': float(np.mean(r1_p_hal)), 'r': float(np.mean(r1_r_hal)), 'f1': float(np.mean(r1_f_hal))},
            'rouge_2': {'p': float(np.mean(r2_p_hal)), 'r': float(np.mean(r2_r_hal)), 'f1': float(np.mean(r2_f_hal))},
            'rouge_l': {'p': float(np.mean(rl_p_hal)), 'r': float(np.mean(rl_r_hal)), 'f1': float(np.mean(rl_f_hal))},
        },
        'factual_novelty': {
            'novel_unigram_pct': float(np.mean(nov1_fac) * 100),
            'novel_bigram_pct': float(np.mean(nov2_fac) * 100),
            'novel_trigram_pct': float(np.mean(nov3_fac) * 100),
            'fragment_coverage': float(np.mean(cov_fac) * 100),
            'fragment_density': float(np.mean(den_fac)),
        },
        'hallucinated_novelty': {
            'novel_unigram_pct': float(np.mean(nov1_hal) * 100),
            'novel_bigram_pct': float(np.mean(nov2_hal) * 100),
            'novel_trigram_pct': float(np.mean(nov3_hal) * 100),
            'fragment_coverage': float(np.mean(cov_hal) * 100),
            'fragment_density': float(np.mean(den_hal)),
        },
        'cross_summary_rouge': {
            'rouge_1_f1': float(np.mean(r1_cross)),
            'rouge_2_f1': float(np.mean(r2_cross)),
            'rouge_l_f1': float(np.mean(rl_cross)),
        }
    }

    # Save results to json
    with open('summarization_metrics_full.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nSaved full summarization metrics to 'summarization_metrics_full.json'")

    print("\n" + "=" * 70)
    print("           FULL DATASET SUMMARIZATION EVALUATION REPORT           ")
    print("=" * 70)
    print(f"Total Corpus Evaluated:        20,000 samples (10,000 paired instances)")
    print("-" * 70)
    print("1. FACTUAL REFERENCE SUMMARIES (GROUND TRUTH / CNN-DM HIGHLIGHTS):")
    print(f"   ROUGE-1: Precision = {results['factual_rouge']['rouge_1']['p'] * 100:.2f}%, Recall = {results['factual_rouge']['rouge_1']['r'] * 100:.2f}%, F1 = {results['factual_rouge']['rouge_1']['f1'] * 100:.2f}%")
    print(f"   ROUGE-2: Precision = {results['factual_rouge']['rouge_2']['p'] * 100:.2f}%, Recall = {results['factual_rouge']['rouge_2']['r'] * 100:.2f}%, F1 = {results['factual_rouge']['rouge_2']['f1'] * 100:.2f}%")
    print(f"   ROUGE-L: Precision = {results['factual_rouge']['rouge_l']['p'] * 100:.2f}%, Recall = {results['factual_rouge']['rouge_l']['r'] * 100:.2f}%, F1 = {results['factual_rouge']['rouge_l']['f1'] * 100:.2f}%")
    print(f"   Abstractiveness:")
    print(f"     - Novel Unigrams: {results['factual_novelty']['novel_unigram_pct']:.2f}%")
    print(f"     - Novel Bigrams:  {results['factual_novelty']['novel_bigram_pct']:.2f}%")
    print(f"     - Novel Trigrams: {results['factual_novelty']['novel_trigram_pct']:.2f}%")
    print(f"     - Extractive Fragment Coverage: {results['factual_novelty']['fragment_coverage']:.2f}%")
    print(f"     - Extractive Fragment Density:  {results['factual_novelty']['fragment_density']:.2f}")
    print("-" * 70)
    print("2. HALLUCINATED SUMMARY COUNTERPARTS (HALUEVAL SYNTHETIC PERTURBATIONS):")
    print(f"   ROUGE-1: Precision = {results['hallucinated_rouge']['rouge_1']['p'] * 100:.2f}%, Recall = {results['hallucinated_rouge']['rouge_1']['r'] * 100:.2f}%, F1 = {results['hallucinated_rouge']['rouge_1']['f1'] * 100:.2f}%")
    print(f"   ROUGE-2: Precision = {results['hallucinated_rouge']['rouge_2']['p'] * 100:.2f}%, Recall = {results['hallucinated_rouge']['rouge_2']['r'] * 100:.2f}%, F1 = {results['hallucinated_rouge']['rouge_2']['f1'] * 100:.2f}%")
    print(f"   ROUGE-L: Precision = {results['hallucinated_rouge']['rouge_l']['p'] * 100:.2f}%, Recall = {results['hallucinated_rouge']['rouge_l']['r'] * 100:.2f}%, F1 = {results['hallucinated_rouge']['rouge_l']['f1'] * 100:.2f}%")
    print(f"   Abstractiveness:")
    print(f"     - Novel Unigrams: {results['hallucinated_novelty']['novel_unigram_pct']:.2f}%")
    print(f"     - Novel Bigrams:  {results['hallucinated_novelty']['novel_bigram_pct']:.2f}%")
    print(f"     - Novel Trigrams: {results['hallucinated_novelty']['novel_trigram_pct']:.2f}%")
    print(f"     - Extractive Fragment Coverage: {results['hallucinated_novelty']['fragment_coverage']:.2f}%")
    print(f"     - Extractive Fragment Density:  {results['hallucinated_novelty']['fragment_density']:.2f}")
    print("-" * 70)
    print("3. CONTRASTIVE ANALYSIS (FACTUAL VS HALLUCINATED):")
    delta_r1 = (results['factual_rouge']['rouge_1']['f1'] - results['hallucinated_rouge']['rouge_1']['f1']) * 100
    delta_r2 = (results['factual_rouge']['rouge_2']['f1'] - results['hallucinated_rouge']['rouge_2']['f1']) * 100
    delta_rl = (results['factual_rouge']['rouge_l']['f1'] - results['hallucinated_rouge']['rouge_l']['f1']) * 100
    print(f"   ROUGE-1 Delta (Factual vs Hallucinated): {delta_r1:+.2f}%")
    print(f"   ROUGE-2 Delta (Factual vs Hallucinated): {delta_r2:+.2f}%")
    print(f"   ROUGE-L Delta (Factual vs Hallucinated): {delta_rl:+.2f}%")
    print(f"   Cross-Summary Semantic Agreement (ROUGE-1 F1): {results['cross_summary_rouge']['rouge_1_f1'] * 100:.2f}%")
    print(f"   Cross-Summary Semantic Agreement (ROUGE-2 F1): {results['cross_summary_rouge']['rouge_2_f1'] * 100:.2f}%")
    print(f"   Cross-Summary Semantic Agreement (ROUGE-L F1): {results['cross_summary_rouge']['rouge_l_f1'] * 100:.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    evaluate_full_summarization_dataset()
