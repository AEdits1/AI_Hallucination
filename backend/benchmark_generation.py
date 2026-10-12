import time
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from backend.summarizer import normalize_input_text
    from backend.pegasus_model import PEGASUS_SUMMARIZER
    from backend.led_model import LED_SUMMARIZER
    from backend.evaluate_summarization_full import compute_rouge_n, compute_rouge_l, tokenize
except ModuleNotFoundError:
    from summarizer import normalize_input_text
    from pegasus_model import PEGASUS_SUMMARIZER
    from led_model import LED_SUMMARIZER
    from evaluate_summarization_full import compute_rouge_n, compute_rouge_l, tokenize

def run_generation_benchmark(num_samples: int = 20):
    print("=" * 70)
    print("           MODEL GENERATION BENCHMARK (PEGASUS vs LED)          ")
    print("=" * 70)

    dataset_path = 'hallucination_dataset_full.csv'
    df = pd.read_csv(dataset_path)
    
    # Select gold factual samples (label == 1)
    factual_df = df[df['label'] == 1].reset_index(drop=True)
    
    # Pick diverse samples (e.g. step by evenly spaced indices)
    step = max(1, len(factual_df) // num_samples)
    indices = [i * step for i in range(num_samples)]
    subset = factual_df.iloc[indices].reset_index(drop=True)
    print(f"Selected {len(subset)} benchmark test samples from {len(factual_df):,} gold articles.")

    results = {}

    # 2. Benchmark PEGASUS
    print("\n[2/3] Benchmarking PEGASUS (google/pegasus-cnn_dailymail)...")
    peg_r1, peg_r2, peg_rl = [], [], []
    peg_words, peg_times = [], []

    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="PEGASUS"):
        article = row['premise']
        gold = row['hypothesis']
        gold_tok = tokenize(gold)

        t0 = time.perf_counter()
        try:
            gen = PEGASUS_SUMMARIZER.summarize(article, summary_length=3)
        except Exception as e:
            print(f"Error in PEGASUS sample {idx}: {e}")
            gen = ""
        elapsed = time.perf_counter() - t0

        gen_tok = tokenize(gen)
        _, _, r1 = compute_rouge_n(gen_tok, gold_tok, 1)
        _, _, r2 = compute_rouge_n(gen_tok, gold_tok, 2)
        _, _, rl = compute_rouge_l(gen_tok, gold_tok)

        peg_r1.append(r1)
        peg_r2.append(r2)
        peg_rl.append(rl)
        peg_words.append(len(gen_tok))
        peg_times.append(elapsed)

    results['PEGASUS (google/pegasus-cnn_dailymail)'] = {
        'rouge_1_f1': float(np.mean(peg_r1)),
        'rouge_2_f1': float(np.mean(peg_r2)),
        'rouge_l_f1': float(np.mean(peg_rl)),
        'avg_word_count': float(np.mean(peg_words)),
        'avg_latency_sec': float(np.mean(peg_times)),
        'throughput_samples_sec': float(1.0 / np.mean(peg_times)) if np.mean(peg_times) > 0 else 0.0
    }

    # 3. Benchmark LED
    print("\n[3/3] Benchmarking LED (pszemraj/led-base-book-summary)...")
    led_r1, led_r2, led_rl = [], [], []
    led_words, led_times = [], []

    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="LED"):
        article = row['premise']
        gold = row['hypothesis']
        gold_tok = tokenize(gold)

        t0 = time.perf_counter()
        try:
            gen = LED_SUMMARIZER.summarize(article, summary_length=3)
        except Exception as e:
            print(f"Error in LED sample {idx}: {e}")
            gen = ""
        elapsed = time.perf_counter() - t0

        gen_tok = tokenize(gen)
        _, _, r1 = compute_rouge_n(gen_tok, gold_tok, 1)
        _, _, r2 = compute_rouge_n(gen_tok, gold_tok, 2)
        _, _, rl = compute_rouge_l(gen_tok, gold_tok)

        led_r1.append(r1)
        led_r2.append(r2)
        led_rl.append(rl)
        led_words.append(len(gen_tok))
        led_times.append(elapsed)

    results['LED (pszemraj/led-base-book-summary)'] = {
        'rouge_1_f1': float(np.mean(led_r1)),
        'rouge_2_f1': float(np.mean(led_r2)),
        'rouge_l_f1': float(np.mean(led_rl)),
        'avg_word_count': float(np.mean(led_words)),
        'avg_latency_sec': float(np.mean(led_times)),
        'throughput_samples_sec': float(1.0 / np.mean(led_times)) if np.mean(led_times) > 0 else 0.0
    }

    # Save results
    with open('model_generation_benchmark.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nSaved generation benchmark results to 'model_generation_benchmark.json'")

    print("\n" + "=" * 70)
    print("           MODEL GENERATION BENCHMARK SUMMARY TABLE              ")
    print("=" * 70)
    print(f"{'Model':<38} | {'ROUGE-1':<9} | {'ROUGE-2':<9} | {'ROUGE-L':<9} | {'Words':<6} | {'Latency':<8}")
    print("-" * 88)
    for model_name, m in results.items():
        print(f"{model_name:<38} | {m['rouge_1_f1']*100:>7.2f}% | {m['rouge_2_f1']*100:>7.2f}% | {m['rouge_l_f1']*100:>7.2f}% | {m['avg_word_count']:>6.1f} | {m['avg_latency_sec']:>6.2f}s")
    print("=" * 70)

if __name__ == '__main__':
    run_generation_benchmark(num_samples=15)
