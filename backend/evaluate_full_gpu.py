import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

def evaluate_full_gpu():
    print("=" * 60)
    print("        HALUEVAL FULL DATASET GPU EVALUATION BENCHMARK       ")
    print("=" * 60)

    # 1. Enforce ONLY GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is not available! Script is strictly configured for ONLY GPU.")
    
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(device)
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"Device: {gpu_name}")
    print(f"Total VRAM: {vram_gb:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(device)}")
    print("-" * 60)

    # 2. Paths
    model_dir = 'trained_model'
    dataset_path = 'hallucination_dataset_full.csv'

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    # 3. Load Dataset
    print(f"Loading complete dataset from '{dataset_path}'...")
    load_start = time.perf_counter()
    df = pd.read_csv(dataset_path)
    total_samples = len(df)
    load_time = time.perf_counter() - load_start
    print(f"Dataset loaded: {total_samples:,} samples in {load_time:.2f} seconds.")

    # 4. Load Model & Tokenizer
    print(f"\nLoading model & tokenizer from '{model_dir}' onto {gpu_name}...")
    model_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    torch.cuda.synchronize()
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded onto GPU in {model_load_time:.2f} seconds.")

    # 5. Warm-up GPU
    print("\nWarming up GPU kernels...")
    dummy_inputs = tokenizer(["Warm up document"], ["Warm up summary"], return_tensors='pt')
    dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
    with torch.no_grad():
        for _ in range(5):
            _ = model(**dummy_inputs)
    torch.cuda.synchronize()
    print("Warm-up complete.")

    # 6. Evaluation Loop with GPU Timing
    batch_size = 32
    all_preds = []
    all_labels = []

    print(f"\nStarting GPU inference on {total_samples:,} samples (Batch Size = {batch_size})...")
    
    # CUDA Events for accurate GPU execution timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    wall_start = time.perf_counter()
    start_event.record()

    with torch.no_grad():
        for start in tqdm(range(0, total_samples, batch_size), desc="GPU Inference Progress"):
            batch = df.iloc[start:start + batch_size]
            premises = batch['premise'].tolist()
            hypotheses = batch['hypothesis'].tolist()
            
            inputs = tokenizer(
                premises,
                hypotheses,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            
            all_preds.extend(preds)
            all_labels.extend(batch['label'].tolist())

    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    # Timing metrics
    gpu_time_ms = start_event.elapsed_time(end_event)
    gpu_time_sec = gpu_time_ms / 1000.0
    wall_time_sec = wall_end - wall_start
    throughput = total_samples / wall_time_sec
    ms_per_sample = (wall_time_sec / total_samples) * 1000.0

    # 7. Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    # 8. Report Results
    print("\n" + "=" * 60)
    print("                    EVALUATION RESULTS                    ")
    print("=" * 60)
    print(f"Total Dataset Size Evaluated: {total_samples:,} samples")
    print(f"GPU Hardware:                {gpu_name} (16 GB VRAM)")
    print("-" * 60)
    print(f"Accuracy:                    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision:             {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Macro Recall:                {recall:.4f} ({recall * 100:.2f}%)")
    print(f"Macro F1-Score:              {f1:.4f} ({f1 * 100:.2f}%)")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  [TN (Hallucination Detected): {cm[0][0]:,}]   [FP: {cm[0][1]:,}]")
    print(f"  [FN: {cm[1][0]:,}]                           [TP (Factually Accurate): {cm[1][1]:,}]")
    print("-" * 60)
    print("Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Hallucinated (0)", "Factual (1)"], digits=4))
    
    print("=" * 60)
    print("                      TIMING BENCHMARK                    ")
    print("=" * 60)
    print(f"Total Wall-Clock Time:       {wall_time_sec:.2f} seconds ({wall_time_sec / 60:.2f} minutes)")
    print(f"Pure GPU Kernel Time:        {gpu_time_sec:.2f} seconds")
    print(f"Inference Throughput:        {throughput:.2f} samples/second")
    print(f"Average Latency Per Sample:  {ms_per_sample:.2f} ms")
    print("=" * 60)

if __name__ == '__main__':
    evaluate_full_gpu()
