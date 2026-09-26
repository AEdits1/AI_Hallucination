import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Paths (relative to project root)
MODEL_DIR = 'trained_model'
DATA_PATH = 'hallucination_dataset_cpu.csv'  # the full dataset you provided

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load full dataset
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Prepare inputs and run inference in batches
labels = []
preds = []
batch_size = 32
for start in range(0, len(df), batch_size):
    batch = df.iloc[start:start+batch_size]
    premises = batch['premise'].tolist()
    hypotheses = batch['hypothesis'].tolist()
    inputs = tokenizer(premises, hypotheses, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
    batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    preds.extend(batch_preds)
    labels.extend(batch['label'].tolist())

# Compute metrics
accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

print(f"Full dataset size: {len(labels)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (macro): {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall (macro): {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-score (macro): {f1:.4f} ({f1*100:.2f}%)")
