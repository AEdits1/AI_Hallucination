import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the fine‑tuned model and tokenizer from the saved checkpoint
model = AutoModelForSequenceClassification.from_pretrained('trained_model')
tokenizer = AutoTokenizer.from_pretrained('trained_model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running inference on device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")
model.to(device)

# Example premise / hypothesis pair (feel‑free to edit)
premise = "The cat sat on the mat."
hypothesis = "A cat is sitting."

# Tokenize the pair
inputs = {k: v.to(device) for k, v in tokenizer(premise, hypothesis, return_tensors='pt').items()}

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
print('Probabilities (Contradiction, Entailment, Unverifiable):', probs)

# Map the predicted index to a human‑readable label (the model uses the same mapping as the training code)
label_map = {0: 'CONTRADICTION', 1: 'ENTAILMENT', 2: 'UNVERIFIABLE'}
pred_idx = int(probs.argmax())
print('Predicted label:', label_map.get(pred_idx, pred_idx))
