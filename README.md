# Text Summarization Web App

A full-stack web application that summarizes long text using extractive or abstractive models. Built with **Flask**, **React**, and **Hugging Face Transformers**.

## Features

- **Three summarization models:** Extractive AI (TextRank / supervised), PEGASUS, and BART
- Large text input with drag-and-drop file upload
- Adjustable summary length (short / medium / long)
- Post-summary analytics: keywords, compression, reading time, content accuracy score
- Heuristic AI-writing signal estimate
- **Hallucination Detection:** Deep NLI verification against source documents using DeBERTa
- **Optimized Inference:** Uses `torch.inference_mode()` and mixed precision `autocast` for fast GPU acceleration
- Voice input via browser speech recognition
- Summary history (browser localStorage)
- Copy to clipboard and PDF export
- Vercel-inspired React UI

## Project Structure

```text
Text_summarization/
├── backend/
│   ├── app.py              # Flask entry point
│   ├── summarizer.py       # TextRank extractive summarizer
│   ├── supervised_model.py # Trainable SGD extractive model
│   ├── bart_model.py       # BART abstractive model
│   ├── pegasus_model.py    # PEGASUS abstractive model
│   ├── analysis.py         # Analytics & heuristics
│   ├── file_utils.py       # File upload parsing
│   ├── train_model.py      # Training CLI
│   └── benchmark_models.py # ROUGE benchmark CLI
├── frontend/
│   ├── src/App.jsx         # Main React app
│   └── vite.config.js      # Builds to ../static/dist/
├── static/dist/            # Compiled frontend assets
├── templates/index.html    # Flask template loading React build
├── model_artifacts/        # Trained models (created after training)
└── requirements.txt
```

## Run Locally

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Start the server

```bash
python -m backend.app
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Frontend development (optional)

```bash
cd frontend
npm run dev
```

Note: the dev server does not proxy to Flask by default; rebuild or run Flask alongside for API calls.

## Models

| `model_type` | Engine | Description |
|--------------|--------|-------------|
| `extractive` | TextRank or supervised SGD | Default. Uses trained artifact if present, otherwise TextRank PageRank on sentences. |
| `pegasus` | `google/pegasus-cnn_dailymail` | Abstractive transformer (downloads on first use). |
| `bart` | `facebook/bart-large-cnn` | Abstractive transformer (downloads on first use). |

BART and PEGASUS require `torch` and `transformers`. CUDA is used automatically when available.

## Train The Supervised Extractive Model

Train on article/abstract CSV pairs (columns: `article`, plus `abstract`, `highlights`, or `summary`):

```bash
python -m backend.train_model --train "path/to/train.csv" --validation "path/to/validation.csv" --test "path/to/test.csv"
```

Artifacts are saved to:

- `model_artifacts/supervised_summarizer.joblib`
- `model_artifacts/supervised_metrics.json`

When this artifact exists, the `extractive` mode prefers the trained model over TextRank.

## Benchmark Models

Compare supervised extractive, PEGASUS, and BART on a test CSV:

```bash
python -m backend.benchmark_models --test "path/to/test.csv" --samples 25 --summary-length 3
```

Requires a trained supervised artifact. Outputs ROUGE-1, ROUGE-2, ROUGE-L, and average runtime per model.

## API

### `GET /`

Serves the React application.

### `POST /summarize`

Accepts **JSON** or **multipart form-data**.

**JSON body:**

```json
{
  "text": "Your long paragraph goes here.",
  "summary_length": 3,
  "model_type": "extractive"
}
```

**Form fields:** `text`, `summary_length`, `model_type`, optional `file` (upload).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | — | Source text to summarize |
| `summary_length` | int | `3` | Target length (1–10 sentences) |
| `model_type` | string | `extractive` | `extractive`, `pegasus`, or `bart` |
| `file` | file | — | Optional upload (PDF, DOCX, etc.) |

**Response:**

```json
{
  "model_type": "bart",
  "overview": "One-line overview of the content.",
  "summary": "Generated summary text.",
  "source_file": "article.pdf",
  "analysis": {
    "input_word_count": 120,
    "summary_word_count": 42,
    "input_character_count": 800,
    "summary_character_count": 280,
    "input_sentence_count": 8,
    "summary_sentence_count": 3,
    "estimated_reading_time_minutes": 0.6,
    "summary_reading_time_minutes": 0.21,
    "compression_ratio": 35.0,
    "lexical_diversity": 0.72,
    "content_accuracy_score": 83.3,
    "top_keywords": ["summarization", "text", "ranking"],
    "ai_writing_signals": {
      "score": 48,
      "label": "Mixed signals",
      "details": "Heuristic estimate based on sentence uniformity, repetition, transitions, and structure."
    }
  },
  "hallucination_analysis": null
}
```

### `POST /analyze_hallucination`

Takes the source text and generated summary to perform a deep NLI (Natural Language Inference) check, detecting hallucinated or ungrounded claims.

**JSON body:**
```json
{
  "text": "The full original text...",
  "summary": "The generated summary..."
}
```

**Response:**
```json
{
  "overall_label": "POTENTIAL_HALLUCINATION",
  "overall_hallucinated": true,
  "overall_confidence": 0.94,
  "total_claims": 3,
  "supported_claims": 2,
  "contradicted_claims": 1,
  "unverifiable_claims": 0,
  "hallucination_ratio": 0.33,
  "processing_time_seconds": 1.45,
  "sentence_analysis": [
    {
      "claim": "The summary claim",
      "label": "CONTRADICTION",
      "is_hallucinated": true,
      "confidence": 0.91,
      "evidence": "The retrieved sentence from the source."
    }
  ]
}
```

## Upload Notes

Supported formats include:

- `TXT`, `MD`, `CSV`, `JSON`, `HTML`, `XML`, `LOG`
- `PDF`, `DOCX`, `PPTX`
- Common code and config files

Legacy `.ppt` files are not supported; convert to `.pptx` first. Empty or unreadable files return a clear error.

## Roadmap

Planned features (not yet implemented):

- Explainable AI: cross-attention heatmaps on source text
- Hallucination detection via NLI/entailment models
- Server-side history and optional authentication

See `anchor.md/TODO.md` for the full task list (local AI context, gitignored).
