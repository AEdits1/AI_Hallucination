# Explainable Abstractive Summarization with Hallucination Detection using Transformer Models

An end-to-end deep learning framework for **long-context abstractive summarization** and **explainable hallucination detection**. Built with **PyTorch**, **Hugging Face Transformers**, **Flask**, and **React**.

---

## Overview

Modern sequence-to-sequence transformer models frequently suffer from two core limitations:
1. **Context Window Bottlenecks:** Hard input truncation at 1,024 tokens causes standard models to lose narrative continuity on long reports and documents.
2. **Factual Hallucinations:** Autoregressive language generation can fabricate facts, misattribute character actions, or distort source data.

This project addresses both challenges by pairing a **16,384-token Long-Context Summarization model (LED)** with an **Explainable DeBERTa-v3 NLI Hallucination Verification pipeline**. The system ingests documents of any size in a single pass, distills the narrative into a cohesive single paragraph under strict compression limits, and verifies every claim against source evidence.

---

## Key Features

* **Native Long-Document Processing ("In One Go"):** Ingests long documents (up to 16,384 tokens / ~12,000 words) in a single tensor forward pass using sparse local sliding-window attention and global attention on token `<s>`. No chunk fragmentation or recursive loss.
* **Narrative Arc Distillation:** Powered by **`pszemraj/led-base-book-summary`** (fine-tuned on the BookSum dataset) to extract core character arcs, pivotal conflicts, and plot resolutions while suppressing minor dialogue banter and conversational filler.
* **Single Cohesive Paragraph Guarantee:** Synthesizes the entire narrative into one unified, uninterrupted paragraph connecting exposition, conflict, and resolution.
* **Strict Word Count & Compression Limits:**
  * **Short Summary:** Strictly $\le 45\%$ of source words.
  * **Medium Summary:** Strictly $\le 60\%$ of source words.
  * **Long Summary:** Strictly $75\%\text{--}80\%$ of source words (scales up to 1,024 output tokens / ~800 words).
* **Explainable Claim-Level Hallucination Detection:**
  * Deconstructs the generated summary into individual factual claims.
  * Uses **`sentence-transformers/all-MiniLM-L6-v2`** to retrieve semantically aligned evidence sentences from the source document.
  * Employs **`cross-encoder/nli-deberta-v3-base`** to classify each claim as **ENTAILMENT** (supported), **CONTRADICTION** (hallucinated), or **UNVERIFIABLE** (neutral/ungrounded).
* **Comprehensive Analytics & Summary Insights:**
  * Real-time **Source Stats** and **Summary Stats** (word counts, sentence counts, exact compression percentage).
  * Overall Hallucination Ratio and claim classification breakdowns.
  * Keyword extraction and lexical diversity metrics.
* **Multi-Format Ingestion:** Drag-and-drop support for PDF, DOCX, PPTX, TXT, MD, CSV, JSON, HTML, XML, and code files.
* **Modern UI & Tooling:** High-performance React interface with instant copy, browser speech recognition (voice input), local history caching, and one-click PDF export.

---

## Models & Architecture

| Task | Model | Source / Checkpoint | Context Window |
| :--- | :--- | :--- | :--- |
| **Primary Summarizer** | **LED (Longformer Encoder-Decoder)** | `pszemraj/led-base-book-summary` | **16,384 tokens** (~12k words) |
| **Alternative Summarizer** | **PEGASUS** | `google/pegasus-cnn_dailymail` | **1,024 tokens** |
| **Evidence Retrieval** | **SentenceTransformer** | `sentence-transformers/all-MiniLM-L6-v2` | Sentence embeddings |
| **Hallucination Verifier** | **DeBERTa-v3 Cross-Encoder** | `cross-encoder/nli-deberta-v3-base` | NLI sequence classification |

---

## Project Structure

```text
Text_summarization/
├── backend/
│   ├── app.py                      # Flask REST API entry point & routing
│   ├── led_model.py                # LED (Longformer) 16k single-pass summarizer
│   ├── pegasus_model.py            # PEGASUS abstractive summarizer
│   ├── hierarchical_summarizer.py  # Word budgeting, narrative filtering & boundary trimming
│   ├── hallucination_detector.py   # DeBERTa NLI claim verification & retrieval engine
│   ├── analysis.py                 # Text metrics, keywords, reading time & signals
│   ├── file_utils.py               # Document extraction (PDF, DOCX, PPTX, etc.)
│   ├── summarizer.py               # Text pre-processing & normalization utilities
│   └── train_hallucination.py      # Hallucination detection evaluation scripts
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main React web application
│   │   ├── LatticeLoader.jsx       # Animated inference loader
│   │   └── main.jsx                # React root mount
│   ├── package.json                # Frontend dependencies
│   └── vite.config.js              # Vite build config (outputs to static/dist/)
├── static/dist/                    # Pre-compiled production frontend bundle
├── templates/
│   ├── index.html                  # Flask Jinja2 template serving React app
│   ├── 404.html                    # Error fallback template
│   └── privacy.html                # Privacy policy page
├── requirements.txt                # Python backend dependencies
└── README.md                       # Project documentation
```

---

## Installation & Setup

### Prerequisites
* Python 3.10+ (CUDA-enabled GPU recommended)
* Node.js 18+ and npm

### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/<your-username>/Text_summarization.git
cd Text_summarization

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Frontend Assets
```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Run the Application
```bash
python -m backend.app
```
Open your browser at **`http://127.0.0.1:5000`**.

> **Note on First Run:** When running LED or PEGASUS for the first time, Hugging Face will automatically download and cache the model weights locally. Subsequent inferences run directly from your local cache on GPU.

---

## API Reference

### `POST /summarize`
Generates an abstractive narrative summary adhering to strict compression boundaries.

**Payload (JSON or `multipart/form-data`):**
```json
{
  "text": "Full document or story text...",
  "summary_preset": "medium",
  "model_type": "led"
}
```

* `summary_preset`: `"short"` ($\le 45\%$), `"medium"` ($\le 60\%$), or `"long"` ($75\text{--}80\%$).
* `model_type`: `"led"` (default, 16k tokens) or `"pegasus"`.
* `file`: Optional uploaded file (PDF, DOCX, TXT, etc.).

**Sample Response:**
```json
{
  "model_type": "led",
  "overview": "Overview of the text.",
  "summary": "A young mechanic named Silas discovers an ancient sub-surface drill and collaborates with his father to save the trapped crew from the cave-in.",
  "source_file": "report.pdf",
  "source_text": "Full extracted source text...",
  "analysis": {
    "input_word_count": 622,
    "summary_word_count": 48,
    "input_sentence_count": 46,
    "summary_sentence_count": 2,
    "compression_ratio": 7.7,
    "top_keywords": ["silas", "drill", "miners", "cave-in"],
    "estimated_reading_time_minutes": 3.11,
    "summary_reading_time_minutes": 0.24
  },
  "hallucination_analysis": null
}
```

---

### `POST /analyze_hallucination`
Performs sentence-level Natural Language Inference verification on the generated summary against the source document.

**Payload (JSON):**
```json
{
  "text": "Full original source document...",
  "summary": "Generated abstractive summary..."
}
```

**Sample Response:**
```json
{
  "overall_label": "SUPPORTED",
  "overall_hallucinated": false,
  "overall_confidence": 0.96,
  "total_claims": 2,
  "supported_claims": 2,
  "contradicted_claims": 0,
  "unverifiable_claims": 0,
  "hallucination_ratio": 0.0,
  "processing_time_seconds": 0.82,
  "sentence_analysis": [
    {
      "claim": "Silas discovered an ancient sub-surface drill.",
      "label": "ENTAILMENT",
      "is_hallucinated": false,
      "confidence": 0.98,
      "evidence": "In the desolate mining settlement, young mechanic Silas discovered an ancient sub-surface drill."
    }
  ]
}
```

---

## License

This project is licensed under the MIT License.
