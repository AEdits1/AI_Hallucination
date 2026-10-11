from dataclasses import dataclass

try:
    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
except Exception:
    torch = None
    PegasusForConditionalGeneration = None
    PegasusTokenizer = None

try:
    from backend.summarizer import normalize_input_text
except ModuleNotFoundError:
    from summarizer import normalize_input_text


DEFAULT_PEGASUS_MODEL = "google/pegasus-cnn_dailymail"


try:
    from backend.hierarchical_summarizer import (
        calculate_word_budget,
        clean_and_filter_narrative,
        enforce_strict_word_limit,
    )
except ModuleNotFoundError:
    from hierarchical_summarizer import (
        calculate_word_budget,
        clean_and_filter_narrative,
        enforce_strict_word_limit,
    )


@dataclass
class PegasusSummarizer:
    model_name: str = DEFAULT_PEGASUS_MODEL

    def __post_init__(self):
        self._tokenizer = None
        self._model = None

    def _load(self):
        if PegasusTokenizer is None or PegasusForConditionalGeneration is None:
            raise RuntimeError(
                "PEGASUS dependencies are not installed. Install transformers, sentencepiece, and torch to use this model."
            )

        if self._tokenizer is None or self._model is None:
            self._tokenizer = PegasusTokenizer.from_pretrained(self.model_name, use_fast=False)
            self._model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
            if torch is not None and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()

    def _raw_generate(self, text_input: str, min_tokens: int, max_tokens: int) -> str:
        inputs = self._tokenizer(
            text_input,
            truncation=True,
            padding="longest",
            max_length=1024,
            return_tensors="pt",
        )
        if torch is not None and torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        with torch.inference_mode():
            device_type = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            with torch.autocast(device_type=device_type):
                summary_ids = self._model.generate(
                    **inputs,
                    num_beams=4,
                    min_length=min_tokens,
                    max_length=max_tokens,
                    length_penalty=1.2,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
        decoded = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # PEGASUS uses <n> as an inline newline token — replace with a space and normalise whitespace
        decoded = decoded.replace("<n>", " ")
        return " ".join(decoded.split()).strip()

    def summarize(self, text: str, summary_length: int = 3, preset: str = None) -> str:
        """
        Summarizes in ONE GO directly through PEGASUS.
        Ensures a single cohesive paragraph adhering strictly to word count limits:
        - short: <= 45% of source words
        - medium: <= 60% of source words
        - long: 75% - 80% of source words
        """
        normalized = normalize_input_text(text)
        if not normalized:
            return ""

        cleaned = clean_and_filter_narrative(normalized)
        if not cleaned:
            cleaned = normalized

        self._load()

        if preset is None:
            if summary_length <= 2:
                preset = "short"
            elif summary_length >= 5:
                preset = "long"
            else:
                preset = "medium"

        source_word_count = len(cleaned.split())
        min_words, max_words = calculate_word_budget(source_word_count, preset)

        gen_max_tokens = min(360, max(28, int(max_words * 1.35)))
        gen_min_tokens = max(14, int(min_words * 1.1))

        # Generate in one go directly through model
        decoded = self._raw_generate(cleaned, gen_min_tokens, gen_max_tokens)
        return enforce_strict_word_limit(decoded, max_words, min_words)


PEGASUS_SUMMARIZER = PegasusSummarizer()
