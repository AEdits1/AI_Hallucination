from dataclasses import dataclass

try:
    import torch
    from transformers import LEDForConditionalGeneration, LEDTokenizer
except Exception:
    torch = None
    LEDForConditionalGeneration = None
    LEDTokenizer = None

try:
    from backend.summarizer import normalize_input_text
    from backend.hierarchical_summarizer import (
        calculate_word_budget,
        clean_and_filter_narrative,
        enforce_strict_word_limit,
    )
except ModuleNotFoundError:
    from summarizer import normalize_input_text
    from hierarchical_summarizer import (
        calculate_word_budget,
        clean_and_filter_narrative,
        enforce_strict_word_limit,
    )


DEFAULT_LED_MODEL = "pszemraj/led-base-book-summary"


@dataclass
class LedSummarizer:
    model_name: str = DEFAULT_LED_MODEL

    def __post_init__(self):
        self._tokenizer = None
        self._model = None

    def _load(self):
        if LEDTokenizer is None or LEDForConditionalGeneration is None:
            raise RuntimeError(
                "LED dependencies are not installed. Install transformers and torch to use this model."
            )

        if self._tokenizer is None or self._model is None:
            self._tokenizer = LEDTokenizer.from_pretrained(self.model_name)
            self._model = LEDForConditionalGeneration.from_pretrained(self.model_name)
            if torch is not None and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()

    def summarize(self, text: str, summary_length: int = 3, preset: str = None) -> str:
        """
        Summarizes long or short documents in ONE GO using LED's native 16,384 token window.
        Fine-tuned on BookSum to distill narrative arcs, pivotal conflicts, and resolutions
        while suppressing dialogue banter and filler. Supports up to 1,024 output tokens.
        Strictly enforces word count limits:
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

        # Approximate token budgets for generation (~1.3 tokens per word)
        # Allows scaling up to 1024 tokens (full decoder capacity) for comprehensive long summaries
        gen_max_tokens = min(1024, max(28, int(max_words * 1.35)))
        gen_min_tokens = max(14, int(min_words * 1.1))

        # Tokenize IN ONE GO (up to 16,384 tokens)
        inputs = self._tokenizer(
            cleaned,
            truncation=True,
            padding="longest",
            max_length=16384,
            return_tensors="pt",
        )

        # LED requires global_attention_mask to be set: token index 0 (<s>) gets global attention
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        inputs["global_attention_mask"] = global_attention_mask

        if torch is not None and torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        with torch.inference_mode():
            device_type = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            with torch.autocast(device_type=device_type):
                summary_ids = self._model.generate(
                    **inputs,
                    num_beams=4,
                    min_length=gen_min_tokens,
                    max_length=gen_max_tokens,
                    length_penalty=1.2,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    encoder_no_repeat_ngram_size=3,
                    repetition_penalty=3.5,
                )

        decoded = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # Ensure single cohesive paragraph formatting and strict word limit enforcement
        return enforce_strict_word_limit(decoded, max_words, min_words)


LED_SUMMARIZER = LedSummarizer()
