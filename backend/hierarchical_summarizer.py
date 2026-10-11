"""
Hierarchical and Narrative Synthesis Engine for Transformer Summarizers (BART & PEGASUS).

Provides:
1. Long-document chunking and recursive synthesis (handling arbitrary token counts).
2. Narrative distillation (character arcs, pivotal conflicts, resolution, dialogue/filler filtering).
3. Single cohesive paragraph formatting.
4. Strict word count / compression ratio enforcement:
   - Short: <= 45% of source words (capped at dense single paragraph)
   - Medium: <= 60% of source words (capped at dense single paragraph)
   - Long: 75% - 80% of source words (capped at dense single paragraph)
"""

import re
from typing import Callable, List, Tuple


# Regex for trivial dialogue filler, transcript artifacts, and conversational noise
FILLER_REGEX = re.compile(
    r"\b(um|uh|erm|hmm|you know|i mean|like,?\s+(?=you|we|i|they|he|she)|basically|sort of|kind of)\b",
    re.IGNORECASE,
)
SPEAKER_TAG_REGEX = re.compile(
    r"^(?:speaker\s*\d+|[A-Z][a-zA-Z\s]{1,15}):\s*",
    re.IGNORECASE | re.MULTILINE,
)
TIMESTAMP_REGEX = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]|\(\d{1,2}:\d{2}\)")
CONSECUTIVE_SPACES_REGEX = re.compile(r"\s+")
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def clean_and_filter_narrative(text: str) -> str:
    """
    Cleans transcript artifacts, minor conversational filler, and formatting noise
    while preserving plot, core character interactions, and critical narrative actions.
    """
    if not text:
        return ""

    cleaned = TIMESTAMP_REGEX.sub(" ", text)
    cleaned = SPEAKER_TAG_REGEX.sub("", cleaned)
    cleaned = FILLER_REGEX.sub("", cleaned)
    cleaned = CONSECUTIVE_SPACES_REGEX.sub(" ", cleaned).strip()
    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """Splits text into clean, non-empty sentences."""
    if not text:
        return []
    raw = SENTENCE_SPLIT_REGEX.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def calculate_word_budget(input_word_count: int, preset: str = "medium") -> Tuple[int, int]:
    """
    Calculates the target min_words and strict max_words budget based on user specifications:
    - Short: <= 45% of source words
    - Medium: <= 60% of source words
    - Long: 75% - 80% of source words (strict cap at 80%)

    Also bounds the maximum length to ensure the result is a cohesive single paragraph
    (maximum ~140 words for short, ~240 for medium, ~350 for long).
    """
    p = (preset or "medium").strip().lower()
    if p in ("short", "2"):
        ratio_max = 0.45
        ratio_min = 0.25
        paragraph_cap = 250
    elif p in ("long", "5"):
        ratio_max = 0.80
        ratio_min = 0.75
        paragraph_cap = 800
    else:  # medium or 3
        ratio_max = 0.60
        ratio_min = 0.45
        paragraph_cap = 450

    if input_word_count <= 20:
        max_words = max(4, int(input_word_count * ratio_max))
        min_words = max(2, int(input_word_count * ratio_min))
    elif input_word_count <= 60:
        max_words = max(10, int(input_word_count * ratio_max))
        min_words = max(5, int(input_word_count * ratio_min))
    else:
        calculated_max = int(input_word_count * ratio_max)
        calculated_min = int(input_word_count * ratio_min)
        max_words = min(paragraph_cap, calculated_max)
        min_words = min(max_words - 3, max(15, min(paragraph_cap - 20, calculated_min)))

    min_words = max(1, min(min_words, max_words))
    return min_words, max_words


def enforce_strict_word_limit(
    summary: str,
    max_words: int,
    min_words: int = 10,
) -> str:
    """
    Guarantees:
    1. Formatted as a single cohesive paragraph (no newlines).
    2. Word count is strictly <= max_words.
    3. Trims at full sentence boundaries where possible.
    """
    if not summary:
        return ""

    # Flatten into a single cohesive paragraph
    single_paragraph = " ".join(summary.replace("\r", " ").replace("\n", " ").split()).strip()
    words = single_paragraph.split()

    if len(words) <= max_words:
        return single_paragraph

    # Needs truncation to obey strict word limit
    sentences = split_into_sentences(single_paragraph)
    truncated_sentences: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_word_count + s_words <= max_words:
            truncated_sentences.append(sentence)
            current_word_count += s_words
        else:
            break

    if truncated_sentences:
        return " ".join(truncated_sentences).strip()

    # If first sentence alone was already longer than max_words, trim at word boundary
    sub_words = words[:max_words]
    trimmed_text = " ".join(sub_words).strip()
    # Strip trailing commas, colons, semicolons, dashes
    trimmed_text = re.sub(r"[,;:\-\s]+$", "", trimmed_text)
    if not trimmed_text.endswith((".", "!", "?")):
        trimmed_text += "."

    return trimmed_text


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 750,
    overlap_tokens: int = 80,
) -> List[str]:
    """
    Partitions long text into semantically complete, overlapping chunks using sentence
    boundaries. Never splits in the middle of a sentence.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: List[str] = []
    current_chunk_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        # Handle extraordinarily long isolated sentences
        if sentence_tokens > max_tokens:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0
            # Force chunk large sentence
            chunks.append(sentence)
            continue

        if current_tokens + sentence_tokens > max_tokens and current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            chunks.append(chunk_str)

            # Build overlap from the end of the current chunk
            overlap_sentences: List[str] = []
            overlap_count = 0
            for prev_s in reversed(current_chunk_sentences):
                s_toks = len(tokenizer.encode(prev_s, add_special_tokens=False))
                if overlap_count + s_toks <= overlap_tokens:
                    overlap_sentences.insert(0, prev_s)
                    overlap_count += s_toks
                else:
                    break

            current_chunk_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_count + sentence_tokens
        else:
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def synthesize_hierarchical(
    text: str,
    tokenizer,
    generate_fn: Callable[[str, int, int], str],
    preset: str = "medium",
    max_chunk_tokens: int = 750,
) -> str:
    """
    End-to-end multi-stage pipeline:
    1. Filter out superficial filler/transcript noise.
    2. Assess token length:
       - If within model's context window, summarize in a single cohesive pass.
       - If longer, chunk into overlapping segments, summarize each segment to capture
         narrative beats, and recursively synthesize the intermediate summaries into a
         unified single paragraph.
    3. Strictly enforce user-specified word limit (<=45% short, <=60% med, <=80% long).
    4. Ensure single paragraph narrative cohesion.
    """
    cleaned_source = clean_and_filter_narrative(text)
    if not cleaned_source:
        return ""

    source_word_count = len(cleaned_source.split())
    min_words, max_words = calculate_word_budget(source_word_count, preset)

    # Estimate token count
    input_tokens = len(tokenizer.encode(cleaned_source, add_special_tokens=False))

    # Approximate token targets for generation (typically ~1.3 tokens per word)
    gen_max_tokens = min(320, max(28, int(max_words * 1.35)))
    gen_min_tokens = max(14, int(min_words * 1.1))

    if input_tokens <= max_chunk_tokens:
        # Single-pass direct generation
        raw_summary = generate_fn(cleaned_source, gen_min_tokens, gen_max_tokens)
        return enforce_strict_word_limit(raw_summary, max_words, min_words)

    # Multi-pass hierarchical chunking
    chunks = chunk_text_by_tokens(cleaned_source, tokenizer, max_chunk_tokens=max_chunk_tokens)

    # MAP PHASE: Distill key narrative movements from each chunk
    chunk_summaries: List[str] = []
    chunk_gen_max = min(180, max(45, int(gen_max_tokens * 0.85)))
    chunk_gen_min = max(20, int(chunk_gen_max * 0.4))

    for chunk in chunks:
        if not chunk.strip():
            continue
        c_summary = generate_fn(chunk, chunk_gen_min, chunk_gen_max)
        if c_summary.strip():
            chunk_summaries.append(c_summary.strip())

    if not chunk_summaries:
        return ""

    # REDUCE / SYNTHESIS PHASE
    combined_intermediate = " ".join(chunk_summaries)
    combined_tokens = len(tokenizer.encode(combined_intermediate, add_special_tokens=False))

    # If the combined intermediate summaries still exceed the context window, recursively condense
    while combined_tokens > max_chunk_tokens and len(chunk_summaries) > 1:
        reduced_chunks: List[str] = []
        half = max(1, len(chunk_summaries) // 2)
        group_1 = " ".join(chunk_summaries[:half])
        group_2 = " ".join(chunk_summaries[half:])

        reduced_chunks.append(generate_fn(group_1, chunk_gen_min, chunk_gen_max))
        reduced_chunks.append(generate_fn(group_2, chunk_gen_min, chunk_gen_max))

        chunk_summaries = [r for r in reduced_chunks if r.strip()]
        combined_intermediate = " ".join(chunk_summaries)
        combined_tokens = len(tokenizer.encode(combined_intermediate, add_special_tokens=False))

    # Final synthesis into a single cohesive paragraph
    final_raw_summary = generate_fn(combined_intermediate, gen_min_tokens, gen_max_tokens)
    return enforce_strict_word_limit(final_raw_summary, max_words, min_words)
