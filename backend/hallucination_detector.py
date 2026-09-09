from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class HallucinationDetector:
    """
    Transformer-based hallucination detector.

    Pipeline:
        Source document
            -> sentence splitting
            -> semantic evidence retrieval
            -> DeBERTa NLI
            -> entailment / contradiction / unverifiable
            -> claim-level + document-level result

    This version is designed for document-grounded
    abstractive summarization.
    """

    def __init__(
        self,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        retrieval_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.top_k = top_k

        print(
            f"[Detector] Device: {self.device}"
        )

        print(
            f"[Detector] Loading NLI model: "
            f"{nli_model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                nli_model_name
            )
        )

        self.nli_model = (
            AutoModelForSequenceClassification
            .from_pretrained(nli_model_name)
            .to(self.device)
        )

        self.nli_model.eval()

        print(
            f"[Detector] Loading retrieval model: "
            f"{retrieval_model_name}"
        )

        self.retrieval_model = (
            SentenceTransformer(
                retrieval_model_name,
                device=self.device,
            )
        )

        self.label_map = self._build_label_map()

        print(
            f"[Detector] Label map: "
            f"{self.label_map}"
        )

    # ---------------------------------------------------------
    # MODEL LABELS
    # ---------------------------------------------------------

    def _build_label_map(self) -> dict[int, str]:
        """
        Convert model labels into project terminology.

        Standard NLI:
            contradiction
            neutral
            entailment

        Project terminology:
            CONTRADICTION
            UNVERIFIABLE
            ENTAILMENT
        """

        result = {}

        for index, label in self.nli_model.config.id2label.items():

            normalized = label.lower()

            if "contradiction" in normalized:
                result[int(index)] = "CONTRADICTION"

            elif "entailment" in normalized:
                result[int(index)] = "ENTAILMENT"

            elif "neutral" in normalized:
                result[int(index)] = "UNVERIFIABLE"

            else:
                result[int(index)] = normalized.upper()

        return result

    # ---------------------------------------------------------
    # SENTENCE SPLITTING
    # ---------------------------------------------------------

    @staticmethod
    def split_sentences(text: str) -> list[str]:

        text = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        if not text:
            return []

        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+(?=[A-Z0-9])",
                text,
            )
            if sentence.strip()
        ]

    # ---------------------------------------------------------
    # EVIDENCE RETRIEVAL
    # ---------------------------------------------------------

    def retrieve_evidence(
        self,
        claim: str,
        source_sentences: list[str],
    ) -> list[dict[str, Any]]:

        if not source_sentences:
            return []

        claim_embedding = (
            self.retrieval_model.encode(
                claim,
                normalize_embeddings=True,
            )
        )

        source_embeddings = (
            self.retrieval_model.encode(
                source_sentences,
                normalize_embeddings=True,
            )
        )

        similarities = np.dot(
            source_embeddings,
            claim_embedding,
        )

        ranked_indices = np.argsort(
            similarities
        )[::-1]

        evidence = []

        for index in ranked_indices[: self.top_k]:

            evidence.append(
                {
                    "sentence": source_sentences[index],
                    "similarity": float(
                        similarities[index]
                    ),
                }
            )

        return evidence

    # ---------------------------------------------------------
    # NLI
    # ---------------------------------------------------------

    def _predict_nli(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict[str, float]:

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.nli_model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]

        scores = {}

        for index, probability in enumerate(
            probabilities.cpu().tolist()
        ):

            label = self.label_map.get(
                index,
                f"LABEL_{index}",
            )

            scores[label] = float(
                probability
            )

        return scores

    # ---------------------------------------------------------
    # SINGLE CLAIM ANALYSIS
    # ---------------------------------------------------------

    def analyze_claim(
        self,
        source_text: str,
        summary_claim: str,
    ) -> dict[str, Any]:

        source_sentences = (
            self.split_sentences(source_text)
        )

        evidence_candidates = (
            self.retrieve_evidence(
                summary_claim,
                source_sentences,
            )
        )

        if not evidence_candidates:

            return {
                "claim": summary_claim,
                "label": "UNVERIFIABLE",
                "is_hallucinated": True,
                "confidence": 1.0,
                "scores": {
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "unverifiable": 1.0,
                },
                "evidence": None,
                "evidence_similarity": 0.0,
            }

        best_result = None

        for evidence_item in evidence_candidates:

            scores = self._predict_nli(
                evidence_item["sentence"],
                summary_claim,
            )

            entailment = scores.get(
                "ENTAILMENT",
                0.0,
            )

            contradiction = scores.get(
                "CONTRADICTION",
                0.0,
            )

            unverifiable = scores.get(
                "UNVERIFIABLE",
                0.0,
            )

            # Strongest relation between
            # evidence and claim.
            relation_strength = max(
                entailment,
                contradiction,
            )

            current = {
                "scores": {
                    "entailment": entailment,
                    "contradiction": contradiction,
                    "unverifiable": unverifiable,
                },
                "relation_strength": relation_strength,
                "evidence": evidence_item["sentence"],
                "similarity": evidence_item["similarity"],
            }

            if (
                best_result is None
                or current["relation_strength"]
                > best_result["relation_strength"]
            ):
                best_result = current

        assert best_result is not None

        scores = best_result["scores"]

        entailment = scores["entailment"]
        contradiction = scores["contradiction"]
        unverifiable = scores["unverifiable"]

        # -----------------------------------------------------
        # CLAIM DECISION
        # -----------------------------------------------------

        if (
            contradiction > entailment
            and contradiction > unverifiable
        ):

            label = "CONTRADICTION"
            is_hallucinated = True
            confidence = contradiction

        elif (
            entailment > contradiction
            and entailment >= unverifiable
        ):

            label = "ENTAILMENT"
            is_hallucinated = False
            confidence = entailment

        else:

            label = "UNVERIFIABLE"
            is_hallucinated = True
            confidence = unverifiable

        return {
            "claim": summary_claim,

            "label": label,

            "is_hallucinated": is_hallucinated,

            "confidence": round(
                float(confidence),
                4,
            ),

            "scores": {
                "entailment": round(
                    entailment,
                    4,
                ),
                "contradiction": round(
                    contradiction,
                    4,
                ),
                "unverifiable": round(
                    unverifiable,
                    4,
                ),
            },

            "evidence": best_result["evidence"],

            "evidence_similarity": round(
                best_result["similarity"],
                4,
            ),
        }

    # ---------------------------------------------------------
    # FULL SUMMARY ANALYSIS
    # ---------------------------------------------------------

    def analyze_summary(
        self,
        source_text: str,
        generated_summary: str,
    ) -> dict[str, Any]:

        summary_sentences = (
            self.split_sentences(
                generated_summary
            )
        )

        source_sentences = (
            self.split_sentences(source_text)
        )

        analyzed_sentences = []

        for sentence in summary_sentences:

            if len(sentence.strip()) < 10:
                continue

            result = self.analyze_claim(
                source_text,
                sentence,
            )

            analyzed_sentences.append(
                result
            )

        total_claims = len(
            analyzed_sentences
        )

        entailed_claims = sum(
            result["label"] == "ENTAILMENT"
            for result in analyzed_sentences
        )

        contradicted_claims = sum(
            result["label"]
            == "CONTRADICTION"
            for result in analyzed_sentences
        )

        unverifiable_claims = sum(
            result["label"]
            == "UNVERIFIABLE"
            for result in analyzed_sentences
        )

        hallucinated_claims = sum(
            result["is_hallucinated"]
            for result in analyzed_sentences
        )

        if total_claims:

            hallucination_ratio = (
                hallucinated_claims
                / total_claims
            )

            average_confidence = float(
                np.mean(
                    [
                        result["confidence"]
                        for result in analyzed_sentences
                    ]
                )
            )

        else:

            hallucination_ratio = 0.0
            average_confidence = 0.0

        # -----------------------------------------------------
        # DOCUMENT DECISION
        # -----------------------------------------------------

        if contradicted_claims > 0:

            overall_label = (
                "HALLUCINATION_DETECTED"
            )

        elif hallucination_ratio >= 0.25:

            overall_label = (
                "POTENTIAL_HALLUCINATION"
            )

        else:

            overall_label = "SUPPORTED"

        return {

            "overall_label": overall_label,

            "overall_hallucinated": (
                overall_label
                != "SUPPORTED"
            ),

            "overall_confidence": round(
                average_confidence,
                4,
            ),

            "total_claims": total_claims,

            "supported_claims": (
                entailed_claims
            ),

            "contradicted_claims": (
                contradicted_claims
            ),

            "unverifiable_claims": (
                unverifiable_claims
            ),

            "hallucinated_claims": (
                hallucinated_claims
            ),

            "hallucination_ratio": round(
                hallucination_ratio,
                4,
            ),

            "source_sentence_count": (
                len(source_sentences)
            ),

            "summary_sentence_count": (
                total_claims
            ),

            "sentence_analysis": (
                analyzed_sentences
            ),
        }