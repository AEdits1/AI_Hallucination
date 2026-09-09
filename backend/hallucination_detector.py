import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download the sentence tokenizer model if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

class HallucinationDetector:
    def __init__(self, model_name="roberta-large-mnli"):
        """
        Initializes the RoBERTa NLI cross-encoder model for entailment scoring.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Loading NLI model '{model_name}' on device: {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def analyze_claim(self, source_text: str, summary_claim: str) -> dict:
        """
        Compares a single summary statement against the source context.
        """
        inputs = self.tokenizer(
            source_text,
            summary_claim,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        contradiction_prob = float(probabilities[0])
        neutral_prob = float(probabilities[1])
        entailment_prob = float(probabilities[2])

        is_hallucinated = contradiction_prob > entailment_prob or contradiction_prob > 0.45

        return {
            "claim": summary_claim,
            "is_hallucinated": is_hallucinated,
            "scores": {
                "contradiction": round(contradiction_prob, 4),
                "neutral": round(neutral_prob, 4),
                "entailment": round(entailment_prob, 4)
            },
            "status": "Hallucinated" if is_hallucinated else ("Factual" if entailment_prob >= 0.5 else "Unverified")
        }

    def analyze_summary(self, source_text: str, generated_summary: str) -> dict:
        """
        Splits the full summary into sentences and analyzes each one.
        """
        # Split paragraph into individual sentences
        sentences = nltk.sent_tokenize(generated_summary)
        analyzed_sentences = []
        
        has_hallucination = False
        total_contradiction = 0.0

        for sentence in sentences:
            # Skip very short fragments
            if len(sentence.strip()) < 10:
                continue
                
            result = self.analyze_claim(source_text, sentence)
            analyzed_sentences.append(result)
            
            total_contradiction += result["scores"]["contradiction"]
            if result["is_hallucinated"]:
                has_hallucination = True
                
        avg_risk = (total_contradiction / len(analyzed_sentences)) if analyzed_sentences else 0.0

        return {
            "overall_hallucinated": has_hallucination,
            "overall_risk_score": round(avg_risk, 4),
            "sentence_analysis": analyzed_sentences
        }

if __name__ == "__main__":
    detector = HallucinationDetector()
    
    context = "Apollo 11 was the spaceflight that first landed humans on the Moon in July 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on July 20, 1969."
    
    # A mix of truth and hallucination
    mixed_summary = "Apollo 11 successfully landed on the Moon in 1969. The mission was commanded by Yuri Gagarin. Neil Armstrong was also part of the crew."

    print("\n--- Testing Full Summary Analysis ---")
    import json
    result = detector.analyze_summary(context, mixed_summary)
    print(json.dumps(result, indent=2))