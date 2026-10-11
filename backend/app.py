from pathlib import Path
try:
    from backend.hallucination_detector import HallucinationDetector
except ModuleNotFoundError:
    from hallucination_detector import HallucinationDetector
from flask import Flask, jsonify, render_template, request
import time
try:
    from backend.analysis import analyze_text
    from backend.led_model import LED_SUMMARIZER
    from backend.file_utils import extract_text_from_upload
    from backend.pegasus_model import PEGASUS_SUMMARIZER
    from backend.summarizer import generate_overview, normalize_input_text
    from backend.supervised_model import load_supervised_summarizer
except ModuleNotFoundError:
    from analysis import analyze_text
    from led_model import LED_SUMMARIZER
    from file_utils import extract_text_from_upload
    from pegasus_model import PEGASUS_SUMMARIZER
    from summarizer import generate_overview, normalize_input_text
    from supervised_model import load_supervised_summarizer


BASE_DIR = Path(__file__).resolve().parent.parent
TRAINED_SUMMARIZER = load_supervised_summarizer()

def get_trained_summarizer():
    global TRAINED_SUMMARIZER
    if TRAINED_SUMMARIZER is None:
        TRAINED_SUMMARIZER = load_supervised_summarizer()
    return TRAINED_SUMMARIZER

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

print("Initializing Hallucination Detector (this might take a moment)...")
hallucination_detector = HallucinationDetector()

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/summarize")
def summarize():
    uploaded_file = request.files.get("file")
    is_form_request = request.content_type and (
        "multipart/form-data" in request.content_type
        or "application/x-www-form-urlencoded" in request.content_type
    )

    if is_form_request:
        text = (request.form.get("text") or "").strip()
        summary_length = request.form.get("summary_length", 3)
        summary_preset = request.form.get("summary_preset")
        model_type = (request.form.get("model_type") or "led").strip().lower()
    else:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        summary_length = payload.get("summary_length", 3)
        summary_preset = payload.get("summary_preset")
        model_type = (payload.get("model_type") or "led").strip().lower()

    file_name = None

    if uploaded_file and uploaded_file.filename:
        try:
            extracted_text, file_name = extract_text_from_upload(uploaded_file)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        if extracted_text.strip():
            text = extracted_text

    if not text:
        return jsonify({"error": "Please enter text or upload a supported file to summarize."}), 400

    text = normalize_input_text(text)

    try:
        summary_length = int(summary_length)
    except (TypeError, ValueError):
        summary_length = 3

    if summary_length < 1 or summary_length > 10:
        summary_length = 3

    try:
        if model_type == "pegasus":
            summary = PEGASUS_SUMMARIZER.summarize(text, summary_length=summary_length, preset=summary_preset)
        else:
            model_type = "led"
            summary = LED_SUMMARIZER.summarize(text, summary_length=summary_length, preset=summary_preset)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        if model_type == "pegasus":
            label = "PEGASUS"
        elif model_type == "led":
            label = "LED"
        else:
            label = "summarization"
        return jsonify({"error": f"{label} failed: {error}"}), 500


    overview = generate_overview(summary or text)
    analysis = analyze_text(text, summary)

    return jsonify(
        {
            "model_type": model_type,
            "overview": overview,
            "summary": summary,
            "source_file": file_name,
            "source_text": text,
            "analysis": analysis,
            "hallucination_analysis": None,
        }
    )

@app.post("/analyze_hallucination")
def analyze_hallucination():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    summary = (payload.get("summary") or "").strip()

    if not text or not summary:
        return jsonify({"error": "Missing text or summary"}), 400

    print("Running Hallucination Analysis...")
    hallucination_start = time.perf_counter()

    try:
        hallucination_analysis = (
            hallucination_detector.analyze_summary(
                text,
                summary
            )
        )
        hallucination_time = (
            time.perf_counter()
            - hallucination_start
        )
        hallucination_analysis[
            "processing_time_seconds"
        ] = round(
            hallucination_time,
            3
        )
    except Exception as error:
        print(
            f"Error during hallucination detection: {error}"
        )
        hallucination_analysis = {
            "error": "Failed to analyze hallucination",
            "details": str(error),
        }

    return jsonify(hallucination_analysis)


@app.get("/privacy")
def privacy():
    return render_template("privacy.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
     app.run(debug=True, use_reloader=False)
