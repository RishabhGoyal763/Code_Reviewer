import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from gemini_client import review_code

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")}


def clean_json_output(raw: str):
    """Clean Gemini output for valid JSON."""
    if not raw:
        return "{}"
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
    cleaned = re.sub(r"```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]
    return cleaned


@app.post("/api/review")
def api_review():
    data = request.get_json(silent=True) or {}
    code = data.get("code", "").strip()
    language = data.get("language")
    if not code:
        return jsonify({"error": "No code provided."}), 400

    try:
        raw = review_code(code, language)
        cleaned = clean_json_output(raw)
        parsed = json.loads(cleaned)
        return jsonify(parsed)
    except json.JSONDecodeError:
        return jsonify({"summary": "Gemini returned non-JSON. Showing raw.", "raw": raw})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)

