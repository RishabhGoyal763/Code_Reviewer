import os
import re
import json
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ------------------------------------------------------------
# Initialize Gemini Client
# ------------------------------------------------------------
def get_client():
    """
    Initialize and return a Gemini API client using the API key in .env.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


# ------------------------------------------------------------
# System Instructions for the AI Reviewer
# ------------------------------------------------------------
SYSTEM_INSTRUCTIONS = """
You are a highly accurate AI programming assistant.

Task:
Analyze the given source code carefully and decide whether it is syntactically and logically correct.
If the code compiles or runs properly without syntax or structural issues, mark it as "Correct".
If it has any errors (syntax, indentation, logical, or structural), mark it as "Incorrect".

IMPORTANT: Pay close attention to the specified programming language and apply language-specific syntax rules:
- Python: Check indentation, colons, proper use of self, correct function definitions
- JavaScript/TypeScript: Check semicolons (optional but consistent), brackets, var/let/const usage
- Java: Check class structure, type declarations, semicolons, access modifiers
- C/C++: Check pointers, memory management, semicolons, header includes
- C#: Check namespaces, type declarations, LINQ syntax
- Go: Check package declarations, error handling patterns, defer statements
- Rust: Check ownership, borrowing, lifetimes, match statements
- PHP: Check $ variable prefix, -> vs =>, correct function syntax
- Ruby: Check blocks, symbols, proper method definitions
- Swift: Check optionals, guard/let syntax, proper type inference
- Kotlin: Check nullable types, when expressions, data classes
- Dart: Check async/await, nullable types, proper constructors
- R: Check <- assignment, proper vectorization
- SQL: Check query structure, JOIN syntax, proper keywords
- HTML: Check tag closing, nesting, attribute syntax
- CSS: Check selector syntax, property-value pairs, semicolons

Return ONLY valid JSON in this exact format:
{
  "code_status": "Correct" | "Incorrect",
  "issues_found": ["List of specific problems if any"],
  "suggestions": ["List of improvement or optimization tips"]
}

Strict rules:
1. Do NOT include ```json fences or text outside JSON.
2. Mark as "Correct" if code runs successfully (even if improvements exist).
3. Mark as "Incorrect" ONLY if there are actual syntax or logic errors that prevent execution.
4. Always include at least one suggestion, even if the code is correct.
5. Do not overthink stylistic issues – correctness is based on syntax and logic, not formatting.
"""


# ------------------------------------------------------------
# Build the prompt to send to Gemini
# ------------------------------------------------------------
def build_prompt(user_code: str, language_hint: Optional[str]) -> str:
    lang = language_hint or "Auto-detect"
    return f"""
Language: {lang}

Analyze this code for syntax or logic errors.

Code:
{user_code}

Now, return JSON exactly as per the format above.
"""


# ------------------------------------------------------------
# Clean model output to extract JSON safely
# ------------------------------------------------------------
def _clean_to_json_str(text: str) -> Optional[str]:
    """
    Cleans the model output to extract valid JSON if possible.
    Returns a JSON string if valid, else None.
    """
    if not text:
        return None

    cleaned = text.strip()
    # Remove code fences if present
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)

    # Extract between first '{' and last '}'
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            json.loads(candidate)  # validate
            return candidate
        except Exception:
            return None
    return None


# ------------------------------------------------------------
# Review Code Function (Main Entry Point)
# ------------------------------------------------------------
def review_code(user_code: str, language_hint: Optional[str]) -> str:
    """
    Sends user code to Gemini, gets structured JSON response,
    and ensures it always returns valid JSON output.
    """
    client = get_client()

    # Combine system instructions and user code
    full_prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{build_prompt(user_code, language_hint)}"

    try:
        # Generate Gemini model response
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,          # Lower = more consistent
                top_p=0.9,
                max_output_tokens=1024
            ),
        )

        # ----------------------------------------------------
        # Safely extract text from Gemini response
        # ----------------------------------------------------
        text = getattr(response, "text", None)

        if not text:
            candidates = getattr(response, "candidates", [])
            if candidates:
                candidate = candidates[0]
                content = getattr(candidate, "content", None)
                if content:
                    parts = getattr(content, "parts", [])
                    if parts and hasattr(parts[0], "text"):
                        text = parts[0].text

        # ----------------------------------------------------
        # Clean and validate JSON output
        # ----------------------------------------------------
        json_str = _clean_to_json_str(text or "")
        if json_str:
            return json_str

        # ----------------------------------------------------
        # Fallback if model returns non-JSON
        # ----------------------------------------------------
        fallback = {
            "code_status": "Incorrect",
            "issues_found": ["Model did not return valid JSON output."],
            "suggestions": [
                "Re-run the review.",
                "Ensure the snippet is complete and includes required imports.",
            ],
        }
        return json.dumps(fallback)

    except Exception as e:
        # ----------------------------------------------------
        # Fallback if the model call itself fails
        # ----------------------------------------------------
        error_json = {
            "code_status": "Incorrect",
            "issues_found": [f"Model call failed: {str(e)}"],
            "suggestions": [
                "Check your GEMINI_API_KEY and internet connectivity.",
                "Ensure google-genai is installed (version 0.3.0).",
            ],
        }
        return json.dumps(error_json)