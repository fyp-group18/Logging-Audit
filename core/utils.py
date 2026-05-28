# backend/core/utils.py
import json
import re


def parse_gemini_json(text: str) -> dict:
    """Strip markdown code fences from a Gemini response and parse it as JSON.

    Gemini sometimes wraps JSON output in ```json...``` fences or emits trailing
    content after the JSON object.  This function handles both cases via a
    two-pass strategy: first strip fences and try json.loads; if that fails,
    use JSONDecoder.raw_decode to extract only the first valid object.
    """
    clean_text = text.strip()
    clean_text = re.sub(r"^```json\s*", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^```\s*", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"```$", "", clean_text, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(clean_text)
        return result
