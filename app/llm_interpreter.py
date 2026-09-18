import json
import os
import re
from app.prompts import SYSTEM_PROMPT, build_user_prompt

class LLMUnavailableError(Exception):
    pass

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()

def _get_default_client():
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUnavailableError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-flash-lite-latest",
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )

def interpret_notes(notes: list[str], battery_capacity_kwh: float, model_client=None) -> list[dict]:
    client = model_client if model_client is not None else _get_default_client()
    try:
        response = client.generate_content(build_user_prompt(notes, battery_capacity_kwh))
        raw_text = _strip_fences(response.text)
        parsed = json.loads(raw_text)
    except LLMUnavailableError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(str(exc)) from exc

    if not isinstance(parsed, list):
        raise LLMUnavailableError("model did not return a JSON array")
    return parsed
