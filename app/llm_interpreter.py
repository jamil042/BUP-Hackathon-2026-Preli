import json
import os
import re
import time
from app.prompts import SYSTEM_PROMPT, build_user_prompt

class LLMUnavailableError(Exception):
    pass

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_RETRYABLE_MARKERS = ("resourceexhausted", "ratelimit", "rate limit", "429", "quota")

def _is_retryable(exc: Exception) -> bool:
    if getattr(exc, "code", None) == 429:
        return True
    try:
        name = type(exc).__name__.lower()
    except Exception:
        name = ""
    message = str(exc).lower()
    return any(marker in name or marker in message for marker in _RETRYABLE_MARKERS)

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

def interpret_notes(notes: list[str], battery_capacity_kwh: float, model_client=None,
                    max_attempts: int = 4, base_delay_s: float = 2.0) -> list[dict]:
    client = model_client if model_client is not None else _get_default_client()
    prompt = build_user_prompt(notes, battery_capacity_kwh)

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = client.generate_content(prompt)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < max_attempts and _is_retryable(exc):
                time.sleep(base_delay_s * (2 ** attempt))
                continue
            raise LLMUnavailableError(str(exc)) from exc

        try:
            raw_text = _strip_fences(response.text)
            parsed = json.loads(raw_text)
        except Exception as exc:
            raise LLMUnavailableError(str(exc)) from exc

        if not isinstance(parsed, list):
            raise LLMUnavailableError("model did not return a JSON array")
        return parsed

    raise LLMUnavailableError(f"rate_limited after {max_attempts} attempts: {last_exc}")