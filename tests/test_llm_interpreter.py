import json
import pytest
from app.llm_interpreter import interpret_notes, LLMUnavailableError

class FakeResponse:
    def __init__(self, text):
        self.text = text

class FakeModel:
    def __init__(self, reply_text=None, raise_exc=None):
        self.reply_text = reply_text
        self.raise_exc = raise_exc

    def generate_content(self, *args, **kwargs):
        if self.raise_exc:
            raise self.raise_exc
        return FakeResponse(self.reply_text)

def test_parses_valid_json_array_from_model():
    reply = json.dumps([
        {"note_index": 0, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [13, 14], "factor": 0.2}, "explanation": "x"}
    ])
    result = interpret_notes(["Solar drops to 20% 1-3pm."], 500.0, model_client=FakeModel(reply_text=reply))
    assert result[0]["directive_type"] == "solar_reduction"

def test_strips_markdown_fences_if_present():
    reply = "```json\n" + json.dumps([{"note_index": 0, "directive_type": "no_op",
                                        "structured_adjustment": None, "explanation": "x"}]) + "\n```"
    result = interpret_notes(["Unrelated note."], 500.0, model_client=FakeModel(reply_text=reply))
    assert result[0]["directive_type"] == "no_op"

def test_raises_llm_unavailable_on_client_exception():
    with pytest.raises(LLMUnavailableError):
        interpret_notes(["Any note."], 500.0, model_client=FakeModel(raise_exc=RuntimeError("api down")))

def test_raises_llm_unavailable_on_unparseable_output():
    with pytest.raises(LLMUnavailableError):
        interpret_notes(["Any note."], 500.0, model_client=FakeModel(reply_text="not json at all"))
