"""Offline tests: the LLM is mocked, so no API key or network is needed.   python -m pytest -q"""
import json

import pytest

import src.ambiguity_gate as gate_mod
import src.pipeline as pipeline_mod
from eval.grader import extract_code, extract_final_number, grade_math, run_humaneval_test
from eval_gate import compute_metrics
from src.ambiguity_gate import AMBIGUOUS, CLEAR, check_ambiguity, parse_gate_output
from src.llm_client import LLMError
from src.pipeline import AACGuardrail, render_history


def gate_json(score, questions=(), missing=(), categories=()):
    return json.dumps({
        "ambiguity_score": score,
        "verdict": "AMBIGUOUS" if score >= 0.5 else "CLEAR",
        "categories": list(categories),
        "missing_info": list(missing),
        "questions": list(questions),
    })


# ------------------------------------------------------------------ gate parsing
def test_parse_clear_json():
    r = parse_gate_output(gate_json(0.05))
    assert r.label == CLEAR and not r.is_ambiguous and r.questions == []


def test_parse_ambiguous_json_with_fences_and_filtering():
    raw = "```json\n" + gate_json(0.9, ["Which language?", " ", "Which input?", "extra?"],
                                  categories=["missing_task_details", "made_up"]) + "\n```"
    r = parse_gate_output(raw)
    assert r.label == AMBIGUOUS
    assert r.questions == ["Which language?", "Which input?"]      # blanks dropped, capped at 2
    assert r.categories == ["missing_task_details"]                # unknown category dropped


def test_parse_json_embedded_in_prose():
    r = parse_gate_output("Sure! Here you go: " + gate_json(0.8, ["What exactly?"]) + " Hope that helps.")
    assert r.label == AMBIGUOUS


def test_threshold_controls_label():
    raw = gate_json(0.6, ["q?"])
    assert parse_gate_output(raw, threshold=0.5).label == AMBIGUOUS
    assert parse_gate_output(raw, threshold=0.7).label == CLEAR
    assert parse_gate_output(raw, threshold=0.7).questions == []   # CLEAR carries no questions


def test_verdict_only_json():
    assert parse_gate_output('{"verdict": "clear"}').ambiguity_score == 0.0
    assert parse_gate_output('{"verdict": "AMBIGUOUS"}').ambiguity_score == 1.0


@pytest.mark.parametrize("raw", ["CLEAR", "clear.", "  **CLEAR**  "])
def test_legacy_clear(raw):
    assert parse_gate_output(raw).label == CLEAR


def test_legacy_questions():
    r = parse_gate_output("1. Which language?\n2. What input should it take?")
    assert r.label == AMBIGUOUS and r.questions == ["Which language?", "What input should it take?"]


@pytest.mark.parametrize("raw", ["I think this is fine.", "{}", '{"verdict": "maybe"}'])
def test_unparseable_raises(raw):
    with pytest.raises(ValueError):
        parse_gate_output(raw)


def test_gate_failure_is_fail_open_by_default(monkeypatch):
    def boom(*a, **k):
        raise LLMError("down")
    monkeypatch.setattr(gate_mod, "call_llm", boom)
    r = check_ambiguity("anything")
    assert r.label == CLEAR and r.error and r.ambiguity_score == 0.0
    assert check_ambiguity("anything", on_error=AMBIGUOUS).label == AMBIGUOUS


def test_gate_sends_context(monkeypatch):
    seen = {}
    monkeypatch.setattr(gate_mod, "call_llm", lambda msgs, **k: seen.setdefault("m", msgs) and gate_json(0.1))
    check_ambiguity("Write a function", context="Assistant asked: lang?\nUser answered: Python")
    assert "<conversation_so_far>" in seen["m"][1]["content"] and "Python" in seen["m"][1]["content"]


# ------------------------------------------------------------------ pipeline
def patch_llm(monkeypatch, gate_replies, answer="FINAL"):
    queue = iter(gate_replies)
    calls = {"gate": 0, "answer_prompt": None}

    def fake_gate(*a, **k):
        calls["gate"] += 1
        return next(queue)

    def fake_answer(messages, **k):
        calls["answer_prompt"] = messages[-1]["content"]
        return answer

    monkeypatch.setattr(gate_mod, "call_llm", fake_gate)
    monkeypatch.setattr(pipeline_mod, "call_llm", fake_answer)
    return calls


def test_clear_request_is_answered_without_questions(monkeypatch):
    calls = patch_llm(monkeypatch, [gate_json(0.02)])
    res = AACGuardrail().run("What is 2+2?", reply_fn=lambda q: pytest.fail("must not ask"))
    assert res.status == "answered" and res.rounds_asked == 0 and calls["gate"] == 1


def test_ambiguous_then_clear_includes_user_reply(monkeypatch):
    calls = patch_llm(monkeypatch, [gate_json(0.9, ["Which language?"]), gate_json(0.1)])
    res = AACGuardrail(max_rounds=2).run("Write a function", reply_fn=lambda q: "Python please",
                                         format_hint="HINT")
    assert res.status == "answered" and res.rounds_asked == 1 and calls["gate"] == 2
    assert "Python please" in calls["answer_prompt"] and "HINT" in calls["answer_prompt"]
    assert res.history == [(["Which language?"], "Python please")]


def test_round_budget_is_respected(monkeypatch):
    calls = patch_llm(monkeypatch, [gate_json(0.9, ["q?"])] * 5)
    asked = []
    res = AACGuardrail(max_rounds=2).run("vague", reply_fn=lambda q: asked.append(q) or "some detail")
    assert len(asked) == 2 and res.rounds_asked == 2 and calls["gate"] == 3
    assert res.status == "answered"                        # answers with assumptions after budget


def test_needs_clarification_without_reply_fn(monkeypatch):
    patch_llm(monkeypatch, [gate_json(0.9, ["Which language?"])])
    res = AACGuardrail().run("Write a function")
    assert res.status == "needs_clarification" and res.questions == ["Which language?"] and res.answer is None


def test_fallback_questions_when_gate_gives_none(monkeypatch):
    patch_llm(monkeypatch, [gate_json(0.9, [], missing=["the input list"]), gate_json(0.1)])
    seen = []
    AACGuardrail().run("Sort it", reply_fn=lambda q: seen.append(q) or "a list of ints")
    assert seen[0] == ["Could you tell me more about: the input list?"]


def test_empty_user_reply_stops_asking(monkeypatch):
    calls = patch_llm(monkeypatch, [gate_json(0.9, ["q?"])] * 3)
    res = AACGuardrail(max_rounds=3).run("vague", reply_fn=lambda q: "   ")
    assert res.status == "answered" and res.history == [] and calls["gate"] == 1


def test_render_history():
    assert render_history([(["a?", "b?"], "yes")]) == "Assistant asked: a? b?\nUser answered: yes"


# ------------------------------------------------------------------ graders
def test_extract_final_number():
    assert extract_final_number("so 3,360 points.\nFinal answer: $3,360.") == 3360
    assert extract_final_number("**Final answer: 12.5**") == 12.5
    assert extract_final_number("Final answer:\n7") == 7
    assert extract_final_number("it is 4 then 9") == 9           # fallback: last number
    assert extract_final_number("no digits") is None


def test_grade_math():
    assert grade_math("Final answer: 1,000", "1000")
    assert not grade_math("Final answer: 999", "1000")


ADD_TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n"


def test_extract_code_prefers_target_function():
    text = "```python\nprint('demo')\n```\n```python\ndef add(a, b):\n    return a + b\n```"
    assert "def add" in extract_code(text, "add")


def test_run_humaneval_test_pass_fail_timeout():
    assert run_humaneval_test("def add(a, b):\n    return a + b", ADD_TEST, "add", timeout=10)
    assert not run_humaneval_test("def add(a, b):\n    return a - b", ADD_TEST, "add", timeout=10)
    assert not run_humaneval_test("def add(a, b):\n    while True: pass", ADD_TEST, "add", timeout=2)
    assert not run_humaneval_test("def add(a b):", ADD_TEST, "add", timeout=10)      # syntax error


# ------------------------------------------------------------------ metrics
def test_compute_metrics():
    rows = ([{"expected": AMBIGUOUS, "score": 0.9}] * 3 + [{"expected": AMBIGUOUS, "score": 0.1}] * 2
            + [{"expected": CLEAR, "score": 0.8}] * 1 + [{"expected": CLEAR, "score": 0.2}] * 4)
    m = compute_metrics(rows, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 2, 1, 4)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.6)
    assert m["accuracy"] == pytest.approx(0.7)
    assert compute_metrics([], 0.5)["f1"] == 0.0
