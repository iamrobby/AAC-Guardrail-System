"""AAC pipeline: gate -> (clarify -> re-gate)* -> answer."""
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

from src import config
from src.ambiguity_gate import GateResult, check_ambiguity
from src.llm_client import call_llm

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's request accurately and completely. "
    "If a detail is genuinely unspecified, state the assumption you are making."
)

History = List[Tuple[List[str], str]]  # [(questions asked, user's reply), ...]


def render_history(history: History) -> str:
    lines = []
    for questions, reply in history:
        lines.append("Assistant asked: " + " ".join(questions))
        lines.append("User answered: " + reply)
    return "\n".join(lines)


def answer_request(request: str, history: Optional[History] = None, format_hint: str = "") -> str:
    """Final answering step. Used by the guardrail AND the baseline so comparisons are fair."""
    content = request
    if history:
        content += (
            "\n\nAdditional details the user provided when asked for clarification:\n"
            + render_history(history)
        )
    if format_hint:
        content += "\n\n" + format_hint
    return call_llm(
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=config.ANSWER_TEMPERATURE,
    )


@dataclass
class PipelineResult:
    status: str                                   # "answered" | "needs_clarification"
    answer: Optional[str] = None
    questions: List[str] = field(default_factory=list)   # pending questions if status != answered
    history: History = field(default_factory=list)
    gates: List[GateResult] = field(default_factory=list)
    rounds_asked: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class AACGuardrail:
    def __init__(self, threshold: Optional[float] = None, max_rounds: Optional[int] = None):
        self.threshold = config.AMBIGUITY_THRESHOLD if threshold is None else threshold
        self.max_rounds = config.MAX_CLARIFY_ROUNDS if max_rounds is None else max_rounds

    @staticmethod
    def _fallback_questions(gate: GateResult) -> List[str]:
        """Used when the gate flagged ambiguity but produced no usable question."""
        if gate.missing_info:
            return [f"Could you tell me more about: {m}?" for m in gate.missing_info[: config.MAX_QUESTIONS]]
        return ["Could you give me more details about exactly what you need?"]

    def run(
        self,
        user_message: str,
        reply_fn: Optional[Callable[[List[str]], str]] = None,
        format_hint: str = "",
    ) -> PipelineResult:
        """Run the guardrail.

        reply_fn(questions) -> the user's reply (a human via input(), or a simulator).
        If reply_fn is None and clarification is needed, returns status="needs_clarification"
        with the questions, so an API/UI can ask the user and call run() again.
        """
        history: History = []
        gates: List[GateResult] = []
        rounds_asked = 0

        while True:
            gate = check_ambiguity(user_message, context=render_history(history), threshold=self.threshold)
            gates.append(gate)
            if not gate.is_ambiguous or rounds_asked >= self.max_rounds:
                break

            questions = gate.questions or self._fallback_questions(gate)
            if reply_fn is None:
                return PipelineResult("needs_clarification", questions=questions, history=history,
                                      gates=gates, rounds_asked=rounds_asked)

            reply = (reply_fn(questions) or "").strip()
            rounds_asked += 1
            if not reply:       # user declined to answer -> proceed with stated assumptions
                break
            history.append((questions, reply))

        answer = answer_request(user_message, history, format_hint)
        return PipelineResult("answered", answer=answer, history=history, gates=gates,
                              rounds_asked=rounds_asked)
