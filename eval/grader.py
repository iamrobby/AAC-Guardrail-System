"""Answer graders: numeric match for GSM8K math, unit tests for HumanEval code."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from src import config

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_HEADER = "from typing import *\nimport math, re, itertools, collections, heapq, string, functools\n\n"


def format_hint(item: dict) -> str:
    """Task-specific output format, identical for baseline, AAC and oracle runs."""
    if item["task"] == "math":
        return "Show brief reasoning, then finish with a final line of the form: Final answer: <number>"
    return (f"Implement this as a single Python function named `{item['func_name']}`. "
            "Return the complete code in one ```python code block.")


# ------------------------------------------------------------------ math
def extract_final_number(text: str) -> Optional[float]:
    marked = re.findall(r"final answer\s*[:\uff1a]?\s*(.+)", text, re.IGNORECASE)
    scope = marked[-1] if marked else text
    numbers = _NUMBER.findall(scope)
    if not numbers:
        return None
    token = numbers[0] if marked else numbers[-1]
    return float(token.replace(",", ""))


def grade_math(response: str, gold_answer: str) -> bool:
    predicted = extract_final_number(response)
    return predicted is not None and abs(predicted - float(gold_answer)) < 1e-6


# ------------------------------------------------------------------ code
def extract_code(response: str, func_name: Optional[str] = None) -> str:
    blocks = _CODE_BLOCK.findall(response)
    if func_name:
        for block in blocks:
            if re.search(rf"def\s+{re.escape(func_name)}\s*\(", block):
                return block
    if blocks:
        return max(blocks, key=len)
    return response


def run_humaneval_test(code: str, test: str, func_name: str, timeout: Optional[int] = None) -> bool:
    """Execute model code + HumanEval `check()` in a subprocess with a timeout.

    NOTE: this runs model-generated code on your machine. It is not a hardened sandbox -
    run evaluations inside a container/VM if you need isolation.
    """
    timeout = config.CODE_EXEC_TIMEOUT if timeout is None else timeout
    program = f"{_CODE_HEADER}{code}\n\n{test}\n\ncheck({func_name})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)], cwd=tmp, capture_output=True, text=True,
                timeout=timeout, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0


def grade_item(item: dict, answer: str) -> bool:
    if item["task"] == "math":
        return grade_math(answer, item["gold_answer"])
    code = extract_code(answer, item["func_name"])
    return run_humaneval_test(code, item["test"], item["func_name"])
