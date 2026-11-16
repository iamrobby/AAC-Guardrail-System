"""Central configuration for the AAC (Ambiguity-Aware Clarification) guardrail."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# --- LLM backend (Groq through its OpenAI-compatible endpoint) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("AAC_MODEL", "openai/gpt-oss-20b")
LLM_RETRIES = int(os.getenv("AAC_LLM_RETRIES", "4"))
LLM_TIMEOUT = float(os.getenv("AAC_LLM_TIMEOUT", "60"))
ANSWER_TEMPERATURE = float(os.getenv("AAC_ANSWER_TEMP", "0.2"))

# --- Guardrail behaviour ---
# A request is AMBIGUOUS when the gate's ambiguity_score >= this threshold.
AMBIGUITY_THRESHOLD = float(os.getenv("AAC_THRESHOLD", "0.5"))
# Maximum number of clarification rounds before answering with stated assumptions.
MAX_CLARIFY_ROUNDS = int(os.getenv("AAC_MAX_ROUNDS", "2"))
# Maximum clarifying questions per round.
MAX_QUESTIONS = 2

# --- Evaluation ---
CODE_EXEC_TIMEOUT = int(os.getenv("AAC_CODE_TIMEOUT", "10"))
