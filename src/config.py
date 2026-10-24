import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# OpenRouter Free LLM Models Configuration
# Selected from confirmed $0/token free tier models on OpenRouter
# OpenRouter Free LLM Models Configuration
# Selected from confirmed $0/token active free tier models on OpenRouter
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "dots-studio/dots-3-note-preview:free")
FALLBACK_OPENROUTER_MODELS = [
    "inclusionai/ling-3.0-flash-sante:free",
    "poolside/laguna-s-2.1:free",
    "qwen/qwen3.8-27b:free",
    "poolside/laguna-xs-2.1:free",
    "thinkingmachines/inkling:free",
    "liquid/lfm-2.5-2.6b:free",
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Embedding Model (Local SentenceTransformer)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Domain Directories
DOMAINS = ["hr", "finance", "it_access"]

POLICY_DIRS = {
    "hr": BASE_DIR / "data" / "policies" / "hr",
    "finance": BASE_DIR / "data" / "policies" / "finance",
    "it_access": BASE_DIR / "data" / "policies" / "it_access",
}

EDITED_POLICY_DIRS = {
    "hr": BASE_DIR / "data" / "policies_edited" / "hr",
    "finance": BASE_DIR / "data" / "policies_edited" / "finance",
    "it_access": BASE_DIR / "data" / "policies_edited" / "it_access",
}

SCENARIOS_FILES = {
    "hr": BASE_DIR / "eval" / "test_scenarios" / "hr_scenarios.json",
    "finance": BASE_DIR / "eval" / "test_scenarios" / "finance_scenarios.json",
    "it_access": BASE_DIR / "eval" / "test_scenarios" / "it_access_scenarios.json",
}

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Random Seeds for Replicability
EVAL_SEEDS = [42, 123, 999]
TOP_K_RETRIEVAL = 3
