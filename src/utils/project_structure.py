"""
Project structure definitions and directory management for GuessArena.
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Define the core directory structure
DIRECTORIES = {
    # Source code
    "SRC": PROJECT_ROOT / "src",
    "EVAL": PROJECT_ROOT / "src" / "eval",
    "BUILD": PROJECT_ROOT / "src" / "build",
    "UTILS": PROJECT_ROOT / "src" / "utils",
    # Configuration
    "CONFIGS": PROJECT_ROOT / "configs",
    # Data files
    "DATA": PROJECT_ROOT / "data",
    "TESTSETS": PROJECT_ROOT / "data" / "testsets",
    "CARDS": PROJECT_ROOT / "data" / "cards",
    "DOCUMENTS": PROJECT_ROOT / "data" / "documents",
    "KNOWLEDGE": PROJECT_ROOT / "data" / "knowledge",
    # Prompts
    "PROMPTS": PROJECT_ROOT / "prompts",
    # Outputs
    "OUTPUTS": PROJECT_ROOT / "outputs",
    "RUN_LOGS": PROJECT_ROOT / "outputs" / "run_logs",
    "OVERALL": PROJECT_ROOT / "outputs" / "overall",
}

# Define specific files
FILES = {
    # Config files
    "MODELS_CONFIG": DIRECTORIES["CONFIGS"] / "models.ini",
    "IND_DOCS_CONFIG": DIRECTORIES["CONFIGS"] / "ind_docs.ini",
    # Prompt files
    "TESTER_SYSTEM": DIRECTORIES["PROMPTS"] / "tester_system.txt",
    "BASIC_TESTEE_SYSTEM": DIRECTORIES["PROMPTS"] / "basic_testee_system.txt",
    "COT_TESTEE_SYSTEM": DIRECTORIES["PROMPTS"] / "cot_testee_system.txt",
    "KNOW_TESTEE_SYSTEM": DIRECTORIES["PROMPTS"] / "know_testee_system.txt",
}


def ensure_project_structure():
    """Create all directories in the project structure if they don't exist."""
    for directory in DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)


def get_testset_path(topic: str, num_cards: int) -> Path:
    """Get the path to a testset file."""
    return DIRECTORIES["TESTSETS"] / f"{topic}_{num_cards}.txt"


def get_knowledge_path(topic: str) -> Path:
    """Get the path to a knowledge background file."""
    return DIRECTORIES["KNOWLEDGE"] / f"{topic}_kbg.txt"


def get_run_log_path(
    topic: str,
    tester: str,
    testee: str,
    prompt_strategy: str,
    num_cards: int,
    chosen_card: str,
    timestamp: str,
) -> Path:
    """Get the path for a run log file."""
    safe_card = chosen_card.replace(" ", "_").replace("/", "_").replace("\\", "_")
    filename = f"{topic}_{tester}_vs_{testee}_{prompt_strategy}_{num_cards}_{safe_card}_{timestamp}.json"
    return DIRECTORIES["RUN_LOGS"] / filename


def get_overall_path(
    topic: str, num_cards: int, testee: str, prompt_strategy: str, timestamp: str
) -> Path:
    """Get the path for an overall results file."""
    filename = f"{topic}_{num_cards}_{testee}_{prompt_strategy}_{timestamp}.json"
    return DIRECTORIES["OVERALL"] / filename


# Initialize the project structure when the module is imported
ensure_project_structure()
