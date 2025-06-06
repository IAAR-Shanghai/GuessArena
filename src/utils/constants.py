"""Centralized constants for the GuessArena project."""

from .project_structure import DIRECTORIES, FILES

# Config files
MODELS_CONFIG = FILES["MODELS_CONFIG"]
IND_DOCS_CONFIG = FILES["IND_DOCS_CONFIG"]

# Prompt files
TESTER_SYSTEM_PATH = FILES["TESTER_SYSTEM"]
BASIC_TESTEE_SYSTEM_PATH = FILES["BASIC_TESTEE_SYSTEM"]
COT_TESTEE_SYSTEM_PATH = FILES["COT_TESTEE_SYSTEM"]
KNOW_TESTEE_SYSTEM_PATH = FILES["KNOW_TESTEE_SYSTEM"]

# Output directories
TESTSETS_DIR = DIRECTORIES["TESTSETS"]
RUN_LOGS_DIR = DIRECTORIES["RUN_LOGS"]
OVERALL_DIR = DIRECTORIES["OVERALL"]
KNOWLEDGE_DIR = DIRECTORIES["KNOWLEDGE"]
CARDS_DIR = DIRECTORIES["CARDS"]
