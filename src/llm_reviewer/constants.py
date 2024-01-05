import os
from enum import Enum

PROJECT_ROOT = os.path.dirname(__file__) + "/"
PROJECT_CODEBASE = PROJECT_ROOT
PATH_TO_CONFIG = os.path.join(PROJECT_CODEBASE, "config.json")


class Roles(Enum):
    # must be the same as keys in the config
    HUMAN = "human"
    LLM = "llm"
