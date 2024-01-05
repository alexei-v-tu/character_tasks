import os
import glob
import pytest
from llm_reviewer.llm_api import LLMAPIFactory
from src.llm_reviewer.utils import load_config
from llm_reviewer.constants import PATH_TO_CONFIG


@pytest.fixture
def llm_configs():
    return load_config(PATH_TO_CONFIG)


@pytest.fixture
def llm_client():
    return LLMAPIFactory().get()


@pytest.fixture
def notebook_samples():
    sample_dir = os.path.join(os.path.dirname(__file__), "samples")
    return sorted(glob.glob(os.path.join(sample_dir, "*.ipynb")))
