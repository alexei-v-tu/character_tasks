import json
from dotenv import load_dotenv, find_dotenv
import os


def load_config(config_file_path: str) -> dict[str, str]:
    with open(config_file_path) as config_file:
        config = json.load(config_file)
    return config


def load_env() -> dict:
    load_dotenv(find_dotenv())
    env_vars = {key: os.getenv(key) for key in os.environ}
    return env_vars
