import os
import json

from src.llm_reviewer.constants import PROJECT_CODEBASE
from src.llm_reviewer.llm_api import make_llm_request


def load_system_prompt(reviewer_name: str) -> str:
    """
    Load the content of a system prompt file.

    :param reviewer_name: The name of the reviewer for which to load the prompt.
    :return: The content of the prompt file.
    """
    prompt_path = os.path.join(PROJECT_CODEBASE, "prompts", f"{reviewer_name}.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def load_rubric(reviewer_name: str) -> str:
    """
    Load the content of a rubric file.

    :param reviewer_name: The name of the reviewer for which to load the rubric.
    :return: The content of the rubric file.
    """
    rubric_path = os.path.join(
        PROJECT_CODEBASE, "prompts", f"{reviewer_name}_rubrics.txt"
    )
    with open(rubric_path, "r", encoding="utf-8") as file:
        return file.read()


def load_schema(reviewer_name: str) -> dict:
    """
    Load the content of a schema file and construct a string that matches the schema.

    :param reviewer_name: The name of the reviewer for which to load the schema.
    :return: The content of the schema file.
    """
    schema_path = os.path.join(
        PROJECT_CODEBASE, "prompts", f"{reviewer_name}_schema.json"
    )
    with open(schema_path, "r", encoding="utf-8") as file:
        schema = json.load(file)

    # Construct a string that matches the schema
    # schema_string = {}
    # for key, value in schema["properties"].items():
    #     if value["type"] == "number":
    #         schema_string[key] = f"number between {value['minimum']} and {value['maximum']}"
    #     elif value["type"] == "array":
    #         if "$ref" in value["items"]:
    #             inner_keys = schema["definitions"]["InnerDict"]["properties"].keys()
    #             schema_string[key] = [{inner_key: "..." for inner_key in inner_keys}]
    #         else:
    #             schema_string[key] = [
    #                 {inner_key: "..." for inner_key in value["items"]["properties"].keys()}
    #             ]

    return schema


def review_turn(
    reviewer_name: str, formatted_turn: str, llm_client, llm_config: dict
) -> str:
    """
    Review a turn using the LLM with the given configuration and texts.

    :param reviewer_name: The name of the reviewer.
    :param texts: The texts to be reviewed - {human: "", llm: ""}
    :param llm_client: The LLM client to use for making requests.
    :param llm_config: The configuration for the LLM client.
    :return: The result of the LLM request.
    """
    prompt = load_system_prompt(reviewer_name)

    response_schema = load_schema(reviewer_name)
    prompt = (
        prompt.replace("@FORMATTED_TURN@", formatted_turn)
        .replace("@RESPONSE_FORMAT@", json.dumps(response_schema))
        .replace("@GRADING_RUBRIC@", load_rubric(reviewer_name))
    )

    messages = [{"role": "system", "content": prompt}]
    result = make_llm_request(client=llm_client, messages=messages, **llm_config)

    return result
