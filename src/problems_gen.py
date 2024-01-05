import json
from typing import List, Dict, Set

import os
from openai import OpenAI

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI()


def load_problems(file_path: str) -> List[dict]:
    """
    Load a list of problems from a JSON file.

    Parameters:
    file_path: str
        The path to the JSON file containing problems data.

    Returns:
    list
        A list of problems loaded from the JSON file, or an empty list if the file
        cannot be found or read as valid JSON.
    """
    existing_problems = []
    try:
        with open(file_path, "r") as f:
            existing_problems = json.load(f)
    except FileNotFoundError:
        print(f"{file_path} not found.")
    except json.JSONDecodeError:
        print(f"{file_path} is not a valid JSON file.")

    return existing_problems


def extract_questions(problems: List[dict]) -> Set[str]:
    """
    Extract unique questions from a list of existing problems.

    Parameters:
    problems: list
        The list of problems, each of which is a dictionary that may contain a
        list of messages under the key "messages".

    Returns:
    set
        A set of unique questions asked by users in the problem messages.
    """
    existing_questions = set()

    for problem in problems:
        messages = problem.get("messages", [])
        for message in messages:
            if message.get("role") == "user":
                user_content = message.get("content")
                if user_content:
                    existing_questions.add(user_content)

    return existing_questions


def find_and_load_all_problems(parent_folder: str) -> List[dict]:
    """
    Look for problems.json files in each child folder, concatenate the content, and return a list of problems.

    Parameters:
    parent_folder: str
        The path to the parent directory.

    Returns:
    list
        A list of problems loaded from all the problems.json files within the child directories.
    """
    all_problems = []
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file == "problems.json":
                json_file_path = os.path.join(root, file)
                loaded_problems = load_problems(json_file_path)
                all_problems.extend(loaded_problems)

    return all_problems


def extract_questions_by_topic(problems: List[dict]) -> Dict[str, Set[str]]:
    """
    Extract questions grouped by topic from a list of existing problems.

    Parameters:
    problems: list
        The list of problems, each of which is a dictionary with "metadata" and "messages".

    Returns:
    dict
        A dictionary where each key is a topic string, and each value is a set of unique
        questions asked by users in the problem messages pertaining to that topic.
    """
    questions_by_topic = {}

    for problem in problems:
        topic = problem.get("metadata", {}).get("topic", "Unknown Topic")
        messages = problem.get("messages", [])
        for message in messages:
            if message.get("role") == "user":
                user_content = message.get("content")
                if user_content:
                    if topic not in questions_by_topic:
                        questions_by_topic[topic] = set()
                    questions_by_topic[topic].add(user_content)

    return questions_by_topic


def generate_human_like_questions(topic, n=5, existing_questions=None):
    SYSTEM_PROMPT = f"""IDENTITY:
You are a world class Python developer. And you're concise and precise.

CONTEXT:
We are trying to generate human-like queries that a user would send to an ai assistant through a chat interface. 
The user's query tone & structure should be diversified as much as possible making sure to include some realistic examples.
We already have batch of previously generated questions and we want to avoid duplication of questions when generating new batch.

CONSTRAINTS:
1. Python Related
2. Easy (Solvable by a median developer in 15 minutes)
3. Questions should elicit a response that includes code.

INSTRUCTION:
You will be given a topic and an ask for number of questions to generate.
You will also be given previously generated questions for respective topic and ask to ignore generating question if exist.
The aim is to maximize diversity.
Act accordingly.

RESPONSE FORMAT:
A JSON-valid list of questions(strings) like {{"questions": ["question1", "question2", ...]}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Topic: {topic} \nNumber of questions: {n} \n existing questions {existing_questions}",
                },
            ],
            temperature=0.0,
            max_tokens=4096,
            seed=42,
            response_format={"type": "json_object"},
        )
        questions = json.loads(response.choices[0].message.content)
        return questions
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def generate_human_like_code_modification_requests(topic, n=5, existing_questions=None):
    SYSTEM_PROMPT = f"""IDENTITY:
You are a world class Python developer. And you're concise and precise.

CONTEXT:
We are trying to generate human-like code modification requests that a user would send to an ai assistant through a chat interface.
The user's query tone & structure should be diversified as much as possible making sure to include some realistic examples.
We already have batch of previously generated questions and we want to avoid duplication of questions when generating new batch.

CONSTRAINTS:
1. Python Related
2. Easy (Solvable by a median developer in 15 minutes)
3. Questions should include code along with a request to modify it.

INSTRUCTION:
You will be given a topic and an ask for number of questions to generate.
You will also be given previously generated questions for respective topic and ask to ignore generating question if exist.
The aim is to maximize diversity.
Act accordingly.

RESPONSE FORMAT:
A JSON-valid list of questions(strings) like {{"questions": ["question1", "question2", ...]}}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Topic: {topic} \nNumber of code modification requests: {n} \n existing topic {existing_questions}",
                },
            ],
            temperature=0.0,
            max_tokens=4096,
            seed=42,
            response_format={"type": "json_object"},
        )
        questions = json.loads(response.choices[0].message.content)
        return questions
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
