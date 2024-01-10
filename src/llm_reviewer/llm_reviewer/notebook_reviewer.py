from typing import Any
from src.llm_reviewer.notebook_parser import parse_notebook
from src.llm_reviewer.turn_reviewer import review_turn
from src.llm_reviewer.llm_api import LLMAPIFactory
from src.llm_reviewer.utils import load_config
from src.llm_reviewer.constants import PATH_TO_CONFIG, Roles
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import pandas as pd


def format_turn_data(turn: dict[str, Any]) -> dict[str, str]:
    """
    Format the turn data for review.

    :param turn: A dictionary representing a turn.
    :return: A formatted dictionary ready for review.
    """
    formatted_turn = ""
    for role_turn in turn:
        if role_turn["role"] == Roles.HUMAN.value:
            formatted_turn += "# HUMAN_REPLY_START\n"
        elif role_turn["role"] == Roles.LLM.value:
            formatted_turn += "# LLM_REPLY_START\n"
        else:
            raise ValueError(f"Unexpected role: {role_turn['role']}")
        for step in role_turn["steps"]:
            if step["type"] == "markdown":
                formatted_turn += step["content"] + "\n"
            elif step["type"] == "code":
                formatted_turn += "```\n" + step["content"] + "\n```\n"
            else:
                raise ValueError(f"Unexpected step type: {step['type']}")
        if role_turn["role"] == Roles.HUMAN.value:
            formatted_turn += "# HUMAN_REPLY_END\n\n"
        elif role_turn["role"] == Roles.LLM.value:
            formatted_turn += "# LLM_REPLY_END\n\n"
    return formatted_turn


def turn_reviewer_worker(
    turn_review_queue: Queue,
    config: dict[str, Any],
    results: list[dict],
    total_reviews: int,
) -> None:
    llm_client = LLMAPIFactory().get()
    while not turn_review_queue.empty():
        turn_id = None
        reviewer = None
        reviews_left = 0
        try:
            reviews_done = len(results)
            reviews_left = turn_review_queue.qsize() - 1
            print(
                f"Reviews done: {reviews_done}, Reviews left after this one: {reviews_left}"
            )
            turn_id, reviewer, turn = turn_review_queue.get()
            r = review_turn(
                reviewer,
                format_turn_data(turn),
                llm_client,
                config[reviewer],
            )
            results.append({"id": turn_id, "reviewer": reviewer, "result": r})
        except Exception as e:
            msg = f"An error occurred while processing {turn_id=} for {reviewer=}: {e}"
            print(msg)
            results.append({"id": turn_id, "reviewer": reviewer, "result": "ERROR"})
        finally:
            turn_review_queue.task_done()
            print(
                f"Review for {turn_id=} by {reviewer=} is done. {len(results)} / {total_reviews} reviews completed."
            )


def populate_queue(queue: Queue, turns: list) -> None:
    for i, turn in enumerate(turns):
        queue.put((i, "english_reviewer", turn))
        queue.put((i, "code_reviewer", turn))


def review_notebook(notebook, max_threads=1) -> dict[str, list[dict[str, Any]]]:
    """
    Review a notebook file and return a list of dictionaries, each representing a review result.
    Each dictionary in the list has two keys: 'turn' and 'review'.
    'turn' is a dictionary with keys 'human' and 'llm', representing the human and LLM assistant parts of the turn.
    'review' is the result of the LLM review for the LLM Assistant part of the turn.

    :param nb_path: The path to the notebook file.
    :return: A list of dictionaries, each representing the review result of a turn.
    """
    config = load_config(PATH_TO_CONFIG)
    turns = parse_notebook(notebook["nb_notebook_parsed"], config["notebook_syntax"])

    turn_queue = Queue()
    populate_queue(turn_queue, turns)
    total_turns = len(turns)
    results = []
    max_threads = min(total_turns * 2, max_threads)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for _ in range(max_threads):
            executor.submit(
                turn_reviewer_worker, turn_queue, config, results, total_turns * 2
            )
    turn_queue.join()

    gathered_results = [{"turn": turn} for turn in turns]
    for result in results:
        turn_id, reviewer, review_result = (
            result["id"],
            result["reviewer"],
            result["result"],
        )
        if reviewer == "english_reviewer":
            gathered_results[turn_id]["english_review"] = review_result
        elif reviewer == "code_reviewer":
            gathered_results[turn_id]["code_review"] = review_result
        else:
            raise Exception(f"Unknown reviewer: {reviewer}")

    print("Review process completed successfully.")

    return {"turns": gathered_results, "nb_path": notebook["file_id"]}


def review_notebooks(notebooks, max_threads_per_notebook=1, max_concurrent_notebooks=1):
    """
    Review multiple notebook files in parallel and return a list of review results for each notebook.

    :param notebooks_paths: A list of paths to the notebook files.
    :param max_threads_per_notebook: Maximum number of threads to use for reviewing each notebook.
    :param max_concurrent_notebooks: Maximum number of notebooks to review concurrently.
    :return: A list of lists, each containing dictionaries of review results for a notebook.
    """
    max_concurrent_notebooks = min(len(notebooks), max_concurrent_notebooks)
    with ThreadPoolExecutor(max_workers=max_concurrent_notebooks) as executor:
        results = executor.map(
            lambda nb_path: review_notebook(
                nb_path, max_threads=max_threads_per_notebook
            ),
            notebooks,
        )
    return list(results)


def review_to_row(review):
    """
    Convert a review dictionary into a row that can be added to a DataFrame, aggregating scores and feedback.

    :param review: A dictionary containing review details for a notebook.
    :return: A dictionary representing the row to be added to the DataFrame.
    """
    nb_path = review.get("nb_path", "")
    turns = review.get("turns", [])

    # Initialize scores and feedback
    code_scores = []
    lang_scores = []
    combined_feedback = []

    # Process each turn and aggregate scores and feedback
    for i, turn in enumerate(turns):
        english_review = turn.get("english_review")
        code_review = turn.get("code_review")

        # Append scores or handle errors
        lang_score = (
            english_review["score"] if isinstance(english_review, dict) else "ERROR"
        )
        code_score = code_review["score"] if isinstance(code_review, dict) else "ERROR"
        if lang_score != "ERROR":
            lang_scores.append(lang_score)
        if code_score != "ERROR":
            code_scores.append(code_score)

        # Combine feedback or show ERROR
        english_feedback = (
            english_review.get("feedback_text", "ERROR")
            if isinstance(english_review, dict)
            else "ERROR"
        )
        code_feedback = (
            code_review.get("feedback_text", "ERROR")
            if isinstance(code_review, dict)
            else "ERROR"
        )
        combined_feedback.append(
            f"#Turn {i+1}:\n\n## Language({lang_score}/5):\n{english_feedback}\n\n## Code({code_score}/5):\n{code_feedback}"
        )

    # Calculate average scores
    avg_code_score = sum(code_scores) / len(code_scores) if code_scores else None
    avg_lang_score = sum(lang_scores) / len(lang_scores) if lang_scores else None
    combined_feedback = "\n\n======\n\n".join(combined_feedback)
    return {
        "nb_path": nb_path,
        "code_score": avg_code_score,
        "lang_score": avg_lang_score,
        "feedback": combined_feedback,
    }


def notebook_reviews_to_df(reviews: list):
    df = pd.DataFrame(
        list(map(review_to_row, reviews)),
        columns=["nb_path", "code_score", "lang_score", "feedback"],
    )
    return df
