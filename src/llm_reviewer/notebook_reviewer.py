from datetime import datetime, timedelta
import traceback
from typing import Any, Optional
from src.llm_reviewer.notebook_parser import notebook_to_turns
from src.llm_reviewer.turn_reviewer import review_turn
from src.llm_reviewer.llm_api import load_config, LLMAPIFactory
from src.llm_reviewer.constants import PATH_TO_CONFIG, PATH_TO_SECRETS, Roles
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import pandas as pd
from threading import Lock, Thread
import threading


# Define a thread-safe counter class
class ThreadSafeProgressCounter:
    def __init__(self, total):
        self.start_time = datetime.now()
        self.success_count = 0
        self.fail_count = 0
        self.total = total
        self._lock = Lock()

    def success(self):
        with self._lock:
            self.success_count += 1

    def fail(self):
        with self._lock:
            self.fail_count += 1

    def report(self):
        with self._lock:
            total_completed = self.success_count + self.fail_count
            time_elapsed = datetime.now() - self.start_time
            if total_completed > 0:
                estimated_total_time = (time_elapsed / total_completed) * self.total
            else:
                estimated_total_time = timedelta(0)
            time_remaining = estimated_total_time - time_elapsed
            return f"Success: {self.success_count}/{self.total}, Fail: {self.fail_count}/{self.total}, Completed: {total_completed}/{self.total}\nTime remaining: {time_remaining}"


from enum import Enum


class IssueLevel(Enum):
    MINOR = 1
    MEDIUM = 2
    CRITICAL = 3


STR_TO_ISSUE_LEVEL = {
    "minor_issues": IssueLevel.MINOR,
    "medium_issues": IssueLevel.MEDIUM,
    "critical_issues": IssueLevel.CRITICAL,
}


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
    verbose: int = 0,
) -> None:
    try:
        llm_client = LLMAPIFactory(PATH_TO_SECRETS).get()
        while not turn_review_queue.empty():
            turn_id = None
            reviewer = None
            reviews_left = 0
            try:
                reviews_done = len(results)
                reviews_left = turn_review_queue.qsize() - 1
                if verbose > 0:
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
                if verbose > 0:
                    print(msg)
                results.append({"id": turn_id, "reviewer": reviewer, "result": "ERROR"})
            finally:
                turn_review_queue.task_done()
                if verbose > 0:
                    print(
                        f"Review for {turn_id=} by {reviewer=} is done. {len(results)} / {total_reviews} reviews completed."
                    )
    except Exception as e:
        if verbose > 0:
            traceback.print_exc()
        return None


def populate_queue(queue: Queue, turns: list) -> None:
    for i, turn in enumerate(turns):
        queue.put((i, "english_reviewer", turn))
        queue.put((i, "code_reviewer", turn))


def review_notebook(
    notebook,
    max_threads=1,
    progress_counter: Optional[ThreadSafeProgressCounter] = None,
    verbose: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Review a notebook file and return a list of dictionaries, each representing a review result.
    Each dictionary in the list has two keys: 'turn' and 'review'.
    'turn' is a dictionary with keys 'human' and 'llm', representing the human and LLM assistant parts of the turn.
    'review' is the result of the LLM review for the LLM Assistant part of the turn.

    :param nb_path: The path to the notebook file.
    :param verbose: Verbosity level of the output. If 1, output notebook reviews; if 0, no output.
    :return: A list of dictionaries, each representing the review result of a turn.
    """
    success = False
    try:
        config = load_config(PATH_TO_CONFIG)

        turns = notebook_to_turns(notebook["nb_parsed_notebook"])

        turn_queue = Queue()
        populate_queue(turn_queue, turns)
        total_turns = len(turns)
        results = []
        max_threads = min(total_turns * 2, max_threads)
        if max_threads == 0:
            if verbose > 0:
                print(f"Review process completed unsuccessfully. {notebook['file_id']}")
            return None
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for _ in range(max_threads):
                executor.submit(
                    turn_reviewer_worker,
                    turn_queue,
                    config,
                    results,
                    total_turns * 2,
                    0 if verbose in [0, 1] else 1,
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
        success = True
        return {"turns": gathered_results, "nb_path": notebook["file_id"]}
    except Exception as e:
        if verbose > 0:
            print(f"Review process completed unsuccessfully. {notebook['file_id']}")
            traceback.print_exc()
        return None
    finally:
        if progress_counter:
            if success:
                progress_counter.success()
            else:
                progress_counter.fail()
            if verbose > 0:
                print("Notebook reviews done:", progress_counter.report())


def review_notebooks(
    notebooks, max_threads_per_notebook=1, max_concurrent_notebooks=1, verbose=1
):
    """
    Review multiple notebook files in parallel and return a list of review results for each notebook.

    :param notebooks_paths: A list of paths to the notebook files.
    :param max_threads_per_notebook: Maximum number of threads to use for reviewing each notebook.
    :param max_concurrent_notebooks: Maximum number of notebooks to review concurrently.
    :return: A list of lists, each containing dictionaries of review results for a notebook.
    """
    max_concurrent_notebooks = min(len(notebooks), max_concurrent_notebooks)
    if not notebooks:
        return []
    with ThreadPoolExecutor(max_workers=max_concurrent_notebooks) as executor:
        progress_counter = ThreadSafeProgressCounter(len(notebooks))
        results = executor.map(
            lambda nb_path: review_notebook(
                nb_path,
                max_threads=max_threads_per_notebook,
                progress_counter=progress_counter,
                verbose=verbose,
            ),
            notebooks,
        )
    return list(results)


def review_to_row(review, issue_level=None):
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
    combined_code_feedback = []
    combined_lang_feedback = []

    # Process each turn and aggregate scores and feedback
    for i, turn in enumerate(turns):
        english_review = turn.get("english_review")
        code_review = turn.get("code_review")

        # Check for score presence and append scores or handle errors
        lang_score = (
            english_review.get("score", None)
            if isinstance(english_review, dict)
            else "ERROR"
        )
        code_score = (
            code_review.get("score", None) if isinstance(code_review, dict) else "ERROR"
        )
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

        # Process feedback_text dict into markdown formatted sections
        if isinstance(english_feedback, dict):
            english_feedback_lines = []
            for k, v in english_feedback.items():
                try:
                    if (
                        v.strip()
                        and k in STR_TO_ISSUE_LEVEL
                        and STR_TO_ISSUE_LEVEL[k].value >= issue_level.value
                    ):
                        english_feedback_lines.append(
                            f"**{k.title()}**\n{v if v.strip() else 'None'}"
                        )
                except Exception as e:
                    print(k, v)
            english_feedback = "\n".join(english_feedback_lines)
        elif issue_level is not None:
            raise Exception("Issue level is supported with dict issues only.")
        if isinstance(code_feedback, dict):
            code_feedback_lines = []
            for k, v in code_feedback.items():
                try:
                    if (
                        v.strip()
                        and k in STR_TO_ISSUE_LEVEL
                        and STR_TO_ISSUE_LEVEL[k].value >= issue_level.value
                    ):
                        code_feedback_lines.append(
                            f"**{k.title()}**\n{v if v.strip() else 'None'}"
                        )
                except Exception as e:
                    print(k, v)
            code_feedback = "\n".join(code_feedback_lines)
        elif issue_level is not None:
            raise Exception("Issue level is supported with dict issues only.")

        if not english_feedback.strip():
            english_feedback = "None"
        if not code_feedback.strip():
            code_feedback = "None"

        combined_feedback.append(
            f"#Turn {i+1}:\n\n## Language({lang_score}/5):\n{english_feedback}\n\n## Code({code_score}/5):\n{code_feedback}"
        )
        combined_code_feedback.append(
            f"#Turn {i+1}:\n\n## Code({code_score}/5):\n{code_feedback}"
        )
        combined_lang_feedback.append(
            f"#Turn {i+1}:\n\n## Language({lang_score}/5):\n{english_feedback}"
        )

    # Calculate average scores with valid values only
    valid_code_scores = [score for score in code_scores if score is not None]
    valid_lang_scores = [score for score in lang_scores if score is not None]
    avg_code_score = (
        sum(valid_code_scores) / len(valid_code_scores) if valid_code_scores else None
    )
    avg_lang_score = (
        sum(valid_lang_scores) / len(valid_lang_scores) if valid_lang_scores else None
    )
    combined_feedback = "\n\n======\n\n".join(combined_feedback)
    return {
        "nb_path": nb_path,
        "code_score": avg_code_score,
        "lang_score": avg_lang_score,
        "comb_feedback": combined_feedback,
        "code_feedback": "\n\n======\n\n".join(combined_code_feedback),
        "lang_feedback": "\n\n======\n\n".join(combined_lang_feedback),
    }


def notebook_reviews_to_df(reviews: list, issue_level=None):
    df = pd.DataFrame(
        list(map(lambda x: review_to_row(x, issue_level), reviews)),
        columns=[
            "nb_path",
            "code_score",
            "lang_score",
            "comb_feedback",
            "code_feedback",
            "lang_feedback",
        ],
    )
    return df
