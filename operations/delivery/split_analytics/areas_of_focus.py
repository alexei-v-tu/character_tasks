AOF_SHEET_ID = "1bsM3nz13BPPqCxmbLYFz5Ed9KOfV51hNoagBP8dP948"
AOF_SHEET_NAME = "Areas status"
from utils import PROJECT_ROOT
from src.sheets_utils import download_sheet_as_df
from pydantic import BaseModel, Field
from typing import List
import os
from llama_index.program import OpenAIPydanticProgram
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv, find_dotenv

api_key = os.environ["OPENAI_API_KEY"]

load_dotenv(find_dotenv())
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from utils import process_batch


def format_topics_with_additional(df):
    topics = []
    last_highlevel = None
    for _, row in df.iterrows():
        highlevel = row["Highlevel_topic"]
        sublevel = row["Sublevel_topic"]
        if highlevel != last_highlevel:
            topics.append(
                f"{last_highlevel} -> [[fill out another topic you think best fits the conversation and is not included]]"
            )

        topics.append(f"{highlevel} -> {sublevel.strip(' - ')}")
        # Add another entry with the same highlevel topic but a placeholder for a new sublevel topic
        # only if the highlevel topic has changed from the last one processed
        last_highlevel = highlevel
    topics.append(
        f"{last_highlevel} -> [[fill out another topic you think best fits the conversation and is not included]]"
    )
    topics.append(
        "Other -> [[fill out another topic you think best fits the conversation and is not included]]"
    )
    return "\n".join(topics)


import json


class DomainCategory(BaseModel):
    """Datamodel for the category and subcategory calssification.
    If the category you determined is not among provided categories or subcategories, set `new_filled_category` to True otherwise to False.
    """

    top_level: str
    sub_level: str
    detailed_level: str
    new_filled_category: bool


def classify_conversation_by_domain(
    conversation: List[dict], domain_categories
) -> DomainCategory:
    prompt_template_str = """
    Categorize the following conversation by domain into one of the following top-level categories, then sub & detailed categories that you think is descriptive & appropriate. If you think that conversation does not best fit into provided classes, you can fill out your own following the provided template [[]].
    
    Categories:

    {categories}

    Conversation:
    {conversation}

    """

    program = OpenAIPydanticProgram.from_defaults(
        llm=OpenAI(api_key=api_key, model="gpt-4-1106-preview", temperature=0),
        output_cls=DomainCategory,
        prompt_template_str=prompt_template_str,
        verbose=True,
    )
    output = program(
        categories=domain_categories,
        conversation=conversation["messages"],
    )
    return output


import concurrent.futures


def process_file(conversation):
    output = classify_conversation_by_domain(conversation)
    record = {
        "id": conversation["id"],
        "colab_link": f"https://colab.research.google.com/drive/{conversation['id']}",
    }
    record.update(output.dict())
    return record


def get_areas_of_focus_data_batch_conversations(selected_batch_folder, max_workers=10):
    aof_df = download_sheet_as_df(PROJECT_ROOT, AOF_SHEET_ID, AOF_SHEET_NAME)
    formatted_topics = format_topics_with_additional(aof_df)
    print(formatted_topics)
    DOMAIN_CATEGORIES = formatted_topics
    selected_conversations = process_batch(selected_batch_folder)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, conversation, DOMAIN_CATEGORIES)
            for conversation in selected_conversations
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    high_level_topics = set(aof_df["Highlevel_topic"].tolist())
    sub_level_topics = set(aof_df["Sublevel_topic"].tolist())

    for record in results:
        record["new_filled_category"] = not (
            record["top_level"] in high_level_topics
            and record["sub_level"] in sub_level_topics
        )

    return results
