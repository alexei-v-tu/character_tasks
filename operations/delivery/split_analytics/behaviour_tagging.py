from tqdm import tqdm
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


import json

import pandas as pd


class BehaviourTag(BaseModel):
    """Datamodel for behaviour tag with top level and sub level category calssification.
    If the Tag you determined is not among provided subcategories, set `new_filled_category` to True otherwise to False and create a new Tag.
    """

    top_level: str
    sub_level: str
    new_filled_category: bool


class BehaviourTagsForConversation(BaseModel):
    """Datamodel for tagging behaviours in a conversation. Conversation can have multiple tags."""

    behaviours: List[BehaviourTag]


def tag_conversation_with_behaviours(
    conversation: List[dict], available_tags
) -> BehaviourTagsForConversation:
    prompt_template_str = """
    Tag the following conversation with behaviour tags using provided categories and subcategories or creating new tags that you think is descriptive & appropriate. If you think that conversation does not best fit into provided classes, you can fill out your own following the provided template [[]].
    
    Available Behaviour Tags:
    {available_tags}

    Conversation:
    {conversation}

    """

    program = OpenAIPydanticProgram.from_defaults(
        llm=OpenAI(api_key=api_key, model="gpt-4-1106-preview", temperature=0),
        output_cls=BehaviourTagsForConversation,
        prompt_template_str=prompt_template_str,
        verbose=False,
    )
    output = program(
        available_tags=available_tags,
        conversation=conversation["messages"],
    )
    return output


import concurrent.futures


def process_file(conversation, available_tags):
    output = tag_conversation_with_behaviours(conversation, available_tags)
    record = {
        "id": conversation["id"],
        "colab_link": f"https://colab.research.google.com/drive/{conversation['id']}",
    }
    record.update(output.dict())
    return record


data = [
    ("Continuation Follow up", "Incrementally Build"),
    ("Continuation Follow up", "Supplement/Extend"),
    ("Continuation Follow up", "Integrate with something else"),
    ("Continuation Follow up", "Request for clarification/elaboration"),
    ("Continuation Follow up", "Request alternatives"),
    ("Pivoting Follow up", "Change in topic"),
    ("Pivoting Follow up", "Change in direction"),
    ("Pivoting Follow up", "Change in focus or goal"),
    ("Respond to Assistant", "Answer a question"),
    ("Respond to Assistant", "End the conversation"),
    ("Respond to Assistant", "Clarify an ambiguity"),
    # ("User emotional state", "User is confused"),
    # ("User emotional state", "User is frustrated"),
    ("User actions", "User contradicts themselves"),
    ("User actions", "User confronts assistant about mistakes"),
    ("User actions", "User makes mistakes"),
]
behavioural_tags_df = pd.DataFrame(data, columns=["top_level", "sub_level"])


def format_topics_with_additional(df):
    topics = []
    last_highlevel = None
    custom_sub_cat = "[[fill out another behaviour tag sub level you think best fits the conversation and is not included, 3 words max]]"
    for _, row in df.iterrows():
        highlevel = row["top_level"]
        sublevel = row["sub_level"]
        if highlevel != last_highlevel and last_highlevel is not None:
            topics.append(f"{last_highlevel} -> {custom_sub_cat}")

        topics.append(f"{highlevel} -> {sublevel.strip(' - ')}")
        # Add another entry with the same highlevel topic but a placeholder for a new sublevel topic
        # only if the highlevel topic has changed from the last one processed
        last_highlevel = highlevel
    topics.append(f"{last_highlevel} -> {custom_sub_cat}")
    topics.append(f"Other -> {custom_sub_cat}")
    return "\n".join(topics)


AVAILABLE_TAGS = format_topics_with_additional(behavioural_tags_df)


def get_behavioural_tags_data_batch_conversations(
    selected_batch_folder, max_workers=10, limit_items_to_first_n=None
):
    selected_conversations = process_batch(
        selected_batch_folder, limit=limit_items_to_first_n
    )
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, conversation, AVAILABLE_TAGS)
            for conversation in selected_conversations
        ]
        progress_bar = tqdm(total=len(futures), desc="Processing Conversations")
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            progress_bar.update(1)
        progress_bar.close()

    high_level_topics = set(behavioural_tags_df["top_level"].tolist())
    sub_level_topics = set(behavioural_tags_df["sub_level"].tolist())

    for record in results:
        for behaviour_tag in record["behaviours"]:
            behaviour_tag["new_filled_category"] = not (
                behaviour_tag["top_level"] in high_level_topics
                and behaviour_tag["sub_level"] in sub_level_topics
            )

    return results
