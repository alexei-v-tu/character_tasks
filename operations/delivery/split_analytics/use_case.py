from pydantic import BaseModel, Field
from typing import List
import os
from llama_index.program import OpenAIPydanticProgram
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from utils import process_batch


api_key = os.environ["OPENAI_API_KEY"]


class SummaryResult(BaseModel):
    """Data model for summary."""

    summary: str = Field(
        description="A short summary containing up to 4 sentences focused on the specific theme."
    )


class SummaryTheme(BaseModel):
    """Data model for the summarization aspect and perspective."""

    theme: str = Field(description="Aspect and theme for which to provide summary.")


def exec_summary(conversation: List[List[dict]], summary_theme: SummaryTheme):
    prompt_template_str = """
    Given the following conversation, please, generate an executive summary for a given theme and through its lense, not of the conversation.
    You are one of many specialized analyzers, so precisely focus on your target summary theme and topic.

    Summary Theme:
    {summary_theme}

    Conversation:
    {conversation}
    """
    program = OpenAIPydanticProgram.from_defaults(
        llm=OpenAI(api_key=api_key, model="gpt-4-1106-preview", temperature=0),
        output_cls=SummaryResult,
        prompt_template_str=prompt_template_str,
        verbose=True,
    )
    output = program(
        summary_theme=summary_theme.model_dump(), conversation=conversation["messages"]
    )
    return output


import json
from tqdm import tqdm
import concurrent.futures


def process_conversation(conversation):
    output = exec_summary(
        conversation, SummaryTheme(theme="User Use Case, how user uses the Assistant")
    )
    record = {
        "id": conversation["id"],
        "colab_link": f"https://colab.research.google.com/drive/{conversation['id']}",
    }
    record.update(output)
    return record


def get_use_case_data_batch_conversations(batch_folder, max_workers=10):
    selected_conversations = process_batch(batch_folder)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_conversation, conversation)
            for conversation in selected_conversations
        ]
        progress_bar = tqdm(total=len(futures))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            progress_bar.update(1)
        progress_bar.close()
    return results
