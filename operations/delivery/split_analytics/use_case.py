import concurrent.futures
import json
import os
from typing import List

from dotenv import find_dotenv, load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.program import OpenAIPydanticProgram
from pydantic import BaseModel, Field
from tqdm import tqdm
from utils import process_batch

load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]


class SummaryResult(BaseModel):
    """Data model for summary."""

    summary: str = Field(
        description="A short summary containing 1 sentence, 15 words max, focused on the specific theme. [super concise language]"
    )
    tags: List[str] = Field(
        description="A list of tags(up to 5) for the conversation from the requested theme perspective."
    )


class SummaryTheme(BaseModel):
    """Data model for the summarization aspect and perspective."""

    theme: str = Field(description="Aspect and theme for which to provide summary.")


def exec_summary(conversation: List[List[dict]], summary_theme: SummaryTheme):
    prompt_template_str = """
    Given the following conversation, please, generate an executive summary for a given theme, not of the conversation.
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
        verbose=False,
    )
    output = program(
        summary_theme=summary_theme.model_dump(), conversation=conversation["messages"]
    )
    return output


def process_conversation(conversation):
    output = exec_summary(
        conversation,
        SummaryTheme(
            theme="User Use Case, why user uses the Assistant in this conversation, in general terms, **for what** the User is using it. Not from a technical perspective, but from a daily life situation perspective. Example: work, homework, exam, studying, inteview, debugging, etc..."
        ),
    )
    record = {
        "id": conversation["id"],
        "colab_link": f"https://colab.research.google.com/drive/{conversation['id']}",
    }
    record.update(output)
    return record


def get_use_case_data_batch_conversations(
    batch_folder, max_workers=10, limit_items_to_first_n=None
):
    selected_conversations = process_batch(batch_folder, limit=limit_items_to_first_n)
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
