from pydantic import Field
from tqdm import tqdm
from utils import PROJECT_ROOT
from src.sheets_utils import download_sheet_as_df
from pydantic import BaseModel, Field
from typing import List
import os
from llama_index.program import OpenAIPydanticProgram
from llama_index.llms.openai import OpenAI
import concurrent.futures
from dotenv import load_dotenv, find_dotenv

api_key = os.environ["OPENAI_API_KEY"]

load_dotenv(find_dotenv())
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from utils import process_batch


class ProgrammingLanguagePresence(BaseModel):
    language: str = Field(description="The programming language.")
    percentage: float = Field(
        description="The percentage of presence, between 0 and 1."
    )


def classify_conversation_by_programming_language(conversation: List[dict]):
    prompt_template_str = """
    Given the following conversation, what "programming languages" are being used? (Must be programming languages like (python, java, SQL, C... etc) NOT libraries, frameworks or tools like (Flask, React, Pandas, Pytorch, keras... etc))

    Conversation:
    {conversation}
    """
    program = OpenAIPydanticProgram.from_defaults(
        llm=OpenAI(api_key=api_key, model="gpt-4-1106-preview", temperature=0),
        output_cls=ProgrammingLanguagePresence,
        prompt_template_str=prompt_template_str,
        allow_multiple=True,
    )
    try:
        output = program(
            conversation=conversation["messages"],
            description="Predict the percentage of presence of programming languages in the given conversation. Total sum of the percentages must sum up to 1.",
        )
        return output
    except ValueError as e:
        print(e)
        return []


def process_file(conversation):
    output = classify_conversation_by_programming_language(conversation)
    record = {
        "id": conversation["id"],
        "colab_link": f"https://colab.research.google.com/drive/{conversation['id']}",
        "language_presence": [o.dict() for o in output],
    }
    return record


def get_programming_language_data_batch_conversations(
    selected_batch_folder, max_workers=10, limit_items_to_first_n=None
):
    selected_conversations = process_batch(
        selected_batch_folder, limit=limit_items_to_first_n
    )
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, conversation)
            for conversation in selected_conversations
        ]
        progress_bar = tqdm(total=len(futures), desc="Processing Conversations")
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            progress_bar.update(1)
        progress_bar.close()

    return results
