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


load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from utils import process_batch


class DependencyUsage(BaseModel):
    dependency: str = Field(description="The tool/library/package imported.")
    percentage: float = Field(
        description="The percentage of the code, that leverages this dependency, between 0 and 1."
    )


def tag_conversations_by_dependency_usage(conversation: List[dict]):
    prompt_template_str = """
    Given the following conversation, What dependencies are used? 
    Dependencies are any imported libraries, packages, frameworks or external services/tools like (Flask, React, Pandas, Pytorch, keras, MySQL... etc)), not specific classes or functions
    For builtin capabilities of a language like language provided dependencies, you can say "vanilla [[programming language]]"

    Conversation:
    {conversation}
    """
    program = OpenAIPydanticProgram.from_defaults(
        llm=OpenAI(api_key=api_key, model="gpt-4-1106-preview", temperature=0),
        output_cls=DependencyUsage,
        prompt_template_str=prompt_template_str,
        allow_multiple=True,
    )
    try:
        output = program(
            conversation=conversation["messages"],
            description="Predict the percentage of presence of dependencies in the given conversation. Total sum of the percentages must sum up to 1!",
        )
        return output
    except ValueError as e:
        print(e)
        return []


def process_file(conversation):
    output = tag_conversations_by_dependency_usage(conversation)
    record = {
        "id": conversation["id"],
        "colab_link": f"https://colab.research.google.com/drive/{conversation['id']}",
        "dependency_presence": [o.dict() for o in output],
    }
    return record


def get_dependency_data_batch_conversations(
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
