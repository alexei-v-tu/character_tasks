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


class DomainCategory(BaseModel):
    top_level: str
    sub_level: str
    detailed_level: str


def classify_conversation_by_domain(conversation: List[dict]) -> DomainCategory:
    DOMAIN_CATEGORIES = """
    - Problem Solving
    - Testing
    - Python Basics & Scripting
    - Databases
    - Data Analysis
    - Machine Learning
    - Web Development
    - System Design
    - Mobile Devlopment
    - Devops
    - Cloud Computing
    - Security
    - Other
    """

    prompt_template_str = """
    Categorize the following conversation by domain into one of the following top-level categories, then sub & detailed categories that you think is descriptive & appropriate:
    {categories}

    Conversation:
    {conversation}
    """

    program = OpenAIPydanticProgram.from_defaults(
        llm=OpenAI(api_key=api_key, model="gpt-4-1106-preview", temperature=0),
        output_cls=DomainCategory,
        prompt_template_str=prompt_template_str,
        verbose=False,
    )
    output = program(
        categories=DOMAIN_CATEGORIES,
        conversation=conversation["messages"],
        description="Categorize a conversation by domain",
    )
    return output


def process_file(conversation):
    output = classify_conversation_by_domain(conversation)
    record = {
        "id": conversation["id"],
        "colab_link": f"https://colab.research.google.com/drive/{conversation['id']}",
    }
    record.update(output.dict())
    return record


def get_domain_data_batch_conversations(
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
