import json
from openai import OpenAI
from src.llm_reviewer.utils import load_env
from collections import defaultdict

import threading


class GlobalUsageManager:
    def __init__(self):
        self._usage_dict = defaultdict(lambda: defaultdict(int))
        self._cost_per_1k_tokens = {
            "gpt-4-1106-preview": {"prompt_tokens": 0.01, "completion_tokens": 0.03},
            "gpt-4": {"prompt_tokens": 0.03, "completion_tokens": 0.06},
            "gpt-3.5-turbo-1106": {"prompt_tokens": 0.001, "completion_tokens": 0.002},
        }
        self._lock = threading.Lock()

    def update_usage(self, model, usage):
        with self._lock:
            for k, v in dict(usage).items():
                self._usage_dict[model][k] += v

    def get_current_costs(self):
        total_cost = 0.0
        for model, usage in self._usage_dict.items():
            if model not in self._cost_per_1k_tokens:
                raise ValueError(f"Cost per token for model '{model}' not found.")
            model_cost = self._cost_per_1k_tokens[model]
            prompt_cost = model_cost["prompt_tokens"] * (usage["prompt_tokens"] / 1000)
            completion_cost = model_cost["completion_tokens"] * (
                usage["completion_tokens"] / 1000
            )
            total_cost += prompt_cost + completion_cost
        return total_cost

    def print_costs(self):
        total_cost = self.get_current_costs()
        print(f"Total cost for all models: ${total_cost:.3f}")

    def reset_usage(self):
        with self._lock:
            self._usage_dict.clear()


global_usage_manager = GlobalUsageManager()


class LLMAPIFactory:
    def __init__(self):
        self._api_key = load_env()["OPENAI_API_KEY"]

    def get(self) -> OpenAI:
        return OpenAI(api_key=self._api_key)


def make_llm_request(
    client,
    messages: list[dict[str, str]],
    model: str = None,
    temperature: float = 1.0,
    max_tokens: int = 4000,
    response_format: str = None,
    retries: int = 3,
    seed=42,
) -> str:
    if response_format not in [{"type": "json_object"}, None]:
        raise ValueError(
            "Unsupported response format. Only 'json_object' or None is allowed."
        )
    if response_format is not None and model not in [
        "gpt-4-1106-preview",
        "gpt-3.5-turbo-1106",
    ]:
        raise ValueError("Model not supported for response_format argument.")

    for retry in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=seed,
            )
            global_usage_manager.update_usage(model, completion.usage)
            if completion.choices[0].finish_reason != "stop":
                raise Exception(
                    "The conversation was stopped for an unexpected reason."
                )

            if response_format == {"type": "json_object"}:
                try:
                    return json.loads(completion.choices[0].message.content)
                except json.JSONDecodeError as e:
                    print("Failed to parse JSON response. Full completion:")
                    print(completion)
                    raise e
            else:
                return completion.choices[0].message.content
        except Exception as e:
            print(
                f"Attempt {retries - retry} of {retries} failed with error: {e}. Retrying..."
            )
    raise Exception("All attempts failed.")
