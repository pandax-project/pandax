import json
import logging
import os
from dataclasses import dataclass

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import types
from openai import OpenAI

from utils.prompts import repair_prompt, test_prompt, translation_prompt

# load the environment variables
load_dotenv()


@dataclass
class RewrittenCellInfo:
    optimized_code: str
    test_code: str


def call_llm(model: str, messages: list, temp: float = 0.7) -> str:
    """
    Calls the appropriate LLM client based on the model parameter.

    Parameters:
        model (str): The model to use (openai/google).
        prompt (str): The input prompt for the LLM.
        temp (float): The temperature for the LLM response.
    Returns:
        str: The response from the LLM.
    """
    if model.lower() == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_api_key)
        outputs = client.chat.completions.create(
            model="o3-mini",  # "chatgpt-4o-latest",
            messages=messages,
            n=1,
            # temperature=0.7,
            # temperature=temp,
            reasoning_effort="high",
        )
        return outputs.choices[0].message.content
    elif model.lower() == "google":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(
            api_key=gemini_api_key,
        )

        generate_content_config = types.GenerateContentConfig(
            temperature=temp,
            top_p=0.95,
            top_k=64,
            max_output_tokens=10000,
            response_mime_type="text/plain",
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=messages,
            config=generate_content_config,
        )
        return response.text
    else:
        raise ValueError(f"Unsupported model: {model}")


def parse_response(response: str) -> str:
    if "```python" in response:
        return response.split("```python")[1].split("```")[0].strip()
    elif "```json" in response:
        return response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        return response.split("```")[1].strip()
    else:
        return response


def get_rewritten_code(
    original_cell_code: str,
    active_vars: list[str],
    future_vars: list[str],
    cudf_profile_output: str,
) -> str | None:
    prompt = translation_prompt(
        original_cell_code, active_vars, future_vars, cudf_profile_output
    )
    messages = [{"role": "user", "content": prompt}]
    rewrite_info = call_llm("openai", messages)
    try:
        rewrite_info = json.loads(rewrite_info)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {rewrite_info}")
        return None
    rewritten_code = rewrite_info["optimized_code"]
    logging.info(f"Original cell code: {original_cell_code}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Rewritten cell code: {rewritten_code}")
    if rewritten_code == "not optimized":
        return None
    return rewritten_code


def get_test_code(original_cell_code: str, optimized_cell_code: str) -> str:
    prompt = test_prompt(original_cell_code, optimized_cell_code)
    messages = [{"role": "user", "content": prompt}]
    test_info = call_llm("openai", messages)
    try:
        test_info = json.loads(test_info)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {test_info}")
        return None
    test_code = test_info["test_code"]
    logging.info(f"Test code: {test_code}")
    return test_code


def get_rewritten_cell_info(
    original_cell_code: str, cudf_profile_output: str
) -> RewrittenCellInfo | None:
    """Rewrite the annotated cell to be faster. Returns None if the cell cannot be rewritten."""
    prompt = translation_prompt(original_cell_code, cudf_profile_output)
    messages = [{"role": "user", "content": prompt}]
    rewritten_cell = call_llm("openai", messages)

    logging.info(f"Original cell code: {original_cell_code}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Rewritten cell: {rewritten_cell}")

    rewritten_cell = parse_response(rewritten_cell)
    try:
        rewritten_cell = json.loads(rewritten_cell)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {rewritten_cell}")
        return None

    rewritten_cell_code = rewritten_cell["optimized_code"]
    rewritten_test_code = rewritten_cell["test_code"]
    if rewritten_cell == "not optimized" or rewritten_test_code == "no test code":
        return None

    rewritten_cell = RewrittenCellInfo(
        optimized_code=rewritten_cell_code,
        test_code=rewritten_test_code,
    )
    print(f"Original cell code: {original_cell_code}")
    print()
    print(f"Rewritten code: {rewritten_cell.optimized_code}")
    print()
    print(f"Test code: {rewritten_cell.test_code}")
    print()

    return rewritten_cell


def get_repaired_cell_code(
    original_cell_code: str, rewritten_cell_code: str | None
) -> str | None:
    # TODO(sahil): return the test harness as well.
    """Repair the rewritten cell to be semantically equivalent to the original cell. Returns None if the cell cannot be repaired."""
    if rewritten_cell_code is None:
        raise RuntimeError("Cannot repair None!")
    prompt = translation_prompt(original_cell_code)
    prompt_repair = repair_prompt(rewritten_cell_code)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "system", "content": rewritten_cell_code},
        {"role": "user", "content": prompt_repair},
    ]

    repaired_cell = call_llm("openai", messages)
    logging.info(f"Prompt repair: {prompt_repair}")

    repaired_cell = parse_response(repaired_cell)
    return repaired_cell
