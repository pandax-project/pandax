import logging
import os

import google.auth.transport.requests
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv
from google.auth import default
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GCP_ACCESS_TOKEN = "ya29.c.c0ASRK0GaLhDwViAGDaR5XuB9mVNsHZ3gZ241PY8mRMlLBG_mp2nVPwo0GVIeixFCABB3uRbayCPp4uvru4o1JzCfyPFmaR9rDABVX7_uBvgb_Pece1530bAcWhmUot3wm0WgHFql0N7L9XEnyPAqhlJR1En5slIIGYT8cucob1x1ZQty1ZQxFD5kcCkjpQa-5E9K1z1QLihcXNus6feciW98P31Y3zec3azZ3pBaPSzEO2WdkKLo-xb6WtsjSjnR5L1h6Y5uwnXHcplBnHMOTY0_RD0-TDFIPJceJMbAIxPFfdwz-P9DIwgALfaueYC_byK9s_JXAC-SR4aSTvx68_eukimiXpT07bPyER8pN1uCJf2vXoe8E0vJRiMtbiZxdx95quUjTAsNK2x9PT409K9Bwmpqo10mOf_sZgrgaFJjIrfBQeuUBR3SaSt466QsYrY3bqMVv2nmh7gSOc_QO8wzprfQ-9ulIZ0v3tQp0d9JMhWjaku2bB1t-Si-5w-YBw_1cJSS_023W1WUblgkpoj9eivkqfW7Xonx3m79SR95mbQfYc95npIR6zsW_lp8aIsn9jyfx6Zw4hj1_pyjItY3BZyv-7S5Woo43o-m-Y5quVRoXq22WeukShX20fu4eo0U0anw2el6WlYnwm732y4hsodfeueamvjgk56F-hXfxgZ3ZSsO856jac0g6cr5JUizeVn47qStXoiIt4ms_iWQpXJrweuzzImpcyj6vWOclYRQOJMOhh260oJ721JanlVO24Ftraq6jtf3JZMO2_iir-feVea4I_z_8djS0kZopkiOst_i3d4WbcWhqW3xrW0qUR8J-0efZYz2knioUUvzwyJOplaQz8p-WqIlZqJbwVJkYy2dp_z9XsQ-oV9bVj6s_MhOp9nn-njuIYo7qe0ekUFqwQWd75-RQv_cX8bbXpUUubVwsmkzeJb1w2tMZBvg8sjh5FUbg6ebekb7Zddd5JqWZk4lugbamYmRrdQBFh5ryMrStWOy"
PROJECT_ID = "goog24-12"
LOCATION_ID = "global"
API_ENDPOINT = "aiplatform.googleapis.com"
MODEL_ID = "gemini-2.5-pro"
GENERATE_CONTENT_API = "streamGenerateContent"
# client = AsyncOpenAI(base_url="https://us-central1-aiplatform.googleapis.com/v1/projects/goog24-12/locations/us-central1/publishers/google/models/gemini-2.5-pro", api_key=GOOGLE_API_KEY)
# client = AsyncOpenAI(base_url="https://aiplatform.googleapis.com/v1/projects/goog24-12/locations/global/publishers/google/models/gemini-2.5-pro:streamGenerateContent", api_key=GCP_ACCESS_TOKEN)
# client = AsyncOpenAI(base_url=f"https://{API_ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{LOCATION_ID}/publishers/google/models/{MODEL_ID}:{GENERATE_CONTENT_API}", api_key=GCP_ACCESS_TOKEN)

credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(google.auth.transport.requests.Request())

# OpenAI Client
client = AsyncOpenAI(
    base_url=f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION_ID}/endpoints/openapi",
    api_key=credentials.token,
)

set_tracing_disabled(True)


class CodeInfo(BaseModel):
    code: str
    profiling_info: str | None = None
    execution_time: float
    execution_output: str | None = None
    active_vars: list[str] | None = None
    future_vars: list[str] | None = None


class RewrittenInfo(BaseModel):
    original_code_info: CodeInfo | None = None
    rewritten_code_info: list[CodeInfo] | None = None


class AgentOutput(BaseModel):
    name: str
    reason: str


class RewrittenOutput(BaseModel):
    code: str
    reason: str


def parse_response(response: str) -> str:
    if "```python" in response:
        return response.split("```python")[1].split("```")[0].strip()
    elif "```json" in response:
        return response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        return response.split("```")[1].strip()
    else:
        return response


def generate_prompt(
    iteration: int,
    original_code_info: CodeInfo,
    rewritten_code_info: CodeInfo | None = None,
) -> str:
    if rewritten_code_info:
        prompt = f"""
        <Original Code>:
        {original_code_info.code}
        <Profiling data>:
        {original_code_info.profiling_info}
        <Execution time>: {original_code_info.execution_time}
        <Active Vars>: {original_code_info.active_vars}
        <Future Vars>: {original_code_info.future_vars}
        <Rewritten Code>:
        {rewritten_code_info.code}
        <Profiling data>:
        {rewritten_code_info.profiling_info}
        <Execution time>: {rewritten_code_info.execution_time}
        <Execution output>: {rewritten_code_info.execution_output}
        <Iteration>: {iteration}/5
    """
        return prompt
    else:
        prompt = f"""
        <Original Code>:
        {original_code_info.code}
        <Profiling data>:
        {original_code_info.profiling_info}

        <Execution time>: {original_code_info.execution_time}
        <Active vars>: {original_code_info.active_vars}

        <Future vars>: {original_code_info.future_vars}

        <Iteration>: {iteration}/5
    """
    return prompt


code_optimizer = Agent(
    name="Code Optimization agent",
    instructions="""
    We are trying to optimize pandas workflow to execute using cudf. We are going to show you one cell from the notebook at a time and your task is to rewrite a given code snippet for cudf. Carefully analyze the provided code and the profiling data to identify the inefficiencies.

    <GENERAL INSTRUCTIONS>:
    1. The notebook already has the (%load_ext cudf.pandas) extension loaded. We can directly use pandas APIS and don't need to replace them with cudf APIS.
    2. Don't import cudf in the code and don't manually move data between CPU and GPU as cudf will handle it automatically.
    3. Don't use any other external libraries.
    4. If no modifications are made to the original code, return code as 'done'.
    5. The cpu operations are listed in the profiling data under the cpu calls column.
    6. The gpu operations are listed in the profiling data under the gpu calls column.

    <CODE OPTIMIZATION INSTRUCTIONS>:
    Try to optimize the code as much as possible for running with cudf. This can be achieved by:
    1. Eliminate loops in favor of vectorized operations
    2. Minimize the data movement between CPU and GPU by replacing CPU operations with GPU operations.
    3. If the code is already using GPU, try to optimize it further by using more efficient GPU operations or exporing alternatives APIs.
    4. Rewritten code should be semantically equivalent to the original code
    5. If the code is complex, try to optimize it piece by piece as there multiple iterations for optimization.
    <VAR DEPENDENCY INSTRUCTIONS>:
    1. Active vars are the variables that are modified in the original code. These variables can be rewritten but needs to be maintained in the optimized code.
    2. Future vars are the variables that are being modified in the next code cells. These variables cannot be reassigned in the optimized code or modified in-place.
    """,
    output_type=RewrittenOutput,
    model=OpenAIChatCompletionsModel(
        model="google/gemini-2.5-pro", openai_client=client
    ),
)

code_fixer = Agent(
    name="Code Repair Agent",
    instructions="""
    Your task is to fix the code for errors. We are trying to optimize pandas workflow to execute using cudf.
    <GENERAL INSTRUCTIONS>:
    1. The notebook already has the (%load_ext cudf.pandas) extension loaded. We can directly use pandas APIs and don't need to replace them with cudf APIs.
    2. Don't import cudf in the code and don't manually move data between CPU and GPU as cudf will handle it automatically.
    3. Don't use any other external libraries.
    4. GPU operations are listed in the profiling data under the gpu calls column.
    5. CPU operations are listed in the profiling data under the cpu calls column.

    <CODE OPTIMIZATION INSTRUCTIONS>:
    Try to optimize the code as much as possible for running with cudf. This can be achieved by:
    1. Eliminate loops in favor of vectorized operations
    2. Minimize the data movement between CPU and GPU by replacing CPU operations with GPU operations.
    3. If the code is already using GPU, try to optimize it further by using more efficient GPU operations or exporing alternatives APIs.
    4. The rewritten code should be semantically equivalent to the original code.
    5. Try to repair the part of the code only for which the assertions are failing. Do not make any changes to the code that is not related to the assertions.
 """,
    output_type=RewrittenOutput,
    model=OpenAIChatCompletionsModel(
        model="google/gemini-2.5-pro", openai_client=client
    ),
)

manager_agent = Agent(
    name="Manager Agent",
    instructions="""
    You are an expert in code optimization. User will provide you with a code and additional information. Your goal is to analyze the code, data and make a decision about the next step. User will also provide the current iteration. Base your decision on the current iteration and iterations left.
    Here are the options:
        -"optimize": we can try to further optimize the rewritten code. Use only if there is a rewritten code.
        -"repair": we can try to fix the rewritten code if there are errors.
        -"new": we can ignore the rewritten code and try to optimize the original code again to generate a "new" code, or if there is no rewritten code.
        -"done": the code is already optimized and no further optimization is needed.

    <GENERAL INSTRUCTIONS FOR DECISION MAKING>:
    1. Even if the code is already using cudf or has been successfully rewritten to be faster, but there exists alternatives APIs or more efficient ways to do the same operation, try to optimize the code then by either using the "optimize" option or "new" option.
    2. If the primary purpose of the code is inspection, debugging, or printing, it cannot be meaningfully optimized. This includes calls to APIs like `.head()`, `.tail()`, `.info()`, `.shape`, `.dtypes`, `.describe()
    3. If there is no rewritten code, use "new" for optimizating the original code.
    4. If the code does not look repairable at all, use "new" to generate a new code.
    """,
    output_type=AgentOutput,
    model=OpenAIChatCompletionsModel(
        model="google/gemini-2.5-pro", openai_client=client
    ),
)


async def call_rewrite_agent(
    num_tries: int, original_code_info: CodeInfo, rewritten_code_info: CodeInfo | None
) -> str | None:
    prompt = generate_prompt(num_tries, original_code_info, rewritten_code_info)
    print("Manager agent prompt: ", prompt)
    print("==========Running the manager agent to decide the next step==========")
    result = await Runner.run(manager_agent, prompt)
    choice = result.final_output.name
    print(f"Choice: {choice}")
    print(f"Reason: {result.final_output.reason}")
    if choice == "optimize" and rewritten_code_info is None:
        choice = "new"

    logging.info(f"Choice: {choice}")
    logging.info(f"Reason: {result.final_output.reason}")
    if choice == "done":
        print("Code is already optimized")
        return None

    elif choice == "optimize":
        prompt_agent = generate_prompt(num_tries, rewritten_code_info)
        print("Code optimizer prompt: ", prompt_agent)
        print("Optimizing the last generated code")
        result = await Runner.run(code_optimizer, prompt_agent)

    elif choice == "repair":
        prompt_agent = generate_prompt(
            num_tries, original_code_info, rewritten_code_info
        )
        print("Code fixer prompt: ", prompt_agent)
        print("Repairing the last generated code")
        result = await Runner.run(code_fixer, prompt_agent)
        print(f"Code: {result.final_output.code}")
        print(f"Reason: {result.final_output.reason}")
        logging.info(f"Code: {result.final_output.code}")
        logging.info(f"Reason: {result.final_output.reason}")

    elif choice == "new":
        prompt_agent = generate_prompt(num_tries, original_code_info)
        print("Generating new code from the original code")
        print(f"Prompt: {prompt_agent}")
        result = await Runner.run(code_optimizer, prompt_agent)
        logging.info(f"output from agent: {result}")
        print(f"Code: {result.final_output.code}")
        print(f"Reason: {result.final_output.reason}")

    try:
        parsed_response = parse_response(result.final_output.code)
        if parsed_response == "done":
            print("No Modifications made to the original code")
            return None
        else:
            print(f"Parsed response: {parsed_response}")
            logging.info(f"Parsed response: {parsed_response}")
            return parsed_response

    except Exception as e:
        logging.error(f"Error parsing the response: {e}")
        logging.error(f"Response: {result.final_output.code}")
        raise ValueError(f"Error parsing the response: {e}")
