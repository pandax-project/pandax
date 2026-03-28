import logging
from pathlib import Path

from agents import Agent, Runner
from agents.model_settings import ModelSettings
from pydantic import BaseModel
from utils.logging_utils import log_agent_token_usage

NUM_TRIES_PER_CELL = 5


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
    num_tries_per_cell: int = NUM_TRIES_PER_CELL,
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
        <Iteration>: {iteration + 1}/{num_tries_per_cell}
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

        <Iteration>: {iteration + 1}/{num_tries_per_cell}
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
    5. If the code is complex or has multiple loops, try to optimize it piece by piece as there are multiple iterations for optimization.
    <VAR DEPENDENCY INSTRUCTIONS>:
    1. Active vars are the variables that are modified in the original code. These variables can be rewritten but needs to be maintained in the optimized code.
    2. Future vars are the variables that are being modified in the next code cells. These variables cannot be reassigned in the optimized code or modified in-place.
    """,
    output_type=RewrittenOutput,
    model="o4-mini",
    model_settings=ModelSettings(reasoning={"summary": "auto", "effort": "high"}),
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
    model_settings=ModelSettings(reasoning={"summary": "auto", "effort": "high"}),
    model="o4-mini",
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
    2. If the primary purpose of the code is debugging or printing, it cannot be meaningfully optimized. This includes calls to APIs like `.head()`, `.tail()`, `.info()`, `.shape`, `.dtypes`, `.describe()
    3. If there is no rewritten code, use "new" for optimizating the original code.
    4. If the code does not look repairable at all, use "new" to generate a new code.
    """,
    output_type=AgentOutput,
    model="gpt-4o",
)


async def call_rewrite_agent(
    num_tries: int,
    original_code_info: CodeInfo,
    rewritten_code_info: CodeInfo | None,
    benchmark_name: str,
    cell_index: int,
    output_dir: Path,
) -> str | None:
    """Run manager + rewrite agents and return rewritten code when available.

    Besides orchestrating agent calls, this function records token usage for
    each model invocation into `rewrite_agent_token_usage.csv` so we can analyze
    prompt/response costs by benchmark, cell, and try.

    Args:
        num_tries: Current rewrite attempt number for the target cell.
        original_code_info: Context for the original cell code/profile.
        rewritten_code_info: Context from the previous rewrite attempt, if any.
        benchmark_name: Stable benchmark identifier used in CSV logs.
        cell_index: Annotated cell index being rewritten.
        output_dir: Directory where token-usage CSV is stored.

    Returns:
        Parsed rewritten code string, or None when manager chooses "done" (or
        when the optimizer returns "done").
    """
    prompt = generate_prompt(num_tries, original_code_info, rewritten_code_info)
    print("Manager agent prompt: ", prompt)
    print("==========Running the manager agent to decide the next step==========")
    result = await Runner.run(manager_agent, prompt)
    log_agent_token_usage(
        output_dir=output_dir,
        benchmark_name=benchmark_name,
        cell_index=cell_index,
        try_number=num_tries,
        category="manager_decision",
        result=result,
    )
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
        log_agent_token_usage(
            output_dir=output_dir,
            benchmark_name=benchmark_name,
            cell_index=cell_index,
            try_number=num_tries,
            category="optimizer_call",
            result=result,
        )

    elif choice == "repair":
        prompt_agent = generate_prompt(
            num_tries, original_code_info, rewritten_code_info
        )
        print("Code fixer prompt: ", prompt_agent)
        print("Repairing the last generated code")
        result = await Runner.run(code_fixer, prompt_agent)
        log_agent_token_usage(
            output_dir=output_dir,
            benchmark_name=benchmark_name,
            cell_index=cell_index,
            try_number=num_tries,
            category="repair_call",
            result=result,
        )
        print(f"Code: {result.final_output.code}")
        print(f"Reason: {result.final_output.reason}")
        logging.info(f"Code: {result.final_output.code}")
        logging.info(f"Reason: {result.final_output.reason}")

    elif choice == "new":
        prompt_agent = generate_prompt(num_tries, original_code_info)
        print("Generating new code from the original code")
        print(f"Prompt: {prompt_agent}")
        result = await Runner.run(code_optimizer, prompt_agent)
        log_agent_token_usage(
            output_dir=output_dir,
            benchmark_name=benchmark_name,
            cell_index=cell_index,
            try_number=num_tries,
            category="new_code_call",
            result=result,
        )
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


# if __name__ == "__main__":
#     asyncio.run(main())
