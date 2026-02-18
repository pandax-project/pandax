import logging

from agents import Agent, Runner, set_default_openai_key

set_default_openai_key("YOUR OPENAI API KEY")

from agents.model_settings import ModelSettings

from utils.agent_flow import (
    AgentOutput,
    CodeInfo,
    RewrittenOutput,
    generate_prompt,
    parse_response,
)

# === CPU Pandas Rewrite Agents ===

code_optimizer_cpu = Agent(
    name="Pandas CPU Optimization Agent",
    instructions="""
    You are an expert in optimizing pandas code for CPU efficiency. Your job is to rewrite code to be faster and more memory-efficient while keeping its behavior exactly the same.

    <GENERAL INSTRUCTIONS>:
    1. Don't use any other external libraries.
    2. Pandas is already imported and can be used with pd.
    3. If no modifications are made to the original code, return code as 'done'.
    4. You must rewrite the entire input code cell, not just the parts you optimize.
    5. Preserve function calls, print statements, tests, and any code outside functions verbatim.

    <CODE OPTIMIZATION INSTRUCTIONS>:
    Try to optimize the code as much as possible for running with Pandas. This can be achieved by:
    1. Eliminate loops in favor of vectorized operations
    2. Chain operations to avoid intermediate copies
    3. Avoid recomputation
    4. Rewritten code should be exactly semantically equivalent to the original code
    5. If the code is complex or has multiple loops, try to optimize it piece by piece as there are multiple iterations for optimization

    <VAR DEPENDENCY INSTRUCTIONS>:
    1. Active vars are the variables that are modified in the original code. These variables can be rewritten but needs to be maintained in the optimized code.
    2. Future vars are the variables that are being modified in the next code cells. These variables cannot be reassigned in the optimized code or modified in-place.
    """,
    output_type=RewrittenOutput,
    model="o4-mini",
    model_settings=ModelSettings(reasoning={"summary": "auto", "effort": "high"}),
)

code_fixer_cpu = Agent(
    name="Pandas CPU Repair Agent",
    instructions="""
    You are an expert debugger specializing in fixing DataFrame pandas code. Your task is to identify and fix errors while maintaining the optimization benefits.
    <GENERAL INSTRUCTIONS>:
    1. Don't use any other external libraries.
    2. Pandas is already imported and can be used with pd.

    <CODE OPTIMIZATION INSTRUCTIONS>:
    1. Eliminate loops in favor of vectorized operations
    2. Chain operations to avoid intermediate copies
    3. Avoid recomputation
    4. Rewritten code should be exactly semantically equivalent to the original code
    5. If the code is complex or has multiple loops, try to optimize it piece by piece as there are multiple iterations for optimization
""",
    output_type=RewrittenOutput,
    model_settings=ModelSettings(reasoning={"summary": "auto", "effort": "high"}),
    model="o4-mini",
)

manager_agent_cpu = Agent(
    name="Pandas Optimization Manager Agent",
    instructions="""
    You are an expert optimization manager who decides the next step in the code optimization process. Analyze the current state and choose the best action.

    Here are the options:
        -"optimize": we can try to further optimize the rewritten code. Use only if there is a rewritten code.
        -"repair": we can try to fix the rewritten code if there are errors.
        -"new": we can ignore the rewritten code and try to optimize the original code again to generate a "new" code, or if there is no rewritten code.
        -"done": the code is already optimized and no further optimization is needed.

    <GENERAL INSTRUCTIONS FOR DECISION MAKING>:
    1. Even if the code has been successfully rewritten to be faster, but there exists alternatives APIs or more efficient ways to do the same operation, try to optimize the code then by either using the "optimize" option or "new" option.
    2. If the primary purpose of the code is debugging or printing, it cannot be meaningfully optimized. This includes calls to APIs like `.head()`, `.tail()`, `.info()`, `.shape`, `.dtypes`, `.describe()
    3. If there is no rewritten code, use "new" for optimizating the original code.
    4. If the code does not look repairable at all, use "new" to generate a new code.

""",
    output_type=AgentOutput,
    model="gpt-4o",
)


async def call_rewrite_agent_cpu(
    num_tries: int,
    original_code_info: CodeInfo,
    rewritten_code_info: CodeInfo | None,
    num_tries_per_cell: int,
) -> str | None:
    prompt = generate_prompt(
        num_tries, original_code_info, rewritten_code_info, num_tries_per_cell
    )
    print("Manager agent prompt: ", prompt)
    print("==========Running the manager agent to decide the next step==========")
    result = await Runner.run(manager_agent_cpu, prompt)
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
        prompt_agent = generate_prompt(
            num_tries, rewritten_code_info, None, num_tries_per_cell
        )
        print("Code optimizer prompt: ", prompt_agent)
        print("Optimizing the last generated code")
        result = await Runner.run(code_optimizer_cpu, prompt_agent)

    elif choice == "repair":
        prompt_agent = generate_prompt(
            num_tries, original_code_info, rewritten_code_info, num_tries_per_cell
        )
        print("Code fixer prompt: ", prompt_agent)
        print("Repairing the last generated code")
        result = await Runner.run(code_fixer_cpu, prompt_agent)
        print(f"Code: {result.final_output.code}")
        print(f"Reason: {result.final_output.reason}")
        logging.info(f"Code: {result.final_output.code}")
        logging.info(f"Reason: {result.final_output.reason}")

    elif choice == "new":
        prompt_agent = generate_prompt(
            num_tries, original_code_info, None, num_tries_per_cell
        )
        print("Generating new code from the original code")
        print(f"Prompt: {prompt_agent}")
        result = await Runner.run(code_optimizer_cpu, prompt_agent)
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
