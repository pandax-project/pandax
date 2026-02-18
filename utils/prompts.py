def translation_prompt(
    unopt_code: str,
    active_vars: list[str],
    future_vars: list[str],
    cudf_profile_output: str,
) -> str:
    prompt = f"""
    We are trying to optimize pandas workflow to execute using cudf. We are going to show you one cell from the notebook at a time and your task is to rewrite a given code snippet for cudf.

    You need to follow the following instructions:
    <CODE OPTIMIZATION INSTRUCTIONS>:
    1. We are using the cudf extension (%load_ext cudf.pandas) for pandas. We can directly use pandas imports and functions, and don't need to import cudf.
    2. Make sure the rewritten code is semantically equivalent to the original code.
    3. Try to optimize the code as much as possible for running with cudf. If there are loops in the code, try to vectorize them. If other libraries are used in the code, try to use functions from pandas which can run faster with cudf.
    4. If no modifications are made to the original code, return
    ```
    {{"optimized_code": "not optimized"}}
    ```
    5. Don't use any other external libraries.
    ```

    <VAR DEPENDENCY INSTRUCTIONS>:
    {active_vars} are the variables that are modified in the original code. These variables can be rewritten but needs to be maintained in the optimized code.
    {future_vars} are the variables that are being modified in the next code cells. These variables cannot be reassigned in the optimized code or modified in-place.


    # Original code
    {unopt_code}

    Here is the cudf profile output for the original code:
    ```
    {cudf_profile_output}
    ```

    <OUTPUT FORMAT>
    Json object with the optimized code snippet.
    {{ "optimized_code": "optimized code snippet"}}

    ```
    """
    return prompt


def test_prompt(unopt_code: str, opt_code: str) -> str:
    prompt = f"""
    Your task is to write test code to compare the outputs of the original code and the optimized code.
    <TEST CODE INSTRUCTIONS>:
    We need to generate code which we will use to test the optimized code. The test code will be executed to compare the outputs of the original code and the optimized code if they are equivalent.
    Instructions for the test code:
    1. We will run the test code in a separate ipython shell after executing the original and rewritten and saving the states.
    2. In the new shell, the code should follow following pattern for only the variables that are modified in the original code. For statements that print the output, no testing is required.
    3. Original variable names are the same as the optimized variable names. Do not change the variable names after checkpoint loading.
    ```
    reassign the optimized variables that need to be compared
    # LOAD CHECKPOINT ORIG
    reassign the original variables that need to be compared
    code to compare the variables
    ```
    Example:
    ```
    df_opt = df['col']
    # LOAD CHECKPOINT ORIG
    df_orig = df['col']
    assert df_opt.equals(df_orig)
    ```
    Do not add any extra comments in the test code.
    4. Make sure the test code has checks to compare all the variables which are modified in the original code.
    5. The testing code should not execute the same code again.
    6. Do not add any imports to the test code, we will setup the environment.

    # Original code
    {unopt_code}

    # Rewritten code
    {opt_code}

    <OUTPUT FORMAT>
    Json object with the test code snippet.
    {{ "test_code": "test code snippet"}}
    """
    return prompt


def repair_prompt(rewritten_code: str) -> str:
    prompt = f"""
    The following code snippet does not match the outputs of the orginal unoptimized code. Your task is to repair the code snippet so that it produces the same output as the original code.
    ```
    # Rewritten code
    {rewritten_code}
    ```
    Return the repaired code snippet with the ```python``` tag.
    """
    return prompt
