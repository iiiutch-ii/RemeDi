import json

def build_zeroshot_instruction(problem, need_code=False):
    func_name = problem["code"].split("(")[0].split(' ')[1]
    example_strs = "\n".join(problem["test_list"])
    return f'''You are an expert Python programmer, and here is your task:

{problem["text"]}

The function is named `{func_name}`, and should pass tests like these:

{example_strs}'''


def format_deepseek_example(problem, need_code=False):
    q, tests, code = problem["text"], problem["test_list"], problem["code"]
    prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
    if need_code:
        code = code.replace("\r", "").replace("\t", "    ")
        prompt += "\n>>> Code:\n{}".format(code)
    return prompt


def format_kod_czy_0419(problem, need_code=False):
    return problem


def format_kod_czy_0420(problem, need_code=False):
    function_name = problem["test_list"][0].split(' ')[1].split('(')[0]
    unit_tests = "\n".join(["    " + item for item in problem["test_list"]])
    # function_declaration = [item for item in problem["code"].split("\r\n")
    #                         if function_name in item and item.startswith("def ")][0]
    import ast
    first_assert = problem["test_list"][0]
    num_args = 0
    try:
        tree = ast.parse(first_assert.strip())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == function_name:
                num_args = len(node.args)
                break
    except Exception as e:
        print("Warning: AST parsing failed, fallback to 2 args.")
        num_args = 2  

    param_names = ", ".join([f"input_param_{i+1}" for i in range(num_args)])
    function_declaration = f"def {function_name}({param_names}):"
    return f"""You are an expert Python programmer. Your task is to complete the implementation of a function named `{function_name}`.

** TARGET FUNCTION **
{problem["text"]}

** UNIT TESTS **
Your code should pass unit tests like:
{unit_tests}

Here is the function to complete:
```python
{function_declaration}
    \"\"\"{problem["text"]}
    \"\"\"
```
"""

def format_MBPP_fewshot_example(problem, examples, need_format=True):
    # q, tests, code = problem["text"], problem["test_list"], problem["code"]
    if need_format:
        prompt = format_deepseek_example(problem, need_code=False)
    else:
        prompt = problem
    prompt_with_shots = '''
Below are example functions based on given problems.
Each example includes a problem, test cases, and a solution.

{}

Now, write a Python function for the following problem.
Return only the function code without markdown, explanations, or extra formatting.

{}
'''.strip().format('\n\n'.join(examples), prompt)
    return prompt_with_shots


def read_MBPP_test_examples(data_path: str, format_test_func=format_deepseek_example, num_examples=0, use_example=True):

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    if use_example:
        for i in range(1, 4):
            ex = examples[i]
            ex_prompt = format_test_func(ex, need_code=True)
            example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
            examples_str += [example_prompt]

    for i in range(10, 510):
    # for i in range(10, 30):
        ex = examples[i]
        if use_example:
            prompt_shots = format_MBPP_fewshot_example(ex, examples_str, use_example)
        else:
            prompt_shots = format_kod_czy_0420(ex)

        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_shots
        }

def read_MBPP_test_examples_deepseekExamples_KodPrompt(data_path: str, num_examples=0,):

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        ex_prompt = format_deepseek_example(ex, need_code=True)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
    # for i in range(10, 30):
        ex = examples[i]
        ploblem_prompt = format_kod_czy_0420(ex)
        prompt_shots = format_MBPP_fewshot_example(ploblem_prompt, examples_str, need_format=False)

        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_shots
        }


if __name__ == "__main__":
    ...