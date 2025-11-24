import json

def extract_docstring_content(code: str) -> str:
    """Extract the full raw content inside triple double-quotes (\"\"\"), preserving line breaks."""
    lines = code.splitlines()
    in_docstring = False
    doc_lines = []

    for line in lines:
        stripped = line.strip()
        if '"""' in stripped:
            quote_count = stripped.count('"""')
            if not in_docstring:
                in_docstring = True
                # Handle single-line docstring
                if quote_count == 2:
                    return stripped.split('"""')[1]
                else:
                    after_quote = stripped.split('"""', 1)[1]
                    doc_lines.append(after_quote)
            else:
                before_quote = stripped.split('"""', 1)[0]
                doc_lines.append(before_quote)
                break
        elif in_docstring:
            doc_lines.append(line)

    return "\n".join(doc_lines).strip()


def extract_asserts(test_code: str, function_name: str = "candidate") -> str:
    """Extracts assert statements from test block and replaces 'candidate' with function_name."""
    lines = test_code.splitlines()
    asserts = [
        line.replace("candidate", function_name) 
        for line in lines 
        if line.strip().startswith("assert")
    ]
    return "\n".join(asserts)


def format_HumanEval_prompt_few_shot(problems):
    task_ids = list(problems.keys())
    for task_id in task_ids:
        problem = problems[task_id]
        function_name = problem['entry_point']
        prompt = problem['prompt'].rstrip()
        docstring = extract_docstring_content(prompt)
        unit_tests = extract_asserts(problem['test'].strip(), function_name)
        HumanEval_prompt = f"""You are an expert Python programmer. Your task is to complete the implementation of a function named `{function_name}`.

** UNIT TESTS **
Your code should pass unit tests like:
{unit_tests}

Here is the function to complete:
```python
{prompt}
```
"""
        problems[task_id]['prompt'] = HumanEval_prompt
    return problems


def format_HumanEval_prompt_zero_shot(problems):
    task_ids = list(problems.keys())
    for task_id in task_ids:
        problem = problems[task_id]
        function_name = problem['entry_point']
        prompt = problem['prompt'].rstrip()
        # docstring = extract_docstring_content(prompt)
        # unit_tests = extract_asserts(problem['test'].strip(), function_name)
        HumanEval_prompt = f"""You are an expert Python programmer. Your task is to complete the implementation of a function named `{function_name}`.

Here is the function to complete:
```python
{prompt}
```
"""
        problems[task_id]['prompt'] = HumanEval_prompt
    return problems

if __name__ == "__main__":
    pass