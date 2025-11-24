import requests
import torch 
import socket
import re 
import os 
from math_verify import parse, verify
from types import SimpleNamespace
import concurrent.futures
from .dataloader import register_reward
from .ifeval.eval_lib import test_instruction_following_loose
port = {
    'gnho037': 38008,
    'gnho040': 38009,
    'gnho012': 38007,
    'gnho011': 38006,
    'gnho010': 38005,
}

model_name_or_path = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
base_url = "http://localhost:{}/classify".format(
    port.get(socket.gethostname(), '0')
)

def catch_verify(*args, **kwargs):
    try:
        return verify(*args, **kwargs)
    except:
        return False

def simplified_template(prompt, response):
    return f"""<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"""


@register_reward('preference')
def reward_func_batch(
    batch, 
    responses, 
):
    prompt = batch['prompts'][0]
    payload = {
        "model": 'Skywork/Skywork-Reward-V2-Llama-3.1-8B',
        "text": [simplified_template(prompt, response) for response in responses]
    }
    rewards = []
    try:
        responses = requests.post(base_url, json=payload, timeout=10).json()
        for response in responses:
            rewards.append(response["embedding"][0])
        assert len(rewards) == len(payload['text']), f"Expected {len(payload['text'])} rewards, got {len(rewards)}"
        return torch.tensor(rewards)
    except Exception as e:
        print(f"Error: {e}")
        return torch.zeros(len(payload['text']))


@register_reward('stem')
def reward_func_stem(
    batch, 
    responses, 
):
    ext_ans = [parse(batch['answers'][0], extraction_mode="first_match",)] * len(responses)
    ext_res = [parse(response, extraction_mode="first_match") for response in responses]
    rewards = torch.zeros(len(responses))
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if catch_verify(ans, res):
            rewards[i] += 1.0

    return rewards  

@register_reward('ifeval')
def reward_func_ifeval(
    batch, 
    responses, 
):
    inp = SimpleNamespace(**batch['args'])
    rewards = torch.zeros(len(responses))
    for ix, resp in enumerate(responses):
        res = test_instruction_following_loose(inp, resp)
        rwd_list = res.follow_instruction_list
        rewards[ix] = sum(rwd_list) / len(rwd_list)
    return rewards  



scp_system_prompt = r"""
Act as an impartial judge to evaluate whether the "Model Response" correctly answers the "Question" and aligns with the core information in the "Reference Answer".

**Evaluation Process:**
- Extract the answer from the "Reference Answer" and "Model Response".
- Focus on whether the model's answer and the reference answer is essentially the same. Do not consider whether there are any errors in the process.

**Final Judgment:**
- If the final answer is fully matched with the reference answer: Output exactly: \boxed{\text{Correct}}
- Else: Output exactly: \boxed{\text{Incorrect}}.
"""
q_template = """
**Question:**
{}

**Reference Answer:**
{}

**Model Response:**
{}
"""


verify_bench_prompt = r"""
Given the following problem and the reference answer. Judge the correctness of the 
answers given later, with some ability to generalize and match the form and format of the 
answer results. The following specific requirements are followed when judging:
 1. Judge only whether the final result of the reference answer and the answer to be judged 
agree; do not consider whether there are any errors in the process. Don't verify the 
correctness of the answer by yourself, please only refer to the reference answer for the 
correctness of the answer.
 2. The reference answer and the answer to be judged only need to be essentially the same, 
ignoring irrelevant details such as units, symbols, whether or not to approximate, and the form 
of expression in the answer. The two answers are considered to be consistent if they are 
equivalently transformable.
 3. All your analysis answer must be in English.
 4. Please analyze the judged answer and try to compare it with the reference answer. 
At the end of all analysis, give the result of the judgment on an extra line at the end of the 
answer in the form "Final Judgment: \boxed{\text{Correct}}/\boxed{\text{Incorrect}}".
"""
extra_info = """
 Problem: {}
 Reference Answer: {}
 Solution to be evaluated: {}
"""



def extract_judgment(text):
    """
    """
    pattern = r'\\boxed{\\text{(Correct|Incorrect)}}'
    
    match = re.search(pattern, text)
    
    if match:
        result = match.group(1)
        if result == "Correct":
            return 1
        else:
            return 0
    else:
        return 0


endpoint = "http://gnho012:49000/v1/chat/completions"


def send_request(payload, max_retry: int = 3):
    retries = 0
    while retries < max_retry:
        try:
            # very slow
            response = requests.post(endpoint, json=payload, timeout=20)
            return extract_judgment(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            retries += 1
            print(f"Error: {e}")
            
    return 0
    
    


@register_reward('scp')
def reward_func_scp(
    batch, 
    responses, 
):
    # return reward_func_batch(batch, responses)
    q = batch['prompts'][0]
    ref = batch['extract_solution']
    rewards = torch.zeros(len(responses))
    # 准备所有payload
    payloads = []
    for resp in responses:
        payload = {
            "model": "/storage/qiguojunLab/qiguojun/home/Models/Qwen/Qwen3-8B",
            "messages": [
                {"role": "system", "content": scp_system_prompt.strip()},
                {"role": "user", "content": q_template.format(q, ref, resp).strip()}
            ],
            # "messages": [
            #     {"role": "user", "content": verify_bench_prompt.strip() + extra_info.format(q, ref, resp).strip()}
            # ],
            "temperature": 0.5,
            "max_tokens": 1024,
            "top_p": 0.8,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        payloads.append(payload)
    
    # 使用线程池并发发送请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(send_request, payload, 5): ix 
            for ix, payload in enumerate(payloads)
        }
        
        # 等待所有任务完成并收集结果
        for future in concurrent.futures.as_completed(future_to_index):
            ix = future_to_index[future]
            try:
                result = future.result()
                rewards[ix] = result
            except Exception as e:
                print(f"Request for index {ix} generated an exception: {e}")
                rewards[ix] = 0
    
    return rewards

@register_reward('mmlu')
def reward_func_mmlu(
    batch, 
    responses, 
):
    answer = parse(batch['response'], extraction_mode='first_match')
    rewards = torch.zeros(len(responses))
    for ix, resp in enumerate(responses):
        res = parse(resp, extraction_mode='first_match')
        if catch_verify(answer, res):
            rewards[ix] = 1

    return rewards


@register_reward('arc')
def reward_func_arc(
    batch, 
    responses, 
):
    answer = batch['response'].split('The best answer is ')[-1][:-1]
    rewards = torch.zeros(len(responses))
    for ix, resp in enumerate(responses):
        res = parse(resp, extraction_mode='first_match')
        if len(res) == 2 and res[1].lower() == answer.lower():
            rewards[ix] = 1
    return rewards