import os
import torch 
import pickle
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import json
import torch_utils.distributed as dist

_reward_functions = {}

def register_reward(name=None):
    def decorator(func):
        key = name if name is not None else func.__name__
        _reward_functions[key] = func
        return func
    return decorator

def get_reward_func(name: str):
    return _reward_functions[name]


code_instruct = r" (Let's first analyse the problem and then implement the code, DO NOT contains unit test in your code)"
math_instruct = r' (Put the final answer in \boxed{})'
class DataCollection(IterableDataset):

    def __init__(
        self,
        index_file: str,
        rank: int = None,
        world_size: int = None,
        index_id: int = None,
    ):
        super().__init__()
        self.index_list = []
        with open(index_file, 'rb') as f:
            self.index_list = pickle.load(f)
        # 缓存文件句柄，避免频繁打开关闭文件
        self._file_handles = None

        self.rank = rank
        self.world_size = world_size
        self.index_id = index_id

    def _load_data(self, file, offset):
        if self._file_handles is None:
            self._file_handles = open(file, 'r')
        f = self._file_handles
        f.seek(offset)
        line = f.readline().strip()
        data = json.loads(line)
        
        file_name = os.path.basename(file)
        # Code Data
        assert 'prompts' not in data, "prompts is already in data"
        if file_name == 'LeetCodeDataset.jsonl':
            data['prompts'] = data['query'] + code_instruct
        elif file_name == 'TACO-verified.jsonl':
            data['prompts'] = data['question'] + code_instruct
            data['input_output'] = json.loads(data['input_output'])
        elif file_name == 'KodCode-V1-SFT-R1.jsonl':
            data['prompts'] = data['problem'] + code_instruct
        # Math Data
        elif 'Big-Math-RL-Verified' in file_name or file_name == 'NuminaMath-1.5-RL-Verifiable.jsonl':
            data['prompts'] = data['problem'] + math_instruct
            data['answers'] = data['answer']
        # Preference Data
        elif file_name == 'Infinity-Preference.jsonl' or file_name == 'Skywork-Reward-Preference-80K-v0.2.jsonl':
            data['prompts'] = data['chosen'][0]['content']
        elif file_name == 'MATH.jsonl':
            data['prompts'] = data['problem'] + math_instruct
            data['answers'] = data['solution']
        elif file_name == 'gsm8k.jsonl':
            data['prompts'] = data['question'] + math_instruct
            data['answers'] = data['answer']
        elif 'nemo-stem' in file_name:
            data['prompts'] = data['messages'][1]['content'] + math_instruct
            data['answers'] = data['messages'][2]['content']
        elif 'nemo-if' in file_name:
            data['prompts'] = data['input'][0]['content']
        elif 'scp-61k' in file_name or 'scp-nomath' in file_name or 'scp-82k' in file_name:
            data['prompts'] = data['problem']
        elif 'mmlu_filtered' in file_name or 'arc_c_filtered' in file_name:
            data['prompts'] = data['transform_question'] + math_instruct
        elif 'rstar' in file_name:
            data['prompts'] = data['question'] + code_instruct
            data['inputs'] = json.loads(data['inputs'])
            data['outputs'] = json.loads(data['outputs'])
        elif 'deepscaler' in file_name:
            data['prompts'] = data['problem'] + math_instruct
            data['answers'] = data['answer']
        else:
            raise ValueError(f"Unsupported dataset file: {file_name}")

        # check if data field is None
        for key, value in data.items():
            if value is None:
                data[key] = 'None'

        return data
    
    def __del__(self):
        """清理文件句柄"""
        del self._file_handles

    def __iter__(self): 
        # split all sub list
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        global_worker_id = worker_id + self.rank * num_workers
        total_workers = self.world_size * num_workers

        sub_list = self.index_list[global_worker_id::total_workers]
        
        # 无限循环，根据权重采样返回数据
        counter = 0
        while True:
            data_item = sub_list[counter % len(sub_list)]
            counter = counter + 1
            
            data = self._load_data(data_item[0], data_item[1])
            data['reward_tag'] = data_item[2]
            data['sample_metadata'] = (data_item[0], data_item[1], self.index_id, 1.0)
            if data is None: continue
            yield data


def collate_fn(batch):
    assert len(batch) == 1, "The batch size must be 1"
    data = batch[0]
    data['prompts'] = [data['prompts']]
    if 'answers' in data:
        data['answers'] = [data['answers']]

    return data


def universal_reward_func(batch, responses):
    rwd_func = get_reward_func(batch['reward_tag'])

    return rwd_func(batch, responses)

def load_rl_dataset(
    index_dir: str,
    splits: list[float] = None,
    num_workers: int = 8,
    prefetch_factor: int = 4,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    index_list = [os.path.join(index_dir, fn) for fn in os.listdir(index_dir) if fn.endswith('.index')]
    index_list.sort()
    assert sum(splits) == world_size, "The sum of splits must be equal to world_size"
    assert len(index_list) == len(splits), "The number of index files and splits must be the same"
    rank_splits = [[] for _ in range(len(index_list))] 
    start_rank = 0
    for i in range(len(index_list)):
        rank_splits[i] = list(range(start_rank, start_rank + splits[i]))
        start_rank += splits[i]
    
    if rank == 0:
        for index, rank_group in zip(index_list, rank_splits):
            print(f'index {os.path.basename(index)} has ranks: {rank_group}')

    # pick 
    # dist.print0(rank_splits)
    for index, rank_group in zip(index_list, rank_splits):
        if rank in rank_group:
            index_to_pick = index
            index_world_size = len(rank_group)
            index_rank = rank_group.index(rank)
            break

    ds = DataCollection(
        index_to_pick, index_rank, index_world_size, index_list.index(index_to_pick))
    dl = DataLoader(
        ds, 
        batch_size=1, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
    )

    return dl

# import os 
# from general_data.dataloader import create_and_save_index
# src_dir = '/storage/qiguojunLab/huangzem/data/general-rl-data/20250905'

# create_and_save_index(
#     file_list=[
#         # math
#         os.path.join(src_dir, 'Big-Math-RL-Verified.jsonl'),
#         os.path.join(src_dir, 'gsm8k.jsonl'),
#         os.path.join(src_dir, 'MATH.jsonl'),
#         os.path.join(src_dir, 'NuminaMath-1.5-RL-Verifiable.jsonl'),
#         # code
#         os.path.join(src_dir, 'KodCode-V1-SFT-R1.jsonl'),
#         os.path.join(src_dir, 'LeetCodeDataset.jsonl'),
#         # general
#         os.path.join(src_dir, 'Infinity-Preference.jsonl'),
#         os.path.join(src_dir, 'Skywork-Reward-Preference-80K-v0.2.jsonl'),
#     ],
#     reward_tag=[
#         # math
#         'general-math',
#         'gsm8k',
#         'general-math',
#         'general-math',
#         # code
#         'kodcode',
#         'leetcode',
#         # general
#         'preference',
#         'preference'
#     ],
#     output_dir='general_data/index',
# )