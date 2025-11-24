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


code_instruct = r" (DO NOT contains unit test in your code)" # Let's first analyse the problem and then implement the code, 
math_instruct = r' (Put the final answer in \boxed{})'
class DataCollection(IterableDataset):

    def __init__(
        self,
        index_list: list[str],
        data_weights: list[float],
        rank: int = None,
        world_size: int = None,
    ):
        super().__init__()
        self.index_list = []
        for fn in index_list:
            if not fn.endswith('.index'): continue
            with open(fn, 'rb') as f:
                sub_list = pickle.load(f)
            self.index_list.append(sub_list)
        self.weights_mapping = data_weights
        self._file_handles = {}

        if rank is None:
            rank = dist.get_rank()
        if world_size is None:
            world_size = dist.get_world_size()
        self.rank = rank
        self.world_size = world_size

    def _get_file_handle(self, file_path):
        if file_path not in self._file_handles:
            self._file_handles[file_path] = open(file_path, 'rb')
        return self._file_handles[file_path]

    def _load_data(self, file, offset):
        f = self._get_file_handle(file)
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
        for f in self._file_handles.values():
            if not f.closed:
                f.close()

    def __iter__(self): 
        # split all sub list
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        global_worker_id = worker_id + self.rank * num_workers
        total_workers = self.world_size * num_workers

        source_iterators = {}
        for ix, sub_list in enumerate(self.index_list):
            sub_list = sub_list[global_worker_id::total_workers]
            if len(sub_list) == 0: continue
            source_iterators[ix] = {
                'data': sub_list,
                'counter': 0,
                'weight': self.weights_mapping[ix]
            }
        
        total_weight = sum(info['weight'] for info in source_iterators.values())
        source_probs = [source_iterators[i]['weight'] / total_weight 
                       for i in source_iterators.keys()]
        source_indices = list(source_iterators.keys())
        print(source_indices)
        while True:
            chosen_idx = np.random.choice(source_indices, p=source_probs)
            
            source_info = source_iterators[chosen_idx]
            data_list = source_info['data']
            counter = source_info['counter']
            
            data_item = data_list[counter % len(data_list)]
            source_info['counter'] = counter + 1
            
            data = self._load_data(data_item[0], data_item[1])
            data['reward_tag'] = data_item[2]
            data['sample_metadata'] = (data_item[0], data_item[1], chosen_idx, source_probs[chosen_idx])
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
    weights: list[float] = None,
    num_workers: int = 8,
    prefetch_factor: int = 4,
):
    index_list = [os.path.join(index_dir, fn) for fn in os.listdir(index_dir) if fn.endswith('.index')]
    index_list.sort()
    if weights is None:
        weights = [1.0] * len(index_list)
    assert len(index_list) == len(weights), "The number of index files and weights must be the same"
    if dist.get_rank() == 0:
        for fn, w in zip(index_list, weights):
            print(f'{os.path.basename(fn)} weight: {w}')
    ds = DataCollection(index_list, weights)
    dl = DataLoader(
        ds, 
        batch_size=1, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
    )

    return dl

def create_index(
    file_list: list[str], 
    reward_tag: list[str],
):
    index_content = []
    for fn, tag in zip(file_list, reward_tag):
        fp = open(fn, 'r')
        offset = 0
        while True:
            line = fp.readline()
            if not line: break
            pair = (fn, offset, tag) 
            offset = fp.tell()
            index_content.append(pair)
        fp.close()
    return index_content

def create_and_save_index(
    file_list: list[str],
    reward_tag: list[str],
    output_dir: str,
):
    for file, rwd_tag in zip(file_list, reward_tag):
        index_content = create_index([file], [rwd_tag])
        print(f'{file} has {len(index_content)} items')
        name_list = '.'.join(os.path.basename(file).split('.')[:-1])
        with open(os.path.join(output_dir, f'{name_list}.index'), 'wb') as f:
            pickle.dump(index_content, f)
