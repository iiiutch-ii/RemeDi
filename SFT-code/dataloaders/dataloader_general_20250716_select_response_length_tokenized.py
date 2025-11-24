from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os, random
import numpy as np
from tqdm import tqdm
from dataloaders.sampler import InfiniteSampler
from typing import List, Tuple
from torch_utils.distributed import get_rank, get_world_size
import re
import json
from collections import Counter
import time

##############################共用区域#################################
#在collate_fn中对批次数据进行填充，选取批次内最长的句子为填充依据，提升处理效率，使用封闭函数，可以给定一些额外的指定参数，比如tokenizer
def make_collate_fn(tokenizer):
    def collate_fn(batch):
        unpadding_text = [item[0] for item in batch]
        mask_start_id = [item[1] for item in batch]
        text_length = [item[2] for item in batch]

        padding_chat_tokens = tokenizer(
        unpadding_text,
        padding="longest",      # 关键参数：按最长序列填充
        truncation=True,       # 超长序列自动截断
        return_tensors="pt",
        max_length=4096,        # 返回PyTorch张量
    )["input_ids"]
        return padding_chat_tokens, mask_start_id, text_length
    return collate_fn


def make_collate_fn_cut(tokenizer):
    def collate_fn(batch):
        unpadding_text = [item[0] for item in batch]
        mask_start_id = [item[1] for item in batch]
        text_length = [item[2] for item in batch]

        len_max = max(text_length)

        padding_chat_tokens = []
        for text in unpadding_text:#text已经是token了
            res = len_max - len(text)
            text = text + [126081 for i in range(res)]
            padding_chat_tokens.append(text)
        padding_chat_tokens = torch.tensor(padding_chat_tokens)
        
        return padding_chat_tokens, mask_start_id, text_length
    return collate_fn
#######################################################################



###########################Math###############################
#定义数据读取格式
class MATH_Dataset(Dataset):
    def __init__(self, file_path,tokenizer,max_length,rank_index):
        
        ############
        # max_length = 1024
        # rank_index = 0w
        # rank_assign = [0,1]
        ###########

        #读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        #整合文件
        self.chat_texts = []
        self.mask_start_id = []
        self.length_of_full_ids_text = []
        self.dataset_id = []
        self.source = []

        for i in tqdm(range(len(data)),desc=f"Rank{rank_index},MATH", position=rank_index, disable=rank_index != 0):
            if data[i]["length_of_full_ids_text"] <= 4096:
                if data[i]["response_len"] <= max_length:
                    self.chat_texts.append(data[i]["chat_texts"])
                    self.mask_start_id.append(data[i]["mask_start_id"])
                    self.length_of_full_ids_text.append(data[i]["length_of_full_ids_text"])
                    self.dataset_id.append(data[i]["dataset_id"])
                    self.source.append(data[i]["source"])
        
        total_rate = len(self.chat_texts)/len(data)
        mylog = "Math:" + str(total_rate)
        print(mylog)
        print(Counter(self.source))
            
    
    def __len__(self):
        return len(self.mask_start_id)

    def __getitem__(self, idx):
        return self.chat_texts[idx],self.mask_start_id[idx],self.length_of_full_ids_text[idx],self.dataset_id[idx]


#定义dataloader
def dataloader_MATH(file_path,tokenizer_path,batch_size,max_length,rank_index,rank_assign):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_set = MATH_Dataset(file_path,tokenizer=tokenizer,max_length=max_length,rank_index=rank_index)
    sampler = InfiniteSampler(data_set, num_replicas=len(rank_assign),rank=rank_assign.index(rank_index),shuffle=False)
    dataloader = DataLoader(
        data_set,
        batch_size=batch_size,      # 每批样本数
        num_workers=16,
        sampler = sampler,
        collate_fn=make_collate_fn_cut(tokenizer=tokenizer)
    )
    return dataloader

###########################################################################




###########################Science###############################
#定义数据读取格式
class SCIENCE_Dataset(Dataset):
    def __init__(self, file_path,tokenizer,max_length,rank_index):
        
        ############# max_length = 1024
        # rank_index = 0
        # rank_assign = [0,1]
        ###########

        #读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        #整合文件
        self.chat_texts = []
        self.mask_start_id = []
        self.length_of_full_ids_text = []
        self.dataset_id = []
        self.source = []

        for i in tqdm(range(len(data)),desc=f"Rank{rank_index},SCIENCE", position=rank_index, disable=rank_index != 0):
            if data[i]["length_of_full_ids_text"] <= 4096:
                if data[i]["response_len"] <= max_length:
                    self.chat_texts.append(data[i]["chat_texts"])
                    self.mask_start_id.append(data[i]["mask_start_id"])
                    self.length_of_full_ids_text.append(data[i]["length_of_full_ids_text"])
                    self.dataset_id.append(data[i]["dataset_id"])
                    self.source.append(data[i]["source"])
        
        total_rate = len(self.chat_texts)/len(data)
        mylog = "SCIENCE:" + str(total_rate)
        print(mylog)
        print(Counter(self.source))
            
    
    def __len__(self):
        return len(self.mask_start_id)

    def __getitem__(self, idx):
        return self.chat_texts[idx],self.mask_start_id[idx],self.length_of_full_ids_text[idx],self.dataset_id[idx]


#定义dataloader
def dataloader_SCIENCE(file_path,tokenizer_path,batch_size,max_length,rank_index,rank_assign):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_set = SCIENCE_Dataset(file_path,tokenizer=tokenizer,max_length=max_length,rank_index=rank_index)
    sampler = InfiniteSampler(data_set, num_replicas=len(rank_assign),rank=rank_assign.index(rank_index),shuffle=False)
    dataloader = DataLoader(
        data_set,
        batch_size=batch_size,      # 每批样本数
        num_workers=16,
        sampler = sampler,
        collate_fn=make_collate_fn_cut(tokenizer=tokenizer)
    )
    return dataloader

###########################################################################






###########################Code###############################
#定义数据读取格式
class CODE_Dataset(Dataset):
    def __init__(self, file_path,tokenizer,max_length,rank_index):
        
        ############# max_length = 1024
        # rank_index = 0
        # rank_assign = [0,1]
        ###########

        #读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        #整合文件
        self.chat_texts = []
        self.mask_start_id = []
        self.length_of_full_ids_text = []
        self.dataset_id = []
        self.source = []

        for i in tqdm(range(len(data)),desc=f"Rank{rank_index},CODE", position=rank_index, disable=rank_index != 0):
            if data[i]["length_of_full_ids_text"] <= 4096:
                if data[i]["response_len"] <= max_length:
                    self.chat_texts.append(data[i]["chat_texts"])
                    self.mask_start_id.append(data[i]["mask_start_id"])
                    self.length_of_full_ids_text.append(data[i]["length_of_full_ids_text"])
                    self.dataset_id.append(data[i]["dataset_id"])
                    self.source.append(data[i]["source"])
        
        total_rate = len(self.chat_texts)/len(data)
        mylog = "CODE:" + str(total_rate)
        print(mylog)
        print(Counter(self.source))
            
    
    def __len__(self):
        return len(self.mask_start_id)

    def __getitem__(self, idx):
        return self.chat_texts[idx],self.mask_start_id[idx],self.length_of_full_ids_text[idx],self.dataset_id[idx]


#定义dataloader
def dataloader_CODE(file_path,tokenizer_path,batch_size,max_length,rank_index,rank_assign):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_set = CODE_Dataset(file_path,tokenizer=tokenizer,max_length=max_length,rank_index=rank_index)
    sampler = InfiniteSampler(data_set, num_replicas=len(rank_assign),rank=rank_assign.index(rank_index),shuffle=False)
    dataloader = DataLoader(
        data_set,
        batch_size=batch_size,      # 每批样本数
        num_workers=16,
        sampler = sampler,
        collate_fn=make_collate_fn_cut(tokenizer=tokenizer)
    )
    return dataloader

###########################################################################





###########################CHAT###############################
#定义数据读取格式
class CHAT_Dataset(Dataset):
    def __init__(self, file_path,tokenizer,max_length,rank_index):
        
        ############# max_length = 1024
        # rank_index = 0
        # rank_assign = [0,1]
        ###########

        #读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        #整合文件
        self.chat_texts = []
        self.mask_start_id = []
        self.length_of_full_ids_text = []
        self.dataset_id = []
        self.source = []

        for i in tqdm(range(len(data)),desc=f"Rank{rank_index},CHAT", position=rank_index, disable=rank_index != 0):
            if data[i]["length_of_full_ids_text"] <= 4096:
                if data[i]["response_len"] <= max_length:
                    self.chat_texts.append(data[i]["chat_texts"])
                    self.mask_start_id.append(data[i]["mask_start_id"])
                    self.length_of_full_ids_text.append(data[i]["length_of_full_ids_text"])
                    self.dataset_id.append(data[i]["dataset_id"])
                    self.source.append(data[i]["source"])
        
        total_rate = len(self.chat_texts)/len(data)
        mylog = "CHAT:" + str(total_rate)
        print(mylog)
        print(Counter(self.source))
            
    
    def __len__(self):
        return len(self.mask_start_id)

    def __getitem__(self, idx):
        return self.chat_texts[idx],self.mask_start_id[idx],self.length_of_full_ids_text[idx],self.dataset_id[idx]


#定义dataloader
def dataloader_CHAT(file_path,tokenizer_path,batch_size,max_length,rank_index,rank_assign):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_set = CHAT_Dataset(file_path,tokenizer=tokenizer,max_length=max_length,rank_index=rank_index)
    sampler = InfiniteSampler(data_set, num_replicas=len(rank_assign),rank=rank_assign.index(rank_index),shuffle=False)
    dataloader = DataLoader(
        data_set,
        batch_size=batch_size,      # 每批样本数
        num_workers=16,
        sampler = sampler,
        collate_fn=make_collate_fn_cut(tokenizer=tokenizer)
    )
    return dataloader

###########################################################################




###########################CHAT###############################
#定义数据读取格式
class SYNTHESIZE_Dataset(Dataset):
    def __init__(self, file_path,tokenizer,max_length,rank_index):
        
        ############# max_length = 1024
        # rank_index = 0
        # rank_assign = [0,1]
        ###########

       #读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        #整合文件
        self.chat_texts = []
        self.mask_start_id = []
        self.length_of_full_ids_text = []
        self.dataset_id = []
        self.source = []

        for i in tqdm(range(len(data)),desc=f"Rank{rank_index},SYNTHESIZE", position=rank_index, disable=rank_index != 0):
            if data[i]["length_of_full_ids_text"] <= 4096:
                if data[i]["response_len"] <= max_length and data[i]["response_len"] > 100:
                    self.chat_texts.append(data[i]["chat_texts"])
                    self.mask_start_id.append(data[i]["mask_start_id"])
                    self.length_of_full_ids_text.append(data[i]["length_of_full_ids_text"])
                    self.dataset_id.append(data[i]["dataset_id"])
                    self.source.append(data[i]["source"])
        
        total_rate = len(self.chat_texts)/len(data)
        mylog = "SYNTHESIZE:" + str(total_rate)
        print(mylog)
        print(Counter(self.source))

        import time
        time.sleep(180)
            
    
    def __len__(self):
        return len(self.mask_start_id)

    def __getitem__(self, idx):
        return self.chat_texts[idx],self.mask_start_id[idx],self.length_of_full_ids_text[idx],self.dataset_id[idx]


#定义dataloader
def dataloader_SYNTHESIZE(file_path,tokenizer_path,batch_size,max_length,rank_index,rank_assign):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_set = SYNTHESIZE_Dataset(file_path,tokenizer=tokenizer,max_length=max_length,rank_index=rank_index)
    sampler = InfiniteSampler(data_set, num_replicas=len(rank_assign),rank=rank_assign.index(rank_index),shuffle=False)
    dataloader = DataLoader(
        data_set,
        batch_size=batch_size,      # 每批样本数
        num_workers=16,
        sampler = sampler,
        collate_fn=make_collate_fn_cut(tokenizer=tokenizer)
    )
    return dataloader

###########################################################################



###########################general###############################
#定义数据读取格式
class General_Dataset(Dataset):
    def __init__(self, file_path,tokenizer,max_length,rank_index):
        
        ############# max_length = 1024
        # rank_index = 0
        # rank_assign = [0,1]
        ###########

       #读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        #整合文件
        self.chat_texts = []
        self.mask_start_id = []
        self.length_of_full_ids_text = []
        self.dataset_id = []
        self.source = []
        min_length = 100
        if "synthetic" in file_path:
            min_length = 50

        for i in tqdm(range(len(data)),desc=f"Rank{rank_index},{file_path}", position=rank_index):
            if data[i]["length_of_full_ids_text"] <= 4096:
                if data[i]["response_len"] <= max_length and data[i]["length_of_full_ids_text"] - data[i]["response_len"] > 10 and data[i]["response_len"] > min_length:
                    self.chat_texts.append(data[i]["chat_texts"])
                    self.mask_start_id.append(data[i]["mask_start_id"])
                    self.length_of_full_ids_text.append(data[i]["length_of_full_ids_text"])
                    self.dataset_id.append(data[i]["dataset_id"])
                    self.source.append(data[i]["source"])
        
        total_rate = len(self.chat_texts)/len(data)
        mylog = f"{file_path}:" + str(total_rate)
        print(mylog)
        print(Counter(self.source))

        #import time
        #time.sleep(180)
            
    
    def __len__(self):
        return len(self.mask_start_id)

    def __getitem__(self, idx):
        return self.chat_texts[idx],self.mask_start_id[idx],self.length_of_full_ids_text[idx],self.dataset_id[idx]


#定义dataloader
def dataloader_General(file_path,tokenizer_path,batch_size,max_length,rank_index,rank_assign):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data_set = General_Dataset(file_path,tokenizer=tokenizer,max_length=max_length,rank_index=rank_index)
    sampler = InfiniteSampler(data_set, num_replicas=len(rank_assign),rank=rank_assign.index(rank_index),shuffle=True)
    dataloader = DataLoader(
        data_set,
        batch_size=batch_size,      # 每批样本数
        num_workers=16,
        sampler = sampler,
        collate_fn=make_collate_fn_cut(tokenizer=tokenizer)
    )
    return dataloader

###########################################################################



###########################################################################

def load_multiple_dataset(
    local_path: List[str],
    num_replicas_per_dataset: List[int],
    batch_size: int,
    num_workers: int = 8,
    max_length: int = 1024,
    tokenizer_path ="GSAI-ML/LLaDA-8B-Instruct",
):
    ##################### batch_size =2
    # num_workers = 8
    # batch_size = 2
    # max_length = 1024
    ####################
    rank = int(dist.get_rank())

    # 为数据分配设备
    dataset_index = [rank in ranks for ranks in num_replicas_per_dataset]
    dataset_index = next((i for i, x in enumerate(dataset_index) if x), None)
    dataset_path = local_path[dataset_index]
    
    # if "math" in dataset_path:
    #     loader_fn = dataloader_MATH
    # elif "code" in dataset_path:
    #     loader_fn = dataloader_CODE
    # elif "chat" in dataset_path:
    #     loader_fn = dataloader_CHAT
    # elif "science" in dataset_path:
    #     loader_fn = dataloader_SCIENCE
    # elif "synthesize" in dataset_path:
    #     loader_fn = dataloader_SYNTHESIZE
    # else:
    #     raise ValueError(f"Unrecognized dataset path: {dataset_path}. Expected to contain 'math' or 'code' or 'chat' or 'science'.")

    loader_fn = dataloader_General
    dataloader = loader_fn(
        file_path=dataset_path,
        tokenizer_path = tokenizer_path,
        batch_size = batch_size,
        max_length = max_length,
        rank_index=rank,
        rank_assign = num_replicas_per_dataset[dataset_index]
    )

    torch.distributed.barrier()
    return dataloader





