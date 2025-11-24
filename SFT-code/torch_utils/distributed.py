# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
import random 
from accelerate import Accelerator, InitProcessGroupKwargs
import datetime
#----------------------------------------------------------------------------

ACCELERATOR = None

def init():

    global ACCELERATOR#​​global ACCELERATOR​​ 的声明通常用于在函数内部访问或修改函数外部定义的全局变量 ACCELERATOR
    
    ACCELERATOR = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(
            timeout=datetime.timedelta(hours=0.5)
        )]
    )#初始化 Accelerator（自动处理分布式设置）

#----------------------------------------------------------------------------

def get_accelerator():
    return ACCELERATOR

def get_rank():
    return ACCELERATOR.process_index

#----------------------------------------------------------------------------

def get_local_rank():
    return ACCELERATOR.local_process_index

#----------------------------------------------------------------------------

def get_world_size():
    return ACCELERATOR.num_processes

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):#通过 if get_rank() == 0 检查当前进程的 rank（进程编号）是否为 0。如果是，则执行 print(*args, **kwargs)；否则跳过。
    if get_rank() == 0:#如果所有进程都执行 print()，相同的日志会被重复输出多次（例如，8个GPU会打印8次相同内容），导致日志冗余和可读性下降
        print(*args, **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    init()
    a = torch.zeros(3, device="cuda") + get_rank()
    aa = ACCELERATOR.gather(a)
    print(aa)
