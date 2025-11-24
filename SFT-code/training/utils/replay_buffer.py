import os
import random
import torch
from torch_utils import distributed as dist


def map_into_device(anything, device):
    if isinstance(anything, torch.Tensor):
        return anything.to(device)
    if isinstance(anything, list):
        return [map_into_device(item, device) for item in anything]
    if isinstance(anything, dict):
        return {k: map_into_device(anything[k], device) for k in anything}
    raise NotImplemented


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.is_valid = []


    def save_in_buffer(self, input_chunks):
        self.buffer.extend(map_into_device(input_chunks, "cpu"))
        self.is_valid.extend([False] * len(input_chunks))
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
            self.is_valid = self.is_valid[-self.buffer_size:]

    def sample_from_buffer(self, k, device):
        valid_indexes = [index for index, flag in enumerate(self.is_valid) if flag]
        sampled_indexes = random.choices(valid_indexes, k=k)
        for idx in sampled_indexes:
            self.is_valid[idx] = False
        return map_into_device([self.buffer[idx] for idx in sampled_indexes], device)

    def is_ready(self, k):
        return sum(self.is_valid) > k

    def reset_flag(self):
        self.is_valid = [True] * len(self.buffer)

    def save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.buffer, f"{path}/replay_buffer_{dist.get_rank()}.pth")

    def load_checkpoint(self, path):
        self.buffer = torch.load(f"{path}/replay_buffer_{dist.get_rank()}.pth")
        self.reset_flag()


if __name__ == "__main__":
    pass