import torch
from accelerate import Accelerator
from typing import List, Dict, Any
from typing import Sequence

def pad_to_max_length_dim1(tensor: torch.Tensor, max_len: int, pad_value: float = 0.0) -> torch.Tensor:
    pad_len = max_len - tensor.size(1)
    if pad_len == 0:
        return tensor
    pad_shape = tensor.shape[0:1] + (pad_len,) + tensor.shape[2:]
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=1)

def reorganize_input_chunks(
    input_chunks: List[Dict[str, Any]], 
    accelerator: Accelerator, 
    pad_fields: Dict[str, float],  # now accepts a dict with padding values per field
    sample_pad_field: str,         # determine which to get the max_len
) -> List[Dict[str, Any]]: 

    # Step 1: Merge filtered chunks
    merged = {}
    for key in input_chunks[0]:
        val = input_chunks[0][key]
        if isinstance(val, torch.Tensor):
            merged[key] = []
        elif isinstance(val, list):
            if len(val) == 0:
                continue
            elif isinstance(val[0], torch.Tensor):
                merged[key] = [[] for _ in range(len(val))]

    for chunk in input_chunks:
        for key, val in chunk.items():
            if isinstance(val, torch.Tensor):
                merged[key].append(val)
            elif isinstance(val, list):
                if len(val) == 0:
                    continue
                else:
                    for i, item in enumerate(val):
                        merged[key][i].append(item)

    for key in merged:
        if isinstance(merged[key], list) and isinstance(merged[key][0], torch.Tensor):
            merged[key] = torch.cat(merged[key], dim=0)
        elif isinstance(merged[key], list) and isinstance(merged[key][0], list):
            merged[key] = [torch.cat(sublist, dim=0) for sublist in merged[key]]

    # Step 3: Gather across processes with padding on dim=1 for specified fields
    max_seq_len = accelerator.gather(
        torch.tensor([merged[sample_pad_field].size(1)], device=accelerator.device)
    ).max().item()

    for key in merged:
        if isinstance(merged[key], torch.Tensor):
            if key in pad_fields:
                padded = pad_to_max_length_dim1(merged[key], max_seq_len, pad_value=pad_fields[key])
            else:
                padded = merged[key]
            merged[key] = accelerator.gather(padded)
        elif isinstance(merged[key], list):
            merged[key] = [
                accelerator.gather(
                    pad_to_max_length_dim1(t, max_seq_len, pad_value=pad_fields.get(key, 0.0))
                ) if key in pad_fields else accelerator.gather(t)
                for t in merged[key]
            ]

     # Step 4: Filter out entries where "advantage" == 0
    advantage = merged["advantages"]
    mask = advantage != 0

    filtered_chunk = {}
    for key, val in merged.items():
        if isinstance(val, torch.Tensor):
            filtered_chunk[key] = val[mask]
        elif isinstance(val, list) and isinstance(val[0], torch.Tensor):
            filtered_chunk[key] = [item[mask] for item in val]
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(val)}")

    total_valid_samples = filtered_chunk['advantages'].shape[0]

    # Step 4: Even redistribution across processes
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    per_process = total_valid_samples // world_size
    remainder = total_valid_samples % world_size

    start = rank * per_process + min(rank, remainder)
    end = start + per_process + (1 if rank < remainder else 0)

    local_batch = {}
    for key, val in filtered_chunk.items():
        if isinstance(val, torch.Tensor):
            local_batch[key] = val[start:end]
        elif isinstance(val, list):
            local_batch[key] = [v[start:end] for v in val]

    # Step 5: Resplit into input_chunks
    num_samples = local_batch['advantages'].shape[0]
    if num_samples > 8:
        num_chunks = (num_samples + 7) // 8
        chunk_sizes = [num_samples // num_chunks] * num_chunks
        for i in range(num_samples % num_chunks):
            chunk_sizes[i] += 1

        input_chunks = []
        idx = 0
        for size in chunk_sizes:
            chunk = {}
            for key, val in local_batch.items():
                if isinstance(val, torch.Tensor):
                    chunk[key] = val[idx:idx+size]
                elif isinstance(val, list):
                    chunk[key] = [v[idx:idx+size] for v in val]
            input_chunks.append(chunk)
            idx += size
    else:
        input_chunks = [local_batch]

    return input_chunks


def split_input_chunks_local(
    input_chunks, # List of Dicts
    fwd_num_generations,
    n: int = 1,
):
    """
    1. List of Tensors 
    2. Tensors (B,...) or (,)
    3. object
    """
    if n == 1:
        return input_chunks
    n_chunks = len(input_chunks) * n
    new_chunks = []
    # 4 * 2
    for idx in range(n_chunks):
        cur_chunk = {}
        cur_chunk_idx = idx // n
        idx_in_chunk = idx % n
        start_idx = idx_in_chunk * fwd_num_generations // n
        end_idx = start_idx + fwd_num_generations // n
        for key in input_chunks[0].keys():
            item = input_chunks[cur_chunk_idx][key]
            if isinstance(item, torch.Tensor):
                if item.dim() == 0:
                    cur_chunk[key] = item
                else:
                    cur_chunk[key] = item[start_idx:end_idx]
            elif isinstance(item, Sequence):
                if isinstance(item[0], torch.Tensor):
                    cur_chunk[key] = [item[start_idx:end_idx] for item in item]
                else:
                    cur_chunk[key] = item[start_idx:end_idx]
            else:
                cur_chunk[key] = input_chunks[cur_chunk_idx][key]
        new_chunks.append(cur_chunk)

    return new_chunks


import unittest
from unittest.mock import MagicMock
import torch
from accelerate import Accelerator

# Assume your reorganize_input_chunks and pad_to_max_length are already imported.

class TestReorganizeInputChunks(unittest.TestCase):
    
    def setUp(self):
        # Setup mock accelerator
        self.accelerator = MagicMock(spec=Accelerator)
        
        # Define world_size and rank for multi-process simulation
        self.accelerator.num_processes = 1
        self.accelerator.process_index = 0  # This would be 0 in the first process

        # Mock gather to simply return the inputs
        def mock_gather(tensor):
            return tensor

        self.accelerator.gather.side_effect = mock_gather
    
    def test_reorganize_input_chunks(self):
        input_chunks = [
            {"advantage": torch.tensor([1, 0]), "data": torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])},
            {"advantage": torch.tensor([1, 1]), "data": torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])}
        ]

        # Call the function
        output = reorganize_input_chunks(input_chunks, self.accelerator, pad_fields=["data"])

        # Assert output is a list of chunks
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

        # Check that advantage 0 is filtered
        output_data = output[0]["data"]
        self.assertEqual(output_data.size(0), 2)  # Only two samples should remain after filtering
        
        # Check that gathering happens correctly
        self.accelerator.gather.assert_called()

    def test_padding_and_shaping(self):
        input_chunks = [
            {"advantage": torch.tensor([1, 0]), "data": torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])},
            {"advantage": torch.tensor([1, 1]), "data": torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])}
        ]

        # Simulate more processes and different data distributions
        self.accelerator.num_processes = 2
        self.accelerator.process_index = 1  # This would be 1 in the second process

        output = reorganize_input_chunks(input_chunks, self.accelerator, pad_fields=["data"])

        # Check reshaped output (should be padded and split correctly)
        output_data = output[0]["data"]
        self.assertEqual(output_data.size(0), 3)  # Should be padded to max length 3 for this example

if __name__ == '__main__':
    # unittest.main()

    accelerator = Accelerator()

    if accelerator.process_index == 0:
        input_chunks = [
            {"advantage": torch.tensor([1, 0], device=accelerator.device), "data": [torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]], device=accelerator.device)], "single": torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]], device=accelerator.device)},
            {"advantage": torch.tensor([2, 0], device=accelerator.device), "data": [torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]], device=accelerator.device)], "single": torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]], device=accelerator.device)},
        ]
    else:
        input_chunks = [
            {"advantage": torch.tensor([3, 0], device=accelerator.device), "data": [torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], device=accelerator.device)], "single": torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 3.0, 4.0]], device=accelerator.device)},
            {"advantage": torch.tensor([4, 0], device=accelerator.device), "data": [torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], device=accelerator.device)], "single": torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 3.0, 4.0]], device=accelerator.device)},
        ]

    output = reorganize_input_chunks(input_chunks, accelerator, pad_fields={"data": 0, "single": 1}, sample_pad_field="single")

