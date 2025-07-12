import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class DiskDataset(Dataset):
    def __init__(self, file_path, max_seq_len, tokenizer, stride_fraction=0.5):
        assert os.path.isfile(file_path), f"File not found: {file_path}"
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.stride = int(max_seq_len * stride_fraction)

        self.data = np.memmap(file_path, dtype="int32", mode="r")
        self.file_size = len(self.data)
        self.start_indices = list(range(0, self.file_size - max_seq_len + 1, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        seq = self.data[start: start + self.max_seq_len].copy()
        return torch.tensor(seq, dtype=torch.long)

    @staticmethod
    def build_from_dataset(dataset_dict, tokenizer, output_dir, add_bos=False, add_eos=True, column="text", num_workers=None):
        os.makedirs(output_dir, exist_ok=True)

        def tokenize_fn(text):
            ids = tokenizer.encode(text)
            if add_bos:
                ids = [tokenizer.bos_token_id] + ids
            if add_eos:
                ids = ids + [tokenizer.eos_token_id]
            return ids

        for split in ["train", "validation", "val", "test"]:
            if split not in dataset_dict:
                continue

            print(f"Building split: {split}")
            dataset = dataset_dict[split]

            print("Estimating total token count...")
            token_count = 0
            for example in tqdm(dataset, desc="Estimating"):
                token_count += len(tokenize_fn(example[column]))

            file_path = os.path.join(output_dir, f"{split if split != 'validation' else 'val'}.bin")
            memmap_array = np.memmap(file_path, dtype="int32", mode="w+", shape=(token_count,))
            write_pointer = 0
            buffer = []
            buffer_size = 8192

            def generator():
                for example in dataset:
                    yield tokenize_fn(example[column])

            print("Tokenizing and writing...")
            for token_ids in tqdm(generator(), total=len(dataset), desc=f"Writing {split}"):
                buffer.extend(token_ids)
                if len(buffer) >= buffer_size:
                    memmap_array[write_pointer:write_pointer + len(buffer)] = buffer
                    write_pointer += len(buffer)
                    buffer = []

            if buffer:
                memmap_array[write_pointer:write_pointer + len(buffer)] = buffer

            memmap_array.flush()
            del memmap_array

    @staticmethod
    def get_splits(base_path, tokenizer, max_seq_len):
        splits = {}
        for split_name, stride in [("train", 0.5), ("val", 1.0), ("test", 1.0)]:
            path = os.path.join(base_path, f"{split_name}.bin")
            if os.path.isfile(path):
                splits[split_name] = DiskDataset(
                    file_path=path,
                    max_seq_len=max_seq_len,
                    tokenizer=tokenizer,
                    stride_fraction=stride,
                )
        return splits