import os
from datasets import load_dataset, DatasetDict
from icl_lm.data import Tokenizer
from icl_lm.data import DiskDataset

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "tinystories_v2")
DATASET_PATH = os.path.abspath(DATASET_PATH)

def initialize_dataset(tokenizer, num_workers):
    
    train_data = load_dataset("roneneldan/TinyStories", data_files="TinyStoriesV2-GPT4-train.txt", split="train")
    valid_data = load_dataset("roneneldan/TinyStories", data_files="TinyStoriesV2-GPT4-valid.txt", split="train")

    val_data = train_data.select(range(20_000))
    train_data = train_data.select(range(20_000, len(train_data)))

    dataset_dict = DatasetDict({
        "train": train_data,
        "val": val_data,
        "test": valid_data
    })

    DiskDataset.build_from_dataset(dataset_dict, tokenizer, "datasets/tinystories_v2", add_bos=True, add_eos=True, num_workers=num_workers)
    
def get_splits(tokenizer, max_seq_len, num_workers=16):
    
    if not os.path.exists(DATASET_PATH):
        print("Building dataset...")
        initialize_dataset(tokenizer, num_workers)
        
    return DiskDataset.get_splits(
        base_path="datasets/tinystories_v2",
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )

if __name__ == "__main__":
    initialize_dataset(tokenizer=Tokenizer(), num_workers=16)