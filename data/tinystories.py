import os
from datasets import load_dataset, DatasetDict
from icl_lm.data import Tokenizer
from icl_lm.data import DiskDataset

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "tinystories_v2")
DATASET_PATH = os.path.abspath(DATASET_PATH)

def initialize_dataset(tokenizer, num_workers):
    
    train_data = load_dataset("roneneldan/TinyStories", data_files="TinyStoriesV2-GPT4-train.txt", split="train")
    valid_data = load_dataset("roneneldan/TinyStories", data_files="TinyStoriesV2-GPT4-valid.txt", split="train")

    val_data = valid_data.select(range(10_000))
    test_data = valid_data.select(range(10_000, len(valid_data)))

    dataset_dict = DatasetDict({
        "train": train_data,
        "val": val_data,
        "test": test_data
    })

    DiskDataset.build_from_dataset(dataset_dict, tokenizer, "datasets/tinystories_v2", num_workers=num_workers)
    
def get_splits(tokenizer, max_seq_len, num_workers=16):
    
    print(DATASET_PATH)
    print(os.path.exists(DATASET_PATH))
    
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