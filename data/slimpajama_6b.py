import os
from datasets import load_dataset, DatasetDict
from icl_lm.data import Tokenizer
from icl_lm.data import DiskDataset

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "slimpajama_6b")
DATASET_PATH = os.path.abspath(DATASET_PATH)

def initialize_dataset(tokenizer, num_workers):
    
    train_data = load_dataset("DKYoon/SlimPajama-6B", split="train")
    val_data = load_dataset("DKYoon/SlimPajama-6B", split="validation")
    test_data = load_dataset("DKYoon/SlimPajama-6B", split="test")
    
    dataset_dict = DatasetDict({
        "train": train_data,
        "val": val_data,
        "test": test_data
    })

    DiskDataset.build_from_dataset(dataset_dict, tokenizer, DATASET_PATH, num_workers=num_workers)
    
def get_splits(tokenizer, max_seq_len, num_workers=16):

    if not os.path.exists(DATASET_PATH):
        print("Building dataset...")
        initialize_dataset(tokenizer, num_workers)
        
    return DiskDataset.get_splits(
        base_path=DATASET_PATH,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )

if __name__ == "__main__":
    initialize_dataset(tokenizer=Tokenizer(), num_workers=16)