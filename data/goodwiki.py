import os
from datasets import load_dataset, DatasetDict
from icl_lm.data import Tokenizer
from icl_lm.data import DiskDataset

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "goodwiki")
DATASET_PATH = os.path.abspath(DATASET_PATH)

def initialize_dataset(tokenizer, num_workers):
    
    train_data = load_dataset("euirim/goodwiki", split="train")
    
    val_data = train_data.select(range(2_000)) # 2k val
    test_data = train_data.select(range(2_000, 4_000)) # 2k test
    train_data = train_data.select(range(4_000, len(train_data))) # 40k train
    
    dataset_dict = DatasetDict({
        "train": train_data,
        "val": val_data,
        "test": test_data
    })

    DiskDataset.build_from_dataset(dataset_dict, tokenizer, "datasets/goodwiki", column="markdown", num_workers=num_workers)
    
def get_splits(tokenizer, max_seq_len, num_workers=16):
    
    if not os.path.exists(DATASET_PATH):
        print("Building dataset...")
        initialize_dataset(tokenizer, num_workers)
        
    return DiskDataset.get_splits(
        base_path="datasets/goodwiki",
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )

if __name__ == "__main__":
    initialize_dataset(tokenizer=Tokenizer(), num_workers=16)