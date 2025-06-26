import os
import torch

import hydra
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from icl_lm.data.tokenizer import Tokenizer
from icl_lm.util.trainer import Trainer

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):
    tokenizer = Tokenizer()
    config.model["vocab_size"] = tokenizer.vocab_size

    model = get_class(config.model._target_)(config.model)
    model.name = generate_model_name(config.model, config.dataset.name)

    splits = get_method(config.dataset._target_)(
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        num_workers=config.training.num_workers
    )

    checkpoint_dir = os.path.join("outputs", "checkpoints")
    log_dir = os.path.join("outputs", "logs")

    device = torch.device(config.device)

    if config.mode == "train":
        trainer = Trainer(config.training, model, splits, tokenizer, checkpoint_dir, log_dir, device)
        if config.training.resume:
            trainer.checkpointing.load_recent()
        trainer.train()

    elif config.mode == "eval":
        print("Eval mode is not yet implemented.")

    elif config.mode == "generate":
        print("Generate mode is not yet implemented.")

    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

def generate_model_name(model_config, dataset_name):
    cfg = OmegaConf.to_container(model_config, resolve=True)
    parts = [cfg["name"]]
    
    for k, v in cfg.items():
        if k.startswith("_") or k == "name":
            continue
        if isinstance(v, bool):
            if v:
                parts.append(k)
        else:
            parts.append(f"{k}={v}")
    
    parts.append(dataset_name)
    return "-".join(parts)

if __name__ == "__main__":
    main()
