import os
import torch

import hydra
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from icl_lm.data.tokenizer import Tokenizer
from icl_lm.util.trainer import Trainer
from icl_lm.util.generator import Generator
from icl_lm.util.llm_eval import Evaluator

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):
    
    tokenizer = Tokenizer()
    config.model["vocab_size"] = tokenizer.vocab_size

    model = get_class(config.model._target_)(config.model)
    model.name = generate_model_name(config)

    splits = get_method(config.dataset._target_)(
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        num_workers=config.training.num_workers
    )
    splits["name"] = config.dataset.name

    checkpoint_dir = os.path.join("outputs", "checkpoints", config.dataset.name, model.name, str(config.training.optimizer.lr))
    generation_dir = os.path.join("outputs", "generations", config.dataset.name, model.name, str(config.training.optimizer.lr))
    llm_eval_dir = os.path.join("outputs", "llm_eval", config.dataset.name, "baseline") if config.generation.use_baseline else os.path.join("outputs", "llm_eval", config.dataset.name, model.name, str(config.training.optimizer.lr))

    device = torch.device(config.device)

    if config.mode == "train":
        trainer = Trainer(config.training, model, splits, tokenizer, checkpoint_dir, device)
        if config.training.resume:
            trainer.checkpointing.load_recent()
        trainer.train()
    elif config.mode == "eval":
        evaluator = Evaluator(config.generation, model, splits, tokenizer, checkpoint_dir, llm_eval_dir, device)
        evaluator.evaluate()
    elif config.mode == "generate":
        generator = Generator(config.generation, model, splits, tokenizer, checkpoint_dir, generation_dir, device)
        
        checkpoint_type = config.generation.checkpoint

        if checkpoint_type == "best":
            generator.checkpointing.load_best()
        elif checkpoint_type == "recent":
            generator.checkpointing.load_recent()
        elif checkpoint_type.startswith("epoch_"):
            epoch_num = int(checkpoint_type.split("_")[1])
            generator.checkpointing.load_epoch(epoch_num)
        elif checkpoint_type is not None and checkpoint_type != "":
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

        generator.generate()
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

def generate_model_name(config):
    
    parts = [config.model.name]
    
    for k, v in config.model.items():
        if k.startswith("_") or k == "name" or k is None:
            continue
        elif isinstance(v, bool) and v:
            parts.append(k)
        
    parts.append(config.dataset.name)
    
    return "-".join(parts)
 
if __name__ == "__main__":
    main()
