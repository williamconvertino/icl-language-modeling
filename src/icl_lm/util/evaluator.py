import os
import json
import math
import statistics
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

class Evaluator:
    def __init__(self, config, model, splits, tokenizer, checkpoint_dir, eval_dir, device):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.test_data = splits["test"]
        self.dataset_name = splits["name"]
        self.eval_dir = eval_dir

        os.makedirs(self.eval_dir, exist_ok=True)
        self.results_path = os.path.join(self.eval_dir, "results.json")

        self.model.to(self.device)

        self.dataloader = DataLoader(
            self.test_data,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False,
        )

        # Load checkpoint
        from .checkpointing import Checkpointing
        self.checkpointing = Checkpointing(
            model=self.model,
            checkpoint_dir=checkpoint_dir,
            device=self.device
        )

        self.autocast_dtype = getattr(torch, config.precision)

    def step_loss(self, batch):
        input_tokens = batch[:, :-1]
        target_tokens = batch[:, 1:]

        with autocast(device_type="cuda", dtype=self.autocast_dtype):
            logits = self.model(input_tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        return loss

    def evaluate(self):
        if os.path.exists(self.results_path):
            with open(self.results_path, "r") as f:
                results = json.load(f)
            self.print_latex(results)
            return results

        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating on test set"):
                batch = batch.to(self.device)
                loss = self.step_loss(batch)
                losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        std_loss = statistics.stdev(losses) if len(losses) > 1 else 0.0
        ppl = math.exp(mean_loss)

        results = {
            "loss": round(mean_loss, 4),
            "ppl": round(ppl, 2),
            "loss_std": round(std_loss, 4),
        }

        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.print_latex(results)
        return results

    def print_latex(self, results):
        latex = (
            f"{results['loss']} ({results['loss_std']}) & "
            f"{results['ppl']}"
        )
        print("Latex format:")
        print(latex)
