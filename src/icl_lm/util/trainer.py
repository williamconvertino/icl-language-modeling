import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.amp import autocast, GradScaler
from .checkpointing import Checkpointing

class Trainer:
    def __init__(self, config, model, splits, tokenizer, checkpoint_dir, device):
        self.config = config
        # self.model = torch.compile(model)
        self.model = model
        self.splits = splits
        self.tokenizer = tokenizer
        self.device = device
        
        self.train_dataloader = DataLoader(
            splits["train"],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True
        )

        self.val_dataloader = DataLoader(
            splits["val"],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False
        )

        self.optimizer = instantiate(config.optimizer, params=self.model.parameters())
        
        training_steps = len(self.train_dataloader) * config.epochs
        warmup_steps = int(training_steps * 0.1)

        self.scheduler = instantiate(
            config.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )

        self.checkpointing = Checkpointing(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_dir=checkpoint_dir,
            device=device
        )

        self.autocast_dtype = getattr(torch, config.precision)
        self.scaler = GradScaler()

    def step_loss(self, batch):
        input_tokens = batch[:, :-1]
        target_tokens = batch[:, 1:]

        with autocast(device_type='cuda', dtype=self.autocast_dtype):
            logits = self.model(input_tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        return loss

    def train(self):
        tqdm.write(f"Training model {self.model.name} on device {self.device}")

        self.model.to(self.device)

        for epoch in range(self.checkpointing.current_epoch + 1, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                loss = self.step_loss(batch)
                self.scaler.scale(loss).backward()

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_train_loss = total_loss / len(self.train_dataloader)
            val_loss = self.validate()

            self.checkpointing.save_epoch(epoch, val_loss)
            self.checkpointing.save_best(epoch, val_loss)

            tqdm.write(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = batch.to(self.device)
                loss = self.step_loss(batch)
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)
