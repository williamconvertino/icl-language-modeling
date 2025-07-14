import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from torch.amp import autocast, GradScaler
from .checkpointing import Checkpointing

class Trainer:
    def __init__(self, config, model, splits, tokenizer, checkpoint_dir, device):
        self.config = config
        self.model = model
        self.splits = splits
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(self.device)
        
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
            checkpoint_dir=checkpoint_dir,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        if config.num_save_steps is not None:
            self.checkpointing.load_recent_step()
        else:
            self.checkpointing.load_recent()

        self.autocast_dtype = getattr(torch, config.precision)
        self.scaler = GradScaler()
        
        if config.compile:
            self.model = torch.compile(model)

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

        prev_val_loss = float('inf')
        patience = 2
        patience_counter = 0
        
        resuming = self.config.num_save_steps is not None

        for epoch in range(self.checkpointing.current_epoch + 1, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            val_loss = 0.0
            ppl = 0.0

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            for i, batch in enumerate(pbar):
                
                if resuming and i < self.checkpointing.current_step:
                    continue
                else:
                    resuming = False
                
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                loss = self.step_loss(batch)

                if i % 5000 == 0 and torch.isnan(loss):
                    raise ValueError(f"NaN loss encountered during training (epoch {epoch}, step {i}).")

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                train_loss = loss.item()
                
                total_loss += train_loss
                
                if self.config.num_save_steps is not None and i % self.config.num_save_steps == 0 and i != 0:
                    
                    val_loss = self.validate()
                    ppl = math.exp(val_loss)
                    
                    self.checkpointing.save_step(
                        epoch=epoch,
                        step_in_epoch=i,
                        val_loss=val_loss
                    )
                
                pbar.set_postfix({
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "ppl": f"{ppl:.2f}"
                })

            avg_train_loss = total_loss / len(self.train_dataloader)
            
            val_loss = self.validate()
            ppl = math.exp(val_loss)

            self.checkpointing.save_epoch(epoch, val_loss)
            self.checkpointing.save_best(epoch, val_loss)

            tqdm.write(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {ppl}")
            
            if val_loss >= prev_val_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                prev_val_loss = val_loss
            
            if patience_counter >= patience:
                tqdm.write("Early stopping triggered.")
                return

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = batch.to(self.device)
                loss = self.step_loss(batch)
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)
