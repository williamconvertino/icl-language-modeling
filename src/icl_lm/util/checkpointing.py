import os
import torch
import re
from glob import glob

class Checkpointing:
    def __init__(self, model, checkpoint_dir, optimizer=None, scheduler=None, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = os.path.join(checkpoint_dir, model.name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = self._get_initial_best_val_loss()
        self.current_epoch = 0

    def _get_initial_best_val_loss(self):
        best_ckpts = glob(os.path.join(self.checkpoint_dir, "best_e*_val=*.pt"))
        best_val = float("inf")
        for ckpt_path in best_ckpts:
            match = re.search(r"val=([0-9.]+)\.pt", ckpt_path)
            if match:
                val = float(match.group(1))
                best_val = min(best_val, val)
        return best_val

    def _save(self, filename, epoch=None, val_loss=None):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {"model": self.model.state_dict()}
        if self.optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss
        torch.save(checkpoint, path)

    def save_checkpoint(self, name, val_loss):
        filename = f"{name}_val={val_loss:.4f}.pt"
        self._save(filename, val_loss=val_loss)

    def save_epoch(self, epoch, val_loss):
        assert self.optimizer is not None and self.scheduler is not None, "Optimizer and scheduler must be defined to save epoch checkpoint."
        filename = f"epoch_{epoch}_val={val_loss:.4f}.pt"
        self._save(filename, epoch=epoch, val_loss=val_loss)
        self.current_epoch = epoch

    def save_best(self, epoch, val_loss):
        if val_loss >= self.best_val_loss:
            return
        
        self.best_val_loss = val_loss

        old_best = glob(os.path.join(self.checkpoint_dir, "best_e*_val=*.pt"))
        for f in old_best:
            os.remove(f)
            
        filename = f"best_e{epoch}_val={val_loss:.4f}.pt"
        self._save(filename, epoch=epoch, val_loss=val_loss)
        
    def load_best(self):
        best_ckpts = glob(os.path.join(self.checkpoint_dir, "best_e*_val=*.pt"))
        if best_ckpts:
            best_ckpt = sorted(best_ckpts)[-1]
            state = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            print(f"Loaded best model from {best_ckpt}")
        else:
            print(f"No best model found")

    def load_recent(self):
        if self.optimizer is None or self.scheduler is None:
            raise ValueError("Optimizer and scheduler must be defined to load recent checkpoint.")
        
        epoch_ckpts = glob(os.path.join(self.checkpoint_dir, "epoch_*_val=*.pt"))
        if epoch_ckpts:
            
            def extract_epoch(path):
                match = re.search(r"epoch_(\d+)_val=", path)
                return int(match.group(1)) if match else -1

            most_recent = max(epoch_ckpts, key=extract_epoch)
            state = torch.load(most_recent, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.current_epoch = state["epoch"]
            print(f"Loaded model, optimizer, and scheduler from {most_recent}")
        else:
            print("No epochs found")
            self.current_epoch = 0