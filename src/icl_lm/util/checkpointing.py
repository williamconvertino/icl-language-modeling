import os
import torch
import re
from glob import glob

class Checkpointing:
    def __init__(self, model, checkpoint_dir, optimizer=None, scheduler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = self._get_initial_best_val_loss()
        self.current_epoch = 0
        self.current_step = 0

    def _get_initial_best_val_loss(self):
        best_ckpts = glob(os.path.join(self.checkpoint_dir, "best_e*_val=*.pt"))
        best_val = float("inf")
        for ckpt_path in best_ckpts:
            match = re.search(r"val=([0-9.]+)\.pt", ckpt_path)
            if match:
                val = float(match.group(1))
                best_val = min(best_val, val)
        return best_val

    def _save(self, filename, epoch=None, val_loss=None, step_in_epoch=None):
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
        if step_in_epoch is not None:
            checkpoint["step_in_epoch"] = step_in_epoch
        torch.save(checkpoint, path)

    def save_checkpoint(self, name, val_loss):
        filename = f"{name}_val={val_loss:.4f}.pt"
        self._save(filename, val_loss=val_loss)

    def save_epoch(self, epoch, val_loss):
        assert self.optimizer is not None and self.scheduler is not None
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

    def save_step(self, epoch, step_in_epoch, val_loss):
        assert self.optimizer is not None and self.scheduler is not None
        filename = f"epoch_{epoch}_step_{step_in_epoch}_val={val_loss:.4f}.pt"
        self._save(filename, epoch=epoch, val_loss=val_loss, step_in_epoch=step_in_epoch)

    def load_step(self, epoch=None, step_in_epoch=None):
        pattern = os.path.join(self.checkpoint_dir, "epoch_*_step_*_val=*.pt")
        step_ckpts = glob(pattern)

        if not step_ckpts:
            print("No step-level checkpoints found")
            return

        def extract_tuple(path):
            match = re.search(r"epoch_(\d+)_step_(\d+)_val=", path)
            if match:
                return int(match.group(1)), int(match.group(2))
            return -1, -1

        if epoch is not None and step_in_epoch is not None:
            matches = [p for p in step_ckpts if f"epoch_{epoch}_step_{step_in_epoch}_" in p]
            if not matches:
                print(f"No checkpoint found for epoch {epoch}, step {step_in_epoch}")
                return
            target_ckpt = max(matches, key=os.path.getmtime)
        else:
            target_ckpt = max(step_ckpts, key=lambda p: extract_tuple(p))

        state = torch.load(target_ckpt, map_location=self.device, weights_only=False)
        self.load_checkpoint_state(state)
        print(f"Loaded step checkpoint from {target_ckpt}")

    def load_checkpoint_state(self, state):
        self.model.load_state_dict(state["model"])
        if self.optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        self.current_epoch = state.get("epoch", 0)
        self.current_step = state.get("step_in_epoch", 0)

    def load_best(self):
        best_ckpts = glob(os.path.join(self.checkpoint_dir, "best_e*_val=*.pt"))
        if best_ckpts:
            best_ckpt = sorted(best_ckpts)[-1]
            state = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.load_checkpoint_state(state)
            print(f"Loaded best model from {best_ckpt}")
        else:
            print("No best model found")

    def load_recent(self):
        epoch_ckpts = glob(os.path.join(self.checkpoint_dir, "epoch_*_val=*.pt"))
        if epoch_ckpts:
            def extract_epoch(path):
                match = re.search(r"epoch_(\d+)_val=", path)
                return int(match.group(1)) if match else -1

            most_recent = max(epoch_ckpts, key=extract_epoch)
            state = torch.load(most_recent, map_location=self.device, weights_only=False)
            self.load_checkpoint_state(state)
            print(f"Loaded model{', optimizer' if self.optimizer else ''}{', scheduler' if self.scheduler else ''} from {most_recent}")
        else:
            print("No epochs found")
            self.current_epoch = 0

    def load_recent_step(self):
        step_ckpts = glob(os.path.join(self.checkpoint_dir, "epoch_*_step_*_val=*.pt"))
        if not step_ckpts:
            print("No recent step checkpoint found")
            return

        def extract_tuple(path):
            match = re.search(r"epoch_(\d+)_step_(\d+)_val=", path)
            return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)

        most_recent = max(step_ckpts, key=lambda p: extract_tuple(p))
        state = torch.load(most_recent, map_location=self.device, weights_only=False)
        self.load_checkpoint_state(state)
        print(f"Loaded most recent step checkpoint from {most_recent}")

    def load_epoch(self, epoch_number):
        pattern = os.path.join(self.checkpoint_dir, f"epoch_{epoch_number}_val=*.pt")
        matching_ckpts = glob(pattern)
        if not matching_ckpts:
            print(f"No checkpoint found for epoch {epoch_number}")
            return

        def extract_val_loss(path):
            match = re.search(r"val=([0-9.]+)\.pt", path)
            return float(match.group(1)) if match else float("inf")

        target_ckpt = max(matching_ckpts, key=extract_val_loss)
        state = torch.load(target_ckpt, map_location=self.device, weights_only=False)
        self.load_checkpoint_state(state)
        print(f"Loaded model{', optimizer' if self.optimizer else ''}{', scheduler' if self.scheduler else ''} from {target_ckpt}")
