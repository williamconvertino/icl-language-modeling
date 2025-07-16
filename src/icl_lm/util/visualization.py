import os
import re
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict

class Visualizer:
    def __init__(self, config, checkpoint_root="outputs/checkpoints", figure_root="outputs/figures"):
        self.config = config

        # Dataset name from config (support both `config.dataset.name` and `config.dataset`)
        dataset = getattr(config.dataset, "name", None) or config.dataset
        self.dataset = dataset
        self.metric = config.visualization.metric
        self.mode = config.visualization.visualization_mode  # "lr" or "model"

        self.checkpoint_dir = os.path.join(checkpoint_root, dataset)
        self.figure_dir = os.path.join(figure_root, dataset)
        os.makedirs(self.figure_dir, exist_ok=True)

    def _gather_losses(self):
        """
        Gathers all validation losses from checkpoint filenames.
        Returns: model -> lr -> list of (epoch, val_loss)
        """
        all_data = defaultdict(lambda: defaultdict(list))
        pattern = os.path.join(self.checkpoint_dir, "*", "*", "epoch_*_val=*.pt")

        for ckpt in glob(pattern):
            match = re.search(rf"{re.escape(self.checkpoint_dir)}/([^/]+)/([^/]+)/epoch_(\d+)_val=([0-9.]+)\.pt", ckpt)
            if match:
                model, lr, epoch, val = match.groups()
                all_data[model][lr].append((int(epoch), float(val)))

        for model in all_data:
            for lr in all_data[model]:
                all_data[model][lr].sort(key=lambda x: x[0])

        return all_data

    def _maybe_convert(self, losses):
        if self.metric == "perplexity":
            return [2.718 ** l for l in losses]
        return losses

    def plot(self):
        all_data = self._gather_losses()
        COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        if self.mode == "lr":
            model_name = self.config.model.name if hasattr(self.config.model, "name") else self.config.model
            if model_name not in all_data:
                print(f"No data found for model '{model_name}'")
                return

            curves = []
            for lr, values in all_data[model_name].items():
                epochs, losses = zip(*values)
                losses = self._maybe_convert(losses)
                final_loss = losses[-1]
                label = f"lr={lr}"
                curves.append((final_loss, epochs, losses, label))

            curves.sort(reverse=True, key=lambda x: x[0])  # Worst loss to best

            plt.figure(figsize=(10, 6))
            for i, (_, epochs, losses, label) in enumerate(curves):
                color = COLORS[i % len(COLORS)]
                plt.plot(epochs, losses, label=label, color=color)

            ylabel = "Validation Perplexity" if self.metric == "perplexity" else "Validation Loss"
            plt.title(f"{model_name} – {ylabel} vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(self.figure_dir, f"{model_name}_{self.metric}_vs_epoch.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot to {save_path}")

        elif self.mode == "model":
            curves = []
            for model, lr_dict in all_data.items():
                best_lr, best_values = None, None
                best_final_loss = float("inf")

                for lr, values in lr_dict.items():
                    final_loss = values[-1][1]
                    if final_loss < best_final_loss:
                        best_final_loss = final_loss
                        best_lr = lr
                        best_values = values

                if best_values:
                    epochs, losses = zip(*best_values)
                    losses = self._maybe_convert(losses)
                    final_loss = losses[-1]
                    label = f"{model} (lr={best_lr})"
                    curves.append((final_loss, epochs, losses, label))

            curves.sort(reverse=True, key=lambda x: x[0])

            plt.figure(figsize=(10, 6))
            for i, (_, epochs, losses, label) in enumerate(curves):
                color = COLORS[i % len(COLORS)]
                plt.plot(epochs, losses, label=label, color=color)

            ylabel = "Validation Perplexity" if self.metric == "perplexity" else "Validation Loss"
            plt.title(f"Best LR – {ylabel} vs Epoch per Model")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(self.figure_dir, f"best_lr_{self.metric}_vs_epoch.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot to {save_path}")

        else:
            raise ValueError(f"Unknown visualization_mode '{self.mode}'")
