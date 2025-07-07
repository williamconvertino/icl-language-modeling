import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

root_dir = "outputs/checkpoints/tinystories"
pattern = re.compile(r"epoch_(\d+)_val=([0-9.]+)\.pt")

model_name_map = {
    "icl-n_layers=8-n_heads=8-hidden_dim=512-max_seq_len=512-vocab_size=50267-icl_use_ln-block_order=None-tinystories": "ICL",
    "icl-n_layers=8-n_heads=8-hidden_dim=512-max_seq_len=512-vocab_size=50267-icl_use_ln-icl_mlp_out-block_order=None-tinystories": "ICL (MLP between layers)",
    "icl-n_layers=8-n_heads=8-hidden_dim=512-max_seq_len=512-vocab_size=50267-icl_use_ln-use_mlp_out-block_order=None-tinystories": "ICL (MLP at output)",
    "transformer-n_layers=8-n_heads=8-hidden_dim=512-max_seq_len=512-vocab_size=50267-tinystories": "Transformer (8L)",
    "transformer-n_layers=6-n_heads=8-hidden_dim=512-max_seq_len=512-vocab_size=50267-tinystories": "Transformer (6L)",
    "transformer-n_layers=4-n_heads=8-hidden_dim=512-max_seq_len=512-vocab_size=50267-tinystories": "Transformer (4L)",
}

# model -> lr -> {epoch -> val_loss}
all_losses = defaultdict(lambda: defaultdict(dict))

# Load data
for model_dir in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_dir)
    if not os.path.isdir(model_path):
        continue

    for lr in os.listdir(model_path):
        lr_path = os.path.join(model_path, lr)
        if not os.path.isdir(lr_path):
            continue

        for fname in os.listdir(lr_path):
            match = pattern.match(fname)
            if match:
                epoch = int(match.group(1))
                val = float(match.group(2))
                all_losses[model_dir][lr][epoch] = val

# Use shared-epoch strategy to determine best LR per model
best_lr_per_model = {}
final_loss_per_model = {}

for model, lr_dict in all_losses.items():
    # Get shared epochs across all LRs
    epoch_sets = [set(losses.keys()) for losses in lr_dict.values() if losses]
    if not epoch_sets:
        continue
    shared_epochs = set.intersection(*epoch_sets)
    if not shared_epochs:
        continue
    max_shared_epoch = max(shared_epochs)

    # Find LR with lowest loss at that epoch
    best_lr = None
    best_loss = float("inf")
    for lr, losses in lr_dict.items():
        if max_shared_epoch in losses and losses[max_shared_epoch] < best_loss:
            best_loss = losses[max_shared_epoch]
            best_lr = lr

    if best_lr is not None:
        best_lr_per_model[model] = best_lr
        final_loss_per_model[model] = lr_dict[best_lr][max_shared_epoch]

# Sort models by performance
sorted_models = sorted(final_loss_per_model.items(), key=lambda x: x[1])

# Plotting
plt.figure(figsize=(10, 6))
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map = {model: color_cycle[i % len(color_cycle)] for i, (model, _) in enumerate(sorted_models)}

for model, _ in sorted_models:
    lrs = all_losses[model]
    best_lr = best_lr_per_model[model]
    color = color_map[model]

    for lr, losses in lrs.items():
        epochs = sorted(losses.keys())
        val_losses = [losses[e] for e in epochs]
        alpha = 1.0 if lr == best_lr else 0.3
        lw = 2.5 if lr == best_lr else 1
        zorder = 2 if lr == best_lr else 1
        label = f"{model_name_map.get(model, model)} (lr={lr})" if lr == best_lr else None
        plt.plot(epochs, val_losses, color=color, alpha=alpha, linewidth=lw, label=label, zorder=zorder)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curves per Model (Best LR Highlighted)")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/best_val_losses.png", dpi=300)
