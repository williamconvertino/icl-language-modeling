import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from .checkpointing import Checkpointing

class Generator:
    def __init__(self, config, model, splits, tokenizer, checkpoint_dir, generation_dir, device):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = splits["test"]
        self.device = device
        self.generation_dir = generation_dir

        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.max_len = config.max_length
        self.num_samples = config.num_samples
        self.temperature = config.temperature
        self.top_p = config.top_p

        os.makedirs(os.path.join(generation_dir, splits["name"]), exist_ok=True)
        self.out_path = os.path.join(generation_dir, splits["name"], f"{model.name}.log")
        
        self.checkpointing = Checkpointing(
            model=self.model,
            checkpoint_dir=checkpoint_dir
        )

        checkpoint_type = config.checkpoint

        if checkpoint_type == "best":
            self.checkpointing.load_best()
        elif checkpoint_type == "recent":
            self.checkpointing.load_recent()
        elif checkpoint_type.startswith("epoch_"):
            epoch_num = int(checkpoint_type.split("_")[1])
            self.checkpointing.load_epoch(epoch_num)
        elif checkpoint_type is not None and checkpoint_type != "":
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

    @torch.no_grad()
    def generate(self):

        self.model.eval()
        self.model.to(self.device)

        with open(self.out_path, "w", encoding="utf-8") as f:
            for i in tqdm(range(self.num_samples), desc="Generating samples"):
                sample = self.test_data[i]
                input_tokens = torch.tensor(sample[:self.max_len], dtype=torch.long, device=self.device)

                if self.tokenizer.eos_token_id in input_tokens:
                    eos_index = (input_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
                    input_tokens = input_tokens[:eos_index]
                
                decoded_full = self.tokenizer.decode(input_tokens.tolist())

                prompt_len = max(1, len(input_tokens) // 2)
                prompt = input_tokens[:prompt_len]

                generated = self.sample(prompt)

                if self.tokenizer.eos_token_id in generated:
                    eos_index = (generated == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
                    generated = generated[:eos_index]

                decoded_prompt = self.tokenizer.decode(prompt.tolist())
                decoded_gen = self.tokenizer.decode(generated.tolist()[prompt_len:])

                f.write("============\n")
                f.write("ORIGINAL SAMPLE:\n")
                f.write("============\n")
                f.write(decoded_full + "\n")
                f.write("============\n")
                f.write("GENERATION:\n")
                f.write("============\n")
                f.write(f"{decoded_prompt} [{decoded_gen}]\n")
                f.write("============\n")
                f.write("++++++++++++\n")

    def sample(self, input_ids):
        input_ids = input_ids.unsqueeze(0)
        while input_ids.shape[1] < self.max_len:
            logits = self.model(input_ids)[:, -1, :] / self.temperature
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > self.top_p

            if cutoff.any():
                cutoff_index = torch.nonzero(cutoff, as_tuple=False)[0, 1] + 1
                probs[0, sorted_indices[0, cutoff_index:]] = 0
                probs /= probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == self.eos_token_id:
                break

        return input_ids.squeeze(0)
