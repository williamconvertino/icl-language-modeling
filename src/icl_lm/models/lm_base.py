import torch
import torch.nn as nn
import torch.nn.functional as F

class LMBase(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def generate(self, input_ids, max_length, temperature=1.0, top_p=0.9, eos_token_id=None):
        self.eval()

        generated = input_ids.clone()
        
        for _ in range(max_length):
            logits = self(generated, return_loss=False)
            logits = logits[:, -1, :] / temperature

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = 0

            logits[torch.arange(logits.size(0)).unsqueeze(1), sorted_indices] = logits.masked_fill(sorted_mask, float('-inf'))[torch.arange(logits.size(0)).unsqueeze(1), sorted_indices]

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Early stopping if EOS is reached
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)