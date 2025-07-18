import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .lm_base import LMBase

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.W_q = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.W_k = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.W_v = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.W_o = nn.Linear(config.hidden_dim, config.embed_dim, bias=False)

        self.attn_scale = 1 / math.sqrt(config.hidden_dim // config.n_heads)

        self.rotary = RotaryPositionalEmbeddings(
            config.hidden_dim // config.n_heads,
            max_seq_len=config.max_seq_len
        )

        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)

        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, S, _ = x.shape

        q = self.W_q(x).view(B, S, self.config.n_heads, -1).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.config.n_heads, -1).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.config.n_heads, -1).transpose(1, 2)

        q = self.rotary(q)
        k = self.rotary(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        scores = scores.masked_fill(self.causal_mask[:S, :S], float('-inf'))

        probs = self.drop_attn(F.softmax(scores, dim=-1))
        output = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, self.config.hidden_dim)

        return self.drop_resid(self.W_o(output))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.embed_dim, 4 * config.mlp_dim)
        self.fc_2 = nn.Linear(4 * config.mlp_dim, config.embed_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc_2(self.drop(self.act(self.fc_1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.ln_attn = nn.LayerNorm(config.embed_dim)
        self.ln_mlp = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

class TransformerFast(LMBase):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_out = nn.LayerNorm(config.embed_dim)

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.apply(self.init_weights)

    def forward(self, input_tokens):
        x = self.embedding(input_tokens)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.lm_head(x)
