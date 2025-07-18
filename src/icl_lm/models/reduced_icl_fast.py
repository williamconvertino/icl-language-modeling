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

        self.W_q = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_k = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_v = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_o = nn.Linear(config.hidden_dim_phi, config.embed_dim_phi, bias=False)

        self.attn_scale = 1 / math.sqrt(config.hidden_dim_phi)
        self.rotary = RotaryPositionalEmbeddings(config.hidden_dim_phi // config.n_heads, max_seq_len=config.max_seq_len)
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
        output = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, self.config.hidden_dim_phi)

        return self.drop_resid(self.W_o(output))

class MLPPhi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.embed_dim_phi, 4 * config.mlp_dim_phi)
        self.fc_2 = nn.Linear(4 * config.mlp_dim_phi, config.embed_dim_phi)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc_2(self.drop(self.act(self.fc_1(x))))

class MLPF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.embed_dim_f, 4 * config.mlp_dim_f)
        self.fc_2 = nn.Linear(4 * config.mlp_dim_f, config.embed_dim_f)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc_2(self.drop(self.act(self.fc_1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLPPhi(config)
        self.ln_attn = nn.LayerNorm(config.embed_dim_phi)
        self.ln_mlp = nn.LayerNorm(config.embed_dim_phi)

    def forward(self, x):
        x = x + self.attn(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

class ICLAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_q = nn.Linear(config.embed_dim_phi, config.hidden_dim_f, bias=False)
        self.W_k = nn.Linear(config.embed_dim_phi, config.hidden_dim_f, bias=False)
        self.W_v = nn.Linear(config.embed_dim_f, config.hidden_dim_f, bias=False)
        self.W_o = nn.Linear(config.hidden_dim_f, config.embed_dim_f, bias=False)

        self.attn_scale = 1 / math.sqrt(config.hidden_dim_f)
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)

        mask = torch.triu(torch.ones(config.max_seq_len + 1, config.max_seq_len + 1), diagonal=0).bool()
        mask[0, 0] = False
        self.register_buffer("causal_mask", mask)

    def forward(self, q, k, v):
        B, S, _ = q.shape

        q = self.W_q(q).view(B, S, self.config.n_heads, -1).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, -1).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        scores = scores.masked_fill(self.causal_mask[:S, :S], float('-inf'))

        probs = self.drop_attn(F.softmax(scores, dim=-1))
        output = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, self.config.hidden_dim_f)

        return self.drop_resid(self.W_o(output))

class ICLBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = ICLAttention(config)
        self.mlp = MLPF(config)
        self.ln_qk = nn.LayerNorm(config.embed_dim_phi)
        self.ln_v = nn.LayerNorm(config.embed_dim_f)

    def forward(self, covariates, targets, f_update):
        q = k = self.ln_qk(covariates)
        v = self.ln_v(targets)

        f_update = f_update + self.attn(q, k, v)
        f_update = f_update + self.mlp(f_update)
        return covariates, targets, f_update

class ReducedICLFast(LMBase):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_phi = nn.Embedding(config.vocab_size, config.embed_dim_phi)
        self.embed_f = nn.Embedding(config.vocab_size, config.embed_dim_f)
        self.x_s = nn.Parameter(torch.randn(1, 1, config.embed_dim_phi))

        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers // 2)])
        self.icl_blocks = nn.ModuleList([ICLBlock(config) for _ in range(config.n_layers // 2)])

        self.mlp_out = MLPF(config)
        self.ln_mlp_out = nn.LayerNorm(config.embed_dim_f)
        self.ln_out = nn.LayerNorm(config.embed_dim_f)

        self.lm_head = nn.Linear(config.embed_dim_f, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_f.weight

        self.apply(self.init_weights)

    def forward(self, input_tokens):
        x = self.embed_phi(input_tokens)
        y = self.embed_f(input_tokens)

        B, S, E = y.shape
        device = y.device

        x_s = self.x_s.expand(B, 1, -1)
        covariates = torch.cat([x_s, x], dim=1)

        y_NP1 = torch.zeros(B, 1, E, device=device)
        targets = torch.cat([y, y_NP1], dim=1)

        f_update = torch.zeros(B, S + 1, E, device=device)

        for t_block, icl_block in zip(self.transformer_blocks, self.icl_blocks):
            covariates = t_block(covariates)
            covariates, targets, f_update = icl_block(covariates, targets, f_update)

        y_out = f_update[:, 1:, :]
        y_out = y_out + self.mlp_out(self.ln_mlp_out(y_out))
        y_out = self.ln_out(y_out)

        return self.lm_head(y_out)
