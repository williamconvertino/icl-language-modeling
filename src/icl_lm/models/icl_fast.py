import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lm_base import LMBase
from torchtune.modules import RotaryPositionalEmbeddings

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_k = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_v = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_o = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim // config.n_heads, max_seq_len=config.max_seq_len)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
        causal_mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        
        self.register_buffer("causal_mask", causal_mask)
    
    def forward(self, x):
        
        q = k = v = x
        
        B, S, E = q.shape
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(self.causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.hidden_dim)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=True)
        self.fc_2 = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=True)
        
        self.activation = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.attention = Attention(config)
        self.ln_attn = nn.LayerNorm(config.hidden_dim)
        
        self.mlp = MLP(config)
        self.ln_mlp = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x):
        x = x + self.attention(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x
    
class ICLAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.W_q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_k = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        if config.icl_use_wv:
            self.W_v = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            self.W_o = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        else:
            self.W_o = nn.Linear(config.n_heads * config.hidden_dim, config.hidden_dim, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
        causal_mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=0).bool()
        causal_mask[0, 0] = False
        
        self.register_buffer("causal_mask", causal_mask)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape
        device = q.device
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(self.causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.view(B, S, self.config.hidden_dim)
            
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output

class ICLBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.mlp = MLP(config)
        self.ln_mlp = nn.LayerNorm(config.hidden_dim)
        
        self.attention = ICLAttention(config)
        self.ln_v = nn.LayerNorm(config.hidden_dim)
        self.ln_qk = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, covariates, targets, functional_update):
        
        v = self.ln_v(targets)
        q = k = self.ln_qk(covariates)
            
        functional_update = functional_update + self.attention(q, k, v)
        
        functional_update = functional_update + self.mlp(functional_update)
        
        return covariates, targets, functional_update
        
class ICLFast(LMBase):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        self.w_s = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers // 2)])
        self.icl_blocks = nn.ModuleList([ICLBlock(config) for _ in range(config.n_layers // 2)])
                
        self.ln_out = nn.LayerNorm(config.hidden_dim)
        
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(self.init_weights)
        
    def forward(self, input_tokens):
        
        w = self.embedding(input_tokens)

        B, S, E = w.shape
        device = w.device
        
        w_s = self.w_s.expand(B, -1, -1)
        covariates = torch.cat([w_s, w], dim=1)
        
        w_NP1 = torch.zeros(B, 1, E, device=device)
        targets = torch.cat([w, w_NP1], dim=1)
        
        functional_update = torch.zeros(B, S+1, E, device=device)

        for transformer_block, icl_block in zip(self.transformer_blocks, self.icl_blocks):
            covariates = transformer_block(covariates)
            covariates, targets, functional_update = icl_block(covariates, targets, functional_update)
        
        x = functional_update[:, 1:, :]
        
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        return logits