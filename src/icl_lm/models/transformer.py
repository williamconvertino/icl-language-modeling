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
        
        self.W_q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_k = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_v = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.W_o = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim // config.n_heads, max_seq_len=config.max_seq_len)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
        self.register_buffer("cached_training_mask", torch.triu(torch.ones(1, 1, config.max_seq_len, config.max_seq_len), diagonal=1).bool(), persistent=False)

    def forward(self, x):
        
        q = k = v = x
        
        B, S, E = q.shape
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        if S == self.config.max_seq_len:
            causal_mask = self.cached_training_mask
        else:
            causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
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

class Transformer(LMBase):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
                
        self.ln_out = nn.LayerNorm(config.hidden_dim)
        
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(self.init_weights)
    
    def forward(self, input_tokens, target_tokens=None, ignore_index=None):
        
        x = self.embedding(input_tokens)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        if target_tokens is not None:
            x = self.ln_out(x)
            logits = self.lm_head(x)
            return self.compute_loss(logits, target_tokens, ignore_index=ignore_index)
        else:
            x = self.ln_out(x[:, [-1], :])
            logits = self.lm_head(x)
            return logits