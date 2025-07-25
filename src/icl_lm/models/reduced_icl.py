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
        
        self.W_q = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_k = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_v = nn.Linear(config.embed_dim_phi, config.hidden_dim_phi, bias=False)
        self.W_o = nn.Linear(config.hidden_dim_phi, config.embed_dim_phi, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim_phi)
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim_phi // config.n_heads, max_seq_len=config.max_seq_len)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
    
    def forward(self, x):
        
        q = k = v = x
        
        B, S, E = q.shape
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.hidden_dim_phi // self.config.n_heads).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.hidden_dim_phi // self.config.n_heads).transpose(1, 2)
        v = self.W_v(v).view(B, S, self.config.n_heads, self.config.hidden_dim_phi // self.config.n_heads).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)

        causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(q.device)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.hidden_dim_phi)
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output
    
class MLPPhi(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.embed_dim_phi, 4 * config.mlp_dim_phi, bias=True)
        self.fc_2 = nn.Linear(4 * config.mlp_dim_phi, config.embed_dim_phi, bias=True)
        
        self.activation = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc_2(x)
        return x
    
class MLPF(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.embed_dim_f, 4 * config.mlp_dim_f, bias=True)
        self.fc_2 = nn.Linear(4 * config.mlp_dim_f, config.embed_dim_f, bias=True)
        
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
        self.ln_attn = nn.LayerNorm(config.embed_dim_phi)
        
        self.mlp = MLPPhi(config)
        self.ln_mlp = nn.LayerNorm(config.embed_dim_phi)
        
    def forward(self, x):
        x = x + self.attention(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

class ICLAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Possible that the W_q and W_k dimensions shouldn't match that of W_v, but I'll leave this for future research
        self.W_q = nn.Linear(config.embed_dim_phi, config.hidden_dim_f, bias=False)
        self.W_k = nn.Linear(config.embed_dim_phi, config.hidden_dim_f, bias=False)
        
        if config.icl_use_wv:
            self.W_v = nn.Linear(config.embed_dim_f, config.hidden_dim_f, bias=False)
            self.W_o = nn.Linear(config.hidden_dim_f, config.embed_dim_f, bias=False)
        elif config.icl_share_wv:
            self.W_v = nn.Linear(config.embed_dim_f, config.hidden_dim_f // config.n_heads, bias=False)
            self.W_o = nn.Linear(config.hidden_dim_f, config.embed_dim_f, bias=False)
        else:
            self.W_o = nn.Linear(config.n_heads * config.embed_dim_f, config.embed_dim_f, bias=False)
        
        self.attn_scale = 1 / math.sqrt(config.hidden_dim_f)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape
        device = q.device
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.hidden_dim_f // self.config.n_heads).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.hidden_dim_f // self.config.n_heads).transpose(1, 2)
        
        if self.config.icl_use_wv:
            v = self.W_v(v).view(B, S, self.config.n_heads, self.config.hidden_dim_f // self.config.n_heads).transpose(1, 2)
        elif self.config.icl_share_wv:
            v = self.W_v(v).unsqueeze(2).expand(B, S, self.config.n_heads, self.config.hidden_dim_f // self.config.n_heads).transpose(1, 2)
        else:
            v = v.unsqueeze(2).expand(B, S, self.config.n_heads, self.config.embed_dim_f).transpose(1, 2)
    
        causal_mask = torch.triu(torch.ones(S, S), diagonal=0).bool().to(device)
        causal_mask[0, 0] = False
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        if self.config.icl_use_wv or self.config.icl_share_wv:
            attn_output = attn_output.view(B, S, self.config.hidden_dim_f)
        else:
            attn_output = attn_output.view(B, S, self.config.n_heads * self.config.embed_dim_f)
            
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output

class ICLBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if self.config.icl_use_mlp:
            self.mlp = MLPF(config)
            self.ln_mlp = nn.LayerNorm(config.embed_dim_f)
        
        self.attention = ICLAttention(config)
        self.ln_v = nn.LayerNorm(config.embed_dim_f)
        self.ln_qk = nn.LayerNorm(config.embed_dim_phi)
        
    def forward(self, covariates, targets, functional_update):
        
        v = self.ln_v(targets)
        q = k = self.ln_qk(covariates)
            
        functional_update = functional_update + self.attention(q, k, v)
        
        if self.config.icl_use_mlp:
            functional_update = functional_update + self.mlp(functional_update)
        
        return covariates, targets, functional_update
        
class ReducedICL(LMBase):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding_phi = nn.Embedding(config.vocab_size, config.embed_dim_phi)
        self.embedding_f = nn.Embedding(config.vocab_size, config.embed_dim_f)
        
        self.x_s = nn.Parameter(torch.randn(1, 1, config.embed_dim_phi))
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers // 2)])
        self.icl_blocks = nn.ModuleList([ICLBlock(config) for _ in range(config.n_layers // 2)])
        
        if self.config.use_output_mlp:
            self.ln_mlp_out = nn.LayerNorm(config.embed_dim_f)
            self.mlp_out = MLPF(config)
                
        self.ln_out = nn.LayerNorm(config.embed_dim_f)
        
        self.lm_head = nn.Linear(config.embed_dim_f, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding_f.weight
        
        self.apply(self.init_weights)
        
    def forward(self, input_tokens):
        
        x = self.embedding_phi(input_tokens)
        y = self.embedding_f(input_tokens)
        
        B, S, E = y.shape
        device = y.device
        
        x_s = self.x_s.expand(B, -1, -1)
        covariates = torch.cat([x_s, x], dim=1)
        
        y_NP1 = torch.zeros(B, 1, E, device=device)
        targets = torch.cat([y, y_NP1], dim=1)
        
        functional_update = torch.zeros(B, S+1, E, device=device)

        for transformer_block, icl_block in zip(self.transformer_blocks, self.icl_blocks):
            covariates = transformer_block(covariates)
            covariates, targets, functional_update = icl_block(covariates, targets, functional_update)
        
        y = functional_update[:, 1:, :]
        
        if self.config.use_output_mlp:
            y = y + self.mlp_out(self.ln_mlp_out(y))
        
        y = self.ln_out(y)
        logits = self.lm_head(y)
        
        return logits