import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .lm_base import LMBase
from .transformer import Attention, MLP, TransformerBlock

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
        
        self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim // config.n_heads, max_seq_len=config.max_seq_len + 1)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
        # Mask diagonal (ignoring the first to prevent softmax issues) 
        cached_training_mask = torch.triu(torch.ones(1, 1, config.max_seq_len + 1, config.max_seq_len + 1), diagonal=0).bool()
        cached_training_mask[0, 0] = False
        
        self.register_buffer("cached_training_mask", cached_training_mask, persistent=False)

    def forward(self, q, k, v):
        
        B, S, E = q.shape
        device = q.device
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        
        if self.config.icl_use_wv:
            v = self.W_v(v).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        else:
            v = v.unsqueeze(2).expand(B, S, self.config.n_heads, self.config.hidden_dim).transpose(1, 2)
        
        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)
        
        if S == self.config.max_seq_len + 1:
            causal_mask = self.cached_training_mask
        else:
            causal_mask = torch.triu(torch.ones(1, 1, S, S), diagonal=0).bool().to(device)
            causal_mask[0, 0] = False
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop_attn(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        if self.config.icl_use_wv:
            attn_output = attn_output.view(B, S, self.config.hidden_dim)
        else:
            attn_output = attn_output.view(B, S, self.config.n_heads * self.config.hidden_dim)
            
        attn_output = self.W_o(attn_output)
        attn_output = self.drop_resid(attn_output)
        
        return attn_output

class ICLBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.mlp = MLP(config)
        
        if config.icl_use_ln_mlp:
            self.ln_mlp = nn.LayerNorm(config.hidden_dim)
        
        self.attention = ICLAttention(config)
        
        if config.icl_use_ln_v:
            self.ln_v = nn.LayerNorm(config.hidden_dim)
        if config.icl_use_ln_qk:
            self.ln_qk = nn.LayerNorm(config.hidden_dim)
        
    def _calculate_ex(self, functional_update):
        
        if self.config.icl_use_ln_mlp:
            ex_term = self.mlp(self.ln_mlp(functional_update))
        else:
            ex_term = self.mlp(functional_update)
        
        if self.config.icl_use_skip_mlp:
            ex_term += functional_update
        
        return ex_term
        
    def forward(self, covariates, targets, functional_update):
        v = targets - self._calculate_ex(functional_update)
        
        if self.config.icl_use_ln_v:
            v = self.ln_v(v)
        
        if self.config.icl_use_ln_qk:
            q = k = self.ln_qk(covariates)
        else:
            q = k = covariates
            
        functional_update = functional_update + self.attention(q, k, v)
        
        return covariates, targets, functional_update
        
class ICL(LMBase):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        if config.block_order is None:
            config.block_order = ["T", "I"] * (config.n_layers // 2)
            if config.n_layers % 2 != 0:
                config.block_order = ["T"] + config.block_order
        
        self.blocks = nn.ModuleList([TransformerBlock(config) if sym.lower() == "t" else ICLBlock(config) for sym in config.block_order])
                
        self.ln_out = nn.LayerNorm(config.hidden_dim)
        
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        if self.config.share_covariate_attn:
            self.share_covariate_attn()
        if self.config.share_covariate_mlp:
            self.share_covariate_mlp()
        if self.config.share_icl_attn:
            self.share_icl_attn()
        if self.config.share_icl_mlp:
            self.share_icl_mlp()
        
        self.apply(self.init_weights)
    
    def share_covariate_attn(self):
        W_q = None
        W_k = None
        W_v = None
        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() == "t":
                if W_q is None:
                    W_q = block.attention.W_q
                    W_k = block.attention.W_k
                    W_v = block.attention.W_v
                else:
                    block.attention.W_q = W_q
                    block.attention.W_k = W_k
                    block.attention.W_v = W_v
    
    def share_icl_attn(self):
        W_q = None
        W_k = None
        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() != "t":
                if W_q is None:
                    W_q = block.attention.W_q
                    W_k = block.attention.W_k
                else:
                    block.attention.W_q = W_q
                    block.attention.W_k = W_k
                    
    def share_covariate_mlp(self):
        fc_1 = None
        fc_2 = None
        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() == "t":
                if fc_1 is None:
                    fc_1 = block.mlp.fc_1
                    fc_2 = block.mlp.fc_2
                else:
                    block.mlp.fc_1 = fc_1
                    block.mlp.fc_2 = fc_2
    
    def share_icl_mlp(self):
        fc_1 = None
        fc_2 = None
        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() != "t":
                if fc_1 is None:
                    fc_1 = block.mlp.fc_1
                    fc_2 = block.mlp.fc_2
                else:
                    block.mlp.fc_1 = fc_1
                    block.mlp.fc_2 = fc_2
    
    def forward(self, input_tokens, target_tokens=None, inference_mode=True):
        
        device = input_tokens.device
        
        covariates = self.embedding(input_tokens)
        targets = self.embedding(target_tokens)
        
        if inference_mode:
            B, S, E = covariates.shape
            covariates = torch.cat([torch.zeros(B, 1, E, device=device), covariates], dim=1)
            targets = torch.cat([targets, torch.zeros(B, 1, E, device=device)], dim=1)
        
        functional_update = torch.zeros_like(covariates)
        
        for block, sym in zip(self.blocks, self.config.block_order):
            if sym.lower() == "t":
                covariates = block(covariates)
            else:
                covariates, targets, functional_update = block(covariates, targets, functional_update)
        
        x = functional_update
        
        if inference_mode:
            x = x[:, [-1], :]
            
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        return logits