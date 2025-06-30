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
        
        # if config.icl_use_rotary_embedding:
        #     self.rotary_embeddings = RotaryPositionalEmbeddings(config.hidden_dim // config.n_heads, max_seq_len=config.max_seq_len + 1)
        
        self.drop_attn = nn.Dropout(0.1)
        self.drop_resid = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        
        B, S, E = q.shape
        device = q.device
        
        q = self.W_q(q).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        k = self.W_k(k).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        
        if self.config.icl_use_wv:
            v = self.W_v(v).view(B, S, self.config.n_heads, self.config.hidden_dim // self.config.n_heads).transpose(1, 2)
        else:
            v = v.unsqueeze(2).expand(B, S, self.config.n_heads, self.config.hidden_dim).transpose(1, 2)
        
        # if self.config.icl_use_rotary_embedding:
        #     q = self.rotary_embeddings(q)
        #     k = self.rotary_embeddings(k)

        causal_mask = torch.triu(torch.ones(S, S), diagonal=0).bool().to(device)
        causal_mask[0, 0] = False
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        
        attn_scores = torch.ones(attn_scores.size(1), attn_scores.size(2), device=device).repeat(B, self.config.n_heads, 1, 1) # Identity q/k for ease of testing

        # attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_scores = attn_scores.masked_fill(causal_mask, float(0.0))
        
        # attn_probs = F.softmax(attn_scores, dim=-1)
        # attn_probs = self.drop_attn(attn_probs)
        print("++++ Attention Scores ++++")
        print(attn_scores)
        attn_output = torch.matmul(attn_scores, v)
        
        print("++++ Attention Outputs ++++")
        print(attn_output)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        if self.config.icl_use_wv:
            attn_output = attn_output.view(B, S, self.config.hidden_dim)
        else:
            attn_output = attn_output.view(B, S, self.config.n_heads * self.config.hidden_dim)
            
        # attn_output = self.W_o(attn_output)
        # attn_output = self.drop_resid(attn_output)
        
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
        
    def calculate_expectation(self, functional_update):
        return functional_update
        # if self.config.icl_use_ln_mlp:
        #     ex_term = self.mlp(self.ln_mlp(functional_update))
        # else:
        #     ex_term = self.mlp(functional_update)
        
        # if self.config.icl_use_skip_mlp:
        #     ex_term += functional_update
        
        # return ex_term
        
    def forward(self, covariates, targets, functional_update):
        
        v = targets + self.calculate_expectation(functional_update)
        
        if self.config.icl_use_ln_v:
            v = self.ln_v(v)
        
        if self.config.icl_use_ln_qk:
            q = k = self.ln_qk(covariates)
        else:
            q = k = covariates
            
        attn_out = self.attention(q, k, v)
        print("=====Attention Output====")
        print(attn_out)
        

        functional_update = functional_update + attn_out
        
        print("=====Updated Functional Update====")
        print(functional_update)
        
        return covariates, targets, functional_update
        
class EmptyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x

class ICLTest(LMBase):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        self.w_s = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        
        self.block_order = config.block_order
        
        if self.block_order is None:
            self.block_order = ["T", "I"] * (config.n_layers // 2)
            if config.n_layers % 2 != 0:
                self.block_order = ["T"] + self.block_order
        print(self.block_order)
        self.blocks = nn.ModuleList([EmptyBlock(config) if sym.lower() == "t" else ICLBlock(config) for sym in self.block_order])
                
        self.ln_out = nn.LayerNorm(config.hidden_dim)
        
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(self.init_weights)
    
    def forward(self, input_tokens):
        
        w = self.embedding(input_tokens)
        
        print("=====W====")
        print(w)
        print("=====W_s====")
        print(self.w_s)

        B, S, E = w.shape
        device = w.device
        
        w_s = self.w_s.expand(B, -1, -1)
        covariates = torch.cat([w_s, w], dim=1)
        
        w_NP1 = torch.zeros(B, 1, E, device=device) 
        targets = torch.cat([w, w_NP1], dim=1)
        
        functional_update = torch.zeros(B, S+1, E, device=device)
        
        print("=====Covariates====")
        print(covariates)
        print("=====Targets====")
        print(targets)
        print("=====Functional Update====")
        print(functional_update)

        for block, sym in zip(self.blocks, self.block_order):
            if sym.lower() == "t":
                covariates = block(covariates)
            else:
                covariates, targets, functional_update = block(covariates, targets, functional_update)
        
        x = functional_update[:, 1:, :]
        
        print("=====Output====")
        print(x)
        
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        return logits