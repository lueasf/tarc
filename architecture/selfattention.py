# selfattention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, output_size, dropout=0.1):
        # output_size is the size of the output vectors (d_k, d_v) as seen in the schema

        super().__init__()
        # Layers with d_model sized vector as input and output_size sized vector as output
        self.query = nn.Linear(d_model, output_size) # matrix W_q and bias b_q
        self.key = nn.Linear(d_model, output_size)
        self.value = nn.Linear(d_model, output_size)

        self.dropout = nn.Dropout(dropout) # only during training
    
    def forward(self, query, key, value, mask=None):
        # query : tensor of shape (batch_size, seq_len, d_model)
        # key : tensor of shape (batch_size, seq_len, d_model)
        # value : tensor of shape (batch_size, seq_len, d_model)

        Q = self.query(query) # Q = query * W_q + b_q (W_q . x + b_q)
        K = self.key(key)
        V = self.value(value)

        d_k = K.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k) # bacth matrix multiplication

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 3 and mask.size(1) == 1 and mask.size(2) == scores.size(2):
                pass
            elif mask.dim() == 3 and mask.shape == scores.shape:
                pass
            else:
                raise ValueError(f"Mask shape {mask.shape}")

            if mask.size(1) == 1 and scores.size(1) > 1:
                mask = mask.expand(-1, scores.size(1), -1)

            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.bmm(weights, V)
        return out