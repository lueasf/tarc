import torch
import torch.nn as nn

from  selfattention import SelfAttention

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):

        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_output_size = self.d_model // self.num_heads

        self.attention = nn.ModuleList([
            SelfAttention(d_model, self.attn_output_size) for _ in range(num_heads)])

        # matrix that takes a vector of size d_model and transforms it to a vector of size d_model
        self.output = nn.Linear(self.d_model, self.d_model)

    def forward(self, query, key, value, mask=None):
        x = torch.cat([
            layer(query, key, value, mask) for layer in self.attention
        ], dim=-1)

        # shuffle the outputs of the different heads together
        x = self.output(x)
        return x