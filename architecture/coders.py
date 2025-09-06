import torch.nn as nn

from  feedforward import FeedForward
from  multihead import MultiHeadedAttention

class Encoder(nn.Module):

    def __init__(self, d_model, num_heads, num_encoders):
        # d_model is the dimension of the vectors / input embeddings
        # num_heads is the number of attention heads
        # num_encoders is the number of encoder layers (Nx in the scheme)

        super().__init__()
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads) for _ in range(num_encoders)])

    def forward(self, src, src_mask):
        # src is the input sequence
        # src_mask is the mask for the input sequence (to ignore padding tokens)

        output = src
        for layer in self.enc_layers:
            output = layer(output, src_mask)
        return output
    

class Decoder(nn.Module):

    def __init__(self, d_model, num_heads, num_decoders):

        super().__init__()
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads) for _ in range(num_decoders)
        ])

    def forward(self, tgt, enc, tgt_mask, enc_mask):
        # tgt is the target sequence (english sentence but the last word is missing)
        # enc is the output of the encoder
        # tgt_mask is the mask for the target sequence
        # enc_mask is the mask for the encoder output 

        output = tgt
        for layer in self.dec_layers:
            output = layer(output, enc, tgt_mask, enc_mask)
        return output
    

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        # d_ff is the dimension of the feedforward network, usually 4 times d_model (4*512=2048)
        # dropout is the dropout rate to prevent overfitting

        super().__init__()
        # Multi-head attention layer
        self.attn = MultiHeadedAttention(d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # Feed-forward network layer
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = src
        x += self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x), mask=src_mask)
        x += self.ffn(self.ffn_norm(x))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        self.masked_attn = MultiHeadedAttention(d_model, num_heads, dropout)
        self.masked_attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadedAttention(d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc, tgt_mask=None, enc_mask=None):
        x = tgt
        x_norm = self.masked_attn_norm(x)
        x += self.masked_attn(x_norm, x_norm, x_norm, mask=tgt_mask)

        x_norm = self.attn_norm(x)
        x += self.attn(x_norm, enc, enc, mask=enc_mask)

        x_norm = self.ffn_norm(x)
        x = x + self.ffn(x_norm)
        
        return x