import torch

from transformer import Transformer

model = Transformer(d_model=64, num_heads=8, num_encoders=2, num_decoders=2,
                    src_vocab_size=100, tgt_vocab_size=100, max_len=50)

# Batch size = 2 sentences, Source length = 7, Target length = 6
B,S,T = 2, 7, 6
src = torch.randint(3, 100, (B,S)) # random integers in [3, 99] for token ids
tgt = torch.randint(3, 100, (B,T))
logits = model(src, tgt)
print(logits.shape)  # torch.Size([2, 6, 100])
