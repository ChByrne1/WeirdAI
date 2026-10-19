import torch
from torch import nn as nn
from src.weird_ai.feed_forward import FeedForward
from src.weird_ai.layer_norm import LayerNorm
from src.weird_ai.attention import CausalAttention


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, num_heads, dropout, qkv_bias):
        super().__init__()
        self.att = CausalAttention(
            embedding_dim=emb_dim,
            output_dim=emb_dim,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias)
        self.ff = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x