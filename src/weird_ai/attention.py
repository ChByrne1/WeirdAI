import torch
from torch import nn as nn


class SimpleSelfAttention(nn.Module):
    """
    A simple self-attention module without trainable query, key, and value projections.

    This version is primarily for learning.
    """

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, embedding_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        torch.manual_seed(123)

        inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
            ]
        )

        query_index = 1
        query = inputs[query_index]

        attention_scores = torch.empty(inputs.shape[0])

        for index, token_embedding in enumerate(inputs):
            attention_scores[index] = torch.dot(query, token_embedding)

        attention_weights = torch.softmax(attention_scores, dim=0)

        context_vector = torch.zeros(query.shape)

        for index, token_embedding in enumerate(inputs):
            context_vector += attention_weights[index] * token_embedding

        context_vector_vectorized = attention_weights @ inputs

        all_attention_scores = inputs @ inputs.T

        all_attention_weights = torch.softmax(all_attention_scores, dim=-1)

        all_context_vectors = all_attention_weights @ inputs

        return all_context_vectors, all_attention_weights


class SelfAttention(nn.Module):
    """
    Trainable self-attention using query, key, and value projections.
    """

    def __init__(self, embedding_dim, output_dim, qkv_bias=False):
        super().__init__()
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, output_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values

        return context_vec, attn_weights


class CausalAttention(nn.Module):
    """
    Self-attention with a causal mask so tokens cannot attend to future tokens.
    """

    def __init__(self, embedding_dim, output_dim, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (batch_size, num_tokens, output_dim)
        """

        b, num_tokens, output_dim = x.shape

        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_( self.mask.bool()[ :num_tokens, :num_tokens ], -torch.inf)
        attention_weights = torch.softmax( attention_scores / keys.shape[-1] ** 0.5, dim=-1 )

        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ values

        return context_vectors