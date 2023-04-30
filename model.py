import numpy as np
import torch
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module

# TODO delete unrequired stuff : del
# TODO add non linearities


class TransformerBlock(Module):
    def __init__(self, dims, heads, device="cpu"):
        super(TransformerBlock, self).__init__()

        if dims % heads != 0:
            raise ValueError(
                "Number of dimensions should be divisible by number of heads"
            )
        self.dims = dims
        self.heads = heads
        self.dims_per_head = dims // heads
        self.device = device

        self.Wq = Linear(self.dims, self.dims)
        self.Wk = Linear(self.dims, self.dims)
        self.Wv = Linear(self.dims, self.dims)

        self.Wout = Linear(self.dims, self.dims)

        self.dropout = Dropout(p=0.2)
        # TODO: Understand LayerNorm
        self.layernorm = LayerNorm(self.dims)

    def forward(self, input):
        B, N, _ = input.shape
        V = self.Wv(input)
        V = V.view(B, N, self.heads, self.dims_per_head).transpose(1, 2)

        attention_matrix = self.calc_attention(input)
        QKV = attention_matrix @ V
        QKV = QKV.transpose(1, 2).contiguous().view(B, N, -1)

        # Add and norm
        QKV += self.dropout(input)
        QKV = self.layernorm(QKV)

        # Feedforward
        output = self.Wout(QKV)

        # Add and norm
        output += self.dropout(QKV)
        output = self.layernorm(output)

        return QKV

    def calc_attention(self, input):
        B, N, _ = input.shape
        Q = self.Wq(input)
        K = self.Wk(input)

        Q = Q.view(B, N, self.heads, self.dims_per_head).transpose(1, 2)
        K = K.view(B, N, self.heads, self.dims_per_head).transpose(1, 2)

        # Matmul Q and K.T
        QK = Q @ torch.transpose(K, -2, -1)
        # Scale
        scaled_QK = QK / np.sqrt(self.dims_per_head)
        not_mask = torch.triu(torch.ones(1, 1, N, N), diagonal=1).to(self.device)
        mask = not_mask == 0
        # TODO Understand this mask creation
        scaled_QK_masked = scaled_QK.masked_fill(mask, -1e9)
        attention_matrix = torch.softmax(scaled_QK_masked, dim=-1)
        # attention_matrix = torch.softmax(scaled_QK, dim=-1)
        return attention_matrix


class LanguageModel(Module):
    def __init__(self, dims, heads, nblocks, vocab_size, maxlen, padding_idx, device):
        super(LanguageModel, self).__init__()

        self.dims = dims
        self.heads = heads
        self.nblocks = nblocks
        self.vocab_size = vocab_size
        self.device = device

        # Word embedding layer
        self.word_emb = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.dims,
            padding_idx=padding_idx,
        ).to(self.device)

        # Understand position embedding layer TODO
        # Copied from: https://nlp.seas.harvard.edu/annotated-transformer/
        self.pos_emb = torch.zeros(maxlen, self.dims).to(self.device)
        position = torch.arange(0, maxlen).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dims, 2) * -(np.log(10000.0) / self.dims)
        )
        self.pos_emb[:, 0::2] = torch.sin(position * div_term)
        self.pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.pos_emb = self.pos_emb.unsqueeze(0)
        self.pos_emb.requires_grad = False

        self.transformer_blocks = [
            TransformerBlock(dims=self.dims, heads=self.heads, device=self.device).to(
                self.device
            )
            for n in range(self.nblocks)
        ]

        # Final logits
        self.Wout = Linear(self.dims, self.vocab_size).to(self.device)

        self.dropout = Dropout(p=0.2).to(self.device)

    def forward(self, x):
        # TODO: Take PAD mask into consideration?
        x = self.word_emb(x) + self.pos_emb[:, : x.size(1)]
        x = self.dropout(x)
        for t_block in self.transformer_blocks:
            x = t_block(x)
        x_out = self.Wout(x)
        return x_out
