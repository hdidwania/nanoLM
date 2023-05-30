import numpy as np
import torch
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module, ReLU

# TODO delete unrequired stuff : del


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
        self.relu = ReLU()

    def forward(self, x_input, pad_mask):
        B, N, _ = x_input.shape
        V = self.Wv(x_input)
        V = V.view(B, N, self.heads, self.dims_per_head).transpose(1, 2)

        attention_matrix = self.calc_attention_weights(x_input, pad_mask)
        QKV = attention_matrix @ V
        QKV = QKV.transpose(1, 2).contiguous().view(B, N, -1)

        # Add and norm
        attention_sublayer_output = self.layernorm(x_input + self.dropout(QKV))

        # Feedforward
        linear_sublayer_output = self.relu(self.Wout(attention_sublayer_output))

        # Add and norm
        final_output = self.layernorm(
            attention_sublayer_output + self.dropout(linear_sublayer_output)
        )

        return final_output

    def calc_attention_weights(self, x_input, pad_mask):
        B, N, _ = x_input.shape
        Q = self.Wq(x_input)
        K = self.Wk(x_input)

        Q = Q.view(B, N, self.heads, self.dims_per_head).transpose(1, 2)
        K = K.view(B, N, self.heads, self.dims_per_head).transpose(1, 2)

        # Matmul Q and K.T
        QK = Q @ torch.transpose(K, -2, -1)
        # Scale
        scaled_QK = QK / np.sqrt(self.dims_per_head)
        causal_mask = torch.triu(torch.ones(1, 1, N, N), diagonal=1).to(self.device)
        causal_mask = causal_mask == 1
        pad_mask = pad_mask.contiguous().view(B, 1, 1, -1).tile([1, 1, N, 1])
        pad_mask = pad_mask == 1
        final_mask = causal_mask | pad_mask
        scaled_QK_masked = scaled_QK.masked_fill(final_mask, -1e9)
        attention_matrix = torch.softmax(scaled_QK_masked, dim=-1)
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

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dims=self.dims, heads=self.heads, device=self.device
                ).to(self.device)
                for n in range(self.nblocks)
            ]
        )

        # Final logits
        self.Wout = Linear(self.dims, self.vocab_size).to(self.device)

    def forward(self, x, pad_mask):
        x = self.word_emb(x) + self.pos_emb[:, : x.size(1)]
        for t_block in self.transformer_blocks:
            x = t_block(x, pad_mask)
        x_out = self.Wout(x)
        return x_out
