from __future__ import annotations

import torch
from torch import Tensor, nn


class SemanticDecoder(nn.Module):
    def __init__(self, *, vocab_size: int, dim: int, layers: int, heads: int,
                 ffn_dim: int, max_tokens: int, dropout: float, pad_token_id: int) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_tokens = max_tokens
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_tokens, dim)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(dim))
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def forward(self, token_ids: Tensor, memory: Tensor, memory_padding_mask: Tensor
                ) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if token_ids.shape[1] > self.max_tokens:
            raise ValueError("semantic sequence exceeds configured maximum")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                token_ids.shape[1], token_ids.shape[1], dtype=torch.bool,
                device=token_ids.device
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            hidden,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=token_ids.eq(self.pad_token_id),
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output_projection(decoded)

    @torch.inference_mode()
    def generate(self, memory: Tensor, memory_padding_mask: Tensor, *, bos_token_id: int,
                 eos_token_id: int) -> Tensor:
        tokens = torch.full(
            (memory.shape[0], 1), bos_token_id, dtype=torch.long, device=memory.device
        )
        finished = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
        for _ in range(self.max_tokens - 1):
            next_token = self(tokens, memory, memory_padding_mask)[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, eos_token_id, next_token)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            finished |= next_token.eq(eos_token_id)
            if bool(finished.all()):
                break
        return tokens
