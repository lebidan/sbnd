# Error-Correction Code Transformer model from https://arxiv.org/abs/2206.14881
# Most of this code is taken from https://github.com/yoniLc/ECCT
# and was released under MIT license. Copyright (c) 2022 yoniLc

# Main changes with respect to the original implementation:
# - use of pytorch's scaled_dot_product_attention
# - mask modified to prevent tokens to attend to themselves
# - disable bias in FFN layers
# - no more intermediate layernorm in the middle
# - syndromes in binary rather than bipolar form

import copy, math
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Callable
from torch import Tensor
from torch.nn import LayerNorm
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer: "EncoderLayer", N: int) -> None:
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: "MultiHeadedAttention",
        feed_forward: "PositionwiseFeedForward",
        dropout: float,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(
        self, h: int, d_model: int, dropout: float = 0.1, use_fast_attn: bool = False
    ) -> None:
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.use_fast_attn = use_fast_attn

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        if self.use_fast_attn:
            # MODIF RLB: Use pytorch fast SDPA implementation
            # According to pytorch's documentation, dropout needs to be manually disabled in eval mode
            p = self.dropout.p if self.training else 0.0
            x = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=p
            )
        else:
            x = self.attention(
                query, key, value, mask
            )  # defaults back to original implementation

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    # original implementation of SDPA operation, from S. Rush blog post "the annotated transformer"
    def attention(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -torch.inf)
        p_attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(p_attn, value)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(
            d_model, d_ff, bias=False
        )  # modif RLB: disable bias for both FFN layers
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class ECCT(nn.Module):
    def __init__(
        self,
        code: LinearCode,
        n_layers: int = 6,
        embed_dim: int = 32,
        n_heads: int = 8,
        res_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0,
        up_proj: int = 4,
        use_fast_attn: bool = True,
        compile: bool = False,
        output: str = "codeword",
    ) -> None:
        super().__init__()

        N_dec, d_model, h = n_layers, embed_dim, n_heads
        output_sz = code.k if output == "message" else code.n

        log.info(f"Building a {n_layers}-layer ECCT decoder")
        log.info(f"Embedding dimension = {embed_dim}")
        log.info(
            f"Self-attention uses {n_heads} heads of dimension {embed_dim // n_heads}"
        )
        if use_fast_attn:
            log.info("Self-attention uses PyTorch's F.scaled_dot_product_attention")
        else:
            log.info("Self-attention uses eager SDPA implementation")
        log.info(f"FFN expansion factor = {up_proj}")

        # build model layers
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, attn_dropout, use_fast_attn)
        ff = PositionwiseFeedForward(d_model, d_model * up_proj, ffn_dropout)
        self.src_embed = torch.nn.Parameter(torch.empty((code.n + code.m, d_model)))
        self.decoder = Encoder(
            EncoderLayer(d_model, c(attn), c(ff), res_dropout), N_dec
        )
        self.oned_final_embed = nn.Sequential(nn.Linear(d_model, 1))
        self.out_fc = nn.Linear(code.n + code.m, output_sz)

        if compile:
            log.info("Compiling model forward for faster training")
            self.compile()

        # setup mask and other parameters

        self.get_mask(code, use_fast_attn)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.example_input_array = torch.zeros(1, code.n), torch.zeros(
            1, code.m
        )  # for model summary

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        x = torch.cat([ym, (1 - s) / 2], dim=1)  # (|y|, s) with s in binary format
        emb = x.unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        emb = self.decoder(emb, self.src_mask)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def get_mask(self, code: LinearCode, invert: bool = False) -> None:

        def build_mask(code: LinearCode, invert: bool = False) -> Tensor:
            mask_size = code.n + code.m
            # MODIF RLB: modify mask to prevent tokens from attending to themselves
            mask = torch.zeros(
                mask_size, mask_size
            )  # torch.eye(mask_size, mask_size) in the original implementation
            for ii in range(code.m):
                idx = torch.where(code.H[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            # mask=True for elements that do NOT participate to attention
            # we need to invert the mask for Pytorch SDPA (opposite convention)
            src_mask = ~(mask > 0)
            return ~src_mask if invert else src_mask

        src_mask = build_mask(code, invert=invert)
        self.register_buffer("src_mask", src_mask)


if __name__ == "__main__":
    pass
