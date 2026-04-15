# Cross Message-Passing Transformer model from https://arxiv.org/abs/2507.01038
# Most of this code is taken from https://github.com/iil-postech/crossmpt
# and was released for non-commercial research purpose by Seong-Joon Park

# Main changes with respect to the original implementation:
# - use of pytorch's scaled_dot_product_attention
# - disable bias in FFN layers
# - no more intermediate layernorm in the middle

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

    def forward(
        self, x1: Tensor, x2: Tensor, mask_VN: Tensor, mask_CN: Tensor
    ) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            x1 = layer(x1, x2, mask_VN)
            x2 = layer(x2, x1, mask_CN)
        return self.norm(x1), self.norm(x2)


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

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor) -> Tensor:
        x1 = self.sublayer[0](x1, lambda x: self.self_attn(x, x2, x2, mask))
        return self.sublayer[1](x1, self.feed_forward)


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
        # modif RLB: disable bias for both FFN layers
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class CrossMPT(nn.Module):
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

        log.info(f"Building a {n_layers}-layer CrossMPT decoder")
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
        self.src_embed_VN = torch.nn.Parameter(torch.empty((code.n, d_model)))
        self.src_embed_CN = torch.nn.Parameter(torch.empty((code.m, d_model)))
        self.decoder = Encoder(
            EncoderLayer(d_model, c(attn), c(ff), res_dropout), N_dec
        )
        self.oned_final_embed = nn.Sequential(nn.Linear(d_model, 1))
        self.out_fc = nn.Linear(code.n + code.m, output_sz)

        if compile:
            log.info("Compiling model layers for faster training")
            self.decoder.compile()

        # setup masks and other parameters

        self.get_masks(code, use_fast_attn)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.example_input_array = torch.zeros(1, code.n), torch.zeros(
            1, code.m
        )  # for model summary

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        VN = ym.unsqueeze(-1) * self.src_embed_VN.unsqueeze(0)
        CN = s.unsqueeze(-1) * self.src_embed_CN.unsqueeze(0)
        emb1, emb2 = self.decoder(VN, CN, self.src_mask_VN, self.src_mask_CN)
        emb = torch.cat([emb1, emb2], dim=1)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    @staticmethod
    def _enforce_stride1_in_last_dim(x: Tensor) -> Tensor:
        # solution from https://github.com/pytorch/pytorch/issues/127523
        if x.stride(-1) != 1:
            x = torch.empty_like(x, memory_format=torch.contiguous_format).copy_(x)
        return x

    def get_masks(self, code: LinearCode, invert: bool = False) -> None:

        def build_mask(code: LinearCode, invert: bool = False):
            mask = code.H > 0
            # Original code: mask=True for elements that DON'T participate to attention
            # Need to invert for Pytorch SDPA: mask=True for elements that DO participate to attention
            src_mask = mask if invert else ~mask
            return src_mask

        src_mask_CN = build_mask(code, invert=invert)
        src_mask_VN = src_mask_CN.T

        # make sure last dim of each mask has stride=1, as this is what fast attention and
        # memory-efficient attention expect for *all* of their input tensors, mask included
        # see: https://github.com/pytorch/pytorch/issues/116333
        src_mask_CN = CrossMPT._enforce_stride1_in_last_dim(src_mask_CN)
        src_mask_VN = CrossMPT._enforce_stride1_in_last_dim(src_mask_VN)
        assert (
            src_mask_CN.stride(-1) == 1
        ), f"The last dim of src_mask_CN must have stride 1 (got {src_mask_CN.stride(-1)})"
        assert (
            src_mask_VN.stride(-1) == 1
        ), f"The last dim of src_mask_VN must have stride 1 (got {src_mask_VN.stride(-1)})"

        self.register_buffer("src_mask_VN", src_mask_VN)
        self.register_buffer("src_mask_CN", src_mask_CN)


if __name__ == "__main__":
    pass
