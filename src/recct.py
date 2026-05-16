# A recurrent ECCT decoder model
#
# rECCT derives from the universal transformer idea of Dehghani et al. (2019):
# https://arxiv.org/abs/1807.03819
#
# Transposition of this idea to the ECCT architecture was first proposed in
# Gaston de Boni Rovella's PhD thesis: https://theses.fr/2024ESAE0065, Chap.3
# See also: https://github.com/gastondeboni/Syndrome_Based_Neural_Decoding
#
# There has been renewed interest recently in recurrent transformers as a
# parameter-efficient architecture (see papers on looped transformers).
#
# The following is my minimal implementation of a recurrent ECCT, built on a
# more up-to-date and readable TF architecture than the original ECCT model


import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor, BoolTensor
from .codes import LinearCode
from .decoder import BaseDecoder
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class GeGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class SwiGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        dim: int,
        expand: float = 4,
        act: nn.Module | None = None,
        dropout: float = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        act = act if act is not None else nn.GELU()
        hidden_dim = int(expand * dim)
        if isinstance(act, GeGLU) or isinstance(act, SwiGLU):
            self.up = nn.Linear(dim, 2 * hidden_dim, bias=bias)  # packed projection
        else:
            self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = act
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.up(x))
        x = self.drop(x)  # optional dropout on the FFN inner activations
        return self.down(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        # Original ECCT approach (assume one embedding vector per dim of x)
        x = torch.cat([ym, s], dim=1)
        return self.embed.weight.unsqueeze(0) * x.unsqueeze(-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        assert (
            input_dim % num_heads == 0
        ), f"input dim ({input_dim}) must be divisible by number of heads ({num_heads})"
        self.head_dim = input_dim // num_heads
        self.attn_dropout = attn_dropout

        # Packed QKV and output projections
        self.W_qkv = nn.Linear(input_dim, 3 * input_dim, bias=bias)
        self.W_out = nn.Linear(input_dim, input_dim, bias=bias)

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:

        # batch size (B), sequence length (S), input dim (D), heads (H), head dim (DH)
        B, S, D = x.shape
        H, DH = self.num_heads, self.head_dim

        # Project input and unpack to obtain Q, K, V
        qkv = self.W_qkv(x)  # [B, S, 3 * D] with D = H * DH
        qkv = qkv.reshape(B, S, 3, H, DH)  # [B, S, 3, H, DH]
        qkv = qkv.transpose(1, 3)  # [B, H, 3, S, DH]
        q, k, v = qkv.unbind(dim=2)  # [B, H, S, DH] each

        # SDPA using PyTorch's scaled_dot_product_attention
        # Output shape after self-attn: [B, H, S, DH]
        # According to pytorch's documentation, dropout needs to be manually disabled in eval mode
        dropout_p = self.attn_dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p
        )

        # Re-assemble heads outputs side-by-side: [B, H, S, DH] -> [B, S, H * DH]
        output = output.transpose(1, 2).reshape(B, S, D)

        # Apply output projection: [B, S, H * DH] -> [B, S, D]
        return self.W_out(output)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        ffn_expand_factor: float = 4,
        ffn_dropout: float = 0,
        res_dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.pre_norm1 = nn.LayerNorm(embed_dim, bias=bias)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_dropout, bias=bias
        )
        self.drop = nn.Dropout(res_dropout) if res_dropout > 0 else nn.Identity()
        self.pre_norm2 = nn.LayerNorm(embed_dim, bias=bias)
        self.ffn = FeedForwardNetwork(
            embed_dim, ffn_expand_factor, act=GeGLU(), dropout=ffn_dropout, bias=bias
        )

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        xn = self.pre_norm1(x)
        x = x + self.drop(self.attn(xn, mask))
        xn = self.pre_norm2(x)
        x = x + self.drop(self.ffn(xn))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, embed_dim: int, fc_in: int, fc_out: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.post_norm = nn.LayerNorm(embed_dim, bias=bias)
        self.squeeze_emb = nn.Linear(embed_dim, 1, bias=bias)
        self.to_logits = nn.Linear(fc_in, fc_out)  # keep bias for final FC layer

    def forward(self, x: Tensor) -> Tensor:
        # normalization layer, followed by embedding dimension reduction (d->1)
        # and final down projection from context size m+n to output size n
        x = self.post_norm(x)
        x = self.squeeze_emb(x).squeeze(-1)
        return self.to_logits(x)


class RECCT(BaseDecoder):
    def __init__(
        self,
        code: LinearCode,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        n_iters: int = 6,
        attn_dropout: float = 0.1,
        ffn_expand_factor: float = 4,
        ffn_dropout: float = 0.0,
        res_dropout: float = 0.1,
        bias: bool = False,
        compile: bool = False,
        error_space: str = "codeword",
    ) -> None:
        super().__init__(code, error_space=error_space, compile=compile)

        log.info(f"Building a {n_layers}-layer recurrent ECCT decoder")
        log.info(f"The rECCT decoder iterates {n_iters} times over itself internally")
        log.info(f"Embedding dimension = {embed_dim}")
        log.info(
            f"Self-attention uses {n_heads} heads of dimension {embed_dim // n_heads}"
        )
        log.info("Self-attention uses PyTorch's F.scaled_dot_product_attention")
        if (embed_dim // n_heads) % 8 != 0:
            log.warning(
                "Fast CUDA fused kernels for SDPA require head dim to be a multiple of 8"
            )
        log.info(f"FFN expansion factor = {ffn_expand_factor:.1f}")

        self.n_iters = n_iters
        self.n_layers = n_layers

        # build attn mask from PCM
        self.register_mask(code)

        # build the different layers of the rECCT decoder
        self.embed = EmbeddingLayer(code.n + code.m, embed_dim)
        self.encoding_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim,
                    n_heads,
                    attn_dropout,
                    ffn_expand_factor,
                    ffn_dropout,
                    res_dropout,
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.decode = DecoderLayer(
            embed_dim, code.n + code.m, self.output_sz, bias=bias
        )

        # initialize parameters
        self.apply(self._init_weights)

        # call last (compiles the forward graph once all submodules/buffers exist)
        self._maybe_compile()

    def register_mask(self, code: LinearCode) -> None:

        mask_size = code.n + code.m
        #  tokens do not attend to themselves (all-zero diagonal in the mask)
        mask = torch.zeros(mask_size, mask_size)
        # code bits tokens and syndrome tokens can attend to each other within parity-check equations
        for row in range(code.m):
            cols = torch.where(code.H[row] > 0)[0]
            for p, v in enumerate(cols):
                mask[code.n + row, v] = 1  # H in the bottom left block
                mask[v, code.n + row] = 1  # H.T in the upper right block
                for vp in cols[p + 1 :]:
                    mask[v, vp] = 1  # var-to-var in the upper left block
                    mask[vp, v] = 1  # (symmetric around the diagonal)
        # Pytorch SDPA requires mask=True for elements that DO participate to attention
        mask = mask.bool()

        # make sure last dim of each mask has stride=1, as this is what fast attention and
        # memory-efficient attention expect for *all* of their input tensors, mask included
        # see: https://github.com/pytorch/pytorch/issues/116333

        def _enforce_stride1_in_last_dim(x: Tensor) -> Tensor:
            # solution from https://github.com/pytorch/pytorch/issues/127523
            if x.stride(-1) != 1:
                x = torch.empty_like(x, memory_format=torch.contiguous_format).copy_(x)
            return x

        mask = _enforce_stride1_in_last_dim(mask)

        assert (
            mask.stride(-1) == 1
        ), f"The last dim of mask must have stride 1 (got {mask.stride(-1)})"

        self.register_buffer("mask", mask)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:

        # generate the embedding vectors from the input
        x = self.embed(ym, s)

        # iterate over the base encoder layers to refine the latent representation of the error pattern
        for layer in self.encoding_layers:
            for _ in range(self.n_iters):
                x = layer(x, self.mask)

        # decode the result to obtain the error pattern
        return self.decode(x)


if __name__ == "__main__":
    pass
