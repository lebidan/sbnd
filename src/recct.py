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
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)
from torch import Tensor
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)

# compile once at module level; called everywhere through this handle
_flex_attention = torch.compile(flex_attention)


class GeGLU(nn.Module):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.scaling = torch.nn.Parameter(torch.ones(dim)) if dim is not None else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate * self.scaling) * x


class SwiGLU(nn.Module):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.scaling = torch.nn.Parameter(torch.ones(dim)) if dim is not None else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return torch.sigmoid(gate * self.scaling) * x  # = SiLU(x) if scaling = 1


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        dim: int,
        expand: float = 4,
        act: nn.Module = nn.GELU(),
        dropout: float = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
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

    def forward(
        self,
        x: Tensor,
        block_mask: BlockMask | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:

        # batch size (B), sequence length (S), input dim (D), heads (H), head dim (DH)
        B, S, D = x.shape
        H, DH = self.num_heads, self.head_dim

        # Project input and unpack to obtain Q, K, V
        qkv = self.W_qkv(x)  # [B, S, 3 * D] with D = H * DH
        qkv = qkv.reshape(B, S, 3, H, DH)  # [B, S, 3, H, DH]
        qkv = qkv.transpose(1, 3)  # [B, H, 3, S, DH]
        q, k, v = qkv.unbind(dim=2)  # [B, H, S, DH] each

        if x.is_cuda and DH >= 16:
            # Attention dropout via score_mod: Bernoulli mask on the pre-softmax scores.
            # Note: FlexAttention has no post-softmax dropout equivalent to SDPA's dropout_p;
            # the surviving scores are simply renormalized by softmax (no 1/(1-p) rescaling).
            score_mod = None
            if self.training and self.attn_dropout > 0:
                keep = torch.rand(B, H, S, S, device=x.device) >= self.attn_dropout

                def score_mod(score, b, h, q_idx, kv_idx):  # type: ignore[misc]
                    return torch.where(keep[b, h, q_idx, kv_idx], score, float("-inf"))

            # FlexAttention (CUDA only, head_dim >= 16): [B, H, S, DH]
            output = _flex_attention(
                q, k, v, score_mod=score_mod, block_mask=block_mask
            )
            assert isinstance(output, Tensor)  # narrow Union return type for mypy
        else:
            # SDPA fallback: CPU, or head_dim < 16 (FlexAttention/Inductor requires head_dim >= 16).
            # Uses the raw bool attention mask (True = participates) rather than the BlockMask.
            p = self.attn_dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=p
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

    def forward(
        self,
        x: Tensor,
        block_mask: BlockMask | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        xn = self.pre_norm1(x)
        x = x + self.drop(self.attn(xn, block_mask, mask))
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
        # and final down projection from context size 2n-k to output size n
        x = self.post_norm(x)
        x = self.squeeze_emb(x).squeeze(-1)
        return self.to_logits(x)


class RECCT(nn.Module):
    def __init__(
        self,
        code: LinearCode,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        n_iters: int = 6,
        attn_dropout: float = 0.1,
        ffn_expand_factor: float = 4,
        ffn_dropout: float = 0,
        res_dropout: float = 0.1,
        bias: bool = False,
        compile: bool = False,
        output: str = "codeword",
    ) -> None:
        super().__init__()

        log.info(f"Building a {n_layers}-layer recurrent ECCT decoder")
        log.info(f"The rECCT decoder iterates {n_iters} times over itself internally")
        log.info(f"Embedding dimension = {embed_dim}")
        log.info(
            f"Self-attention uses {n_heads} heads of dimension {embed_dim // n_heads}"
        )
        log.info("Self-attention uses PyTorch's FlexAttention")
        if (embed_dim // n_heads) % 8 != 0:
            log.warning(
                "Fast CUDA fused attention kernels require head dim to be a multiple of 8"
            )
        log.info(f"FFN expansion factor = {ffn_expand_factor:.1f}")

        self.n_iters = n_iters
        self.n_layers = n_layers

        # build attn mask from PCM (registered as a buffer so Lightning moves it to GPU)
        self.register_buffer("mask", self._build_mask(code))
        # BlockMask is not a Tensor, so it can't be a buffer; built lazily at first
        # forward, by which time self.mask is already on the correct device.
        self._block_mask: BlockMask | None = None

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
        output_size = code.k if output == "message" else code.n
        self.decode = DecoderLayer(embed_dim, code.n + code.m, output_size, bias=bias)

        if compile:
            log.info("Compiling model forward for faster training")
            self.compile()

        # initialize parameters
        self.apply(self._init_weights)

        # no example_input_array here: Lightning's ModelSummary calls forward through
        # hooks in a code path that prevents torch.compile from tracing into the
        # flex_attention call; the eager fallback then fails on mask_mod indexing a
        # closure-captured buffer. Dropping example_input_array disables per-layer
        # shape info in the summary but keeps param counts.
        self.example_input_array = None

    def _build_mask(self, code: LinearCode):
        mask_size = code.n + code.m
        mask = torch.zeros(mask_size, mask_size)  # token do not attend to themselves
        for ii in range(code.m):
            idx = torch.where(code.H[ii] > 0)[0]
            for jj in idx:
                for kk in idx:
                    if jj != kk:
                        mask[jj, kk] += 1
                        mask[kk, jj] += 1
                        mask[code.n + ii, jj] += 1
                        mask[jj, code.n + ii] += 1
        return (
            mask > 0
        )  # Pytorch SDPA convention: mask=True for elements that DO participate to attention

    @torch.compiler.disable  # keep block_mask construction out of the compiled graph
    def _get_block_mask(self) -> BlockMask | None:
        # FlexAttention is CUDA-only; skip block_mask creation on CPU (e.g. during
        # Lightning's model summary, which traces the forward before .to(device)).
        mask: Tensor = self.mask  # type: ignore[assignment]
        if mask.device.type != "cuda":
            return None
        if self._block_mask is None:
            S = mask.size(0)

            def mask_mod(b, h, q_idx, kv_idx):
                return mask[q_idx, kv_idx]

            self._block_mask = create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=S,
                KV_LEN=S,
                device=mask.device,
                _compile=True,
            )
        return self._block_mask

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:

        # lazy-build the FlexAttention BlockMask on first call (right device via buffer)
        block_mask = self._get_block_mask()
        mask: Tensor = self.mask  # type: ignore[assignment]

        # generate the embedding vectors from the input
        x = self.embed(ym, s)

        # iterate over the base encoder layers to refine the latent representation of the error pattern
        for layer in self.encoding_layers:
            for _ in range(self.n_iters):
                x = layer(x, block_mask, mask)

        # decode the result to obtain the error pattern
        return self.decode(x)

    # BlockMask holds a closure (mask_mod) that cannot be pickled. Lightning pickles
    # the decoder into the checkpoint via self.hparams.decoder, so exclude the cached
    # _block_mask from the pickled state (it will be rebuilt lazily on next forward).
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_block_mask"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._block_mask = None


if __name__ == "__main__":
    pass
