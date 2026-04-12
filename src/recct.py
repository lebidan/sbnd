# A recurrent ECCT decoder model
#
# rECCT derives from the universal transformer idea of Dehghani et al. (2019): 
# https://arxiv.org/abs/1807.03819 
#
# Transposition of this idea to the ECCT architecture was first proposed in 
# Gaston de Boni Rovella's PhD thesis: https://theses.fr/2024ESAE0065, Chap.3
# See also: https://github.com/gastondeboni/Syndrome_Based_Neural_Decoding 
#
# The following is my minimal implementation of a recurrent ECCT, built on a 
# more up-to-date and readable TF architecture than the original ECCT model


import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import Tensor, BoolTensor
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class GeGLU(nn.Module):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.scaling = torch.nn.Parameter(torch.ones(dim)) if dim is not None else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu( gate * self.scaling ) * x


class SwiGLU(nn.Module):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.scaling = torch.nn.Parameter(torch.ones(dim)) if dim is not None else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.sigmoid( gate * self.scaling ) * x     # = SiLU(x) if scaling = 1


class FeedForwardNetwork(nn.Module):
    def __init__(self, 
                 dim: int, 
                 expand: float = 4,
                 act: nn.Module = nn.GELU(),
                 dropout: float = 0, 
                 bias: bool = False
                 ) -> None:
        super().__init__()
        hidden_dim = int(expand * dim)
        if isinstance(act, GeGLU) or isinstance(act, SwiGLU):
            self.up = nn.Linear(dim, 2 * hidden_dim, bias=bias) # packed projection
        else:
            self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = act
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act( self.up(x) )
        x = self.drop(x)    # optional dropout on the FFN inner activations
        return self.down(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        # Original implementation (assume one embedding vector per dim of x)
        x = torch.cat([ym, s], dim=1)
        return self.embed.weight.unsqueeze(0) * x.unsqueeze(-1)

        # Alternative implementation, with 2 embed (one for mag, another for synd)
        # emb_ym = self.embed.weight[0].unsqueeze(0) * ym.unsqueeze(-1)
        # emb_s  = self.embed.weight[1].unsqueeze(0) * s.unsqueeze(-1)
        # return torch.cat([emb_ym, emb_s], dim=1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int, 
                 attn_dropout: float = 0.0, 
                 bias: bool = False
                 ) -> None:
        super().__init__()

        self.num_heads = num_heads
        assert input_dim % num_heads == 0, \
            f"input dim ({input_dim}) must be divisible by number of heads ({num_heads})" 
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
        qkv = self.W_qkv(x)                 # [B, S, 3 * D] with D = H * DH
        qkv = qkv.reshape(B, S, 3, H, DH)   # [B, S, 3, H, DH]
        qkv = qkv.transpose(1, 3)           # [B, H, 3, S, DH]
        q, k, v = qkv.unbind(dim=2)         # [B, H, S, DH] each
        
        # SDPA using PyTorch's scaled_dot_product_attention
        # Output shape after self-attn: [B, H, S, DH]
        # According to pytorch's documentation, dropout needs to be manually disabled in eval mode
        dropout_p = self.attn_dropout if self.training else 0.0
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

        # Re-assemble heads outputs side-by-side: [B, H, S, DH] -> [B, S, H * DH]
        output = output.transpose(1, 2).reshape(B, S, D)
        
        # Apply output projection: [B, S, H * DH] -> [B, S, D]
        return self.W_out(output)


class EncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim: int = 64, 
                 num_heads: int = 4,
                 attn_dropout: float = 0.1,
                 ffn_expand_factor: float = 4,
                 ffn_dropout: float = 0,
                 res_dropout: float = 0.1,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        self.pre_norm1 = nn.LayerNorm(embed_dim, bias=bias)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout, bias=bias)
        self.drop = nn.Dropout(res_dropout) if res_dropout > 0 else nn.Identity()
        self.pre_norm2 = nn.LayerNorm(embed_dim, bias=bias)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_expand_factor, act=GeGLU(), dropout=ffn_dropout, bias=bias)

    def forward(self, x: Tensor, mask: BoolTensor | None = None, layer_idx: int = 1) -> Tensor:
        xn = self.pre_norm1(x)
        x = x + self.drop( self.attn(xn, mask) )
        xn = self.pre_norm2(x)
        x = x + self.drop( self.ffn(xn) )
        return x
    
    
class DecoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 fc_in: int, 
                 fc_out: int, 
                 bias: bool = False
                 ) -> None:
        super().__init__()
        self.post_norm = nn.LayerNorm(embed_dim, bias=bias)
        self.squeeze_emb = nn.Linear(embed_dim, 1, bias=bias)
        self.to_logits = nn.Linear(fc_in, fc_out)   # keep bias for final FC layer

    def forward(self, x: Tensor) -> Tensor:
        # normalization layer, followed by embedding dimension reduction (d->1)
        # and final down projection from context size 2n-k to output size n
        x = self.post_norm(x)
        x = self.squeeze_emb(x).squeeze(-1)
        return self.to_logits(x)


class RECCT(nn.Module):
    def __init__(self, 
                 code: LinearCode, 
                 embed_dim: int = 64, 
                 num_heads: int = 4, 
                 num_layers: int = 6, 
                 attn_dropout: float = 0.1,
                 ffn_expand_factor: float = 4,
                 ffn_dropout: float = 0,
                 res_dropout: float = 0.1, 
                 bias: bool = False,
                 compile: bool = False,
                 output: str = "codeword"
                 ) -> None:
        super().__init__()

        log.info(f"Building a {num_layers}-layer recurrent ECCT decoder")
        log.info(f"Embedding dimension = {embed_dim}")
        log.info(f"Self-attention uses {num_heads} heads of dimension {embed_dim // num_heads}")
        log.info("Self-attention uses PyTorch's F.scaled_dot_product_attention")
        assert ((embed_dim // num_heads) % 8 ) == 0, \
            "CUDA fast SDPA fused kernels require head dimension to be a multiple of 8"  
        log.info(f"FFN expansion factor = {ffn_expand_factor:.1f}")

        self.num_layers = num_layers
        
        # build attn mask from PCM
        self.register_buffer('mask', self._build_mask(code))
        
        # build the different layers of the rECCT decoder
        # self.embed = EmbeddingLayer(2, embed_dim)
        self.embed = EmbeddingLayer(code.n + code.m, embed_dim)
        self.encode = EncoderLayer(embed_dim, num_heads, attn_dropout, \
            ffn_expand_factor, ffn_dropout, res_dropout, bias=bias)
        output_size = code.k if output == "message" else code.n
        self.decode = DecoderLayer(embed_dim, code.n + code.m, output_size, bias=bias)
       
        if compile:
            log.info("Compiling model layers for faster training")
            self.embed.compile()
            self.encode.compile()
            self.decode.compile()
        
        # initialize parameters
        self.apply( self._init_weights )        
        
        # for model summary
        self.example_input_array = torch.zeros(1, code.n), torch.zeros(1, code.m) # for model summary


    def _build_mask(self, code: LinearCode):
        mask_size = code.n + code.m
        mask = torch.zeros(mask_size, mask_size)    # token do not attend to themselves
        for ii in range(code.m):
            idx = torch.where(code.H[ii] > 0)[0]
            for jj in idx:
                for kk in idx:
                    if jj != kk:
                        mask[jj, kk] += 1
                        mask[kk, jj] += 1
                        mask[code.n + ii, jj] += 1
                        mask[jj, code.n + ii] += 1
        return mask > 0     # Pytorch SDPA convention: mask=True for elements that DO participate to attention


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
        
        # iterate over the base encoder layer to refine the latent representation of the error pattern
        for layer_idx in range(self.num_layers):
            x = self.encode(x, self.mask, layer_idx)
        
        # decode the result to obtain the error pattern
        return self.decode(x)
    

if __name__ == '__main__': 
    pass
