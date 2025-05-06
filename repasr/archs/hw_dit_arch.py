import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange 

from basicsr.utils.registry import ARCH_REGISTRY

from torch.nn.attention.flex_attention import flex_attention
torch._dynamo.config.force_parameter_static_shapes = False
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Optional, Sequence, Literal

import math


ATTN_TYPE = Literal['Naive', 'SDPA', 'Flex']
"""
Naive Self-Attention: 
    - Numerically stable
    - Choose this for train if you have enough time and GPUs
    - Training ESC with Naive Self-Attention: 33.46dB @Urban100x2

Flex Attention:
    - Fast and memory efficient
    - Choose this for train/test if you are using Linux OS
    - Training ESC with Flex Attention: 33.44dB @Urban100x2

SDPA with memory efficient kernel:
    - Memory efficient (not fast)
    - Choose this for train/test if you are using Windows OS
    - Training ESC with SDPA: 33.43dB @Urban100x2
"""


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        nn.init.trunc_normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.trunc_normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb.unsqueeze(2).unsqueeze(3)  # to (B, C, 1, 1) 


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / q.shape[-1]**0.5
    score = score + bias
    score = F.softmax(score, dim=-1)
    out = score @ v
    return out


def apply_rpe(table: torch.Tensor, window_size: int):
    def bias_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int):
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]
    return bias_mod


def feat_to_win(x: torch.Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x, 'b (heads c) (h wh) (w ww) -> (b h w) heads (wh ww) c',
        heads=heads, wh=window_size[0], ww=window_size[1]
    )


def feat_to_qkvwin(x: torch.Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x, 'b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww) c',
        heads=heads, wh=window_size[0], ww=window_size[1], qkv=3
    )


def win_to_feat(x, window_size: Sequence[int], h_div: int, w_div: int):
    return rearrange(
        x, '(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)',
        h=h_div, w=w_div, wh=window_size[0], ww=window_size[1]
    )


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if affine else None
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if affine else None
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            if self.training:
                return F.layer_norm(x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
            else:
                return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
            

class S3(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.scale_1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0)
        self.scale_2 = nn.Conv2d(dim_in, dim_out, 1, 1, 0)
        self.shift = nn.Conv2d(dim_in, dim_out, 1, 1, 0)

        nn.init.zeros_(self.scale_1.weight)
        nn.init.zeros_(self.scale_1.bias)
        nn.init.zeros_(self.scale_2.weight)
        nn.init.zeros_(self.scale_2.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: input features with shape of (B, C, 1, 1)
        """
        scale_1 = self.scale_1(t) + 1.0
        scale_2 = self.scale_2(t) + 1.0
        shift = self.shift(t)
        return scale_1, scale_2, shift


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int, exp_ratio: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, int(dim*exp_ratio), 1, 1, 0)
        self.dwc = nn.Conv2d(int(dim*exp_ratio), int(dim*exp_ratio), kernel_size, 1, kernel_size//2, groups=int(dim*exp_ratio))
        self.aggr = nn.Conv2d(int(dim*exp_ratio), dim, 1, 1, 0)

        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.dwc.weight, std=0.02)
        nn.init.zeros_(self.dwc.bias)
        nn.init.trunc_normal_(self.aggr.weight, std=0.02)
        nn.init.zeros_(self.aggr.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        x = self.aggr(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
            self, dim: int, window_size: int, num_heads: int, 
            attn_func=None, attn_type: ATTN_TYPE = 'Flex'
        ):
        super().__init__()
        self.dim = dim
        window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.to_qkv = nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        
        self.attn_type = attn_type
        self.attn_func = attn_func
        self.relative_position_bias = nn.Parameter(
            torch.randn(num_heads, (2*window_size[0]-1)*(2*window_size[1]-1)).to(torch.float32) * 0.001
        )
        if self.attn_type == 'Flex':
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)
        self.is_mobile = False 

        nn.init.trunc_normal_(self.to_qkv.weight, std=0.02)
        nn.init.zeros_(self.to_qkv.bias)
        nn.init.trunc_normal_(self.to_out.weight, std=0.02)
        nn.init.zeros_(self.to_out.bias)

    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
        # Transposed idxs of original Swin Transformer
        # But much easier to implement and the same relative position distance anyway
        idxs_window = []
        for head in range(heads):
            for h in range(window_size**2):
                for w in range(window_size**2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        idxs = torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)
        return idxs
        
    def pad_to_win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x
    
    def to_mobile(self):
        bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
        self.rpe_bias = nn.Parameter(bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1]))
        
        del self.relative_position_bias
        del self.rpe_idxs
        
        self.is_mobile = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        h_div, w_div = x.shape[2] // self.window_size[0], x.shape[3] // self.window_size[1]
        
        qkv = self.to_qkv(x)
        dtype = qkv.dtype
        qkv = feat_to_qkvwin(qkv, self.window_size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.attn_type == 'Flex':
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = self.attn_func(q, k, v, score_mod=self.get_rpe)
        elif self.attn_type == 'SDPA':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1])
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                out = self.attn_func(q, k, v, attn_mask=bias, is_causal=False)
        elif self.attn_type == 'Naive':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1])
            out = self.attn_func(q, k, v, bias)
        else:
            raise NotImplementedError(f'Attention type {self.attn_type} is not supported.')
        
        out = win_to_feat(out, self.window_size, h_div, w_div)
        out = self.to_out(out.to(dtype)[:, :, :h, :w])
        return out   

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
    

class WindowCrossAttention(nn.Module):
    def __init__(
            self, dim: int, window_size: int, num_heads: int, 
            attn_func=None, attn_type: ATTN_TYPE = 'Flex'
        ):
        super().__init__()
        self.dim = dim
        window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.window_size = window_size
        self.num_heads = num_heads
        # self.to_qkv = nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.to_q = nn.Conv2d(dim, dim, 1, 1, 0)     # Source
        self.to_kv = nn.Conv2d(dim, dim*2, 1, 1, 0)  # Target
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        
        self.attn_type = attn_type
        self.attn_func = attn_func
        self.relative_position_bias = nn.Parameter(
            torch.randn(num_heads, (2*window_size[0]-1)*(2*window_size[1]-1)).to(torch.float32) * 0.001
        )
        if self.attn_type == 'Flex':
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)
        self.is_mobile = False 

        nn.init.trunc_normal_(self.to_q.weight, std=0.02)
        nn.init.zeros_(self.to_q.bias)
        nn.init.trunc_normal_(self.to_kv.weight, std=0.02)
        nn.init.zeros_(self.to_kv.bias)
        nn.init.trunc_normal_(self.to_out.weight, std=0.02)
        nn.init.zeros_(self.to_out.bias)

    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
        # Transposed idxs of original Swin Transformer
        # But much easier to implement and the same relative position distance anyway
        idxs_window = []
        for head in range(heads):
            for h in range(window_size**2):
                for w in range(window_size**2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        idxs = torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)
        return idxs
        
    def pad_to_win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x
    
    def to_mobile(self):
        bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
        self.rpe_bias = nn.Parameter(bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1]))
        
        del self.relative_position_bias
        del self.rpe_idxs
        
        self.is_mobile = True
    
    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        ref = self.pad_to_win(ref, h, w)
        h_div, w_div = x.shape[2] // self.window_size[0], x.shape[3] // self.window_size[1]
        
        q = self.to_q(x)
        k, v = self.to_kv(ref).chunk(2, dim=1)
        q = feat_to_win(q, self.window_size, self.num_heads)
        k = feat_to_win(k, self.window_size, self.num_heads)
        v = feat_to_win(v, self.window_size, self.num_heads)
        
        if self.attn_type == 'Flex':
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = self.attn_func(q, k, v, score_mod=self.get_rpe)
        elif self.attn_type == 'SDPA':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1])
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                out = self.attn_func(q, k, v, attn_mask=bias, is_causal=False)
        elif self.attn_type == 'Naive':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1])
            out = self.attn_func(q, k, v, bias)
        else:
            raise NotImplementedError(f'Attention type {self.attn_type} is not supported.')
        
        out = win_to_feat(out, self.window_size, h_div, w_div)
        out = self.to_out(out[:, :, :h, :w])
        return out   

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class Block(nn.Module):
    def __init__(
            self, dim: int, window_sizes: Sequence[int], num_heads: int,
            exp_ratio: int, use_cross_attn: bool = False, use_timestep_emb: bool = False,
            attn_func=None, attn_type: ATTN_TYPE = 'Flex',
    ):
        super().__init__()
        self.dim = dim
        self.window_sizes = window_sizes
        self.use_cross_attn = use_cross_attn
        self.use_timestep_emb = use_timestep_emb

        self.ln_attns = nn.ModuleList([
            LayerNorm(dim) for _ in window_sizes
        ])
        self.ln_ffns = nn.ModuleList([
            LayerNorm(dim) for _ in window_sizes
        ])
        if use_cross_attn:
            self.ln_cross_attns = nn.ModuleList([
                LayerNorm(dim) for _ in window_sizes
            ])

        self.attns = nn.ModuleList([
            WindowAttention(dim, window_size, num_heads, attn_func=attn_func, attn_type=attn_type)
            for window_size in window_sizes
        ])
        self.ffns = nn.ModuleList([
            ConvFFN(dim, 3, exp_ratio)
            for _ in window_sizes
        ])
        if use_cross_attn:
            self.cross_attns = nn.ModuleList([
                WindowCrossAttention(dim, window_size, num_heads, attn_func=attn_func, attn_type=attn_type)
                for window_size in window_sizes
            ])
            self.refine_ref = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1, 0),
            )
            nn.init.trunc_normal_(self.refine_ref[0].weight, std=0.02)
            nn.init.zeros_(self.refine_ref[0].bias)
            nn.init.trunc_normal_(self.refine_ref[1].weight, std=0.02)
            nn.init.zeros_(self.refine_ref[1].bias)
            nn.init.trunc_normal_(self.refine_ref[3].weight, std=0.02)
            nn.init.zeros_(self.refine_ref[3].bias)
        
        if use_timestep_emb:
            self.s3 = nn.ModuleList([
                S3(dim, dim) for _ in range(len(window_sizes) * 2)
            ])

        self.layer_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x: torch.Tensor, ref: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip = x

        for i in range(len(self.window_sizes)):
            s = x
            # 1. Self-Attention
            x = self.ln_attns[i](x)
            if self.use_timestep_emb:
                scale_1, scale_2, shift = self.s3[i * 2](t)
                x = x * scale_1 + shift
            x = self.attns[i](x)
            if self.use_timestep_emb:
                x = x * scale_2
            x = x + s

            # 2. Cross-Attention
            if self.use_cross_attn:
                s = x
                ref = self.refine_ref(ref)
                x = self.ln_cross_attns[i](x)  # Always use LN not timestep emb
                x = self.cross_attns[i](x, ref)
                x = x + s

            # 3. Feed Forward
            s = x
            x = self.ln_ffns[i](x)
            if self.use_timestep_emb:
                scale_1, scale_2, shift = self.s3[i * 2 + 1](t)
                x = x * scale_1 + shift
            x = self.ffns[i](x)
            if self.use_timestep_emb:
                x = x * scale_2
            x = x + s
        
        return x * self.layer_scale + skip
    

class SimpleCNN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.conv1 = ConvFFN(dim, 3, 2)
        self.conv2 = ConvFFN(dim, 3, 2)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv1(self.ln1(x))
        x = x + self.conv2(self.ln2(x))
        return x
    

@ARCH_REGISTRY.register()
class HierachicalWindowDiT(nn.Module):
    def __init__(
            self, dim: int, window_sizes: Sequence[int], num_heads: int,
            exp_ratio: int, num_blocks: int, align_layer: float = 0.33,
            align_dim: int = 1024
    ):
        super().__init__()

        # self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.to_feat = nn.Sequential(
            nn.Conv2d(3, dim//2, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(dim//2, dim, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        self.lr_to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.refine_lr = SimpleCNN(dim)
        self.t_embed = TimestepEmbedder(dim, dim)
        self.d_embed = TimestepEmbedder(dim, dim)

        # always use flex attention
        attn_func = torch.compile(flex_attention, dynamic=True)

        self.nth_align_layer = int(num_blocks * align_layer)
        self.blocks = nn.ModuleList([
            Block(
                dim, window_sizes, num_heads,
                exp_ratio, use_cross_attn=i <= self.nth_align_layer,
                use_timestep_emb=i > self.nth_align_layer,
                attn_func=attn_func, attn_type='Flex'
            ) for i in range(num_blocks)
        ])

        # upsampler  always 4 
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(dim, 3, 3, 1, 1),
        )

        # to dim x 16 x 16 for alignment
        # training patch dim is 64. So, downsampling factor is 4
        self.align_conv = nn.Sequential(  
            nn.Conv2d(dim, align_dim, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(align_dim, align_dim, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(align_dim, align_dim, 1),
        )
        nn.init.trunc_normal_(self.to_feat[0].weight, std=0.02)
        nn.init.zeros_(self.to_feat[0].bias)
        nn.init.trunc_normal_(self.to_feat[2].weight, std=0.02)
        nn.init.zeros_(self.to_feat[2].bias)
        nn.init.zeros_(self.to_feat[4].weight)
        nn.init.zeros_(self.to_feat[4].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, d: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (B, 3, H, W)
            t: timestep embedding with shape of (B,)
            ref: lr input features with shape of (B, 3, H, W)
        """
        # 1. Timestep embedding
        x = self.to_feat(x)
        ref = self.refine_lr(self.lr_to_feat(ref))
        c = self.t_embed(t) + self.d_embed(d)

        # 2. Hierarchical blocks
        for i, block in enumerate(self.blocks):
            x = block(x, ref, c)
            if i == self.nth_align_layer:
                align_feat = x
        
        # 3. Upsample
        x = self.upsampler(x)
        if self.training:
            return x, self.align_conv(align_feat).permute(0, 2, 3, 1).flatten(1, 2).contiguous()
        else:
            return x


if __name__ == '__main__':
    model = HierachicalWindowDiT(
        dim=64, window_sizes=[4, 8, 12, 24, 32, 64], num_heads=4,
        exp_ratio=2, num_blocks=6
    ).cuda()
    model.eval()
    print(model)
    x = torch.randn(4, 3, 256, 256).cuda()
    t = torch.randn(4).cuda()
    d = torch.randn(4).cuda()
    ref = torch.randn(4, 3, 64, 64).cuda()
    out = model(x, t, d, ref)
    print(out.shape)  # (1, 3, 256, 256)
