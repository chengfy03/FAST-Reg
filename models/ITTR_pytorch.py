import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat


# helper functions

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t):
    return F.normalize(t, dim=-1)


# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


# classes
#Cross-scale dual branch 

class HPB(nn.Module):
    """ Hybrid Perception Block """

    def __init__(
            self,
            dim,
            dim_head=32,
            heads=8,
            ff_mult=4,
            attn_height_top_k=16,
            attn_width_top_k=16,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()

        self.attn = DPSA(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            height_top_k=attn_height_top_k,
            width_top_k=attn_width_top_k,
            dropout=attn_dropout
        )

       # self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        #self.attn_parallel_combine_out = nn.Conv2d(dim * 2, dim, 1)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.attn_parallel_combine_out = nn.Conv2d(dim * 2, dim, 1)

        ff_inner_dim = dim * ff_mult

        self.ff = nn.Sequential(
            nn.Conv2d(dim, ff_inner_dim, 1),
            nn.InstanceNorm2d(ff_inner_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            Residual(nn.Sequential(
                nn.Conv2d(ff_inner_dim, ff_inner_dim, 3, padding=1, groups=ff_inner_dim),
                nn.InstanceNorm2d(ff_inner_dim),
                nn.GELU(),
                nn.Dropout(ff_dropout)
            )),
            nn.Conv2d(ff_inner_dim, dim, 1),
            nn.InstanceNorm2d(ff_inner_dim)
        )

        def forward(self, x):
        attn_branch_out = self.attn(x)
        conv_branch_out = self.dwconv(x)

        concatted_branches = torch.cat((attn_branch_out, conv_branch_out), dim=1)
        attn_out = self.attn_parallel_combine_out(concatted_branches) + x

        return self.ff(attn_out)
 

class DPSA(nn.Module):
    """ Dual-pruned Self-attention Block """

    def __init__(
            self,
            dim,
            height_top_k=16,
            width_top_k=16,
            dim_head=32,
            heads=8,
            dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.norm = ChanLayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        # print("DPSA:", x.shape)
        b, c, h, w = x.shape
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        # fold out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c x y', h=self.heads), (q, k, v))
        # print('b (h c) x y -> (b h) c x y :', q.size(), k.size(), v.size())

        # q.size() = torch.Size([16, 16, 64, 64]) b d h w 转为 torch.Size([16, 64, 64, 16]) b h w d
        q, k, v = map(lambda t: rearrange(t, 'b d h w->b h w d'), (q, k, v))
        # print('b d h w->b h w d :', q.size(), k.size(), v.size())

        # they used l2 normalized queries and keys, cosine sim attention basically

        q, k = map(l2norm, (q, k))

        # calculate whether to select and rank along height and width

        need_height_select_and_rank = self.height_top_k < h
        need_width_select_and_rank = self.width_top_k < w

        # select and rank keys / values, probing with query (reduced along height and width) and keys reduced along row and column respectively

        if need_width_select_and_rank or need_height_select_and_rank:
            q_probe = reduce(q, 'b h w d -> b d', 'sum')

        # gather along height, then width

        if need_height_select_and_rank:
            k_height = reduce(k, 'b h w d -> b h d', 'sum')

            top_h_indices = einsum('b d, b h d -> b h', q_probe, k_height).topk(k=self.height_top_k, dim=-1).indices
            top_h_indices = repeat(top_h_indices, 'b h -> b h w d', d=self.dim_head, w=k.shape[-2])

            k, v = map(lambda t: t.gather(1, top_h_indices), (k, v))  # first gather across height

        if need_width_select_and_rank:
            k_width = reduce(k, 'b h w d -> b w d', 'sum')

            top_w_indices = einsum('b d, b w d -> b w', q_probe, k_width).topk(k=self.width_top_k, dim=-1).indices
            top_w_indices = repeat(top_w_indices, 'b w -> b h w d', d=self.dim_head, h=k.shape[1])

            k, v = map(lambda t: t.gather(2, top_w_indices), (k, v))  # then gather along width

        # select the appropriate keys and values

        q, k, v = map(lambda t: rearrange(t, 'b ... d -> b (...) d'), (q, k, v))

        # cosine similarities

        sim = einsum('b i d, b j d -> b i j', q, k)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate out

        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge heads and combine out

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x=h, y=w, h=self.heads)
        return self.to_out(out)





class PixelLevelMacroMicroInterleaving(nn.Module):
    """
    Pixel-level Macro--Micro Interleaving.

    输入：
        macro_feature: DPSA 宏观拓扑分支特征 A，形状 [B, C, H, W]
        micro_feature: DW 微观纹理分支特征 B，形状 [B, C, H, W]

    交织过程：
        Stage 1：沿高度方向交织，分别构造 H 和 H_tilde；
        Stage 2：沿宽度方向交织 H 和 H_tilde。

    最终每个 2×2 空间块遵循：

        A  B
        B  A

    输出形状保持为 [B, C, H, W]。
    该操作没有可学习参数。
    """

    def forward(self, macro_feature, micro_feature):
        # 两个分支必须具有完全相同的尺寸，才能进行逐像素交织
        if macro_feature.shape != micro_feature.shape:
            raise ValueError(
                "Macro and micro features must have identical shapes, "
                f"but got {tuple(macro_feature.shape)} and "
                f"{tuple(micro_feature.shape)}."
            )

        _, _, height, width = macro_feature.shape
        device = macro_feature.device

        # ==============================================================
        # Stage 1: Heightwise Weaving
        # 沿高度方向进行纵向交织
        # ==============================================================

        # 偶数行掩码：
        # 第 0、2、4、... 行为 True；
        # 第 1、3、5、... 行为 False。
        #
        # 掩码形状为 [1, 1, H, 1]，
        # 会自动广播到 batch、channel 和 width 维度。
        even_row_mask = (
            torch.arange(height, device=device)
            .view(1, 1, height, 1) % 2 == 0
        )

        # H：
        # 偶数行来自宏观拓扑特征 A，
        # 奇数行来自微观纹理特征 B。
        #
        # 行排列为：
        # A, B, A, B, ...
        height_woven = torch.where(
            even_row_mask,
            macro_feature,
            micro_feature
        )

        # H_tilde：
        # 偶数行来自微观纹理特征 B，
        # 奇数行来自宏观拓扑特征 A。
        #
        # 行排列为：
        # B, A, B, A, ...
        height_woven_complement = torch.where(
            even_row_mask,
            micro_feature,
            macro_feature
        )

        # ==============================================================
        # Stage 2: Widthwise Weaving
        # 沿宽度方向进行横向交织
        # ==============================================================

        # 偶数列掩码：
        # 第 0、2、4、... 列为 True；
        # 第 1、3、5、... 列为 False。
        #
        # 掩码形状为 [1, 1, 1, W]，
        # 会自动广播到 batch、channel 和 height 维度。
        even_col_mask = (
            torch.arange(width, device=device)
            .view(1, 1, 1, width) % 2 == 0
        )

        # 偶数列从 H 中选取，奇数列从 H_tilde 中选取。
        #
        # 因此最终的每个 2×2 块为：
        #
        # A  B
        # B  A
        #
        # 即图中所示的 2×2 woven rule。
        interleaved_feature = torch.where(
            even_col_mask,
            height_woven,
            height_woven_complement
        )

        return interleaved_feature