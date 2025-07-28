import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Attention(nn.Module):
    def __init__(
            self,
            d_model,
            window_size,
            d_state=8,
            d_conv=3,
            expand=2.66,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Modified: Use single projection for concatenated channels
        self.in_proj = nn.Linear(self.d_model * 2, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Modified: Use single conv for concatenated channels
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        # Modified: Use single output projection and norm
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model * 2, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def create_patches(self, image_tensor, w, order='ltr_utd'):
        """
        按照指定顺序生成图像patch
        order参数:
        - 'ltr_utd': 从左到右，再从上到下
        - 'rtl_dtu': 从下到上，再从右到左
        - 'utd_ltr': 从上到下，再从左到右
        - 'dtu_rtl': 从右到左，再从下到上
        """
        B, C, H, W = image_tensor.shape
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)

        # 确保图像尺寸能被w整除
        assert H % w == 0 and W % w == 0, f"图像尺寸({H}x{W})必须能被patch大小({w})整除"
        
        # 重塑为patch网格
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous() # 转为BHWC
        patches = image_tensor.view(B, Hg, w, Wg, w, C)  # [B, Hg, w, Wg, w, C]
        
        # 根据不同顺序调整patch排列
        if order == 'ltr_utd':  # 从左到右，再从上到下
            # 原始顺序：patch按行优先排列，每个patch内按行优先
            patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, Hg, Wg, w, w]
        
        elif order == 'rtl_dtu':  # 从下到上，再从右到左
            # 翻转patch顺序和每个patch内的像素顺序
            patches = patches.flip(1, 3).permute(0, 5, 1, 3, 2, 4).flip(4, 5).contiguous()
        
        elif order == 'utd_ltr':  # 从上到下，再从左到右
            # patch按列优先排列，每个patch内按列优先
            patches = patches.permute(0, 5, 3, 1, 4, 2).contiguous()  # [B, C, Wg, Hg, w, w]
        
        elif order == 'dtu_rtl':  # 从右到左，再从下到上
            # 翻转patch顺序和每个patch内的像素顺序
            patches = patches.flip(1, 3).permute(0, 5, 3, 1, 4, 2).flip(4, 5).contiguous()
        
        else:
            raise ValueError(f"Unsupported order: {order}")
        
        return patches
    
    def get_scan(self, x_concat, orders):
        """
        Modified: Process concatenated channels directly without interleaving
        """
        output = []
        for order in orders:
            patches = self.create_patches(x_concat, self.window_size, order)
            
            # 展平每个patch为向量，但保留patch的网格结构
            B, C, Hg, Wg, w1, w2 = patches.shape
            patches_flat = patches.reshape(B, C, Hg, Wg, w1*w2)  # [B, C, Hg, Wg, w*w]
            
            # 展平所有patch
            sequence = patches_flat.reshape(B, C, -1).unsqueeze(1)
            output.append(sequence)
        output = torch.cat(output, dim=1)
        return output

    def reconstruct_image(self, final_result, w, x_concat, orders):
        """
        Modified: Reconstruct single concatenated image
        """
        B, C, H, W = x_concat.shape
        L = H * W
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)
        patch_size = w * w
        sequences = [final_result[:, i] for i in range(4)]

        y_concat = None

        for i, order in enumerate(orders):
            seq = sequences[i].reshape(B, C, Hg, Wg, patch_size)
            patches = seq.reshape(B, C, Hg, Wg, w, w)

            if order == 'ltr_utd':
                patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
                img = patches.view(B, C, Hg*w, Wg*w)
            elif order == 'rtl_dtu':
                patches = patches.flip(2, 3).flip(4, 5)
                img = patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg*w, Wg*w)
            elif order == 'utd_ltr':
                patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
                img = patches.view(B, C, Hg*w, Wg*w).transpose(-1, -2)
            elif order == 'dtu_rtl':
                patches = patches.flip(2, 3).flip(4, 5)
                img = patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg*w, Wg*w).transpose(2, 3)

            if y_concat is None:
                y_concat = img.contiguous().view(B, -1, L).contiguous()
            else:
                y_concat = y_concat + img.contiguous().view(B, -1, L).contiguous()

        return y_concat

    def forward_core(self, x_concat):
        """
        Modified: Process concatenated channels
        """
        B, C, H, W = x_concat.shape
        L = H * W  # Half the original sequence length
        K = 4

        orders = ['ltr_utd', 'rtl_dtu', 'utd_ltr', 'dtu_rtl']
        xs = self.get_scan(x_concat, orders)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        
        return self.reconstruct_image(out_y, self.window_size, x_concat, orders)

    def forward(self, visible, infrared):
        """
        Modified: Concatenate channels at input and split at output
        """
        x_vis = rearrange(visible, 'b c h w -> b h w c')
        x_inf = rearrange(infrared, 'b c h w -> b h w c')

        B, H, W, C = x_vis.shape

        # Concatenate channels
        x_concat = torch.cat([x_vis, x_inf], dim=-1)  # [B, H, W, 2*C]

        xz = self.in_proj(x_concat)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y = self.forward_core(x)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        
        # Split channels back to visible and infrared
        out_vis, out_inf = out.chunk(2, dim=-1)
        
        out_vis = rearrange(out_vis, 'b h w c -> b c h w')
        out_inf = rearrange(out_inf, 'b h w c -> b c h w')
        
        return out_vis, out_inf