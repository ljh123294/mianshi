import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class Rotator:
    """根据hidden_dim，和position_ids 生成对应的旋转位置编码, 和论文中定义略有不同，一个个二维的子空间被
    分割到了前后两部分，分别进行旋转，然后拼接起来
    """
    def __init__(self, D, position_ids):
        """ position_ids: [seq_len], D 和单个头的hidden_dim对应 """
        base = 10000
        d = D / 2
        B = base ** (1/d)
        theta_base = 1.0 / (B ** (torch.arange(0, d)))    # 等比数列， $\Theta$
        thetas = position_ids.outer(theta_base)  # [seq_len, D/2]
        full_thetas = torch.cat((thetas, thetas), dim=-1)  # [seq_len, D]
        self.cos = full_thetas.cos()
        self.sin = full_thetas.sin()

    def rotate(self, x):
        """ trick1
        x: [bs, num_attention_heads, seq_len, D]
        q: [bs, num_attention_heads, seq_len, D]
        cos: [seq_len, D]
        [x,y] @ [[cos, sin], [-sin, cos]] = [x*cos-y*sin, ycos+x*sin] =[x,y]*cos+[-y, x]*sin
        """
        return x * self.cos + self.reverse_half(x) * self.sin

    def reverse_half(self, q):
        """ q: [bs, num_attention_heads, seq_len, D] trick2 """
        u = q[..., : q.shape[-1] // 2]
        v = q[..., q.shape[-1] // 2:]
        return torch.cat((-v, u), dim=-1)


class SelfAttentionWithRoPE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.H = config["n_head"]
        self.F = config["hidden_dim"]  # F
        self.D = self.F // self.H  # D
        # 一次把qkv 全部映射完成，对应W_Q, W_K, W_V
        self.qkv_proj = nn.Linear(self.F, 3 * self.F)
        # 最后的投影，对应于 $W_O$
        self.out_proj = nn.Linear(self.F, self.F)

    def forward(self, x, position_ids):
        # position_ids: [seq_len]
        B, N, _ = x.size()
        q, k, v = self.qkv_proj(x).split(self.F, dim=-1)
        # matmul 只能在最后两个维度相乘，需要对NxD的矩阵相乘，做1,2维度的交换
        k = k.view(B, N, self.H, self.D).transpose(1, 2)
        q = q.view(B, N, self.H, self.D).transpose(1, 2)
        v = v.view(B, N, self.H, self.D).transpose(1, 2)
        # 旋转位置编码
        rotator = Rotator(self.D, position_ids)
        q = rotator.rotate(q)
        k = rotator.rotate(k)
        # 计算相似性
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        # 多头拼接
        y = y.transpose(1, 2).contiguous().view(B, N, self.F)
        y = self.out_proj(y)
        return y


config = {"n_head": 2, "hidden_dim": 16, "batch_size": 3, "seq_len": 5}
attn = SelfAttentionWithRoPE(config)
x = torch.rand(config["batch_size"], config["seq_len"], config["hidden_dim"])
position_ids = torch.arange(config["seq_len"])
y = attn(x, position_ids)
