from torch import nn
import torch

class base_layer(nn.Module):
    def __init__(self, channel_1=3, channel_2=128, channel_3=768):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(channel_1, channel_2, 1),
            nn.BatchNorm1d(channel_2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(channel_2, channel_3, 1)
        )
    def forward(self,x):
        return self.layer(x)

class Baseconv(nn.Module):
    def __init__(self, channel1, channel2, channel3, channel4=None):
        super().__init__()
        self.conv1 = nn.Conv1d(channel1, channel2, 1)
        self.conv2 = nn.Conv1d(channel2, channel3, 1)
        if channel4 is not None:
            self.conv3 = nn.Conv1d(channel4, channel3, 1)

    def forward(self, x, _type):
        if _type == 'norm':
            x = torch.relu(self.conv1(x))
            out = self.conv2(x)
        elif _type == 'res':
            x_1 = self.conv3(x)
            x_2 = torch.relu(self.conv1(x))
            out = self.conv2(x_2)+x_1
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flg=False, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # 1 for mask, 0 for not mask
            # mask shape N, N
            mask_value = -torch.finfo(attn.dtype).max
            mask = (mask > 0)  # convert to boolen, shape torch.BoolTensor[N, N]
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        xx = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Offset-Attention
        if flg:
            x = x - xx
        else:
            x = xx
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CRAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.at_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        at = (q @ k.transpose(-2, -1)) * self.scale
        at = at.softmax(dim=-1)
        at = self.at_drop(at)

        x = (at @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x