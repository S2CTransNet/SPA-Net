import torch
from torch import nn
from timm.layers import DropPath
from net.model import Attention, Mlp, CRAttention
from utils.tools import index_points, knn_point


class TransformerDecoder_(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CRAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim * 2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim * 2, dim)

    def get_graph_feature(self, x, knn_index, x_q=None):
        k = 8
        batch_size, num_points, num_dims = x.size()
        num_query = x_q.size(1) if x_q is not None else num_points
        feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
        feature = feature.view(batch_size, k, num_query, num_dims)
        x = x_q if x_q is not None else x
        x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
        feature = torch.cat((feature - x, x), dim=-1)
        return feature

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        norm_q = self.norm1(q)
        if self_knn_index is not None:
            q_1 = self.self_attn(norm_q,True)
            knn_f = self.get_graph_feature(norm_q, self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)
        else:
            q_1 = self.self_attn(norm_q)
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        if cross_knn_index is not None:
            knn_f = self.get_graph_feature(norm_v, cross_knn_index, norm_q)
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.drop_path(q_2)
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q

class TransformerDecoder(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.k = 8
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CRAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim * 2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim * 2, dim)

    def get_graph_feature(self,  q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        if denoise_length is not None:
            assert idx is None
            v = q
            v_pos = q_pos
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length])  # B N_r k
            local_v_r = index_points(v[:, :-denoise_length], idx)  # B N_r k C
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:])  # B N_n k
            local_v_n = index_points(v, idx)  # B N_n k C
            local_v = torch.cat([local_v_r, local_v_n], dim=1)
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((local_v - q, q), dim=-1)  # B N k C
        else:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos)
            assert idx.size(-1) == self.k
            local_v = index_points(v, idx)  # B N k C
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1)  # B N k C
            feature = torch.cat((local_v - q, q), dim=-1)  # B N k C
        return feature

    def forward(self, q, v, q_pos, v_pos, self_knn_index=None, cross_knn_index=None, denoise_length=None):
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.
        if cross_knn_index is not None:
            q_1 = []
            norm_q = self.norm1(q)
            q_1.append(self.self_attn(norm_q, flg=True, mask=mask))
            f = self.get_graph_feature(norm_q, q_pos, idx=self_knn_index, denoise_length=denoise_length)
            q_1.append(self.knn_map(f).max(-2)[0])
            q_1 = torch.cat(q_1, dim=-1)
            q_1 = self.merge_map(q_1)
        else:
            norm_q = self.norm1(q)
            q_1 = self.self_attn(norm_q, mask=mask)
        q = q + self.drop_path(q_1)

        if cross_knn_index is not None:
            q_2 = []
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            q_2.append(self.attn(norm_q, norm_v))
            f = self.get_graph_feature(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_knn_index)
            q_2.append(self.knn_map_cross(f).max(-2)[0])
            q_2 = torch.cat(q_2, dim=-1)
            q_2 = self.merge_map_cross(q_2)
        else:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            q_2 = self.attn(norm_q, norm_v)
        q = q + self.drop_path(q_2)
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q