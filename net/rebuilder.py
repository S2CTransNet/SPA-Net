import torch
from torch import nn
from utils.tools import *
from net.model import Baseconv
from timm.layers import DropPath
from net.model import Attention, Mlp, CRAttention

class Folding(nn.Module):
    def __init__(self, in_channel, step, hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0)

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )


    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2



class Rebuilder(nn.Module):
    def __init__(self, in_channel, up_scale, hidden_dim=512, embed_dim=768, num_heads=6, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.up_scale = up_scale
        self.tansformer = Transformer(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
        self.up_sampler = nn.Upsample(scale_factor=up_scale)
        self.conv = Baseconv(in_channel, hidden_dim, in_channel)
        self.fusion = Baseconv(in_channel*2, hidden_dim, in_channel, in_channel*2)
        self.pred_offset = Baseconv(in_channel, 64, 3)
        self.convT = nn.ConvTranspose1d(in_channel, in_channel, up_scale, up_scale, bias=False)

    def forward(self, q, v, q_pos, v_pos, self_knn_index=None, cross_knn_index=None):
        q = self.tansformer(q=q.transpose(1, 2), v=v, q_pos=q_pos.transpose(1, 2), v_pos=v_pos,
                            self_knn_index=self_knn_index, cross_knn_index=cross_knn_index).transpose(1, 2)
        f_1 = self.conv(q, 'norm')
        f_1 = self.convT(f_1)
        f_2 = self.up_sampler(q)
        q = self.fusion(torch.cat([f_1, f_2], 1), 'res')
        offset = self.pred_offset(q, 'norm')
        child_p = self.up_sampler(q_pos) + offset
        return q, child_p

class TransformerRebuilder(nn.Module):
    def __init__(self, embed_dim=384, up_scale=[4, 4], num_heads=6, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.k = 8
        self.knn_layer = 1
        rebuilder = []
        for i in range(len(up_scale)):
            rebuilder.append(Rebuilder(in_channel=384, up_scale=up_scale[i], embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate))
        self.rebuilder = nn.ModuleList(rebuilder)

    def forward(self, q, coor, coarse_point_cloud, v, points):
        inp_sparse = fps(points, 128)
        sparse_pcd = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        PC = [sparse_pcd]
        self_knn_index = knn_point(self.k, coarse_point_cloud, coarse_point_cloud)
        cross_knn_index = knn_point(self.k, coor, coarse_point_cloud)
        for i, dec in enumerate(self.rebuilder):
            if i < self.knn_layer:
                q, coarse_point_cloud = dec(q=q.transpose(1, 2), v=v, q_pos=coarse_point_cloud.transpose(1, 2), v_pos=coor, self_knn_index=self_knn_index,
                            cross_knn_index=cross_knn_index)
            else:
                q, coarse_point_cloud = dec(q=q, v=v, q_pos=coarse_point_cloud, v_pos=coor)
            _,_,np =coarse_point_cloud.shape
            inp_sparse = fps(points, np//3)
            sparse_pcd = torch.cat([coarse_point_cloud.transpose(1, 2), inp_sparse], dim=1).contiguous()
            PC.append(sparse_pcd)
        return PC


class Transformer(nn.Module):
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

    def get_graph_feature(self,  q, q_pos, v=None, v_pos=None, idx=None):
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

    def forward(self, q, v, q_pos, v_pos, self_knn_index=None, cross_knn_index=None):
        if self_knn_index is not None:
            q_1 = []
            norm_q = self.norm1(q)
            q_1.append(self.self_attn(norm_q))
            f = self.get_graph_feature(norm_q, q_pos, idx=self_knn_index)
            q_1.append(self.knn_map(f).max(-2)[0])
            q_1 = torch.cat(q_1, dim=-1)
            q_1 = self.merge_map(q_1)
        else:
            norm_q = self.norm1(q)
            q_1 = self.self_attn(norm_q)
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