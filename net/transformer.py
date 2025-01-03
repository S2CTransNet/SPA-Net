import torch
from torch import nn
from timm.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from net.modules import Mlp, Block
knn = KNN(k=16, transpose_mode=False)


class Encode(nn.Module):
    def __init__(self, in_chans=3, embed_dim=2048, num_heads=8, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., num_f=None, i=0):
        super().__init__()
        '''
        K has to be 16
        '''
        if num_f[0] ==3:
            self.input_trans = nn.Conv1d(num_f[0], int(num_f[1]/4), 1)
            self.layer = nn.Sequential(nn.Conv2d(int(num_f[1]/2), num_f[1]*2, kernel_size=1, bias=False),
                                        nn.GroupNorm(4, num_f[1]*2),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Conv2d(num_f[1]*2, num_f[1], kernel_size=1, bias=False),
                                        nn.GroupNorm(4, num_f[1]),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        )
        else:
            self.layer = nn.Sequential(nn.Conv2d(num_f[1], num_f[1] * 2, kernel_size=1, bias=False),
                                       nn.GroupNorm(4, num_f[1] * 2),
                                       nn.LeakyReLU(negative_slope=0.2),
                                       nn.Conv2d(num_f[1] * 2, num_f[1], kernel_size=1, bias=False),
                                       nn.GroupNorm(4, num_f[1]),
                                       nn.LeakyReLU(negative_slope=0.2),
                                       )
        embed_dim = int(embed_dim/2**i)
        self.embed_dim = embed_dim
        self.encoder1 = transform_block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, num_f=num_f)
        self.encoder2 = transform_block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate)
        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, coor1=None):
        # x: bs, 3, np
        # bs 3 N(128)   bs C(224)128 N(128)
        if coor1 == None:
            coor1 = x
            f1 = self.input_trans(x)
        else:
            f1 = x
        f1 = self.get_graph_feature(coor1, f1, coor1, f1)
        f1 = self.layer(f1)
        f1 = f1.max(dim=-1, keepdim=False)[0]
        f1 = self.encoder1(x, f1)
        f1 = self.encoder2(f1)
        if self.embed_dim == 512:
            self.embed_dim = self.embed_dim/2
        coor_q, f_q = self.fps_downsample(coor1, f1, int(self.embed_dim/2))

        return coor_q, f_q


class transform_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_f=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim * 2, dim)
        if num_f is not None:
            self.layer = nn.Sequential(nn.Conv1d(in_channels=num_f[0], out_channels=num_f[1]*4, kernel_size=1, bias=False),
                                        nn.GELU(),
                                        nn.Conv1d(in_channels=num_f[1]*4, out_channels=num_f[1], kernel_size=1, bias=False),
            )
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_f=None):
        norm_x = self.norm1(x)
        if knn_f is not None:
            x_1 = self.attn(norm_x, True)
            x = self.layer(x_1)
            x_1 = torch.cat([x, knn_f], dim=-1)
            x_1 = self.merge_map(x_1)
        else:
            x_1 = self.attn(norm_x, False)
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_graph_feature(x, knn_index, x_q=None):
    k = 8
    batch_size, num_points, num_dims = x.size()
    num_query = x_q.size(1) if x_q is not None else num_points
    feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
    feature = feature.view(batch_size, k, num_query, num_dims)
    x = x_q if x_q is not None else x
    x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
    feature = torch.cat((feature - x, x), dim=-1)
    return feature

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

    def forward(self, x, flg=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
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

def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        _, idx = knn(coor_k, coor_q)  # bs k np
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)

    return idx


class basenet(nn.Module):
    def __init__(self, num_f, knn_layer=1, in_chans=3, embed_dim=768, depth=[6, 6], num_heads=6, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 num_query=224,):
        super().__init__()
        self.knn_layer = knn_layer
        self.encoder = nn.ModuleList([
            Encode(
                in_chans=in_chans, embed_dim=2048, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, num_f=num_f[i], i=i)
            for i in range(3)])
        self.connet = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 384, 1)
        )

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, 384, 1),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(384, 384, 1)
        )
    def forward(self,x):

        # encode
        for i, enc in enumerate(self.encoder):
            if i >= self.knn_layer:
                coor,x = enc(x, coor)  # B N C
            else:
                coor, x = enc(x)
        pos = self.pos_embed(coor).transpose(1, 2)
        x = self.input_proj(x).transpose(1, 2)
        knn_index = get_knn_index(coor)

        return coor, x

# num_f = [[3,32],[32,64],[64,128]]
# mode = basenet(num_f)
# x = torch.randn(8,3,2048).to('cuda')
# mode.to('cuda')
# mode.eval()
# out = mode(x)
