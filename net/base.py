import torch
from net.aggregator import *
from net.encoder import *
from net.decoder import *
from net.rebuilder import *
from net.model import *
from utils.tools import *
from torch import nn
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
knn = KNN(k=8, transpose_mode=False)
from timm.layers import DropPath, trunc_normal_


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

class ModuleBuilder:
    def __init__(self, config):
        self.config = config
    def aggregator(self, mlp_ratio=2., qkv_bias=False,qk_scale=None, drop_rate=0., attn_drop_rate=0., num_f=[[3,32],[32,64],[64,128]]):
        if self.config.aggregator == 'Transformer':
            aggregator = nn.ModuleList([
                AggregationBlock(
                embed_dim=2048, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, num_f=num_f[i], i=i)
            for i in range(3)])
        elif self.config.aggregator == 'DGCNN':
            aggregator = DGCNN_Grouper()
        else:
            raise KeyError('Not found aggregator type')
        return aggregator
    def encoder(self, embed_dim=768, depth=[6, 6], num_heads=6, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        if self.config.encoder == 'Transformer':
            encoder = nn.ModuleList([TransformerEncoder(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])
        else:
            raise KeyError('Not found encoder type')
        return encoder
    def decoder(self, embed_dim=768, depth=[6, 6], num_heads=6, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        if self.config.decoder == 'Transformer':
            decoder = nn.ModuleList([TransformerDecoder(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])
        else:
            raise KeyError('Not found decoder type')
        return decoder

    def rebuilder(self, name, num_query ,fold_step):
        if name == 'FoldingNet':
            rebuilder = Folding(num_query, step=fold_step, hidden_dim=256)
        elif name == 'MLP':
            rebuilder = nn.Sequential(nn.Linear(1024, 1024),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, 3 * num_query))
        elif name == 'Transformer':
            rebuilder = TransformerRebuilder()
        else:
            raise KeyError('Not found rebuilder type')
        return rebuilder

    def generator(self, trans_dim=None):
        mlp_query = nn.Sequential(nn.Conv1d(1024 + 3, 1024, 1),
                                       # nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2),
                                       nn.Conv1d(1024, 1024, 1),
                                       # nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2),
                                       nn.Conv1d(1024, trans_dim, 1))
        if self.config.generator == 'Norm':
            query_ranking = None
        elif self.config.generator == 'Auto':
            query_ranking = nn.Sequential(nn.Linear(3, 256),
                                               nn.GELU(),
                                               nn.Linear(256, 256),
                                               nn.GELU(),
                                               nn.Linear(256, 1),
                                               nn.Sigmoid())
        else:
            raise KeyError('Not found generator type')
        return mlp_query, query_ranking



class Rebuilder(nn.Module):
    def __init__(self,modulebuilder=None, name=None, embed_dim=384, np = 0, fold_step=8, num_query=384):
        super().__init__()
        self.increase_dim = base_layer(channel_1=384, channel_2=1024, channel_3=1024)
        self.reduce_map = nn.Linear((embed_dim + 1024 + np), num_query)
        self.rebuilder = modulebuilder.rebuilder(name=name, fold_step=fold_step, num_query=num_query)
        self.name = name

    def forward(self,q, coarse_point_cloud=None, coor=None, v=None, points=None, feature_last=None):
        bs, s1, s2 = q.shape
        if coarse_point_cloud is not None:
            if self.name == 'FoldingNet':
                _feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)
                _feature = torch.max(_feature, dim=1)[0]
                building_feature = torch.cat([_feature.unsqueeze(-2).expand(-1, s1, -1), q, coarse_point_cloud], dim=-1)
                if feature_last is not None:
                    building_feature += feature_last.repeat(1, 4, 1)
                building_feature_ = self.reduce_map(building_feature.reshape(bs * s1, -1))
                Final_xyz = self.rebuilder(building_feature_).reshape(bs, s1, 3, -1)
                build_points = (Final_xyz[:, :, :3, :] + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)
            elif self.name == 'Transformer':
                build_points = self.rebuilder(q, coor, coarse_point_cloud, v, points)
            return build_points, building_feature
        else:
            if self.name == 'FoldingNet':
                global_feature = self.increase_dim(q)
                global_feature = torch.max(global_feature, dim=-1)[0]
                building_feature = torch.cat([global_feature.unsqueeze(-2).expand(-1, s1, -1), q], dim=-1)
                building_feature = building_feature.reshape(bs * s1, -1)
                building_feature = self.reduce_map(building_feature)
                coarse_point_cloud = self.rebuilder(building_feature).reshape(bs, s1, 3)
            elif self.name == 'MLP':
                global_feature = self.increase_dim(q)
                global_feature = torch.max(global_feature, dim=-1)[0]
                coarse_point_cloud = self.rebuilder(global_feature).reshape(bs, -1, 3)
            return coarse_point_cloud, global_feature


class Generator(nn.Module):
    def __init__(self, modulebuilder=None,trans_dim=384, num_query=384, noise=False):
        super().__init__()
        self.num_query= num_query
        self.noise = noise
        self.mlp_query, self.query_ranking = modulebuilder.generator(trans_dim=trans_dim)

    def forward(self,points, global_feature, coarse):
        #global_feature:(bx1024)  coarse:(bx384x3)
        denoise_length = None
        if self.query_ranking is not None:
            coarse_inp = fps(points, self.num_query // 2)
            coarse = torch.cat([coarse, coarse_inp], dim=1)
            query_ranking = self.query_ranking(coarse)  # b np 1
            idx = torch.argsort(query_ranking, dim=1, descending=True)  # b np 1
            coarse = torch.gather(coarse, 1, idx[:, :self.num_query].expand(-1, -1, coarse.size(-1)))
            if self.noise:
                picked_points = fps(points, 64)
                picked_points = jitter_points(picked_points)
                coarse = torch.cat([coarse, picked_points], dim=1)  # B 256+64 3?
                denoise_length = 64
        query_feature = torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim=-1)
        q = self.mlp_query(query_feature.transpose(1, 2)).transpose(1, 2)
        return q, coarse, denoise_length


class net_builder_(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._int_params()
        self._int_layer()

    def _int_params(self):
        self.noise = self.config.noise if self.config.noise else False
        self.trans_dim = self.config.trans_dim if self.config.trans_dim else None
        self.knn_layer = self.config.knn_layer if self.config.knn_layer else None
        self.num_points = self.config.num_points if self.config.num_points else None
        self.num_query = self.config.num_query if self.config.num_query else None
        self.depth = self.config.depth if self.config.depth else [6, 8]
        self.num_f = self.config.num_f if self.config.num_f else [[3, 32], [32, 64], [64, 128]]
        self.in_chans = self.config.in_chans if self.config.in_chans else 3
        self.k = self.config.k if self.config.k else 8
        self.drop_rate = self.config.drop_rate if self.config.drop_rate else 0.0
        self.mlp_ratio = self.config.mlp_ratio if self.config.mlp_ratio  else 2.0
        self.qkv_bias = self.config.qkv_bias if self.config.qkv_bias else False
        self.qk_scale = self.config.qk_scale if self.config.qk_scale else None
        self.attn_drop_rate = self.config.attn_drop_rate if self.config.attn_drop_rate else 0.0
        self.num_heads = self.config.num_heads if self.config.num_heads else 6
        self.rate = self.config.rate
        self.subset = self.config.subset


    def _int_layer(self):
        modulebuilder = ModuleBuilder(self.config)
        self.pos_embed = base_layer(channel_1=self.in_chans, channel_2=128, channel_3=self.trans_dim)
        self.trans_layer = base_layer(channel_1=128, channel_2=self.trans_dim, channel_3=self.trans_dim)
        self.aggregator = modulebuilder.aggregator(mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0.,
                                                   attn_drop_rate=self.attn_drop_rate, num_f=self.num_f)
        self.encoder = modulebuilder.encoder(embed_dim=self.trans_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                                             qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                             drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate, depth=self.depth)
        self.decoder = modulebuilder.decoder(embed_dim=self.trans_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                                             qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                             drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate, depth=self.depth)
        self.generator = Generator(modulebuilder, trans_dim=self.trans_dim, num_query=self.num_query, noise=self.noise)
        self.rebuilder1 = Rebuilder(modulebuilder,self.config.rebuilder1,embed_dim=self.trans_dim//3, fold_step=self.rate[0])
        self.rebuilder2 = Rebuilder(modulebuilder,self.config.rebuilder2,embed_dim=self.trans_dim, np=3, fold_step=self.rate[1])
        self.rebuilder3 = Rebuilder(modulebuilder, self.config.rebuilder3,embed_dim=self.trans_dim, np=3, fold_step=self.rate[2])
        self.conv = Baseconv(self.trans_dim, 512, self.trans_dim)
        self.fusion = Baseconv(self.trans_dim * 2, 512, self.trans_dim, self.trans_dim * 2)
        self.convT = nn.ConvTranspose1d(self.trans_dim, self.trans_dim, 4, 4, bias=False)
        self.up_sampler = nn.Upsample(scale_factor=4)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self,points, gt=None):
        # aggregation
        x = points.transpose(1, 2).contiguous()
        for i, agg in enumerate(self.aggregator):
            if i >= self.knn_layer:
                coor, x, x_ = agg(x, coor, x_)
            else:
                coor, x, x_ = agg(x)
        # coor：Known key points(bx3x128)
        # x:The features of clusters with known key points as center points(bx128xnp)

        # encode
        pos = self.pos_embed(coor).transpose(1, 2)
        v = self.trans_layer(x).transpose(1, 2)
        coor = coor.transpose(1, 2).contiguous()
        knn_index = knn_point(self.k, coor, coor)
        for i, enc in enumerate(self.encoder):
            if i < self.knn_layer:
                v = enc(v + pos, coor, knn_index)
            else:
                v = enc(v + pos, coor)
        v_ = v.transpose(1, 2)
        #v_:The features of clusters with unknown key points as center points(bxnpx128)

        # first rebuild and q generator
        coarse_point_cloud, global_feature = self.rebuilder1(v_)
        q, coarse_point_cloud, denoise_length = self.generator(points,global_feature,coarse_point_cloud)
        #q: The first generation query of clusters with unknown key points as center points (bxnpx384)

        # decoder
        if denoise_length is None:
            self_knn_index = knn_point(self.k, coarse_point_cloud, coarse_point_cloud)
        else:
            self_knn_index = None
        cross_knn_index = knn_point(self.k, coor, coarse_point_cloud)
        Q = []
        for i, dec in enumerate(self.decoder):
            if i < self.knn_layer:
                q = dec(q=q, v=v, q_pos=coarse_point_cloud, v_pos=coor, self_knn_index=self_knn_index,
                        cross_knn_index=cross_knn_index, denoise_length=denoise_length)
            else:
                q = dec(q=q, v=v, q_pos=coarse_point_cloud, v_pos=coor, denoise_length=denoise_length)
            if i == 5 or i == 7:
                Q.append(q)
        # q: The last generation query of clusters with unknown key points as center points (bxnpx384)

        # second rebuild
        B = points.shape[0]
        if self.config.rebuilder2 == 'Transformer':
            output = self.rebuilder2(q=q, coarse_point_cloud=coarse_point_cloud, coor=coor, v=v, points=points)
        elif self.config.rebuilder2 == 'FoldingNet':
            build_points_, building_feature = self.rebuilder2(q=Q[0], coarse_point_cloud=coarse_point_cloud)
            f_1 = self.conv(Q[-1].transpose(1, 2), 'norm')
            f_1 = self.convT(f_1)
            f_2 = self.up_sampler(Q[-1].transpose(1, 2))
            q = self.fusion(torch.cat([f_1, f_2], 1), 'res').transpose(1, 2)
            build_points, _ = self.rebuilder3(q=q, coarse_point_cloud=build_points_.reshape(B, -1, 3), feature_last=building_feature)
            if self.subset in ['MVP']:
                if denoise_length is not None:
                    pred_fine = build_points[:, : -denoise_length*self.rate[-1]].reshape(B, -1, 3).contiguous()
                    pred_medium = build_points_[:, : -denoise_length].reshape(B, -1, 3).contiguous()
                    pred_coarse = coarse_point_cloud[:, : -denoise_length].contiguous()
                    denoised_fine = build_points[:, -denoise_length*self.rate[-1]:].reshape(B, -1, 3).contiguous()
                    denoised_medium = build_points_[:, -denoise_length:].reshape(B, -1, 3).contiguous()
                    denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()
                    output = (pred_coarse,denoised_coarse, denoised_medium, pred_medium, denoised_fine, pred_fine)
                else:
                    build_points_ = build_points_.reshape(B, -1, 3)
                    build_points = build_points.reshape(B, -1, 3)
                    output=(coarse_point_cloud.contiguous(), build_points_.contiguous(), build_points.contiguous())
            elif self.subset in ['ShapeNet-34','ShapeNet-55']:
                build_points_ = build_points_.reshape(B, -1, 3)
                build_points = build_points.reshape(B, -1, 3)
                inp_sparse = fps(points, 128)
                inp_sparse_ = fps(points, build_points.shape[1]//3)
                sparse_pcd = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
                sparse_pcd_ = torch.cat([build_points_, inp_sparse_], dim=1).contiguous()
                completion_points = torch.cat([build_points, points], dim=1).contiguous()
                # sparse_pcd:All center points
                # completion_points:All points
                output = (sparse_pcd, sparse_pcd_, completion_points)

        return output
