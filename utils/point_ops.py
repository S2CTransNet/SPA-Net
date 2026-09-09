"""Point-cloud FPS / gather / KNN.

Uses pointnet2_ops and KNN_CUDA when they are installed. Otherwise falls back
to PyTorch ops so the model can run on CUDA 13.x / RTX 50-series without those
unmaintained extensions.
"""
import warnings

import torch

try:
    from pointnet2_ops import pointnet2_utils as _pn2
except Exception:
    _pn2 = None

try:
    from knn_cuda import KNN as _CudaKNN
except Exception:
    _CudaKNN = None

if _pn2 is None:
    warnings.warn(
        "pointnet2_ops is not installed; using a PyTorch FPS/gather fallback.",
        RuntimeWarning,
        stacklevel=1,
    )
if _CudaKNN is None:
    warnings.warn(
        "KNN_CUDA is not installed; using a PyTorch cdist/topk fallback.",
        RuntimeWarning,
        stacklevel=1,
    )


def _fps_pytorch(xyz, npoint):
    """xyz: (B, N, 3) -> indices (B, npoint)."""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def _gather_pytorch(features, idx):
    """features: (B, C, N), idx: (B, npoint) -> (B, C, npoint)."""
    C = features.size(1)
    idx = idx.long().unsqueeze(1).expand(-1, C, -1)
    return torch.gather(features, 2, idx)


class _Pointnet2Utils:
    @staticmethod
    def furthest_point_sample(xyz, npoint):
        if _pn2 is not None:
            return _pn2.furthest_point_sample(xyz, npoint)
        return _fps_pytorch(xyz, npoint)

    @staticmethod
    def gather_operation(features, idx):
        if _pn2 is not None:
            return _pn2.gather_operation(features, idx)
        return _gather_pytorch(features, idx)


class _PytorchKNN:
    def __init__(self, k=16, transpose_mode=False):
        self.k = k
        self.transpose_mode = transpose_mode

    def __call__(self, ref, query):
        if self.transpose_mode:
            dist = torch.cdist(query, ref)
            dist, idx = dist.topk(self.k, dim=-1, largest=False)
            return dist, idx
        dist = torch.cdist(
            query.transpose(1, 2).contiguous(),
            ref.transpose(1, 2).contiguous(),
        )
        dist, idx = dist.topk(self.k, dim=-1, largest=False)
        return dist.transpose(1, 2).contiguous(), idx.transpose(1, 2).contiguous()


class KNN:
    def __init__(self, k=16, transpose_mode=False):
        if _CudaKNN is not None:
            self._impl = _CudaKNN(k=k, transpose_mode=transpose_mode)
        else:
            self._impl = _PytorchKNN(k=k, transpose_mode=transpose_mode)

    def __call__(self, ref, query):
        return self._impl(ref, query)


pointnet2_utils = _Pointnet2Utils()
