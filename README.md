<br>

### Introduction
![Image](https://github.com/S2CTransNet/SPA-Net/tree/main/fig/pipeline.png)
This repository serves as the inference implementation for validating the paper's claims, with the latest inference outputs preserved in the result directory.

This branch targets recent NVIDIA GPUs (for example RTX 50-series). For older GPUs, use the **main** branch.
### Structure

Datasets download links:
[MVP](https://mvp-dataset.github.io/MVP/Registration.html),
[ShapeNet-55/34](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md ), The _data_ contains ten KITTI vehicle point clouds for testing.

After downloading the dataset, place it according to the following directory:
```
|-- SPA-Net
    |-- data
        |-- KITTI
            |-- README.tex
            |--partial_0.npy
            |--partial_1.npy
            |--...
        |-- ShapeNet
            |-- shapenet_pc
        |-- MVP_Benchmark
            |-- Completion
   
```
We uploaded the final test results including all categories of datasets stored in _.csv_ format. You can found it in _results_.
```
|-- SPA-Net
    |-- results
        |-- MVP
            |-- Overall.csv
        |-- ShapeNet-55
            |-- Hard.csv
            |-- Medium.csv
            |-- AVG.csv
            |-- Simple.csv
        |-- ShapeNet-34
            |-- Unseen-21
                |-- Hard.csv
                |-- Medium.csv
                |-- AVG.csv
                |-- Simple.csv
            |-- Seen-34
                |-- Hard.csv
                |-- Medium.csv
                |-- AVG.csv
                |-- Simple.csv
```
We also upload four pretrained weights, which you can download on [Google Drive](https://drive.google.com/file/d/1fUr3C1xoc4PtUV5UAvO0zUct7r-o6anT/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1V56LM15zNZ4fppC73QoHNQ?pwd=wxs5) (password:wxs5).  
```
|-- SPA-Net
    |-- pointnet2_ops
    |-- KNN_CUDA
    |-- weight
        |-- KITTI
            |-- KITTI_best.pth #pretraned on other dataset
        |-- MVP
            |-- MVP_best.pth
        |-- ShapeNet-55
            |-- ShapeNet-55_best.pth
        |-- ShapeNet-34
            |-- ShapeNet-34_best.pth
```

### Requirement

This branch is for newer GPUs. Older GPUs should use **main**.

Need Python 3.10, an NVIDIA driver that supports the installed GPU, `git`, and `g++` (`sudo apt install -y build-essential git`).

```bash
conda create -n spanet python=3.10 -y
conda activate spanet
cd /path/to/SPA-Net
bash install.sh
```

`install.sh` installs PyTorch (CUDA wheels from pytorch.org), Python deps, the CUDA compiler headers, then Chamfer / EMD / PointNet++ / KNN_CUDA. It prefers `./pointnet2_ops` and `./KNN_CUDA` (a local `.whl` or source tree). If those directories are missing, it clones them.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_available())"
python -c "from utils.point_ops import _pn2, _CudaKNN; print('pointnet2', _pn2 is not None, 'knn_cuda', _CudaKNN is not None); import chamfer; print('chamfer ok')"
```
### Evaluation

To evaluate a pre-trained SPA-Net on the these dataset with single GPU, run:

```
# It is necessary to create a cfg files.
python cfgs/create_cfgs.py

# Evaluate on KITTI
python main.py --dataset_name KITTI

# Evaluate on ShapeNet-55
python main.py --dataset_name ShapeNet-55

# Evaluate on ShapeNet-34 with Seen type
python main.py --seen_type Seen-34 --dataset_name ShapeNet-34

```
### Check Results
After evaluation, you can go to the _results_ directory to view the results, or use _tool.draw()_ to draw point clouds in real time. If you want to have detailed results, run:
```
# Click the generated link to view the results, which are usually: http://localhost:6006/
tensorboard --logdir=logs
```
### Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@ARTICLE{SPA-Net,
  author={Qin, Xiaofei and Yi, Anluo and Wang, Wei and He, Changxiang and Wang, Lin and Tao, Shiwei and Zhang, Xuedian and Dong, Qiulei},
  journal={IEEE Transactions on Multimedia}, 
  title={SPA-Net: Skeletal-to-Whole Point Cloud Completion Via Progressive Off-attention Neighboring Feature Aggregation}, 
  year={2026},
  doi={10.1109/TMM.2026.3718221}
}
