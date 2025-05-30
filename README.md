<br>

### Introduction
![Image](https://github.com/S2CTransNet/SPA-Net/tree/main/fig/pipeline.png)
This project is a demo version, used for testing and verifying the effectiveness of SPA-Net. The Complete version will be uploaded after acceptance.

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

- h5py==3.12.1
- numpy==1.24.3
- open3d==0.18.0
- plotly==5.24.1
- setuptools==75.1.0
- timm==1.0.12
- torch==1.13.0
- torchvision==0.14.0
- tqdm==4.66.5
- transforms3d==0.4.2
- ninja==1.11.1.3
- python == 3.10
- CUDA == 11.7
```
pip install -r requirements.txt

# Suggest using the following command to install torch and torchvision
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```
In addition, pointnet2_ops and KNN_CUDA are also necessary.
```
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
Chamfer-dist and EMD are used for metric evaluation, and the source code is from [PoinTR](https://github.com/yuxumin/PoinTr/tree/master).
```
# Chamfer Distance and EMD
bash install.sh
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

