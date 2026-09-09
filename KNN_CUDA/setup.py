import os
from setuptools import setup, find_packages

version = {}
with open("knn_cuda/version.py") as fp:
    exec(fp.read(), version)

with open('requirements.txt') as f:
    required = f.read().splitlines()

os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0")
setup(
    name='KNN_CUDA',
    version=version['__version__'],
    description='pytorch version knn support cuda.',
    author='Shuaipeng Li',
    author_email='sli@mail.bnu.edu.cn',
    packages=find_packages(),
    package_data={
        'knn_cuda': ["csrc/cuda/knn.cu", "csrc/cuda/knn.cpp"]
    },  
    install_requires=required
)

