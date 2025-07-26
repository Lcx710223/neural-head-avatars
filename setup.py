from setuptools import setup, find_packages
###为了JULES模拟github环境，把COLAB与本仓库的依赖统一如下，LCX20250727。
setup(
    name="nha",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ConfigArgParse",
        ### 原带版本:"numpy==1.21",修改为不带版本的numpy
        "numpy",
        "torch@https://download.pytorch.org/whl/cu113/torch-1.10.2%2Bcu113-cp39-cp39-linux_x86_64.whl",
        "torchvision@https://download.pytorch.org/whl/cu113/torchvision-0.11.3%2Bcu113-cp39-cp39-linux_x86_64.whl",
        "pytorch3d@https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1100/pytorch3d-0.6.1-cp39-cp39-linux_x86_64.whl",
        "matplotlib",
        "tensorboard",
        "scipy",
        "opencv-python",
        ### 原设定 "chumpy",修改为下面的!pip install git+https://github.com/mattloper/chumpy.git
        "face-alignment",
        "face-detection-tflite",
        ### 原设定 "pytorch-lightning==1.2.4",改版本如下
        pytorch-lightning==1.9.5
        "lpips",
        "pytorch_msssim",
        "cpbd@git+https://github.com/wl2776/python-cpbd.git",
        "scikit-learn",
        "torchscope@git+https://github.com/Tramac/torchscope.git",
        "git+https://github.com/mattloper/chumpy.git"
        "jupyter"
    ],
)
