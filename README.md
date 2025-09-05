# [Resilient Multimodal Industrial Surface Defect Detection with Uncertain Sensors Availability](https://svyj.github.io/MISDD-MM/)

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2509.02962)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)


<hr style="border: 2px solid gray;"></hr>

## Quick Installation

- Run the following script in sequence or execute `bash install.sh` to quickly install the required packages

```bash
conda create -n promptad python=3.11
source activate promptad
conda install cuda-nvcc=12.1.105 -c nvidia
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
pip install KNN_CUDA-0.2-py3-none-any.whl
pip install ninja
pip install open3d
pip install --upgrade pip
pip install --upgrade setuptools
export CPATH=/home/js/.conda/envs/promptad/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/home/js/.conda/envs/promptad/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/home/js/.conda/envs/promptad/bin:$PATH
pip install Pointnet2_PyTorch/pointnet2_ops_lib/.
pip install loguru
pip install opencv-python
pip install tifffile
pip install scikit-image
pip install seaborn
pip install open_clip_torch
```

## Download Dataset

- Download the MVTec 3D-AD dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) and decompress it, keep the source folder structure unchanged
- Specify the dataset path `MVTEC3D_DIR` in `datasets/mvtec3d.py`

## Train

- For defect detection, run `bash scripts/run_train_cls.sh`

```bash
python train_cls.py --dataset mvtec3d --class_name bagel --missing_type both --missing_rate 0.3
```

- For defect segmentation, run `bash scripts/run_train_seg.sh`

```bash
python train_cls.py --dataset mvtec3d --class_name bagel --missing_type both --missing_rate 0.3
```

## Evaluation

For defect detection, run `bash scripts/run_test_cls.sh`

```bash
python test_cls.py --dataset mvtec3d --class_name bagel
```

For defect segmentation, run `bash scripts/run_test_seg.sh`

```bash
python test_seg.py --dataset mvtec3d --class_name bagel --missing_type both --missing_rate 0.3
```

## Citation

```tex
@article{jiang2025category,
  title={Resilient Multimodal Industrial Surface Defect Detection with Uncertain Sensors Availability},
  author={Jiang, Shuai and Ma, Yunfeng and Zhou, Jingyu and Bian, Yuan and Wang, Yaonan and Liu, Min},
  journal={IEEE/ASME Transactions on Mechatronics},
  year={2025},
  publisher={IEEE}
}

@article{jiang2025category,
  title={Resilient Multimodal Industrial Surface Defect Detection with Uncertain Sensors Availability},
  author={Jiang, Shuai and Ma, Yunfeng and Zhou, Jingyu and Bian, Yuan and Wang, Yaonan and Liu, Min},
  journal={arXiv preprint arXiv:2509.02962},
  year={2025}
}
```

## Thanks

Our repository is built on excellent works include  [PromptAD](https://github.com/FuNz-0/PromptAD), [CLIPAD](https://github.com/ByChelsea/CLIP-AD), and [WinCLIP](https://github.com/caoyunkang/WinClip).

