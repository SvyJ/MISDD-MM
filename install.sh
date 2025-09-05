conda create -n misdd-mm python=3.11
source activate misdd-mm
conda install cuda-nvcc=12.1.105 -c nvidia
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
pip install KNN_CUDA-0.2-py3-none-any.whl
pip install ninja
pip install open3d
pip install --upgrade pip
pip install --upgrade setuptools
export CPATH=/home/js/.conda/envs/misdd-mm/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/home/js/.conda/envs/misdd-mm/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/home/js/.conda/envs/misdd-mm/bin:$PATH
pip install Pointnet2_PyTorch/pointnet2_ops_lib/.

pip install loguru
pip install opencv-python
pip install tifffile
pip install scikit-image
pip install seaborn
pip install open_clip_torch