export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4

python test_cls.py --dataset mvtec3d --class_name bagel
python test_cls.py --dataset mvtec3d --class_name cable_gland
python test_cls.py --dataset mvtec3d --class_name carrot
python test_cls.py --dataset mvtec3d --class_name cookie
python test_cls.py --dataset mvtec3d --class_name dowel
python test_cls.py --dataset mvtec3d --class_name foam
python test_cls.py --dataset mvtec3d --class_name peach
python test_cls.py --dataset mvtec3d --class_name potato
python test_cls.py --dataset mvtec3d --class_name rope
python test_cls.py --dataset mvtec3d --class_name tire
