export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4

python test_seg.py --dataset mvtec3d --class_name bagel --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name cable_gland --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name carrot --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name cookie --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name dowel --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name foam --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name peach --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name potato --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name rope --missing_type both --missing_rate 0.3
python test_seg.py --dataset mvtec3d --class_name tire --missing_type both --missing_rate 0.3
