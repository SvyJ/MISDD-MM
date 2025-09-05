import glob
import os
import random

mvtec3d_classes = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                   'foam', 'peach', 'potato', 'rope', 'tire']


MVTEC3D_DIR = '../anomaly_detection/mvtec3d'


def load_mvtec3d(category, k_shot, missing_type, missing_rate=0.3):
    def load_phase(root_path):
        img_tot_paths = []
        pc_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/rgb/*.png")
                pc_path = glob.glob(os.path.join(root_path, defect_type) + "/xyz/*.tiff")
                img_tot_paths.extend(img_paths)
                pc_tot_paths.extend(pc_path)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/rgb/*.png")
                pc_paths = glob.glob(os.path.join(root_path, defect_type) + "/xyz/*.tiff")
                gt_paths = glob.glob(os.path.join(root_path, defect_type) + "/gt/*.png")
                img_paths.sort()
                pc_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                pc_tot_paths.extend(pc_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths) and len(pc_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, pc_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in mvtec3d_classes

    train_path = os.path.join(MVTEC3D_DIR, category, 'train')
    test_path = os.path.join(MVTEC3D_DIR, category, 'test')

    train_img_tot_paths, train_pc_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = load_phase(train_path)
    test_img_tot_paths, test_pc_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = load_phase(test_path)

    train_tot_missing_indxs = missing_setting(len(train_img_tot_paths), missing_type, missing_rate)
    test_tot_missing_indxs = missing_setting(len(test_img_tot_paths), missing_type, missing_rate)

    return (train_img_tot_paths, train_pc_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types, train_tot_missing_indxs), \
           (test_img_tot_paths, test_pc_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types, test_tot_missing_indxs)


def missing_setting(length, missing_type, missing_rate):
    tot_missing_indx = [0]*length
    missing_indices = random.sample(range(length), int(length * missing_rate))
    if missing_type == 'img':
        for idx in missing_indices:
            tot_missing_indx[idx] = 1
    elif missing_type == 'depth':
        for idx in missing_indices:
            tot_missing_indx[idx] = 2
    elif missing_type == 'both':
        for idx in missing_indices:
            tot_missing_indx[idx] = 1
        missing_indices = random.sample(missing_indices, len(missing_indices) // 2)
        for idx in missing_indices:
            tot_missing_indx[idx] = 2

    return tot_missing_indx