import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils.mvtec3d_utils import read_tiff_organized_pc, organized_pc_to_depth_map, resize_organized_pc


class CLIPDataset(Dataset):
    def __init__(self, load_function, category, phase, k_shot, missing_type, missing_rate):

        self.load_function = load_function
        self.phase = phase

        self.category = category

        # load datasets
        self.img_paths, self.pc_paths, self.gt_paths, self.labels, self.types, self.missing_indxs = self.load_dataset(k_shot, missing_type, missing_rate)  # self.labels => good : 0, anomaly : 1

    def load_dataset(self, k_shot, missing_type, missing_rate):

        (train_img_tot_paths, train_pc_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types, train_tot_missing_indxs), \
        (test_img_tot_paths, test_pc_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types, test_tot_missing_indxs) = self.load_function(self.category, k_shot, missing_type, missing_rate)
        if self.phase == 'train':
            return train_img_tot_paths, train_pc_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types, train_tot_missing_indxs
        else:
            return test_img_tot_paths, test_pc_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types, test_tot_missing_indxs

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, pc_path, gt, label, img_type = self.img_paths[idx], self.pc_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        organized_pc = read_tiff_organized_pc(pc_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel, target_height=240, target_width=240)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=240, target_width=240)
        resized_organized_pc = resized_organized_pc.clone().detach().float()

        if gt == 0:
            gt = np.zeros([img.shape[0], img.shape[0]])
        else:
            gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255

        img = cv2.resize(img, (240, 240))
        gt = cv2.resize(gt, (240, 240), interpolation=cv2.INTER_NEAREST)

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        if self.missing_indxs[idx] == 1:
            img = np.zeros_like(img)
        elif self.missing_indxs[idx] == 2:
            resized_depth_map_3channel = np.zeros_like(resized_depth_map_3channel)

        return img, torch.Tensor(resized_organized_pc), torch.Tensor(resized_depth_map_3channel), gt, label, img_name, img_type, self.missing_indxs[idx]
