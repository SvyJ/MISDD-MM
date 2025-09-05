from builtins import len
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from .dataset import CLIPDataset
from .mvtec3d import load_mvtec3d, mvtec3d_classes


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec3d': load_mvtec3d,
}

dataset_classes = {
    'mvtec3d': mvtec3d_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def denormalization_depth(x):
    x = ((x.transpose(1, 2, 0)) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, **kwargs):

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        missing_type=kwargs['missing_type'],
        missing_rate=kwargs['missing_rate'],
    )

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True, num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=1, shuffle=False, num_workers=0)

    return data_loader, dataset_inst, len(dataset_inst)