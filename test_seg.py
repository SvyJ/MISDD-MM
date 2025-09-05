import argparse

import torch.optim.lr_scheduler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from MISDD_MM import *
from utils.eval_utils import *
from torchvision import transforms
from tqdm import tqdm

TASK = 'SEG'

def test(model,
        args,
        dataloader: DataLoader,
        device: str,
        img_dir: str,
        check_path: str,
        ):

    # change the model into eval mode
    model.eval_mode()

    model.load_state_dict(torch.load(check_path), strict=False)

    score_maps = []
    test_imgs = []
    test_depths = []
    gt_mask_list = []
    names = []

    for (img, pc, depth, mask, label, name, img_type, missing_flag) in tqdm(dataloader, ncols=100, desc=f'{args.dataset}/{args.class_name}, missing {args.missing_type}-{args.missing_rate}, testing'):

        img = [model.img_transform(Image.fromarray(f.numpy())) for f in img]
        # pc = [p for p in pc]
        depth = [d for d in depth]
        img = torch.stack(img, dim=0)
        # pc = torch.stack(pc, dim=0)
        depth = torch.stack(depth, dim=0)
        all_prompts_image, all_prompts_depth = model.missing_prompt_learner(missing_flag)

        for d, t, n, l, m in zip(img, depth, name, label, mask):
            test_imgs += [denormalization(d.cpu().numpy())]
            test_depths += [denormalization_depth(t.cpu().numpy())]
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_mask_list += [m]

        img = img.to(device)
        # pc = pc.to(device)
        depth = depth.to(device)
        # score_map = model(args, img, pc, 'seg')
        score_map = model(args, img, depth, 'seg', all_prompts_image, all_prompts_depth, missing_flag)
        score_maps += score_map
        print(len(score_map))
        score_map = score_map[0]*255
        score_map = score_map.astype(np.uint8)
        score_map = cv2.resize(score_map, (400, 400), interpolation=cv2.INTER_CUBIC)
        print(score_map.shape, np.unique(score_map))
        cv2.imwrite(os.path.join(img_dir, f'{n}_score.jpg'), score_map)
        print(os.path.join(img_dir, f'{n}_score.jpg'))

    # test_imgs, test_depths, score_maps, gt_mask_list = specify_resolution(test_imgs, test_depths, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
    # result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)

    # torch.save(model.state_dict(), check_path)
    # if args.vis:
        # plot_sample_cv2(names, test_imgs, test_depths, {'MISDD_MM': score_maps}, gt_mask_list, save_folder=img_dir)

    # return result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    
    train_dataloader, train_dataset_inst, train_dataset_len = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst, test_dataset_len = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    kwargs['size'] = train_dataset_len

    # get the model
    model = MISDD_MM(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    test(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path)

    # p_roc = round(metrics['p_roc'], 2)
    # pro_auc = round(metrics['pro_auc'], 2)
    # object = kwargs['class_name']
    # print(f'Object:{object} =========================== Pixel-AUROC:{p_roc}, AUPRO:{pro_auc}\n')

    # save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
    #             kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'mvtec3d'])
    parser.add_argument('--class_name', type=str, default='transistor')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument('--missing_type', type=str, default='both')
    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--img_lambda", type=float, default=0.5)
    parser.add_argument("--pc_lambda", type=float, default=0.5)
    parser.add_argument("--depth_lambda", type=float, default=0.5)
    parser.add_argument("--missing_prompt_length", type=int, default=36)
    parser.add_argument("--missing_prompt_depth", type=int, default=6)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
