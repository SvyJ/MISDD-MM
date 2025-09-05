from builtins import len
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
import random
import time
import itertools
import wandb
from tqdm import tqdm

TASK = 'CLS'

def save_check_point(model, path):
    selected_keys = [
        'img_feature_gallery1',
        'img_feature_gallery2',
        'pc_feature_gallery1',
        'pc_feature_gallery2',
        'depth_feature_gallery1',
        'depth_feature_gallery2',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)
    print(f"Model saved to {path}")

def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        check_path: str,
        train_data: DataLoader,
        ):

    # change the model into eval mode
    model.eval_mode()

    img_features1 = []
    img_features2 = []
    # pc_features1 = []
    # pc_features2 = []
    depth_features1 = []
    depth_features2 = []
    for (img, pc, depth, mask, label, name, img_type, missing_flag) in tqdm(train_data, ncols=100, desc=f'{args.dataset}/{args.class_name}, missing {args.missing_type}-{args.missing_rate}, building feature gallery:'):

        img = [model.img_transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in img]
        # pc = [p for p in pc]
        depth = [d for d in depth]
        img = torch.stack(img, dim=0).to(device)
        # pc = torch.stack(pc, dim=0).to(device)
        depth = torch.stack(depth, dim=0).to(device)
        _, _, img_feature_map1, img_feature_map2 = model.encode_image(img)
        # _, _, pc_feature_map1, pc_feature_map2 = model.encode_pc(pc)
        _, _, depth_feature_map1, depth_feature_map2 = model.encode_image(depth)

        img_features1.append(img_feature_map1)
        img_features2.append(img_feature_map2)
        # pc_features1.append(pc_feature_map1)
        # pc_features2.append(pc_feature_map2)
        depth_features1.append(depth_feature_map1)
        depth_features2.append(depth_feature_map2)

    img_features1 = torch.cat(img_features1, dim=0)
    img_features2 = torch.cat(img_features2, dim=0)
    # pc_features1 = torch.cat(pc_features1, dim=0)
    # pc_features2 = torch.cat(pc_features2, dim=0)
    depth_features1 = torch.cat(depth_features1, dim=0)
    depth_features2 = torch.cat(depth_features2, dim=0)
    model.build_image_feature_gallery(img_features1, img_features2)
    # model.build_pc_feature_gallery(pc_features1, pc_features2)
    model.build_depth_feature_gallery(depth_features1, depth_features2)

    all_params = itertools.chain(model.img_prompt_learner.parameters(), model.depth_prompt_learner.parameters(), model.missing_prompt_learner.parameters())

    optimizer = torch.optim.SGD(all_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    best_result_dict = None
    for epoch in range(args.Epoch):
        # desc=f'Epoch {args.dataset}/{args.class_name}, {args.k-shot}'
        # TRAIN
        for (img, pc, depth, mask, label, name, img_type, missing_flag) in tqdm(train_data, ncols=100, desc=f'{args.dataset}/{args.class_name}, missing {args.missing_type}-{args.missing_rate}, Epoch {epoch}/{args.Epoch}, training'):
            img = [model.img_transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in img]
            # pc = [p for p in pc]
            depth = [d for d in depth]
            # print(type(img), len(img))
            # print(type(pc), len(pc))
            # print(type(depth), len(depth))
            img = torch.stack(img, dim=0).to(device)
            # pc = torch.stack(pc, dim=0).to(device)
            depth = torch.stack(depth, dim=0).to(device)

            # img = img[0:1, :, :, :].to(device)
            img = img.to(device)
            # pc = pc.to(device)
            depth = depth.to(device)

            # normal_text_prompt, abnormal_text_prompt_manual, abnormal_text_prompt_learned = model.prompt_learner()
            img_normal_text_prompt, img_abnormal_text_prompt_manual, img_abnormal_text_prompt_learned = model.img_prompt_learner()
            depth_normal_text_prompt, depth_abnormal_text_prompt_manual, depth_abnormal_text_prompt_learned = model.depth_prompt_learner()
            all_prompts_image, all_prompts_depth = model.missing_prompt_learner(missing_flag)

            optimizer.zero_grad()

            img_feature, _, _, _ = model.encode_image_missing(img, all_prompts_image, missing_flag)
            img_normal_text_features = model.encode_text_embedding(img_normal_text_prompt, model.img_tokenized_normal_prompts)
            img_abnormal_text_features_manual = model.encode_text_embedding(img_abnormal_text_prompt_manual, model.img_tokenized_abnormal_prompts_manual)
            img_abnormal_text_features_learned = model.encode_text_embedding(img_abnormal_text_prompt_learned, model.img_tokenized_abnormal_prompts_learned)
            img_abnormal_text_features = torch.cat([img_abnormal_text_features_manual, img_abnormal_text_features_learned], dim=0)
            img_mean_ad_manual = torch.mean(F.normalize(img_abnormal_text_features_manual, dim=-1), dim=0)
            img_mean_ad_learned = torch.mean(F.normalize(img_abnormal_text_features_learned, dim=-1), dim=0)
            img_loss_match_abnormal = (img_mean_ad_manual - img_mean_ad_learned).norm(dim=0) ** 2.0
            img_normal_text_features_ahchor = img_normal_text_features.mean(dim=0).unsqueeze(0)
            img_normal_text_features_ahchor = img_normal_text_features_ahchor / img_normal_text_features_ahchor.norm(dim=-1, keepdim=True)
            img_abnormal_text_features_ahchor = img_abnormal_text_features.mean(dim=0).unsqueeze(0)
            img_abnormal_text_features_ahchor = img_abnormal_text_features_ahchor / img_abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            img_abnormal_text_features = img_abnormal_text_features / img_abnormal_text_features.norm(dim=-1, keepdim=True)
            img_l_pos = torch.einsum('nc,cm->nm', img_feature, img_normal_text_features_ahchor.transpose(0, 1))
            img_l_neg_v2t = torch.einsum('nc,cm->nm', img_feature, img_abnormal_text_features.transpose(0, 1))
            img_logits_v2t = torch.cat([img_l_pos, img_l_neg_v2t], dim=-1) * model.model.logit_scale
            img_target_v2t = torch.zeros([img_logits_v2t.shape[0]], dtype=torch.long).to(device)
            img_loss_v2t = criterion(img_logits_v2t, img_target_v2t)
            img_trip_loss = criterion_tip(img_feature, img_normal_text_features_ahchor, img_abnormal_text_features_ahchor)
            img_loss = img_loss_v2t + img_trip_loss + img_loss_match_abnormal * args.lambda1

            depth_feature, _, _, _ = model.encode_image_missing(depth, all_prompts_depth, missing_flag)
            depth_normal_text_features = model.encode_text_embedding(depth_normal_text_prompt, model.depth_tokenized_normal_prompts)
            depth_abnormal_text_features_manual = model.encode_text_embedding(depth_abnormal_text_prompt_manual, model.depth_tokenized_abnormal_prompts_manual)
            depth_abnormal_text_features_learned = model.encode_text_embedding(depth_abnormal_text_prompt_learned, model.depth_tokenized_abnormal_prompts_learned)
            depth_abnormal_text_features = torch.cat([depth_abnormal_text_features_manual, depth_abnormal_text_features_learned], dim=0)
            depth_mean_ad_manual = torch.mean(F.normalize(depth_abnormal_text_features_manual, dim=-1), dim=0)
            depth_mean_ad_learned = torch.mean(F.normalize(depth_abnormal_text_features_learned, dim=-1), dim=0)
            depth_loss_match_abnormal = (depth_mean_ad_manual - depth_mean_ad_learned).norm(dim=0) ** 2.0
            depth_normal_text_features_ahchor = depth_normal_text_features.mean(dim=0).unsqueeze(0)
            depth_normal_text_features_ahchor = depth_normal_text_features_ahchor / depth_normal_text_features_ahchor.norm(dim=-1, keepdim=True)
            depth_abnormal_text_features_ahchor = depth_abnormal_text_features.mean(dim=0).unsqueeze(0)
            depth_abnormal_text_features_ahchor = depth_abnormal_text_features_ahchor / depth_abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            depth_abnormal_text_features = depth_abnormal_text_features / depth_abnormal_text_features.norm(dim=-1, keepdim=True)
            depth_l_pos = torch.einsum('nc,cm->nm', depth_feature, depth_normal_text_features_ahchor.transpose(0, 1))
            depth_l_neg_v2t = torch.einsum('nc,cm->nm', depth_feature, depth_abnormal_text_features.transpose(0, 1))
            depth_logits_v2t = torch.cat([depth_l_pos, depth_l_neg_v2t], dim=-1) * model.model.logit_scale
            depth_target_v2t = torch.zeros([depth_logits_v2t.shape[0]], dtype=torch.long).to(device)
            depth_loss_v2t = criterion(depth_logits_v2t, depth_target_v2t)
            depth_trip_loss = criterion_tip(depth_feature, depth_normal_text_features_ahchor, depth_abnormal_text_features_ahchor)
            depth_loss = depth_loss_v2t + depth_trip_loss + depth_loss_match_abnormal * args.lambda1
            
            loss = args.img_lambda * img_loss + args.depth_lambda * depth_loss

            wandb.log({
                'loss': loss.item(), 
                'img_loss_v2t': img_loss_v2t.item(),
                'img_trip_loss': img_trip_loss.item(), 
                'depth_loss_v2t': depth_loss_v2t.item(), 
                'depth_trip_loss': depth_trip_loss.item(), 
                'img_loss_match_abnormal': img_loss_match_abnormal.item(),
                'depth_loss_match_abnormal': depth_loss_match_abnormal.item()
            })

            loss.backward()
            optimizer.step()
        scheduler.step()
        model.build_img_text_feature_gallery()
        model.build_depth_text_feature_gallery()

        # TEST
        scores_img = []
        score_maps = []
        test_imgs = []
        test_depths = []
        gt_list = []
        gt_mask_list = []
        names = []
        for (img, pc, depth, mask, label, name, img_type, missing_flag) in tqdm(dataloader, ncols=100, desc=f'{args.dataset}/{args.class_name}, missing {args.missing_type}-{args.missing_rate}, Epoch {epoch}/{args.Epoch}, testing'):

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
                l = l.numpy()
                m = m.numpy()
                m[m > 0] = 1

                names += [n]
                gt_list += [l]
                gt_mask_list += [m]

            img = img.to(device)
            # pc = pc.to(device)
            depth = depth.to(device)
            # score_img, score_map = model(args, img, pc, 'cls')
            score_img, score_map = model(args, img, depth, 'cls', all_prompts_image, all_prompts_depth, missing_flag)
            score_maps += score_map
            scores_img += score_img

        test_imgs, test_depths, score_maps, gt_mask_list = specify_resolution(test_imgs, test_depths, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
        result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))

        if best_result_dict is None:
            print(f'===========================Image-AUROC: {result_dict["i_roc"]}')
            save_check_point(model, check_path)
            best_result_dict = result_dict

        elif best_result_dict['i_roc'] < result_dict['i_roc']:
            print(f'===========================Image-AUROC: {result_dict["i_roc"]}')
            save_check_point(model, check_path)
            best_result_dict = result_dict

        wandb.log({
            'Image-AUROC': best_result_dict['i_roc'],
        })

    return best_result_dict


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

    wandb.init(
        project = 'Prompt-RGB_DEPTH-CLS-Missing-V3',
        name = f'{args.dataset}-{args.class_name}-{args.missing_type}-{args.missing_rate}-{args.seed}-{args.img_lambda}-{args.pc_lambda}-{time.time()}',
    )

    # prepare the experiment dir
    _, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the train dataloader
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
    metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader)

    i_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Image-AUROC:{i_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'mvtec3d'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=1)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument('--missing_type', type=str, default='both')
    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=50)
    parser.add_argument("--img_lambda", type=float, default=0.5)
    parser.add_argument("--pc_lambda", type=float, default=0.5)
    parser.add_argument("--depth_lambda", type=float, default=0.5)
    parser.add_argument("--missing_prompt_length", type=int, default=36)
    parser.add_argument("--missing_prompt_depth", type=int, default=6)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
