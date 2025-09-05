from builtins import print
import torch
import random
import copy
import torch.nn as nn
import numpy as np
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import *
from PIL import Image
from scipy.ndimage import gaussian_filter
from utils.mvtec3d_utils import organized_pc_to_unorganized_pc
from utils.pointnet2_utils import interpolating_points

from .CLIPAD import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()   # local tokenizer, no padding, no sos, no eos

valid_backbones = ['ViT-B-16-plus-240', "ViT-B-16"]
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]


def _convert_to_rgb(image):
    return image.convert('RGB')


class PromptLearner(nn.Module):
    def __init__(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre):
        super().__init__()

        if pre == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        state_anomaly1 = state_anomaly + class_state_abnormal[classname]

        if classname in class_mapping:
            classname = class_mapping[classname]

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        normal_ctx_vectors = torch.empty(n_pro, n_ctx, ctx_dim, dtype=dtype)
        abnormal_ctx_vectors = torch.empty(n_pro_ab, n_ctx_ab, ctx_dim, dtype=dtype)

        nn.init.normal_(normal_ctx_vectors, std=0.02)
        nn.init.normal_(abnormal_ctx_vectors, std=0.02)

        normal_prompt_prefix = " ".join(["N"] * n_ctx)
        abnormal_prompt_prefix = " ".join(["A"] * n_ctx_ab)

        self.normal_ctx = nn.Parameter(normal_ctx_vectors)  # to be optimized
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)  # to be optimized

        # normal prompt
        normal_prompts = [normal_prompt_prefix + " " + classname + "." for _ in range(n_pro)]

        # abnormal prompt
        self.n_ab_manual = len(state_anomaly1)
        abnormal_prompts_manual = [normal_prompt_prefix + " " + state.format(classname) + "." for state in state_anomaly1 for _ in range(n_pro)]
        abnormal_prompts_learned = [normal_prompt_prefix + " " + classname + " " + abnormal_prompt_prefix + "." for _ in range(n_pro_ab) for _ in range(n_pro)]

        # abnormal_prompts = abnormal_prompts_learned + abnormal_prompts_manual
        # print("Normal prompts: ", normal_prompts)
        # print("Abnormal prompts Manual: ", abnormal_prompts_manual)
        # print("Abnormal prompts Learned: ", abnormal_prompts_learned)

        tokenized_normal_prompts = CLIPAD.tokenize(normal_prompts)
        tokenized_abnormal_prompts_manual = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_manual])
        tokenized_abnormal_prompts_learned = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_learned])

        with torch.no_grad():
            normal_embedding = clip_model.token_embedding(tokenized_normal_prompts).type(dtype)
            abnormal_embedding_manual = clip_model.token_embedding(tokenized_abnormal_prompts_manual).type(dtype)
            abnormal_embedding_learned = clip_model.token_embedding(tokenized_abnormal_prompts_learned).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("normal_token_prefix", normal_embedding[:, :1, :])  # SOS
        self.register_buffer("normal_token_suffix", normal_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_manual", abnormal_embedding_manual[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_manual", abnormal_embedding_manual[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_learned", abnormal_embedding_learned[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_learned", abnormal_embedding_learned[:, 1 + n_ctx + n_ctx_ab:, :])  # CLS, EOS

        self.n_pro = n_pro
        self.n_ctx = n_ctx
        self.n_pro_ab = n_pro_ab
        self.n_ctx_ab = n_ctx_ab
        self.tokenized_normal_prompts = tokenized_normal_prompts  # torch.Tensor
        self.tokenized_abnormal_prompts_manual = tokenized_abnormal_prompts_manual  # torch.Tensor
        self.tokenized_abnormal_prompts_learned = tokenized_abnormal_prompts_learned  # torch.Tensor
        # self.tokenized_abnormal_prompts = torch.cat([tokenized_abnormal_prompts_manual, tokenized_abnormal_prompts_learned], dim=0)
        # self.tokenized_abnormal_prompts = tokenized_abnormal_prompts_manual
        # self.name_lens = name_lens

    def forward(self):

        # learned normal prompt
        normal_ctx = self.normal_ctx

        normal_prefix = self.normal_token_prefix
        normal_suffix = self.normal_token_suffix

        normal_prompts = torch.cat(
            [
                normal_prefix,  # (n_pro, 1, dim)
                normal_ctx,     # (n_pro, n_ctx, dim)
                normal_suffix,  # (n_pro, *, dim)
            ],
            dim=1,
        )

        # manual abnormal prompt
        n_ab_manual = self.n_ab_manual

        n_pro, n_ctx, dim = normal_ctx.shape
        normal_ctx1 = normal_ctx.unsqueeze(0).expand(n_ab_manual, -1, -1, -1).reshape(-1, n_ctx, dim)

        abnormal_prefix_manual = self.abnormal_token_prefix_manual
        abnormal_suffix_manual = self.abnormal_token_suffix_manual

        abnormal_prompts_manual = torch.cat(
            [
                abnormal_prefix_manual,     # (n_pro * n_ab_manual, 1, dim)
                normal_ctx1,                # (n_pro * n_ab_manual, n_ctx, dim)
                abnormal_suffix_manual,     # (n_pro * n_ab_manual, *, dim)
            ],
            dim=1,
        )

        # learned abnormal prompt
        abnormal_prefix_learned = self.abnormal_token_prefix_learned
        abnormal_suffix_learned = self.abnormal_token_suffix_learned
        abnormal_ctx = self.abnormal_ctx
        n_pro_ad, n_ctx_ad, dim_ad = abnormal_ctx.shape
        normal_ctx2 = normal_ctx.unsqueeze(0).expand(self.n_pro_ab, -1, -1, -1).reshape(-1, n_ctx, dim)
        abnormal_ctx = abnormal_ctx.unsqueeze(0).expand(self.n_pro, -1, -1, -1).reshape(-1, n_ctx_ad, dim_ad)

        abnormal_prompts_learned = torch.cat(
            [
                abnormal_prefix_learned,        # (n_pro * n_pro_ab, 1, dim)
                normal_ctx2,                    # (n_pro * n_pro_ab, n_ctx, dim)
                abnormal_ctx,                   # (n_pro * n_pro_ab, n_ctx_ab, dim)
                abnormal_suffix_learned,        # (n_pro * n_pro_ab, *, dim)
            ],
            dim=1,
        )

        # abnormal_prompts = torch.cat([abnormal_prompts_manual, abnormal_prompts_learned], dim=0)
        # abnormal_prompts = abnormal_prompts_manual

        return normal_prompts, abnormal_prompts_manual, abnormal_prompts_learned

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Missing_PromptLearner(nn.Module):
    def __init__(self, prompt_length, prompt_depth):
        super().__init__()
        prompt_length_half = prompt_length//3 # use half length for generating static prompts, and the other for generating dynamic prompts
        # Default is 1, which is compound shallow prompting
        self.prompt_depth = prompt_depth  # max=12, but will create 11 such shared prompts
        self.image_prompt_complete  = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 896, dtype=torch.float32), std=0.02))
        self.image_prompt_missing   = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 896, dtype=torch.float32), std=0.02))
        self.depth_prompt_complete  = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 896, dtype=torch.float32), std=0.02))
        self.depth_prompt_missing   = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 896, dtype=torch.float32), std=0.02))
        self.common_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 896, dtype=torch.float32), std=0.02))
        self.common_prompt_image    = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 896, dtype=torch.float32), std=0.02))
        self.common_prompt_depth    = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 896, dtype=torch.float32), std=0.02))
        # Also make corresponding projection layers, for each prompt
        embed_dim_depth = 896
        embed_dim_image = 896
        embed_dim = embed_dim_depth + embed_dim_image
        r = 16
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_depth),
                )
        self.compound_prompt_projections_depth = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_depth = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_image),
                )
        self.compound_prompt_projections_image = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        self.common_prompt_projection_image = nn.Sequential(
                nn.Linear(embed_dim_depth, embed_dim_depth//r),
                nn.GELU(),
                nn.Linear(embed_dim_depth//r, embed_dim_image),
                )
        self.common_prompt_projection_depth = nn.Sequential(
                nn.Linear(embed_dim_depth, embed_dim_depth//r),
                nn.GELU(),
                nn.Linear(embed_dim_depth//r, embed_dim_depth),
                )

    def forward(self, missing_type):

        # Before returning, need to transform
        # prompts to 768 for the visual side
        all_prompts_image = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        all_prompts_depth = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        for i in range(len(missing_type)):
            # set initial prompts for each modality
            if missing_type[i]==0:  # modality complete
                initial_prompt_image = self.image_prompt_complete
                initial_prompt_depth = self.depth_prompt_complete
                common_prompt = self.common_prompt_complete
            elif missing_type[i]==1:  # missing image 
                initial_prompt_image = self.image_prompt_missing
                initial_prompt_depth = self.depth_prompt_complete
                common_prompt = self.common_prompt_depth
            elif missing_type[i]==2:  # missing depth 
                initial_prompt_image = self.image_prompt_complete
                initial_prompt_depth = self.depth_prompt_missing
                common_prompt = self.common_prompt_image
            # generate the prompts of the first layer
            all_prompts_image[0].append(self.compound_prompt_projections_image[0](self.layernorm_image[0](torch.cat([initial_prompt_image, initial_prompt_depth], -1))))
            all_prompts_depth[0].append(self.compound_prompt_projections_depth[0](self.layernorm_depth[0](torch.cat([initial_prompt_image, initial_prompt_depth], -1))))
            # generate the prompts of the rest layers
            for index in range(1, self.prompt_depth):
                all_prompts_image[index].append(
                    self.compound_prompt_projections_image[index](self.layernorm_image[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_depth[index-1][-1]], -1))))
                all_prompts_depth[index].append(
                    self.compound_prompt_projections_depth[index](self.layernorm_depth[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_depth[index-1][-1]], -1))))
            all_prompts_image[0][i] = torch.cat([all_prompts_image[0][i], self.common_prompt_projection_image(common_prompt)], 0)
            all_prompts_depth[0][i] = torch.cat([all_prompts_depth[0][i], self.common_prompt_projection_depth(common_prompt)], 0)
        # generate the prompts in each layer as a tensor [B, L, C]
        all_prompts_image = [torch.stack(prompts) for prompts in all_prompts_image]
        all_prompts_depth = [torch.stack(prompts) for prompts in all_prompts_depth]
        return all_prompts_image, all_prompts_depth   


class MISDD_MM(torch.nn.Module):
    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name,  precision='fp16', **kwargs):
        '''
        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(MISDD_MM, self).__init__()

        self.shot = kwargs['size']
        self.missing_prompt_length = kwargs['missing_prompt_length']
        self.missing_prompt_depth = kwargs['missing_prompt_depth']

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision = 'fp32' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop

        self.device = device
        self.get_model(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset)
        self.phrase_form = '{}'

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = 'V1' # V1:
        # visual textual, textual_visual

        self.img_transform = transforms.Compose([
            # transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            # transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])
    
        self.pc_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.depth_transform = transforms.Compose([
            transforms.ToTensor()])

        self.gt_transform = transforms.Compose([
            # transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            # transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])
        

    def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval()

        self.img_prompt_learner = PromptLearner(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, model, self.precision)
        self.depth_prompt_learner = PromptLearner(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, model, self.precision)
        self.missing_prompt_learner = Missing_PromptLearner(self.missing_prompt_length, self.missing_prompt_depth)
        self.model = model.to(self.device)
        self.pc_model = CLIPAD.PointTransformer(group_size=128, num_group=1024).to(self.device)

        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None

        img_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("img_feature_gallery1", img_gallery1)
        img_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("img_feature_gallery2", img_gallery2)
        pc_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.pc_model.encoder_dims))
        self.register_buffer("pc_feature_gallery1", pc_gallery1)
        pc_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.pc_model.encoder_dims))
        self.register_buffer("pc_feature_gallery2", pc_gallery2)
        depth_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("depth_feature_gallery1", depth_gallery1)
        depth_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("depth_feature_gallery2", depth_gallery2)

        img_text_features = torch.zeros((2, self.model.visual.output_dim))
        self.register_buffer("img_text_features", img_text_features)
        depth_text_features = torch.zeros((2, self.model.visual.output_dim))
        self.register_buffer("depth_text_features", depth_text_features)

        if self.precision == 'fp16':
            self.img_feature_gallery1  = self.img_feature_gallery1.half()
            self.img_feature_gallery2  = self.img_feature_gallery2.half()
            self.pc_feature_gallery1  = self.pc_feature_gallery1.half()
            self.pc_feature_gallery2  = self.pc_feature_gallery2.half()
            self.depth_feature_gallery1  = self.depth_feature_gallery1.half()
            self.depth_feature_gallery2  = self.depth_feature_gallery2.half()
            self.img_text_features  = self.img_text_features.half()
            self.depth_text_features  = self.depth_text_features.half()

        # # for testing
        # p1, p2 = self.prompt_learner()
        self.img_tokenized_normal_prompts = self.img_prompt_learner.tokenized_normal_prompts
        self.img_tokenized_abnormal_prompts_manual = self.img_prompt_learner.tokenized_abnormal_prompts_manual
        self.img_tokenized_abnormal_prompts_learned = self.img_prompt_learner.tokenized_abnormal_prompts_learned
        self.img_tokenized_abnormal_prompts = torch.cat([self.img_tokenized_abnormal_prompts_manual, self.img_tokenized_abnormal_prompts_learned], dim=0)

        self.depth_tokenized_normal_prompts = self.depth_prompt_learner.tokenized_normal_prompts
        self.depth_tokenized_abnormal_prompts_manual = self.depth_prompt_learner.tokenized_abnormal_prompts_manual
        self.depth_tokenized_abnormal_prompts_learned = self.depth_prompt_learner.tokenized_abnormal_prompts_learned
        self.depth_tokenized_abnormal_prompts = torch.cat([self.depth_tokenized_abnormal_prompts_manual, self.depth_tokenized_abnormal_prompts_learned], dim=0)

        self.average = torch.nn.AvgPool2d(3, stride=1) # torch.nn.AvgPool2d(1, stride=1) #
        self.resize = torch.nn.AdaptiveAvgPool2d((self.grid_size[0], self.grid_size[1]))
        self.proj = nn.Parameter((self.pc_model.trans_dim ** -0.5) * torch.randn(self.pc_model.trans_dim, self.pc_model.output_dim))

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_image_missing(self, image: torch.Tensor, all_prompts_image: torch.Tensor, missing_type: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image, all_prompts_image, missing_type)
        
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        # return [f / f.norm(dim=-1, keepdim=True) for f in text_features]
        return text_features
    

    def encode_text_embedding(self, text_embedding, original_tokens):
        text_features = self.model.encode_text_embeddings(text_embedding, original_tokens)
        return text_features

    @torch.no_grad()
    def build_img_text_feature_gallery(self):
        normal_text_embeddings, abnormal_text_embeddings_manual, abnormal_text_embeddings_learned = self.img_prompt_learner()
        abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_manual, abnormal_text_embeddings_learned], dim=0)

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.img_tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_text_embeddings, self.img_tokenized_abnormal_prompts)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.img_tokenized_normal_prompts)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0)
            abnormal_text_features = []
            for phrase_id in range(abnormal_text_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), self.img_tokenized_abnormal_prompts)
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0)
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        text_features_all = torch.cat([normal_text_features, abnormal_text_features], dim=0)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.img_text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))
    
    @torch.no_grad()
    def build_depth_text_feature_gallery(self):
        normal_text_embeddings, abnormal_text_embeddings_manual, abnormal_text_embeddings_learned = self.depth_prompt_learner()
        abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_manual, abnormal_text_embeddings_learned], dim=0)

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.depth_tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_text_embeddings, self.depth_tokenized_abnormal_prompts)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.depth_tokenized_normal_prompts)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0)
            abnormal_text_features = []
            for phrase_id in range(abnormal_text_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), self.depth_tokenized_abnormal_prompts)
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0)
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        text_features_all = torch.cat([normal_text_features, abnormal_text_features], dim=0)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.depth_text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))

    def build_image_feature_gallery(self, features1, features2):
        b1, n1, d1 = features1.shape
        self.img_feature_gallery1.copy_(F.normalize(features1.reshape(-1, d1), dim=-1, eps=1e-6))

        b2, n2, d2 = features2.shape
        self.img_feature_gallery2.copy_(F.normalize(features2.reshape(-1, d2), dim=-1, eps=1e-6))

    def build_depth_feature_gallery(self, features1, features2):
        b1, n1, d1 = features1.shape
        self.depth_feature_gallery1.copy_(F.normalize(features1.reshape(-1, d1), dim=-1, eps=1e-6))

        b2, n2, d2 = features2.shape
        self.depth_feature_gallery2.copy_(F.normalize(features2.reshape(-1, d2), dim=-1, eps=1e-6))

    def build_pc_feature_gallery(self, features1, features2):

        b1, n1, d1 = features1.shape
        self.pc_feature_gallery1.copy_(F.normalize(features1.reshape(-1, d1), dim=-1, eps=1e-6))

        b2, n2, d2 = features2.shape
        self.pc_feature_gallery2.copy_(F.normalize(features2.reshape(-1, d2), dim=-1, eps=1e-6))

    def calculate_img_textual_anomaly_score(self, visual_features, task):
        # t = 100
        t = self.model.logit_scale
        # t = self.t
        N = visual_features.shape[0]

        if task == 'seg':
            # ############################################## local tokens scores ############################
            # token_features = self.cross_attention(visual_features[1])
            token_features = visual_features
            local_normality_and_abnormality_score = (t * token_features @ self.img_text_features.T).softmax(dim=-1)

            local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            local_abnormality_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

            return local_abnormality_score.detach()

        elif task == 'cls':
            # ################################################ global cls token scores ##########################
            # global_feature = self.cross_attention(visual_features[0].unsqueeze(dim=1)).squeeze(dim=1)
            global_feature = visual_features
            global_normality_and_abnormality_score = (t * global_feature @ self.img_text_features.T).softmax(dim=-1)

            global_abnormality_score = global_normality_and_abnormality_score[:, 1]

            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_depth_textual_anomaly_score(self, visual_features, task):
        # t = 100
        t = self.model.logit_scale
        # t = self.t
        N = visual_features.shape[0]

        if task == 'seg':
            # ############################################## local tokens scores ############################
            # token_features = self.cross_attention(visual_features[1])
            token_features = visual_features
            local_normality_and_abnormality_score = (t * token_features @ self.depth_text_features.T).softmax(dim=-1)

            local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            local_abnormality_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

            return local_abnormality_score.detach()

        elif task == 'cls':
            # ################################################ global cls token scores ##########################
            # global_feature = self.cross_attention(visual_features[0].unsqueeze(dim=1)).squeeze(dim=1)
            global_feature = visual_features
            global_normality_and_abnormality_score = (t * global_feature @ self.depth_text_features.T).softmax(dim=-1)

            global_abnormality_score = global_normality_and_abnormality_score[:, 1]

            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_img_anomaly_score(self, img_mid_features1, img_mid_features2):
        N = img_mid_features1.shape[0]

        score1, _ = (1.0 - img_mid_features1 @ self.img_feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0

        score2, _ = (1.0 - img_mid_features2 @ self.img_feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
    
    def calculate_depth_anomaly_score(self, depth_mid_features1, depth_mid_features2):
        N = depth_mid_features1.shape[0]

        score1, _ = (1.0 - depth_mid_features1 @ self.depth_feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0

        score2, _ = (1.0 - depth_mid_features2 @ self.depth_feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
    
    def calculate_pc_anomaly_score(self, pc_mid_features1, pc_mid_features2):
        N = pc_mid_features1.shape[0]

        score1, _ = (1.0 - pc_mid_features1 @ self.pc_feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0

        score2, _ = (1.0 - pc_mid_features2 @ self.pc_feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

    def forward(self, args, imgs, depths, task, all_prompts_image, all_prompts_depth, missing_flag):
        
        if task == 'seg':
            img_features, img_features_maps, img_mid_features1, img_mid_features2 = self.encode_image_missing(imgs, all_prompts_image, missing_flag)
            
            depth_features, depth_features_maps, depth_mid_features1, depth_mid_features2 = self.encode_image_missing(depths, all_prompts_depth, missing_flag)
        
            img_textual_anomaly_map = self.calculate_img_textual_anomaly_score(img_features_maps, 'seg')
            depth_textual_anomaly_map = self.calculate_depth_textual_anomaly_score(depth_features_maps, 'seg')

            img_anomaly_map = self.calculate_img_anomaly_score(img_mid_features1, img_mid_features2)
            depth_anomaly_map = self.calculate_depth_anomaly_score(depth_mid_features1, depth_mid_features2)

            anomaly_map = torch.max(
                1. / (1. / img_textual_anomaly_map + 1. / img_anomaly_map),
                1. / (1. / depth_textual_anomaly_map + 1. / depth_anomaly_map)
            )

            anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            am_pix = anomaly_map.squeeze(1).detach().numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
                am_pix_list.append(am_pix[i])

            return am_pix_list

        elif task == 'cls':
            img_features, img_features_maps, img_mid_features1, img_mid_features2 = self.encode_image(imgs)

            depth_features, depth_features_maps, depth_mid_features1, depth_mid_features2 = self.encode_image(depths)

            img_textual_anomaly = self.calculate_img_textual_anomaly_score(img_features, 'cls')
            depth_textual_anomaly = self.calculate_depth_textual_anomaly_score(depth_features, 'cls')

            img_anomaly_map = self.calculate_img_anomaly_score(img_mid_features1, img_mid_features2)
            depth_anomaly_map = self.calculate_depth_anomaly_score(depth_mid_features1, depth_mid_features2)

            anomaly_map = torch.max(img_anomaly_map, depth_anomaly_map)
            anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            am_pix = anomaly_map.squeeze(1).detach().numpy()

            am_pix_list = []
            for i in range(am_pix.shape[0]):
                am_pix_list.append(am_pix[i])

            am_img_list = []
            for i in range(img_textual_anomaly.shape[0]):
                am_img_list.append(np.maximum(img_textual_anomaly[i], depth_textual_anomaly[i]))

            return am_img_list, am_pix_list
        else:
            assert 'task error'

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
