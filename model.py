import math
from torch import nn
import torch.nn.functional as F
from torchvision import models
import torch
import numpy as np

import time

from efficientnet_pytorch import EfficientNet
#from models import rgb_resnet152
from network import resnet101


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class FrameFeats(nn.Module):
    def __init__(self, out_dim, backbone='efficientnetb0'):
        super().__init__()
        
        # self.pretrained_backbone = rgb_resnet152(pretrained=False, num_classes=101)
        # self.pretrained_backbone = models.resnet152(pretrained=False, num_classes=101)
        # self.pretrained_backbone.fc_action = self.pretrained_backbone.fc
        # pretrained_params_ucf101 = torch.load('checkpoints/ucf101_s1_rgb_resnet152.pth.tar')

        if backbone == 'efficientnetb0':
            self.pretrained_backbone = EfficientNet.from_name('efficientnet-b0')
            self.pretrained_backbone._fc = nn.Linear(self.pretrained_backbone._bn1.num_features, 101)
            pretrained_params_ucf101 = torch.load(
                'ucf101_efficientnetb0.pth.tar',
                map_location=device,
            )
            self.pretrained_backbone.load_state_dict(pretrained_params_ucf101['state_dict'])
            num_ftrs = self.pretrained_backbone._fc.in_features
            self.pretrained_backbone._fc = nn.Linear(num_ftrs, out_dim)
        elif backbone == 'resnet101':
            self.pretrained_backbone = resnet101(pretrained=False, channel=3)
            pretrained_params_ucf101 = torch.load('ucf101_resnet101.pth.tar', map_location=device)
            num_ftrs = self.pretrained_backbone.fc_custom.in_features
            self.pretrained_backbone.fc_custom = nn.Linear(num_ftrs, out_dim)

        # print('efficientnetb0 params:', sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.pretrained_backbone.parameters())]))
        # print('resnet101 params:', sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, resnet101(pretrained=False, channel=3).parameters())]))
        
        #self.resnet_wo_last = nn.Sequential(*list(pretrained_backbone.children())[:-1])
        #self.fc = nn.Linear(pretrained_backbone.fc.in_features, out_dim)

    def forward(self, frame):
        #return self.fc(self.resnet_wo_last(frame))
        return self.pretrained_backbone(frame)

class TransformationNet(nn.Module):
    def __init__(self, input_dim, dim, n_actions, instance_norm=True, embed_dropout=.5, trans_dropout=0):
        super().__init__()
        self.dim = dim
        self.n_actions = n_actions
        self.precondition_proj = nn.Linear(input_dim, dim)
        self.effect_proj = nn.Linear(input_dim, dim)
        # T_list = []
        # for _ in range(n_actions):
        #     T_list.append(nn.Linear(dim, dim))
        # self.T_list = nn.ModuleList(T_list)
        self.W_tranformations = nn.Parameter(torch.Tensor(n_actions, dim, dim))
        nn.init.kaiming_uniform_(self.W_tranformations, a=math.sqrt(5))
        self.default_action = torch.Tensor(list(range(self.n_actions)))
        if instance_norm:
            # self.instance_norm = nn.InstanceNorm1d(self.dim, affine=True)
            self.instance_norm = nn.GroupNorm(1, 1)
        else:
            self.instance_norm = nn.Identity()
        self.embed_dropout = nn.Dropout(p=embed_dropout, inplace=True)
        self.trans_dropout = nn.Dropout(p=trans_dropout, inplace=True)

    def forward(self, precondition, effect, action=None):
        batch_size = precondition.shape[0]
        p_avg = precondition.sum(1).float() / (precondition != 0).sum(1).float()
        e_avg = effect.sum(1).float() / (effect != 0).sum(1).float()

        p_avg = self.instance_norm(p_avg.unsqueeze(1)).squeeze(1)
        e_avg = self.instance_norm(e_avg.unsqueeze(1)).squeeze(1)

        p_embed = self.precondition_proj(p_avg)
        p_embed = self.embed_dropout(p_embed)
        e_embed = self.effect_proj(e_avg)
        e_embed = self.embed_dropout(e_embed)

        # if action is None:
        #     action = self.default_action
        if action is not None:
            selected_transformations = self.W_tranformations.index_select(0, action)
            #p_transformed = torch.bmm(selected_transformations, p_embed.unsqueeze(2)).unsqueeze(1)
            p_transformed = torch.bmm(selected_transformations, p_embed.unsqueeze(2)).squeeze(dim=2)
        else:
            # No he revisado esta parte 
            p_transformed = torch.empty((batch_size, self.n_actions, self.dim)).to(device)
            # e_embed_copy = torch.empty((batch_size, self.n_actions, self.dim)).to(device)
            for i in range(self.n_actions):
                p_transformed[:, i, :] = torch.bmm(
                    self.W_tranformations[i].expand(batch_size, self.dim, self.dim), 
                    p_embed.unsqueeze(2)).squeeze()
                # e_embed_copy[:, i, :] = e_embed
            # e_embed = e_embed_copy
            e_embed = e_embed.unsqueeze(1).expand(batch_size, self.n_actions, self.dim).contiguous()

        p_transformed = self.trans_dropout(p_transformed)

        # results = []
        # for action_idx in actions_idx:
        #     results.append(self.T_list[action_idx](p_embed))
        return p_transformed, e_embed

class ActTransNet(nn.Module):
    def __init__(self, frame_feats_dim, model_dim, n_actions, zp_limits, ze_limits, criterion):
        super().__init__()
        self.frame_feats_dim = frame_feats_dim
        self.criterion = criterion

        self.zp_limit_end = zp_limits[1]
        self.ze_limit_start = ze_limits[0]
        self.zp_possible = list(range(zp_limits[0], zp_limits[1] + 1))
        self.ze_possible = list(range(ze_limits[0], ze_limits[1] + 1))
        self.n_zp_possible = zp_limits[1] - zp_limits[0] + 1
        self.n_ze_possible = ze_limits[1] - ze_limits[0] + 1

        self.input_dim = (3, 224, 224)
        self.frame_net_p = FrameFeats(frame_feats_dim)
        self.frame_net_e = FrameFeats(frame_feats_dim)
        self.transformation_net = TransformationNet(frame_feats_dim, model_dim, n_actions)

    def forward(self, frames_p, frames_e, action):
        batch_size = frames_p.shape[0]
        n_frames = frames_p.shape[1]

        frames_feats_p = self.frame_net_p(frames_p.view(-1, *self.input_dim)).view(batch_size, self.zp_limit_end, self.frame_feats_dim)
        frames_feats_e = self.frame_net_e(frames_e.view(-1, *self.input_dim)).view(batch_size, n_frames - self.ze_limit_start, self.frame_feats_dim)

        # Search latent variables
        self.frame_net_p.train(False)
        self.frame_net_e.train(False)
        self.transformation_net.train(False)
        with torch.no_grad():
            best_zp = torch.empty((batch_size,)).to(device)
            best_ze = torch.empty((batch_size,)).to(device)
            frames_p_mask = torch.ones((batch_size, n_frames)).to(device)
            frames_e_mask = torch.ones((batch_size, n_frames)).to(device)
            min_distance = torch.full((batch_size,), torch.finfo(torch.float).max).to(device)
            for zp in self.zp_possible:
                for ze in range(self.n_ze_possible):
                    precondition = frames_feats_p[:, :zp, :]
                    effect = frames_feats_e[:, ze:,:]
                    p_transformed, e_embed = self.transformation_net(precondition, effect, action)
                    # Obtain distance
                    is_positive = torch.ones((batch_size,)).to(device)
                    loss = self.criterion(p_transformed, e_embed, is_positive)
                    # Update min_distance, and best zp and ze
                    better_mask = (loss < min_distance).float()
                    zp_mask = torch.zeros((n_frames,)).to(device)
                    zp_mask[:zp] = 1
                    better_mask_zp = better_mask.unsqueeze(1).expand((batch_size, n_frames))
                    ze_mask = torch.zeros((n_frames,)).to(device)
                    ze_mask[ze:] = 1
                    better_mask_ze = better_mask.unsqueeze(1).expand((batch_size, n_frames))
                    frames_p_mask = frames_p_mask * (1 - better_mask_zp) + zp_mask * better_mask_zp
                    frames_e_mask = frames_e_mask * (1 - better_mask_ze) + ze_mask * better_mask_ze
                    min_distance = min_distance * (1 - better_mask) + loss * better_mask

        self.frame_net_p.train(self.training)
        self.frame_net_e.train(self.training)
        self.transformation_net.train(self.training)
        precondition = frames_feats_p * frames_p_mask.unsqueeze(2).expand((batch_size, n_frames, self.frame_feats_dim))
        effect = frames_feats_e * frames_e_mask.unsqueeze(2).expand((batch_size, n_frames, self.frame_feats_dim))
        p_transformed, e_embed = self.transformation_net(precondition, effect)

        return p_transformed, e_embed