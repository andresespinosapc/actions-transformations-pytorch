import math
from torch import nn
from torchvision import models
import torch
import numpy as np

import time

#from models import rgb_resnet152
from network import resnet101


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class FrameFeats(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.pretrained_resnet = resnet101(pretrained=False, channel=3)
        # self.pretrained_resnet = rgb_resnet152(pretrained=False, num_classes=101)
        # self.pretrained_resnet = models.resnet152(pretrained=False, num_classes=101)
        # self.pretrained_resnet.fc_action = self.pretrained_resnet.fc
        # pretrained_params_ucf101 = torch.load('checkpoints/ucf101_s1_rgb_resnet152.pth.tar')
        pretrained_params_ucf101 = torch.load('ucf101_resnet101.pth.tar')
        self.pretrained_resnet.load_state_dict(pretrained_params_ucf101['state_dict'])
        # num_ftrs = self.pretrained_resnet.fc.in_features
        # self.pretrained_resnet.fc = nn.Linear(num_ftrs, out_dim)
        num_ftrs = self.pretrained_resnet.fc_custom.in_features
        self.pretrained_resnet.fc_custom = nn.Linear(num_ftrs, out_dim)
        #self.resnet_wo_last = nn.Sequential(*list(pretrained_resnet.children())[:-1])
        #self.fc = nn.Linear(pretrained_resnet.fc.in_features, out_dim)

    def forward(self, frame):
        #return self.fc(self.resnet_wo_last(frame))
        return self.pretrained_resnet(frame)

class TransformationNet(nn.Module):
    def __init__(self, input_dim, dim, n_actions):
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

    def forward(self, precondition, effect, action=None):
        batch_size = precondition.shape[0]
        p_avg = precondition.mean(1)
        e_avg = effect.mean(1)

        p_embed = self.precondition_proj(p_avg)
        e_embed = self.effect_proj(e_avg)

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
            best_zp = None
            best_ze = None
            min_distance = float('inf')
            for zp in self.zp_possible:
                for ze in range(self.n_ze_possible):
                    precondition = frames_feats_p[:, :zp, :]
                    effect = frames_feats_e[:, ze:,:]
                    p_transformed, e_embed = self.transformation_net(precondition, effect, action)
                    # Obtain distance
                    is_positive = torch.ones((batch_size,)).to(device)
                    loss = self.criterion(p_transformed, e_embed, is_positive)
                    # If it is better than the last one, update best zp and ze
                    if loss < min_distance:
                        best_zp, best_ze = zp, ze
                        min_distance = loss

        self.frame_net_p.train(self.training)
        self.frame_net_e.train(self.training)
        self.transformation_net.train(self.training)
        precondition = frames_feats_p[:, :best_zp, :]
        effect = frames_feats_e[:, best_ze:, :]
        p_transformed, e_embed = self.transformation_net(precondition, effect)
        return p_transformed, e_embed