import math
from torch import nn
from torchvision import models
import torch
import numpy as np

from models import rgb_resnet152

class FrameFeats(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.pretrained_resnet = rgb_resnet152(pretrained=False, num_classes=101)
        # self.pretrained_resnet = models.resnet152(pretrained=False, num_classes=101)
        # self.pretrained_resnet.fc_action = self.pretrained_resnet.fc
        pretrained_params_ucf101 = torch.load('checkpoints/ucf101_s1_rgb_resnet152.pth.tar')
        self.pretrained_resnet.load_state_dict(pretrained_params_ucf101['state_dict'])
        # num_ftrs = self.pretrained_resnet.fc.in_features
        # self.pretrained_resnet.fc = nn.Linear(num_ftrs, out_dim)
        num_ftrs = self.pretrained_resnet.fc_action.in_features
        self.pretrained_resnet.fc_action = nn.Linear(num_ftrs, out_dim)
        #self.resnet_wo_last = nn.Sequential(*list(pretrained_resnet.children())[:-1])
        #self.fc = nn.Linear(pretrained_resnet.fc.in_features, out_dim)

    def forward(self, frame):
        #return self.fc(self.resnet_wo_last(frame))
        return self.pretrained_resnet(frame)

class ActTransNet(nn.Module):
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
            p_transformed = np.squeeze(torch.bmm(selected_transformations, p_embed.unsqueeze(2)))
        else:
            # No he revisado esta parte 
            p_transformed = torch.empty((batch_size, self.n_actions, self.dim))
            e_embed_copy = torch.empty((batch_size, self.n_actions, self.dim))
            for i in range(self.n_actions):
                p_transformed[:, i, :] = np.squeeze(torch.bmm(
                    self.W_tranformations[i].expand(batch_size, self.dim, self.dim), 
                    p_embed.unsqueeze(2)))
                e_embed_copy[:, i, :] = e_embed
            e_embed = e_embed_copy
            # e_embed = e_embed.unsqueeze(1).expand(batch_size, self.n_actions, self.dim)

        # results = []
        # for action_idx in actions_idx:
        #     results.append(self.T_list[action_idx](p_embed))
        return p_transformed, e_embed