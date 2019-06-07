from torch import nn
from torchvision.models import resnet50


class FrameFeats(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        pretrained_resnet = resnet50(pretrained=True)
        self.resnet_wo_last = nn.Sequential(*list(pretrained_resnet.children())[:-1])
        self.fc = nn.Linear(pretrained_resnet.children()[-1].in_features, out_dim)

    def forward(self, frame):
        return self.resnet_wo_last(self.fc(frame))

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
            p_transformed = torch.bmm(selected_transformations, p_embed.unsqueeze(2)).unsqueeze(1)
        else:
            p_transformed = torch.empty((batch_size, self.n_actions, self.dim))
            for i in range(self.n_actions):
                p_transformed[:, i, :] = torch.bmm(
                    self.W_tranformations[i].expand(batch_size, self.dim, self.dim),
                    p_embed)
        
        # results = []
        # for action_idx in actions_idx:
        #     results.append(self.T_list[action_idx](p_embed))

        return p_transformed, e_embed