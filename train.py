import math
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ActTransNet, FrameFeats
from dataset import UCF101


input_dim = (224, 224, 3)
frame_feats_dim = 512
model_dim = 512
n_actions = 101
batch_size = 32
n_frames = 25
zp_limits = (1/3 * n_frames, math.ceil(1/2 * n_frames - 1))
ze_limits = (math.floor(1/2 * n_frames + 1), 2/3 * n_frames)
zp_possible = list(range(zp_limits[0], zp_limits[1] + 1))
ze_possible = list(range(ze_limits[0], ze_limits[1] + 1))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def train():
    ucf101 = UCF101(root=config.data_dir)
    train_set = DataLoader(
        ucf101, batch_size=batch_size, num_workers=4)
    dataset = iter(train_set)
    pbar = tqdm(dataset)

    for frames, action in pbar:
        frames, action = frames.to(device), action.to(device)

        # Search latent variables
        net.train(False)
        with torch.no_grad():
            frames_feats = frame_net(frames.view(-1, *input_dim)).view(batch_size, n_frames, frame_feats_dim)

            # best_z = torch.empty((batch_size, 2))
            # min_distance = torch.full((batch_size,), float('inf'))
            # for zp in zp_possible:
            #     for ze in ze_possible:
            #         precondition = frames_feats[:, :zp+1]
            #         effect = frames_feats[:, ze:]
            #         transformed_embeds, effect_embed = net(precondition, effect, action)
            #         # Obtain distance
            #         loss = criterion(transformed_embeds[0], effect_embed, torch.ones((batch_size,)))
            #         # Get mask of instances where the new loss is better than the previous ones
            #         better_mask = loss < min_distance
            #         already_best_mask = 1 - better_mask
            #         # Update best_z and min_distance according to that mask
            #         best_z[:, 0] = best_z[:, 0] * already_best_mask + zp * better_mask
            #         best_z[:, 1] = best_z[:, 1] * already_best_mask + ze * better_mask
            #         min_distance = min_distance * already_best_mask + loss * better_mask

            best_zp = None
            best_ze = None
            min_distance = float('inf')
            for zp in zp_possible:
                for ze in ze_possible:
                    precondition = frames_feats[:, zp+1]
                    effect = frames_feats[:, ze:]
                    p_transformed, e_embed = net(precondition, effect, action)
                    # Obtain distance
                    loss = criterion(p_transformed[0], effect_embed, torch.ones((batch_size,)))
                    # If it is better than the last one, update best zp and ze
                    if loss < min_distance:
                        best_zp, best_ze = zp, ze
                        min_distance = loss

        net.zero_grad()
        net.train(True)
        precondition = frames_feats[:, zp+1]
        effect = frames_feats[:, ze+1]
        p_transformed, e_embed = net(precondition, effect)
        loss = criterion(p_transformed[0], effect_embed, torch.ones((batch_size,)))

  

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Actions Transformations')
    parser.add_argument('data_dir', required=True,
        help='the path to the frames')
    config = parser.parse_args()

    frame_net = FrameFeats(frame_feats_dim)
    net = ActTransNet(frame_feats_dim, model_dim, n_actions)

    criterion = nn.CosineEmbeddingLoss(margin=.5)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train()