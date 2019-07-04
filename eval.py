import os
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from model import ActTransNet
from dataset import UCF101


frame_feats_dim = 512 * 4
model_dim = 512
n_actions = 101
n_frames = 25
zp_limits = (int(math.ceil(1/3 * n_frames)), int(math.ceil(1/2 * n_frames - 1)))
ze_limits = (int(math.floor(1/2 * n_frames + 1)), int(math.floor(2/3 * n_frames)))
zp_possible = list(range(zp_limits[0], zp_limits[1] + 1))
ze_possible = list(range(ze_limits[0], ze_limits[1] + 1))
n_zp_possible = zp_limits[1] - zp_limits[0] + 1
n_ze_possible = ze_limits[1] - ze_limits[0] + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Actions Transformations on UCF101')
    parser.add_argument('data_dir', help='the path to the frames')
    parser.add_argument('output_dir', help='the path to store the output file')
    parser.add_argument('--checkpoint', required=True, help='the path to the checkpoint file to use')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--split_path', default='./val_rgb_split1.txt', help='the path to the evaluation split file')
    config = parser.parse_args()


    net = ActTransNet(
        frame_feats_dim,
        model_dim,
        n_actions,
        zp_limits,
        ze_limits,
        nn.CosineEmbeddingLoss(margin=.5, reduction='none'),
        instance_norm=False,
        # embed_dropout=config.embed_dropout,
        # trans_dropout=config.trans_dropout,
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net = net.to(device)

    net.load_state_dict(torch.load(config.checkpoint))

    ucf101 = UCF101(
        zp_limits[1], ze_limits[0],
        root=config.data_dir,
        split_file_path=config.split_path,
        n_frames=n_frames)
    valid_set = DataLoader(
        ucf101, batch_size=config.batch_size, num_workers=4)
    dataset = iter(valid_set)
    pbar = tqdm(dataset)
  
    predictions = []
    net.eval()
    with torch.no_grad():
        for frames_p, frames_e, action in pbar:
            cur_batch_size = frames_p.shape[0]
            frames_p, frames_e, action = frames_p.to(device), frames_e.to(device), action.to(device)

            p_transformed, e_embed = net(frames_p, frames_e, action)
            sim = F.cosine_similarity(p_transformed, e_embed, dim=2)
            prediction = sim.argmax(dim=1)

            predictions.extend(list(map(lambda x: str(x.item()), prediction)))

    with open(os.path.join(config.output_dir, 'act_trans_predictions.txt'), 'w') as f:
        f.write('\n'.join(predictions))