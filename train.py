import os
import sys
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np
from torch.optim import lr_scheduler

from model import ActTransNet, FrameFeats
from dataset import UCF101
import time

LR_DECAY = 0.1
DECAY_PATIENCE = 5
frame_feats_dim = 512*4
model_dim = 512
n_actions = 101
batch_size = 2
batch_size_to_step = 50
iter_to_step = int(batch_size_to_step / batch_size)
n_frames = 25
zp_limits = (int(math.ceil(1/3 * n_frames)), int(math.ceil(1/2 * n_frames - 1)))
ze_limits = (int(math.floor(1/2 * n_frames + 1)), int(math.floor(2/3 * n_frames)))
zp_possible = list(range(zp_limits[0], zp_limits[1] + 1))
ze_possible = list(range(ze_limits[0], ze_limits[1] + 1))
n_zp_possible = zp_limits[1] - zp_limits[0] + 1
n_ze_possible = ze_limits[1] - ze_limits[0] + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def train(epoch, log_path):
    ucf101 = UCF101(
        zp_limits[1], ze_limits[0],
        root=config.data_dir,
        split_file_path=config.train_split_path,
        n_frames=n_frames)
    train_set = DataLoader(
        ucf101, batch_size=batch_size, num_workers=4, shuffle=True)
    dataset = iter(train_set)
    pbar = tqdm(dataset)

    train_log_path = os.path.join(log_path, 'train_log.txt')
    log_data = ''
    net.train()
    optimizer.zero_grad()
    loss_list = []
    acc_list = []
    for i, (frames_p, frames_e, action) in enumerate(pbar):
        cur_batch_size = frames_p.shape[0]
        frames_p, frames_e, action = frames_p.to(device),frames_e.to(device),action.to(device)
    
        p_transformed, e_embed = net(frames_p, frames_e, action)
        y = -1 * torch.ones((cur_batch_size, p_transformed.shape[1]))
        y[list(range(cur_batch_size)), action] = 1
        y = y.to(device)
        p_transformed_shape = p_transformed.shape
        output_loss_dim = p_transformed_shape[0] * p_transformed_shape[1]
        loss = criterion(
            p_transformed.view(output_loss_dim,  p_transformed_shape[2]).to(device),
            e_embed.view(output_loss_dim, p_transformed_shape[2]).to(device),
            y.view(output_loss_dim).to(device)
        ) / iter_to_step
        # print("loss: ", loss.item())
        
        loss.backward()
        if (i+1) % iter_to_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        sim = F.cosine_similarity(p_transformed, e_embed, dim=2)
        prediction = sim.argmax(dim=1).to(device)
        accuracy = ((prediction == action).sum() / action.shape[0]).item()

        loss_list.append(loss.item())
        acc_list.append(accuracy)
        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                epoch + 1, np.mean(loss_list), np.mean(acc_list)
            )
        )

        train_log_path = os.path.join(log_path, 'train_log.txt')
        log_data += '{} - Epoch: {}; Loss: {:.5f};\n'.format(
            datetime.today().replace(microsecond=0),
            epoch + 1, loss.item(),
            np.mean(acc_list),
        )
        sys.stdout.flush()

    with open(train_log_path, 'a+') as f:
        f.write(log_data)

def valid(epoch, log_path):
    ucf101 = UCF101(
        zp_limits[1], ze_limits[0],
        root=config.data_dir,
        split_file_path=config.val_split_path,
        n_frames=n_frames)
    valid_set = DataLoader(
        ucf101, batch_size=batch_size, num_workers=4)
    dataset = iter(valid_set)
    pbar = tqdm(dataset)

    net.eval()
    valid_log_path = os.path.join(log_path, 'val_log.txt')
    log_data = ''
    loss_list = []
    acc_list = []
    with torch.no_grad():
        for frames_p, frames_e, action in pbar:
            cur_batch_size = frames_p.shape[0]
            frames_p, frames_e, action = frames_p.to(device), frames_e.to(device), action.to(device)
            
            p_transformed, e_embed = net(frames_p, frames_e, action)
            y = -1 * torch.ones((cur_batch_size, p_transformed.shape[1]))
            y[list(range(cur_batch_size)), action] = 1
            y = y.to(device)
            p_transformed_shape = p_transformed.shape
            output_loss_dim = p_transformed_shape[0] * p_transformed_shape[1]
            loss = criterion(
                p_transformed.view(output_loss_dim,  p_transformed_shape[2]).to(device),
                e_embed.view(output_loss_dim, p_transformed_shape[2]).to(device),
                y.view(output_loss_dim).to(device)
            )

            # Calculate accuracy
            # accuracy_2 = 0
            # for kin in range(cur_batch_size):
            #     sim_2 = F.cosine_similarity(p_transformed[kin,:,:], e_embed[kin, :,:], dim=1)
            #     print("similaridad: ", sim_2)
            #     prediction_2 = sim_2.argmax().to(device)
            #     accuracy_2 += (prediction_2 == action[kin])
            # accuracy_2 = (accuracy_2 / action.shape[0]).item()
            # print("accuracy_2: ", accuracy_2)
            sim = F.cosine_similarity(p_transformed, e_embed, dim=2)
            prediction = sim.argmax(dim=1).to(device)
            accuracy = ((prediction == action).sum() / action.shape[0]).item()
            
            loss_list.append(loss.item())
            acc_list.append(accuracy)
            pbar.set_description(
                'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                    epoch + 1, np.mean(loss_list), np.mean(acc_list)
                )
            )

            log_data += '{} - Epoch: {}; Loss: {:.5f}; Acc: {:.5f}\n'.format(
                datetime.today().replace(microsecond=0),
                epoch + 1, loss.item(), np.mean(acc_list)
            )
        
        exp_lr_scheduler.step(np.mean(loss_list))
    
    with open(valid_log_path, 'a+') as f:
        f.write(log_data)
    print('Avg acc: {:.5f}'.format(np.mean(acc_list)))
    sys.stdout.flush()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Actions Transformations')
    parser.add_argument('data_dir', help='the path to the frames')
    parser.add_argument('output_dir', help='the path to store the outputs')
    parser.add_argument('--train_split_path', default='./train_rgb_split1.txt', help='the path to the train split file')
    parser.add_argument('--val_split_path', default='./val_rgb_split1.txt', help='the path to the validation split file')
    parser.add_argument('--experiment', required=False, help='the number of experiment')
    parser.add_argument('--n_epochs', required=False, default=162, type=int, help='the number of epochs to run')
    parser.add_argument('--instance_norm', action='store_true')
    parser.add_argument('--embed_dropout', default=.5, type=float)
    parser.add_argument('--trans_dropout', default=0, type=float)
    config = parser.parse_args()

    print('instance_norm:', config.instance_norm)
    print('embed_dropout:', config.embed_dropout)
    print('trans_dropout:', config.trans_dropout)
    sys.stdout.flush()

    criterion = nn.CosineEmbeddingLoss(margin=.5)
    
    net = ActTransNet(
        frame_feats_dim,
        model_dim,
        n_actions,
        zp_limits,
        ze_limits,
        nn.CosineEmbeddingLoss(margin=.5, reduction='none'),
        instance_norm=config.instance_norm,
        embed_dropout=config.embed_dropout,
        trans_dropout=config.trans_dropout,
    )
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net = net.to(device)

    optimizer = optim.Adam(
        net.parameters(),
        lr=1e-4,
        # weight_decay=1e-4,
    )
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=LR_DECAY, patience=DECAY_PATIENCE,verbose=True)

    if config.experiment is None:
        print('Starting new experiment...')
        new_exp = int(max(filter(
            lambda x: not x.startswith('.'),
            os.listdir(os.path.join(config.output_dir, 'checkpoint'))
        ), default=-1)) + 1
        os.mkdir(os.path.join(config.output_dir, 'checkpoint', str(new_exp)))
        os.mkdir(os.path.join(config.output_dir, 'log', str(new_exp)))
        experiment = new_exp
    else:
        experiment = config.experiment

    # Create experiment dir if not exists
    experiment_path = os.path.join(config.output_dir, str(experiment))
    log_path = os.path.join(experiment_path, 'log')
    checkpoint_path = os.path.join(experiment_path, 'checkpoint')
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        os.makedirs(log_path)
        os.makedirs(checkpoint_path)
    # Obtain current epoch from checkpoint file names
    cur_epoch = int(max(map(
        lambda file_name: int(file_name.split('_')[1]) - 1,
        filter(
            lambda x: not x.startswith('.'),
            os.listdir(checkpoint_path))
        ),
        default=-1
    )) + 1
    if cur_epoch != 0:
        print('Loading checkpoint {}...'.format(str(cur_epoch + 1).zfill(2)))
        file_name = next(filter(
            lambda x: 'checkpoint_{}'.format(str(cur_epoch).zfill(2)) in x,
            os.listdir(checkpoint_path)
        ))
        #net.load_state_dict(torch.load(
        #    os.path.join(checkpoint_path, file_name)
        #))
        checkpoint = torch.load(os.path.join(checkpoint_path, file_name))
        cur_epoch+=1
        net.load_state_dict(checkpoint['model_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    # Run epochs
    for epoch in range(cur_epoch, config.n_epochs):
        train(epoch, log_path)
        valid(epoch, log_path)

        with open(
            os.path.join(checkpoint_path, 'checkpoint_{}_{}.model').format(
                str(epoch).zfill(2),
                datetime.today().replace(microsecond=0)
            ), 'wb'
        ) as f:
            #torch.save(net.state_dict(), f)
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': exp_lr_scheduler.state_dict()
            }, f)
