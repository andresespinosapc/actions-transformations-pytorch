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

from model import ActTransNet, FrameFeats
from dataset import UCF101


input_dim = (3, 224, 224)
frame_feats_dim = 512
model_dim = 512
n_actions = 101
batch_size = 5
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
    ucf101 = UCF101(zp_limits[1], ze_limits[0], root=config.data_dir, split='train')
    train_set = DataLoader(
        ucf101, batch_size=batch_size, num_workers=4, shuffle=True)
    dataset = iter(train_set)
    pbar = tqdm(dataset)

    train_log_path = os.path.join(log_path, 'train_log.txt')
    log_data = ''
    for frames_p, frames_e, action in pbar:
        cur_batch_size = frames_p.shape[0]
        frames_p, frames_e, action = frames_p.to(device),frames_e.to(device),action.to(device)
        #frames_p = frames[:, :zp_limits[1]+1].contiguous()
        #frames_e = frames[:, ze_limits[0]:].contiguous()
        #del frames

        optimizer.zero_grad()
        frames_feats_p = frame_net_p(frames_p.view(-1, *input_dim)).view(cur_batch_size, zp_limits[1], frame_feats_dim)
        frames_feats_e = frame_net_e(frames_e.view(-1, *input_dim)).view(cur_batch_size, n_frames - ze_limits[0], frame_feats_dim)

        # Search latent variables
        net.train(False)
        with torch.no_grad():
            best_zp = None
            best_ze = None
            min_distance = float('inf')
            for zp in zp_possible:
                for ze in range(n_ze_possible):
                    precondition = frames_feats_p[:, :zp, :]
                    effect = frames_feats_e[:, ze:,:]
                    p_transformed, e_embed = net(precondition, effect, action)
                    # Obtain distance
                    is_positive = torch.ones((cur_batch_size,)).to(device)
                    loss = criterion(p_transformed, e_embed, is_positive)
                    # If it is better than the last one, update best zp and ze
                    if loss < min_distance:
                        best_zp, best_ze = zp, ze
                        min_distance = loss

        net.train(True)
        precondition = frames_feats_p[:, :best_zp, :]
        effect = frames_feats_e[:, best_ze:, :]
        p_transformed, e_embed = net(precondition, effect)
        y = -1 * torch.ones((cur_batch_size, p_transformed.shape[1]))
        y[:,action] = 1
        y = y.to(device)
        p_transformed_shape = p_transformed.shape
        output_loss_dim = p_transformed_shape[0] * p_transformed_shape[1]
        loss = criterion(
            p_transformed.view(output_loss_dim,  p_transformed_shape[2]).to(device),
            e_embed.view(output_loss_dim, p_transformed_shape[2]).to(device),
            y.view(output_loss_dim).to(device)
        )
        # print("loss: ", loss.item())
        loss.backward()
        optimizer.step()
        
        sim = F.cosine_similarity(p_transformed, e_embed, dim=2)
        print('sim:', sim)
        prediction = sim.argmax(dim=1).to(device)
        print('prediction:', prediction)
        print('action:', action)
        accuracy = ((prediction == action).sum() / action.shape[0]).item()
        # print('Accuracy:', accuracy)

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                epoch + 1, loss.item(), accuracy
            )
        )

        train_log_path = os.path.join(log_path, 'train_log.txt')
        log_data += '{} - Epoch: {}; Loss: {:.5f}; Acc: {:.5f}\n'.format(
            datetime.today().replace(microsecond=0),
            epoch + 1, loss.item(), accuracy
        )
        sys.stdout.flush()

    with open(train_log_path, 'a+') as f:
        f.write(log_data)

def valid(epoch, log_path):
    ucf101 = UCF101(zp_limits[1], ze_limits[0], root=config.data_dir, split='val')
    valid_set = DataLoader(
        ucf101, batch_size=batch_size, num_workers=4)
    dataset = iter(valid_set)
    pbar = tqdm(dataset)

    acc_list = []
    net.train(False)
    valid_log_path = os.path.join(log_path, 'val_log.txt')
    log_data = ''
    with torch.no_grad():
        for frames_p, frames_e, action in pbar:
            cur_batch_size = frames_p.shape[0]
            frames_p, frames_e, action = frames_p.to(device), frames_e.to(device), action.to(device)
            #frames_p = frames[:, :zp_limits[1]+1].contiguous()
            #frames_e = frames[:, ze_limits[0]:].contiguous()

            # Calculate optimal zp and ze values
            frames_feats_p = frame_net_p(frames_p.view(-1, *input_dim)).view(cur_batch_size, zp_limits[1], frame_feats_dim)
            frames_feats_e = frame_net_e(frames_e.view(-1, *input_dim)).view(cur_batch_size, n_frames - ze_limits[0], frame_feats_dim)

            best_zp = None
            best_ze = None
            min_distance = float('inf')
            for zp in zp_possible:
                for ze in range(n_ze_possible):
                    precondition = frames_feats_p[:, :zp, :]
                    effect = frames_feats_e[:, ze:,:]
                    p_transformed, e_embed = net(precondition, effect, action)
                    # Obtain distance
                    is_positive = torch.ones((cur_batch_size,)).to(device)
                    loss = criterion(p_transformed, e_embed, is_positive)
                    # If it is better than the last one, update best zp and ze
                    if loss < min_distance:
                        best_zp, best_ze = zp, ze
                        min_distance = loss

            # Evaluate network
            precondition = frames_feats_p[:, :best_zp, :]
            effect = frames_feats_e[:, best_ze:, :]
            p_transformed, e_embed = net(precondition, effect)
            y = -1 * torch.ones((cur_batch_size, p_transformed.shape[1]))
            y[:,action] = 1
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
            acc_list.append(accuracy)

            pbar.set_description(
                'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                    epoch + 1, loss.item(), accuracy
                )
            )

            log_data += '{} - Epoch: {}; Loss: {:.5f}; Acc: {:.5f}\n'.format(
                datetime.today().replace(microsecond=0),
                epoch + 1, loss.item(), accuracy
            )
    
    with open(valid_log_path, 'a+') as f:
        f.write(log_data)
    print('Avg acc: {:.5f}'.format(np.mean(acc_list)))
    sys.stdout.flush()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Actions Transformations')
    parser.add_argument('data_dir', help='the path to the frames')
    parser.add_argument('output_dir', help='the path to store the outputs')
    parser.add_argument('--experiment', required=False, help='the number of experiment')
    parser.add_argument('--n_epochs', required=False, default=20, help='the number of epochs to run')
    config = parser.parse_args()

    frame_net_p = FrameFeats(frame_feats_dim)
    frame_net_e = FrameFeats(frame_feats_dim)
    net = ActTransNet(frame_feats_dim, model_dim, n_actions)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        frame_net_p = nn.DataParallel(frame_net_p)
        frame_net_e = nn.DataParallel(frame_net_e)
        #net = nn.DataParallel(net)
    frame_net_p = frame_net_p.to(device)
    frame_net_e = frame_net_e.to(device)
    net = net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    criterion = nn.CosineEmbeddingLoss(margin=.5)
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

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
        lambda file_name: int(file_name.split('_')[1]),
        filter(
            lambda x: not x.startswith('.'),
            os.listdir(checkpoint_path))
        ),
        default=-1
    )) + 1
    if cur_epoch != 0:
        file_name = next(filter(
            lambda x: 'checkpoint_{}'.format(str(cur_epoch - 1).zfill(2)) in x,
            os.listdir(checkpoint_path)
        ))
        net.load_state_dict(torch.load(
            os.path.join(checkpoint_path, file_name)
        ))
    # Run epochs
    for epoch in range(cur_epoch, cur_epoch + config.n_epochs):
        train(epoch, log_path)
        valid(epoch, log_path)

        with open(
            os.path.join(checkpoint_path, 'checkpoint_{}_{}.model').format(
                str(epoch + 1).zfill(2),
                datetime.today().replace(microsecond=0)
            ), 'wb'
        ) as f:
            torch.save(net.state_dict(), f)
