from PIL import Image
import torch
from torch.utils.data import Dataset
import os, pickle
from torchvision import transforms
import numpy as np
import video_transforms
import cv2

import sys


class UCF101(Dataset):
    def __init__(self, zp_limit, ze_limit, root, split_file_path, n_frames=25, transform = None):
        self.transform = transform
        if transform is None:
            # self.transform = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.CenterCrop(224),
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # ])
            is_color = True
            scale_ratios = [1.0, 0.875, 0.75, 0.66]
            clip_mean = [0.485, 0.456, 0.406]
            clip_std = [0.229, 0.224, 0.225]
            normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)
            if 'train' in split_file_path:
                self.transform = video_transforms.Compose([
                    # video_transforms.Scale((256)),
                    video_transforms.MultiScaleCrop((224, 224), scale_ratios),
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = video_transforms.Compose([
                    # video_transforms.Scale((256)),
                    video_transforms.CenterCrop((224)),
                    video_transforms.ToTensor(),
                    normalize,
                ])
        self.zp_limit = zp_limit
        self.ze_limit = ze_limit
        self.root = root
        data = []
        for line in open(split_file_path):
            file_name, duration_str, target_str = line.split(' ')
            duration, target = int(duration_str), int(target_str)
            # TEMP
            if 'HandStand' in file_name:
                file_name = file_name.replace('HandStand', 'Handstand')
            frames_path = os.path.join(self.root, file_name)
            diff = (duration - 1) / (n_frames - 1)
            frames_ids = (np.arange(n_frames) * diff + 1).astype(np.int)
            new_data = [frames_path, frames_ids, target]
            data.append(new_data)
        self.data = np.array(data)
    
    def get_transformed_frame(self, path):
        # img = Image.open(path).convert('RGB')
        # return self.transform(img)

        cv_read_flag = cv2.IMREAD_COLOR
        interpolation = cv2.INTER_LINEAR
        cv_img_origin = cv2.imread(path, cv_read_flag)
        new_width = 340
        new_height = 256
        cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        return self.transform(cv_img)

    def __getitem__(self, index):
        frames_path, frames_ids, target = self.data[index]
        print("frames_path: ", frames_path)
        print("frames_ids: ", frames_ids)
        frames_zp = []
        frames_ze = []
        for i in range(len(frames_ids)):
            frame_path = os.path.join(frames_path, 'frame{}.jpg'.format(str(frames_ids[i]).zfill(6)))
            print("frames_path", frame_path)
            sys.stdout.flush()
            if i < self.zp_limit: 
                img = self.get_transformed_frame(frame_path)
                frames_zp.append(img)
            elif i >= self.ze_limit:
                img = self.get_transformed_frame(frame_path)
                frames_ze.append(img)
        return torch.stack(frames_zp), torch.stack(frames_ze), target

    def __len__(self):
        return self.data.shape[0]
    