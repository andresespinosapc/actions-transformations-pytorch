from PIL import Image
import torch
from torch.utils.data import Dataset
import os, pickle
from torchvision import transforms

class UCF101(Dataset):
    def __init__(self, zp_limit, ze_limit, root, split, transform = None):
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.zp_limit = zp_limit
        self.ze_limit = ze_limit
        self.root = root
        # self.path_frames = os.path.join(self.root, 'frames')
        self.path_frames = self.root
        filename = os.path.join(self.path_frames, 'class_video_{}.pkl'.format(split))
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        self.id_videos = list(self.data.keys())
    
    def get_transformed_frame(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __getitem__(self, index):
        id_video = self.id_videos[index]
        class_video = self.data[id_video]
        path_frame_video = os.path.join(self.path_frames, str(id_video))
        frames_zp = []
        frames_ze = []
        list_frames = os.listdir(path_frame_video)
        for fid in range(len(list_frames)):
            if fid < self.zp_limit: 
                img = self.get_transformed_frame(os.path.join(path_frame_video, list_frames[fid]))
                frames_zp.append(img)
            elif fid >= self.ze_limit:
                img = self.get_transformed_frame(os.path.join(path_frame_video, list_frames[fid]))
                frames_ze.append(img)
        return torch.stack(frames_zp), torch.stack(frames_ze), class_video

    def __len__(self):
        return len(self.id_videos)
    