from PIL import Image
import torch
from torch.utils.data import Dataset
import os, pickle
from torchvision import transforms

class ActionTransformation(Dataset):
    def __init__(self, root, transform = None):
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.root = root
        self.path_frames = os.path.join(self.root, 'frames')
        filename = os.path.join(self.path_frames, 'class_video.pkl')
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        class_video = self.data[index]
        path_frame_video = os.path.join(self.path_frames, str(index))
        frames = []
        for name_frame in os.listdir(path_frame_video):
            img = Image.open(os.path.join(path_frame_video, name_frame)).convert('RGB')
            img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames)
        return frames, class_video

    def __len__(self):
        return len(self.data)
    