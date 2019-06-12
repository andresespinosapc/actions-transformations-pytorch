import numpy as np
import cv2
import pickle
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_videos_dir', required=True)
parser.add_argument('--output_frame_dir', required=True)
parser.add_argument('--n_frames_final', required=True)

def save_frame(path_frames_video, id_video, f, frame):
    path_frame = os.path.join(path_frames_video, "frame_{}_{}.jpg".format(str(id_video), str(f)))
    cv2.imwrite(path_frame, frame) 

def read_video(videoFile, n_frames_final, id_video, path_frames_video):
        cap = cv2.VideoCapture(videoFile)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        for f in range(n_frames):
            success,frame = cap.read()
            if frame is not None and success: 
                size_frame = frame.shape
                width_image = size_frame[1]
                height_image = size_frame[0]
                if width_image == width and height_image == height:
                    frames.append(frame)
        n_frames_filtered = len(frames)
        n_sa = n_frames_filtered/n_frames_final
        list_nf = list((np.arange(1,n_frames_final+1)*n_sa).astype(int))
        frames_final = []
        for k in range(n_frames_filtered):
            if ((k + 1) in list_nf):
                frames_final.append(frames[k]) 
                save_frame(path_frames_video, id_video, k, frames[k])
        if len(frames_final) != n_frames_final:
            print("El video {} tiene {} frames".format(id_video, len(frames_final)))
        return frames_final


def main(args):
    path = args.input_videos_dir
    output_frame_dir = args.output_frame_dir
    list_class = os.listdir(path)
    path_frames = os.path.join(output_frame_dir, "frames")
    os.mkdir(path_frames) 
    id_video = 0
    id_class = 0
    name_videos ={}
    class_video_train = {}
    class_video_val = {}
    class_video_test = {}
    class_to_id = {}
    per_train = 0.7
    per_val = 0.15
    for name_class in tqdm(list_class):
        class_to_id[id_class] = name_class
        video_folder = os.path.join(path , name_class)
        list_video_class = os.listdir(video_folder)
        lim_video_train = int(len(list_video_class)*per_train)
        lim_video_val = int(len(list_video_class)*(per_train + per_val))
        for video_name_id in range(len(list_video_class)):
            video_name = list_video_class[video_name_id]
            name_videos[id_video] = video_name
            if video_name_id < lim_video_train: 
                class_video_train[id_video] = id_class
            elif video_name_id > lim_video_train and video_name_id < lim_video_val:
                class_video_val[id_video] = id_class
            else:
                class_video_test[id_video] = id_class
            path_frames_video = os.path.join(path_frames, str(id_video))
            os.mkdir(path_frames_video) 
            video_path = os.path.join(video_folder , video_name)
            frames = read_video(video_path, int(args.n_frames_final), id_video, path_frames_video)
            id_video+=1
        id_class+=1
    with open(os.path.join(path_frames,'name_videos.pkl'),'wb') as f:
        pickle.dump(name_videos, f)
    with open(os.path.join(path_frames,'class_video_train.pkl'),'wb') as f:
        pickle.dump(class_video_train, f)
    with open(os.path.join(path_frames,'class_video_val.pkl'),'wb') as f:
        pickle.dump(class_video_val, f)
    with open(os.path.join(path_frames,'class_video_test.pkl'),'wb') as f:
        pickle.dump(class_video_test, f)
    with open(os.path.join(path_frames,'class_to_id.pkl'),'wb') as f:
        pickle.dump(class_to_id, f)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)