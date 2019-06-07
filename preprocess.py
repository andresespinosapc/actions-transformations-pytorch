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

def read_video(videoFile, n_frames_final, id_video, path_frames_video):
        cap = cv2.VideoCapture(videoFile)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_sa = n_frames/n_frames_final
        frames = []
        list_nf = list((np.arange(1,n_frames_final+1)*n_sa).astype(int))
        for f in range(n_frames):
            success,frame = cap.read()
            if (f + 1) in list_nf and success:
                frames.append(frame)
                path_frame = os.path.join(path_frames_video, "frame_{}_{}.jpg".format(str(id_video), str(f)))
                cv2.imwrite(path_frame, frame) 
        return frames


def main(args):
    path = args.input_videos_dir
    output_frame_dir = args.output_frame_dir
    list_class = os.listdir(path)
    path_frames = os.path.join(output_frame_dir, "frames")
    os.mkdir(path_frames) 
    id_video = 0
    id_class = 0
    name_videos ={}
    class_video = {}
    class_to_id = {}
    for name_class in tqdm(list_class):
        class_to_id[id_class] = name_class
        video_folder = os.path.join(path , name_class)
        for video_name in os.listdir(video_folder):
            name_videos[id_video] = video_name
            class_video[id_video] = id_class
            path_frames_video = os.path.join(path_frames, str(id_video))
            os.mkdir(path_frames_video) 
            video_path = os.path.join(video_folder , video_name)
            frames = read_video(video_path, int(args.n_frames_final), id_video, path_frames_video)
            id_video+=1
        id_class+=1
    with open(os.path.join(path_frames,'name_videos.pkl'),'wb') as f:
        pickle.dump(name_videos, f)
    with open(os.path.join(path_frames,'class_video.pkl'),'wb') as f:
        pickle.dump(class_video, f)
    with open(os.path.join(path_frames,'class_to_id.pkl'),'wb') as f:
        pickle.dump(class_to_id, f)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)