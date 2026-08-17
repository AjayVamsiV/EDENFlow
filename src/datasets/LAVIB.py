import os
import pandas as pd
import torch, torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import warnings
import random

warnings.filterwarnings("ignore")


class LAVIBDataset(Dataset):
    def __init__(self, split_name, data_dir="datasets/LAVIB", height=256, width=256, dur_list=(3, 5, 7),
                 dur_weights=(1.0, 0.0, 0.0)):
        self.dataset_name = split_name
        self.h = height
        self.w = width
        self.dur_list = dur_list
        self.dur_weights = dur_weights
        self.data_root = data_dir
        self.meta_data = self.read_data()

    def __len__(self):
        return len(self.meta_data)

    def read_data(self):
        df = pd.read_csv(os.path.join(self.data_root, "annotations", f"{self.dataset_name}.csv"))
        if self.h <= 540:
            split = "segments_downsampled"
        else:
            split = "segments"
        videos = [os.path.join(self.data_root, split,
                               f"{int(row['name'])}_shot{int(row['shot'])}_{int(row['tmp_crop'])}_{int(row['vrt_crop'])}_{int(row['hrz_crop'])}")
                  for _, row in df.iterrows()]

        videos = sorted(videos)
        vid_list = []
        for vid in videos:
            if self.dataset_name == "train":
                dur = random.choices(self.dur_list, weights=self.dur_weights, k=1)[0]
            else:
                dur = self.dur_list[0]
            for i in range(1, 61 - (dur * 2), dur * 2):
                vid_list.append([vid, (i, i + dur * 2)])
        return vid_list
    
    def aspect_crop(self, frames):
        _, _, H, W = frames.shape

        target_ar = self.w / self.h
        current_ar = W / H

        if current_ar > target_ar:
            # crop width
            new_W = int(H * target_ar)
            start_x = random.randint(0, W - new_W)
            frames = frames[:, :, :, start_x:start_x + new_W]
        else:
            # crop height
            new_H = int(W / target_ar)
            start_y = random.randint(0, H - new_H)
            frames = frames[:, :, start_y:start_y + new_H, :]

        return frames

    def __getitem__(self, index):
        vid_path = self.meta_data[index]
        video_fr = torchvision.io.read_video(f"{vid_path[0]}/vid.mp4")[0]
        video_frames = [video_fr[i] for i in range(vid_path[1][0], vid_path[1][1], 2)]
        video_frames = torch.stack(video_frames).float().permute(0, 3, 1, 2)
        if video_frames.shape[-2] < self.h or video_frames.shape[-1] < self.w:
            video_frames = v2.Resize(size=(self.h, self.w), antialias=True)(video_frames)
        frames_num = len(video_frames)
        mid_frame_index = (frames_num - 1) // 2
        frame0 = video_frames[0, ...]
        frame1 = video_frames[-1, ...]
        gt = video_frames[mid_frame_index, ...]
        frames = torch.stack((frame0, frame1, gt), dim=0)
        if self.dataset_name == "train":
            frames = self.aspect_crop(frames)

        frames = F.interpolate(
            frames,
            size=(self.h, self.w),
            mode="bilinear",
            align_corners=False
        )
        return frames

## OR Convert Video to Frames and Load Frames

# import os
# import torch
# from torch.utils.data import Dataset
# import torchvision
# import torch.nn.functional as F
# import random

# class LAVIBDataset(Dataset):
#     def __init__(self, split_name, data_dir, height=256, width=448, dur_list=(3,5,7)):
#         self.dataset_name = split_name
#         self.data_root = data_dir
#         self.h = height
#         self.w = width
#         self.dur_list = dur_list

#         self.video_dirs = self._load_video_dirs()

#     def _load_video_dirs(self):
#         video_dirs = []
#         for root, dirs, files in os.walk(self.data_root):
#             video_dirs.append(root)
#         return sorted(video_dirs)

#     def __len__(self):
#         return len(self.video_dirs)

#     # -----------------------------
#     # Aspect ratio crop
#     # -----------------------------
#     def aspect_crop(self, frames):
#         _, _, H, W = frames.shape
#         target_ar = self.w / self.h
#         current_ar = W / H

#         if current_ar > target_ar:
#             new_W = int(H * target_ar)
#             start_x = random.randint(0, W - new_W)
#             frames = frames[:, :, :, start_x:start_x + new_W]
#         else:
#             new_H = int(W / target_ar)
#             start_y = random.randint(0, H - new_H)
#             frames = frames[:, :, start_y:start_y + new_H, :]

#         return frames

#     def __getitem__(self, index):
#         vid_dir = self.video_dirs[index]

#         # -----------------------------
#         # Load frames
#         # -----------------------------
#         frame_paths = sorted(os.listdir(vid_dir))
#         frames = [torchvision.io.read_image(os.path.join(vid_dir, f)) for f in frame_paths]
#         frames = torch.stack(frames).float()  # (T, 3, H, W)

#         # -----------------------------
#         # Multi-dur sampling
#         # -----------------------------
#         if self.dataset_name == "train":
#             dur = random.choice(self.dur_list)
#         else:
#             dur = self.dur_list[0]

#         step = (dur - 1) // 2  # key formula

#         # ensure valid start, (assuming 30 alternate frames precomputed from each video from 60 frames)
#         max_start = 30 - 2 * step - 1
#         start = random.randint(0, max_start)

#         # -----------------------------
#         # Triplet
#         # -----------------------------
#         frame0 = frames[start]
#         gt     = frames[start + step]
#         frame1 = frames[start + 2 * step]

#         out = torch.stack([frame0, frame1, gt], dim=0)

#         # -----------------------------
#         # Aspect crop → resize
#         # -----------------------------
#         out = self.aspect_crop(out)

#         out = F.interpolate(
#             out,
#             size=(self.h, self.w),
#             mode='bilinear',
#             align_corners=False
#         )

#         return out