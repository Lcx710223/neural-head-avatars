# This code is intended to replace the original data loading logic in the
# nha/data/real.py file to include 'frame_id' in the batch dictionary and
# include definitions for CLASS_IDCS and digitize_segmap.

# You will need to manually replace the content of nha/data/real.py with the
# code below.

"""
Copyright (c) 2023 Lcx710223 20250709 V2
This software is licensed under the MIT License.
"""

import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Add definitions for CLASS_IDCS and digitize_segmap
CLASS_IDCS = {
    "background": 0,
    "face": 1,
    "right_eyebrow": 2,
    "left_eyebrow": 3,
    "right_eye": 4,
    "left_eye": 5,
    "nose": 6,
    "mouth": 7,
    "right_lip": 8,
    "left_lip": 9,
    "neck": 10,
    "hair": 11,
    "cloth": 12,
    "ear": 13,
    "neck_l": 14,
    "eye_glass": 15,
    "headwear": 16,
    "earring": 17,
    "necklace": 18,
    "nose_tip": 19,
    "jaw": 20,
}


def digitize_segmap(segmap):
    """
    Converts a segmentation map with varying pixel values to a map with discrete class IDs.

    Args:
        segmap (np.ndarray): The input segmentation map.

    Returns:
        np.ndarray: The digitized segmentation map with class IDs.
    """
    unique_values = np.unique(segmap)
    digitized_map = np.zeros_like(segmap, dtype=np.int64)
    for i, value in enumerate(unique_values):
        digitized_map[segmap == value] = i
    return digitized_map


def get_subdirs(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def load_split(data_path, split_config_path):
    with open(split_config_path, "r") as f:
        split = json.load(f)

    train_dirs = [os.path.join(data_path, f"frame_{i}") for i in split["train"]]
    val_dirs = [os.path.join(data_path, f"frame_{i}") for i in split["val"]]

    return train_dirs, val_dirs


def load_all_samples(data_path):
    frame_dirs = sorted(glob.glob(f"{data_path}/frame_*"))
    return frame_dirs


class RealDataModule(Dataset):
    def __init__(
        self,
        data_path,
        split_config,
        tracking_results_path,
        load_lmk=True,
        load_seg=True,
        load_camera=True,
        load_flame=True,
        load_normal=True,
        load_parsing=True,
        augment=True,
        img_res=(500, 500),
        max_rot=20,
        max_transl=0.1,
        noise=0.01,
        jitter=0.02,
    ):
        self.data_path = data_path
        self.split_config = split_config
        self.tracking_results_path = tracking_results_path
        self.load_lmk = load_lmk
        self.load_seg = load_seg
        self.load_camera = load_camera
        self.load_flame = load_flame
        self.load_normal = load_normal
        self.load_parsing = load_parsing
        self.augment = augment
        self.img_res = img_res
        self.max_rot = max_rot
        self.max_transl = max_transl
        self.noise = noise
        self.jitter = jitter

        # Load frame IDs based on split config
        with open(self.split_config, "r") as f:
            split = json.load(f)

        # Determine if it's a training or validation dataset based on split_config
        # Assuming the dataset is instantiated for either train or validation
        # Filter out frame IDs that don't have a corresponding directory
        all_frame_dirs = glob.glob(f"{data_path}/frame_*")
        if any(frame_dir.endswith(f"frame_{i}") for i in split["train"] for frame_dir in all_frame_dirs):
             self.frame_ids = [i for i in split["train"] if os.path.exists(os.path.join(data_path, f"frame_{i}"))]
             print(f"[92m[07/06 02:41:29 nha.data.real]: [0mCollected real training dataset containing: {len(self.frame_ids)} samples.")
        elif any(frame_dir.endswith(f"frame_{i}") for i in split["val"] for frame_dir in all_frame_dirs):
             self.frame_ids = [i for i in split["val"] if os.path.exists(os.path.join(data_path, f"frame_{i}"))]
             print(f"[92m[07/06 02:41:29 nha.data.real]: [0mCollected real validation dataset containing: {len(self.frame_ids)} samples.")
        else:
            # Fallback: load all samples if split doesn't match
            self.frame_ids = sorted([int(d.split('_')[-1]) for d in all_frame_dirs])
            print(f"[93m[WARNING]: Split config does not match, loading all samples: {len(self.frame_ids)} samples.[0m")


        # Load tracking results
        if self.load_flame:
            self.tracking_results = np.load(self.tracking_results_path, allow_pickle=True)
            # Convert tracking results to a dictionary for easier lookup by frame_id
            self.tracking_results_dict = {int(k): v for k, v in self.tracking_results.item().items()}
        else:
            self.tracking_results_dict = {}


    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        frame_dir = os.path.join(self.data_path, f"frame_{frame_id}")

        data = {"frame_id": frame_id} # Add frame_id to the data dictionary

        # Load image
        img_path = os.path.join(frame_dir, "image.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_res)
            data["image"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
             print(f"[93m[WARNING]: Image not found for frame_id {frame_id} at {img_path}. Skipping this sample.[0m")
             return None


        # Load landmarks
        if self.load_lmk:
            lmk_path = os.path.join(frame_dir, "landmarks.npy")
            if os.path.exists(lmk_path):
                data["landmarks"] = torch.from_numpy(np.load(lmk_path)).float()
            else:
                print(f"[93m[WARNING]: Landmarks not found for frame_id {frame_id} at {lmk_path}.[0m")


        # Load segmentation
        if self.load_seg:
            seg_path = os.path.join(frame_dir, "segmentation.png")
            if os.path.exists(seg_path):
                seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                seg = cv2.resize(seg, self.img_res, interpolation=cv2.INTER_NEAREST)
                data["segmentation"] = torch.from_numpy(seg).float() / 255.0
            else:
                print(f"[93m[WARNING]: Segmentation not found for frame_id {frame_id} at {seg_path}.[0m")


        # Load camera
        if self.load_camera:
            camera_path = os.path.join(frame_dir, "camera.json")
            if os.path.exists(camera_path):
                with open(camera_path, "r") as f:
                    camera = json.load(f)
                data["camera_t"] = torch.tensor(camera["translation"]).float()
                data["camera_R"] = torch.tensor(camera["rotation"]).float()
                data["camera_f"] = torch.tensor(camera["focal_length"]).float()
                data["camera_c"] = torch.tensor(camera["center"]).float()
            else:
                print(f"[93m[WARNING]: Camera parameters not found for frame_id {frame_id} at {camera_path}.[0m")


        # Load FLAME parameters from tracking results using frame_id
        if self.load_flame and frame_id in self.tracking_results_dict:
            flame_params = self.tracking_results_dict[frame_id]
            data["flame_shape"] = torch.tensor(flame_params["shape"]).float()
            data["flame_expr"] = torch.tensor(flame_params["expression"]).float()
            data["flame_pose"] = torch.tensor(flame_params["pose"]).float()
        elif self.load_flame:
            # Handle cases where frame_id is not in tracking_results if necessary
            print(f"[93m[WARNING]: FLAME parameters not found for frame_id {frame_id} in tracking results.[0m")
            # You might want to skip this sample or handle it differently
            return None # Example: return None and filter later


        # Load normal
        if self.load_normal:
            normal_path = os.path.join(frame_dir, "normal.png")
            if os.path.exists(normal_path):
                normal = cv2.imread(normal_path)
                normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
                normal = cv2.resize(normal, self.img_res)
                data["normal"] = torch.from_numpy(normal).permute(2, 0, 1).float() / 255.0
            else:
                print(f"[93m[WARNING]: Normal map not found for frame_id {frame_id} at {normal_path}.[0m")


        # Load parsing
        if self.load_parsing:
            parsing_path = os.path.join(frame_dir, "parsing.png")
            if os.path.exists(parsing_path):
                parsing = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
                parsing = cv2.resize(parsing, self.img_res, interpolation=cv2.INTER_NEAREST)
                data["parsing"] = torch.from_numpy(parsing).long()
            else:
                print(f"[93m[WARNING]: Parsing not found for frame_id {frame_id} at {parsing_path}.[0m")


        # Filter out None values if any of the required files were missing
        if any(value is None for value in data.values()):
            return None
        return data
