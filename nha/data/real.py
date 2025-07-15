# This code is intended to replace the original data loading logic in the
# nha/data/real.py file to include 'frame_id' in the batch dictionary and
# include definitions for CLASS_IDCS and digitize_segmap, and add the
# add_argparse_args method to RealDataModule.

# You will need to manually replace the content of nha/data/real.py with the
# code below.

"""
Copyright (c) 2023 Lcx710223 20250710修改。
LCX20250715修改149-150行：{i}修改为：{i:04d},补齐前导零，使正确加载数据集。
This software is licensed under the MIT License.
"""

import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader # Import DataLoader
import argparse # Import argparse
from pytorch_lightning import LightningDataModule # Import LightningDataModule


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


class RealDataModule(LightningDataModule): # Inherit from LightningDataModule
    def __init__(
        self,
        data_path,
        split_config,
        tracking_results_path,
        data_worker=0, # Add data_worker here
        train_batch_size=[2, 1, 1], # Add batch sizes here
        validation_batch_size=[2, 1, 1], # Add batch sizes here
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
        super().__init__() # Call parent constructor
        self.data_path = data_path
        self.split_config = split_config
        self.tracking_results_path = tracking_results_path
        self.data_worker = data_worker # Store data_worker
        self.train_batch_size = train_batch_size # Store batch sizes
        self.validation_batch_size = validation_batch_size # Store batch sizes
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

        self.train_frame_ids = []
        self.val_frame_ids = []
        self.all_frame_ids = [] # Added to store all frame IDs
        self.max_frame_id = 0 # Added to store the maximum frame ID
        self.tracking_results = None
        self.tracking_results_dict = {}


    def setup(self, stage=None):
        """Load data for training and validation."""
        with open(self.split_config, "r") as f:
            split = json.load(f)

        all_frame_dirs = glob.glob(f"{self.data_path}/frame_*")
        self.all_frame_ids = sorted([int(os.path.basename(d).split('_')[-1]) for d in all_frame_dirs if os.path.isdir(d)]) # Get all frame IDs

        if self.all_frame_ids:
            self.max_frame_id = max(self.all_frame_ids) # Calculate max frame ID

        self.train_frame_ids = [i for i in split["train"] if os.path.exists(os.path.join(self.data_path, f"frame_{i:04d}"))]
        self.val_frame_ids = [i for i in split["val"] if os.path.exists(os.path.join(self.data_path, f"frame_{i:04d}"))]

        print(f"[92m[07/06 02:41:29 nha.data.real]: [0mCollected real training dataset containing: {len(self.train_frame_ids)} samples.")
        print(f"[92m[07/06 02:41:29 nha.data.real]: [0mCollected real validation dataset containing: {len(self.val_frame_ids)} samples.")


        # Load tracking results during setup
        if self.load_flame:
            self.tracking_results = np.load(self.tracking_results_path, allow_pickle=True)
            # Convert tracking results to a dictionary for easier lookup by frame_id
            # Filter out keys that cannot be converted to integers (non-frame IDs)
            self.tracking_results_dict = {}
            for k, v in self.tracking_results.items():
                try:
                    self.tracking_results_dict[int(k)] = v
                except ValueError:
                    # Skip keys that are not integer frame IDs
                    pass


    def prepare_data(self):
        """Download data if needed. Not implemented for this dataset."""
        pass

    def train_dataloader(self):
        """Returns the training DataLoader."""
        train_dataset = RealDataset( # Create a separate Dataset class for train/val
            self.data_path,
            self.train_frame_ids,
            self.tracking_results_dict, # Pass the loaded tracking results dict
            load_lmk=self.load_lmk,
            load_seg=self.load_seg,
            load_camera=self.load_camera,
            load_flame=self.load_flame,
            load_normal=self.load_normal,
            load_parsing=self.load_parsing,
            augment=self.augment,
            img_res=self.img_res,
            max_rot=self.max_rot,
            max_transl=self.max_transl,
            noise=self.noise,
            jitter=self.jitter,
        )
        # Use the first element of train_batch_size list for the DataLoader
        return DataLoader(train_dataset, batch_size=self.train_batch_size[0], num_workers=self.data_worker)

    def val_dataloader(self):
        """Returns the validation DataLoader."""
        val_dataset = RealDataset( # Create a separate Dataset class for train/val
            self.data_path,
            self.val_frame_ids,
            self.tracking_results_dict, # Pass the loaded tracking results dict
            load_lmk=self.load_lmk,
            load_seg=self.load_seg,
            load_camera=self.load_camera,
            load_flame=self.load_flame,
            load_normal=self.load_normal,
            load_parsing=self.load_parsing,
            augment=False, # No augmentation for validation
            img_res=self.img_res,
            max_rot=self.max_rot,
            max_transl=self.max_transl,
            noise=self.noise,
            jitter=self.jitter,
        )
         # Use the second element of validation_batch_size list for the DataLoader
        return DataLoader(val_dataset, batch_size=self.validation_batch_size[0], num_workers=self.data_worker)


    @staticmethod # Add staticmethod decorator
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
        parser.add_argument('--split_config', type=str, required=True, help='Path to the split config JSON file')
        parser.add_argument('--tracking_results_path', type=str, required=True, help='Path to the tracking results NPZ file')
        # Add the missing arguments
        parser.add_argument('--data_worker', type=int, default=0, help='Number of data workers')
        # Change nargs for batch sizes to handle list input
        parser.add_argument('--train_batch_size', type=int, nargs='+', default=[2, 1, 1], help='Training batch size for different stages')
        parser.add_argument('--validation_batch_size', type=int, nargs='+', default=[2, 1, 1], help='Validation batch size for different stages')
        parser.add_argument('--load_lmk', type=bool, default=True, help='Load landmarks')
        parser.add_argument('--load_seg', type=bool, default=True, help='Load segmentation')
        parser.add_argument('--load_camera', type=bool, default=True, help='Load camera parameters')
        parser.add_flame = parser.add_argument('--load_flame', type=bool, default=True, help='Load FLAME parameters')
        parser.add_argument('--load_normal', type=bool, default=True, help='Load normal maps')
        parser.add_argument('--load_parsing', type=bool, default=True, help='Load parsing maps')
        parser.add_argument('--augment', type=bool, default=True, help='Apply data augmentation')
        # img_res should be a tuple, parse it accordingly or keep as default and handle in __init__
        parser.add_argument('--img_res', type=int, nargs=2, default=[500, 500], help='Image resolution (width, height)')
        parser.add_argument('--max_rot', type=float, default=20, help='Maximum rotation for augmentation')
        parser.add_argument('--max_transl', type=float, default=0.1, help='Maximum translation for augmentation')
        parser.add_argument('--noise', type=float, default=0.01, help='Noise level for augmentation')
        parser.add_argument('--jitter', type=float, default=0.02, help='Jitter level for augmentation')
        return parser

# Create a separate Dataset class for handling individual samples
class RealDataset(Dataset):
    def __init__(
        self,
        data_path,
        frame_ids,
        tracking_results_dict, # Pass the loaded dictionary
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
        self.frame_ids = frame_ids
        self.tracking_results_dict = tracking_results_dict # Store the dictionary
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
