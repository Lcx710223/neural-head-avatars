### 本文由COPILOT根据PL.DATAMODUL进行注释。20250606。

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from nha.util.log import get_logger
from nha.util.general import get_mask_bbox

logger = get_logger("nha")

# 类别ID，用于分割掩码和标注
CLASS_IDCS = dict(
    background=0,
    face=1, l_brow=2, r_brow=3, l_eye=4, r_eye=5,
    l_ear=7, r_ear=8, nose=10, mouth=11, u_lip=12, l_lip=13,
    neck=14, necklace=15, cloth=16, hair=17, headwear=18,
)

# ----------------------------- RealDataset类 -----------------------------------
class RealDataset(Dataset):
    """
    该类负责从指定的目录结构中读取所有样本（每帧一个样本），
    并在 __getitem__ 按需加载各类数据（图片、分割、关键点、法线等）。
    """

    def __init__(
        self,
        path,
        frame_filter=None,
        tracking_results_path=None,
        tracking_resolution=None,
        load_uv=False,
        load_normal=False,
        load_lmk=False,
        lmk_dynamic=False,
        load_seg=False,
        load_parsing=False,
        load_bbx=False,
        load_flame=False,
        load_camera=False,
        load_light=False,
    ):
        """
        初始化数据集。参数控制是否加载分割、关键点、法线、FLAME参数等。
        :param path: 数据集根目录
        :param frame_filter: 若指定，则只加载部分帧
        :param tracking_results_path: 跟踪器输出的FLAME参数NPZ路径
        :param tracking_resolution: 跟踪图像分辨率
        :param load_xxx: 是否加载xxx类型数据
        """
        super().__init__()
        self._path = Path(path)
        self._has_lmks = load_lmk
        self._has_flame = load_flame
        self._has_camera = load_camera
        self._lmks_dynamic = lmk_dynamic
        self._has_seg = load_seg
        self._has_light = load_light
        self._has_normal = load_normal
        self._has_parsing = load_parsing

        self._eye_info_cache = {}

        # 检查未实现的部分
        if load_uv:
            raise NotImplementedError("Real datasets don't contain uv-maps")
        if load_bbx:
            raise NotImplementedError("Real datasets don't contain bounding boxes")
        if load_flame and tracking_results_path is None:
            raise ValueError("To load flame, must provide tracking_results_path")
        if load_camera and tracking_results_path is None:
            raise ValueError("To load camera, must provide tracking_results_path")

        self._views = []  # 保存所有帧的图片路径
        self._load(frame_filter)  # 填充self._views
        self._tracking_results = (
            dict(np.load(tracking_results_path)) if tracking_results_path is not None else None
        )
        self._tracking_resolution = tracking_resolution

        # 若未指定tracking_resolution但已加载跟踪结果，则自动从中获取分辨率
        if self._tracking_resolution is None and self._tracking_results is not None:
            assert "image_size" in self._tracking_results
            self._tracking_resolution = self._tracking_results["image_size"]

        # 按帧号排序
        self._views = sorted(self._views, key=lambda x: frame2id(x.parent.name))
        self.max_frame_id = max([frame2id(x.parent.name) for x in self._views])

    def _load(self, frame_filter):
        """
        加载所有帧图片路径到self._views。若frame_filter非空，只加载指定帧。
        """
        if frame_filter is None:
            frames = [f for f in os.listdir(self._path) if f.startswith("frame_")]
        else:
            frames = [f"frame_{f:04d}" for f in frame_filter]

        for f in frames:
            self._views.append(self._path / f / "image_0000.png")

        # 检查指定的帧是否都存在
        if frame_filter is not None:
            for f in frame_filter:
                if f not in self.frame_list:
                    raise FileNotFoundError(f"Couldn't find specified frame {f}")

    def __len__(self):
        # 若有tracking_results，长度为两者最小值
        if self._tracking_results is None:
            return len(self._views)
        else:
            return min(len(self._views), len(self._tracking_results["expr"]))

    def _get_eye_info(self, sample):
        """
        解析眼睛区域，计算眼睛分割与关键点的IoU等信息，用于数据质量过滤。
        """
        if not sample["frame"] in self._eye_info_cache:
            eye_info = {}
            parsing = sample["parsing"][0].numpy()
            left_eye_parsing = parsing == CLASS_IDCS["l_eye"]
            right_eye_parsing = parsing == CLASS_IDCS["r_eye"]
            lmks = sample["lmk2d"].numpy().astype(int)

            right_eye = lmks[None, 36:42, :2]
            left_eye = lmks[None, 42:48, :2]

            left_eye_lmks = np.zeros(parsing.shape, dtype=np.uint8)
            right_eye_lmks = np.zeros(parsing.shape, dtype=np.uint8)
            cv2.fillPoly(right_eye_lmks, right_eye, 1)
            cv2.fillPoly(left_eye_lmks, left_eye, 1)

            is_good = []
            for l, p in zip((left_eye_lmks, right_eye_lmks), (left_eye_parsing, right_eye_parsing)):
                intersection = (l > 0) & (p > 0)
                union = (l > 0) | (p > 0)
                sum_union = np.sum(union)
                if sum_union == 0:
                    is_good.append(True)
                    continue
                iou = np.sum(intersection) / sum_union
                if iou > 0.5:
                    is_good.append(True)

            eye_info["valid_eyes"] = len(is_good) == 2

            left_bb = (
                get_mask_bbox(left_eye_parsing)
                if np.sum(left_eye_parsing) > 0
                else [0] * 4
            )
            right_bb = (
                get_mask_bbox(right_eye_parsing)
                if np.sum(right_eye_parsing) > 0
                else [0] * 4
            )
            eye_info["eye_distance"] = [
                left_bb[1] - left_bb[0],
                right_bb[1] - right_bb[0],
            ]

            self._eye_info_cache[sample["frame"]] = eye_info

        return self._eye_info_cache[sample["frame"]]

    def __getitem__(self, i):
        """
        加载单个样本，包括图片、分割、关键点、法线、FLAME参数等。按实际需求灵活加载。
        """
        view = self._views[i]
        frame_path = view.parent
        sample = {}

        # 解析frame号和subject名
        sample["frame"] = frame2id(frame_path.name)
        subject = frame_path.parent.name
        sample["subject"] = subject

        # 加载RGB图像（归一化到-1~1）
        rgba = ttf.to_tensor(Image.open(view).convert("RGBA"))
        sample["rgb"] = (rgba[:3] - 0.5) / 0.5

        # 分割掩码
        if self._has_seg:
            seg_path = view.parent / view.name.replace("image", "seg")
            sample["seg"] = torch.from_numpy(np.array(Image.open(seg_path))).unsqueeze(0)

        # 静态/动态关键点
        if self._has_lmks:
            path = frame_path / view.name.replace("image", "keypoints_static").replace(".png", ".json")
            if self._lmks_dynamic:
                path = frame_path / view.name.replace("image", "keypoints_dynamic").replace(".png", ".json")
            with open(path, "r") as f:
                lmks_info = json.load(f)
                lmks_view = lmks_info["people"][0]["face_keypoints_2d"]
                lmks_iris = lmks_info["people"][0].get("iris_keypoints_2d", None)

            sample["lmk2d"] = (
                torch.from_numpy(np.array(lmks_view)).float()[:204].view(-1, 3)
            )
            if lmks_iris is not None:
                sample["lmk2d_iris"] = torch.from_numpy(np.array(lmks_iris)).float()[:204]
                sample["lmk2d_iris"] = sample["lmk2d_iris"].view(-1, 3)[[1, 0]]

            if not self._lmks_dynamic:
                sample["lmk2d"][:, 2:] = 1.0
                if lmks_iris is not None:
                    if torch.sum(sample["lmk2d_iris"][:, :2] == -1) > 0:
                        sample["lmk2d_iris"][:, 2:] = 0.0
                    else:
                        sample["lmk2d_iris"][:, 2:] = 1.0

        # FLAME参数
        if self._has_flame:
            tr = self._tracking_results
            j = np.where(tr["frame"] == sample["frame"])[0][0]
            sample["flame_shape"] = torch.from_numpy(tr["shape"]).float()
            sample["flame_expr"] = torch.from_numpy(tr["expr"][j]).float()
            sample["flame_pose"] = torch.from_numpy(
                np.concatenate(
                    [
                        tr["rotation"][j],
                        tr["neck_pose"][j],
                        tr["jaw_pose"][j],
                        tr["eyes_pose"][j],
                    ],
                    axis=0,
                )
            ).float()
            sample["flame_trans"] = torch.from_numpy(tr["translation"][j]).float()

        # 相机参数
        if self._has_camera:
            tr = self._tracking_results
            track_h, track_w = self._tracking_resolution
            img_h, img_w = sample["rgb"].shape[-2], sample["rgb"].shape[-1]

            fx_scale = max(track_h, track_w) * img_w / track_w
            fy_scale = max(track_h, track_w) * img_h / track_h
            cx_scale = img_w
            cy_scale = img_h
            if len(tr["K"].shape) == 1:
                sample["cam_intrinsic"] = create_intrinsics_matrix(
                    fx=tr["K"][0] * fx_scale,
                    fy=tr["K"][0] * fy_scale,
                    px=tr["K"][1] * cx_scale,
                    py=tr["K"][2] * cy_scale,
                )
            else:
                assert tr["K"].shape[0] == 3 and tr["K"].shape[1] == 3
                sample["cam_intrinsic"] = torch.from_numpy(tr["K"]).float()
            sample["cam_extrinsic"] = torch.from_numpy(tr["RT"]).float()

        # 光照参数
        if self._has_light:
            tr = self._tracking_results
            if len(tr["light"].shape) == 3:
                sample["light"] = torch.from_numpy(tr["light"][0]).float()
            else:
                sample["light"] = torch.from_numpy(tr["light"]).float()

        # 法线贴图
        if self._has_normal:
            normal_path = view.parent / view.name.replace("image", "normals")
            normal = ttf.to_tensor(Image.open(normal_path).convert("RGB"))
            sample["normal"] = (normal - 0.5) / 0.5
            sample["normal"][1] *= -1

        # parsing掩码
        if self._has_parsing:
            parsing_path = view.parent / view.name.replace("image", "parsing")
            parsing = torch.from_numpy(np.array(Image.open(parsing_path)))[None]
            sample["parsing"] = parsing
            sample.update(self._get_eye_info(sample))

        return sample

    @property
    def frame_list(self):
        frames = []
        for view in self._views:
            frames.append(frame2id(view.parent.name))
        return frames

    @property
    def view_list(self):
        views = []
        for view in self._views:
            views.append(view.name)
        return views

# --------------------------- RealDataModule类 -----------------------------------
class RealDataModule(pl.LightningDataModule):
    """
    用于管理训练/验证划分，按需创建DataLoader并支持PL训练流程。
    """
    def __init__(
        self,
        data_path,
        split_config=None,
        tracking_results_path=None,
        tracking_resolution=None,
        train_batch_size=64,
        validation_batch_size=64,
        data_worker=8,
        load_bbx=False,
        load_seg=False,
        load_flame=False,
        load_lmk=False,
        lmk_dynamic=False,
        load_uv=False,
        load_normal=False,
        load_camera=False,
        load_light=False,
        load_parsing=False,
        **kwargs,
    ):
        super().__init__()
        self._path = Path(data_path)
        self._train_batch = train_batch_size
        self._val_batch = validation_batch_size
        self._workers = data_worker
        self._tracking_results_path = tracking_results_path
        self._tracking_resolution = tracking_resolution
        self._train_set = None
        self._val_set = None
        self._split_config = split_config
        # 控制加载哪些组件
        self._load_components = dict(
            load_bbx=load_bbx,
            load_seg=load_seg,
            load_flame=load_flame,
            load_lmk=load_lmk,
            lmk_dynamic=lmk_dynamic,
            load_uv=load_uv,
            load_camera=load_camera,
            load_normal=load_normal,
            load_light=load_light,
            load_parsing=load_parsing,
        )

    def setup(self, stage=None):
        """
        读取split文件，分别构建训练集和验证集。
        """
        train_split, val_split = self._read_splits(self._split_config)

        self._train_set = RealDataset(
            self._path,
            frame_filter=train_split,
            tracking_results_path=self._tracking_results_path,
            tracking_resolution=self._tracking_resolution,
            **self._load_components,
        )

        logger.info(
            f"Collected real training dataset containing: "
            f"{len(self._train_set)} samples."
        )

        self._val_set = RealDataset(
            self._path,
            frame_filter=val_split,
            tracking_results_path=self._tracking_results_path,
            tracking_resolution=self._tracking_resolution,
            **self._load_components,
        )

        logger.info(
            f"Collected real validation dataset containing: "
            f"{len(self._val_set)} samples."
        )

    def swap_split(self):
        # 交换train与val（用于交叉验证等场景）
        val_set = self._val_set
        train_set = self._train_set
        self._train_set = val_set
        self._val_set = train_set

    @property
    def max_frame_id(self):
        return max(self._train_set.max_frame_id, self._val_set.max_frame_id)

    @staticmethod
    def _read_splits(split_config):
        """
        读取split配置文件，返回训练集和验证集帧号列表。
        """
        if split_config is None:
            return None, None
        else:
            with open(split_config, "r") as f:
                splits = json.load(f)
                train_split = splits["train"]
                val_split = splits["val"]
        return train_split, val_split

    @classmethod
    def add_argparse_args(cls, parser):
        """为命令行参数添加数据集相关选项"""
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--data_worker", type=int, default=8)
        parser.add_argument("--split_config", type=str, required=False, default=None)
        parser.add_argument("--train_batch_size", type=int, default=8, nargs=3)
        parser.add_argument("--validation_batch_size", type=int, default=8, nargs=3)
        parser.add_argument("--tracking_results_path", type=Path, default=None)
        parser.add_argument("--tracking_resolution", type=int, default=None, nargs=2)
        parser.add_argument("--load_uv", action="store_true")
        parser.add_argument("--load_normal", action="store_true")
        parser.add_argument("--load_flame", action="store_true")
        parser.add_argument("--load_bbx", action="store_true")
        parser.add_argument("--load_lmk", action="store_true")
        parser.add_argument("--lmk_dynamic", action="store_true")
        parser.add_argument("--load_seg", action="store_true")
        parser.add_argument("--load_camera", action="store_true")
        parser.add_argument("--load_light", action="store_true")
        parser.add_argument("--load_parsing", action="store_true")
        return parser

    def train_dataloader(self, batch_size=None):
        """返回训练Dataloader"""
        batch_size = batch_size or self._train_batch
        return DataLoader(
            self._train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._workers,
        )

    def val_dataloader(self, batch_size=None):
        """返回验证Dataloader"""
        batch_size = batch_size or self._val_batch
        return DataLoader(
            self._val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._workers,
        )

    def prepare_data(self, *args, **kwargs):
        # 可选：下载、预处理等
        pass

# ------------------------------------------------------------------------------

def frame2id(frame_folder_name):
    """辅助函数，将frame_0001转为int(1)"""
    return int(frame_folder_name.split("_")[-1])
