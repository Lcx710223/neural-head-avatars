### LCX ,COLAB环境GPU受限环境用，由COPILOT编写代码。20250606。
import sys
sys.path.append("/content/neural-head-avatars/")

from nha.util.log import get_logger
from nha.data.real import RealDataset, CLASS_IDCS, frame2id, SEGMENTATION_LABELS
from nha.util.general import get_mask_bbox

from pathlib import Path
from PIL import Image
from torchvision.transforms import *
import torchvision.transforms.functional as ttf

import cv2
import face_alignment
import numpy as np
import json
import subprocess
import shutil
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

from fdlite import (
    FaceDetection,
    FaceLandmark,
    face_detection_to_roi,
    IrisLandmark,
    iris_roi_from_face_landmarks,
)

sys.path.append("deps")

# include dependencies for face normal detection
sys.path.append("deps/face_normals/resnet_unet")
### from face_normals.resnet_unet import ResNetUNet  LCX:有错误，修改如下：
from resnet_unet_model import ResNetUNet

# remove that path and delete the resnet module entry
# because face_parsing has a module with the same name
sys.path.remove("deps/face_normals/resnet_unet")
sys.modules.pop('resnet')

# include dependency for face parsing annotation
sys.path.append("deps/face_parsing")
from face_parsing.model import BiSeNet

# include dependency for segmentation
from RobustVideoMatting.model import MattingNetwork

from nha.data.real import RealDataModule, id2frame, id2view

# set paths to model weights
PARSING_MODEL_PATH = "./assets/face_parsing/model.pth"
NORMAL_MODEL_PATH = "./assets/face_normals/model.pth"
SEG_MODEL_PATH = "./assets/rvm/rvm_mobilenetv3.pth"

# setup logger
logger = get_logger("nha", root=True)

class Video2DatasetConverter:

    IMAGE_FILE_NAME = "image_0000.png"
    ORIGINAL_IMAGE_FILE_NAME = "original_image_0000.png"
    TRANSFORMS_FILE_NAME = "transforms.json"
    LMK_FILE_NAME = "keypoints_static_0000.json"
    SEG_FILE_NAME = "seg_0000.png"
    PARSING_FILE_NAME = "parsing_0000.png"
    NORMAL_FILE_NAME = "normals_0000.png"

    def __init__(
        self,
        video_path,
        dataset_path,
        scale=512,
        force_square=False,
        keep_original_frames=False,
    ):
        self._video_path = Path(video_path)
        self._data_path = Path(dataset_path)
        self._scale = scale
        self._force_square = force_square
        self._keep_original_frames = keep_original_frames
        self._no_iris_landmarks = [-1] * 6
        self._transforms = {}

        assert self._video_path.exists()
        self._data_path.mkdir(parents=True, exist_ok=True)

    def extract_frames(self):
        cap = cv2.VideoCapture(str(self._video_path))
        count = 0

        logger.info("Extracting all frames")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logger.info(f"Extracting frame {count:04d}")

                frame_dir = self._data_path / f"frame_{count:04d}"
                frame_dir.mkdir(exist_ok=True)

                img_file = frame_dir / Video2DatasetConverter.IMAGE_FILE_NAME
                cv2.imwrite(str(img_file), frame)
                count = count + 1

            else:
                break
        cap.release()

    def _get_frame_list(self):
        frame_paths = []
        for frame_dir in self._data_path.iterdir():
            if "frame" in frame_dir.name and frame_dir.is_dir():
                for file in frame_dir.iterdir():
                    if (
                        Video2DatasetConverter.IMAGE_FILE_NAME == file.name
                        and file.is_file()
                    ):
                        frame_paths.append(file)
                        break

        frame_paths = sorted(frame_paths, key=lambda k: frame2id(k.parent.name))
        return frame_paths

    def _get_frame_gen(self):
        def frame_generator():
            for frame_dir in self._data_path.iterdir():
                if "frame" in frame_dir.name and frame_dir.is_dir():
                    for file in frame_dir.iterdir():
                        if (
                            Video2DatasetConverter.IMAGE_FILE_NAME == file.name
                            and file.is_file()
                        ):
                            yield file
                            break

        return frame_generator

    def _write_transforms(self):
        path = self._data_path / Video2DatasetConverter.TRANSFORMS_FILE_NAME
        with open(path, "w") as f:
            json.dump(self._transforms, f)

    def _get_aggregate_bbox(self, bboxes, height, width, padding=20):
        min_l, min_u, min_r, min_b = np.min(bboxes, axis=0)
        max_l, max_u, max_r, max_b = np.max(bboxes, axis=0)

        if self._force_square:
            diff_x, diff_y = (max_r - min_l), (max_b - min_u)
            center_x, center_y = (max_r + min_l) / 2, (max_b + min_u) / 2
            offset = max(diff_x, diff_y)

            if offset >= height // 2 or offset >= width // 2:
                max_offset_x = min(width - center_x, center_x)
                max_offset_y = min(height - center_y, center_y)
                offset = min(max_offset_y, max_offset_x)

            l, r = center_x - offset, center_x + offset
            u, b = center_y - offset, center_y + offset

            if l < 0:
                l, r = 0, r - l
            if r > width:
                l, r = l - (r - width), width
            if u < 0:
                u, b = 0, b - u
            if b > height:
                u, b = u - (b - height), height
        else:
            l, u, r, b = min_l, min_u, max_r, max_b

        l = int(max(0, l - padding))
        r = int(min(width - 1, r + padding))
        u = int(max(0, u - padding))
        b = int(min(height - 1, b + padding))

        assert 0 <= l < r and r < width and 0 <= u < b and b < height

        return l, u, r, b

    def _crop_box_around_seg(self):
        bboxes = []
        for frame in self._get_frame_list():
            parsing_path = frame.parent / Video2DatasetConverter.PARSING_FILE_NAME
            parsing = np.array(Image.open(parsing_path))

            mask = (
                (parsing != CLASS_IDCS["cloth"])
                & (parsing != CLASS_IDCS["background"])
                & (parsing != CLASS_IDCS["neck"])
                & (parsing != CLASS_IDCS["necklace"])
            )
            bbox = get_mask_bbox(mask)
            bboxes.append(bbox)

        assert len(bboxes) > 0
        bboxes = np.stack(bboxes)
        bboxes = bboxes[:, [2, 0, 3, 1]]
        height, width = parsing.shape[:2]
        crop_box = self._get_aggregate_bbox(bboxes, height, width)
        return crop_box

    def apply_transforms(self, crop_seg=True, scale=True, pad_to_square=True):
        crop_box = None
        target_res = None
        pad_dims = None

        for frame in self._get_frame_list():
            frame_id = int(frame.parent.name.split("_")[-1])
            logger.info(f"Transforming frame: {frame_id}")

            if self._keep_original_frames:
                original_filename = (
                    frame.parent / Video2DatasetConverter.ORIGINAL_IMAGE_FILE_NAME
                )
                original_filename = frame.rename(original_filename)
            else:
                original_filename = frame

            img = ttf.to_tensor(Image.open(original_filename))
            x_dim, y_dim = img.shape[-1], img.shape[-2]

            if crop_seg:
                if crop_box is None:
                    crop_box = self._crop_box_around_seg()
                    l, t, r, b = crop_box
                    self._transforms["crop"] = {
                        "x0": l,
                        "y0": t,
                        "w": r - l,
                        "h": b - t,
                    }

                l, t, r, b = crop_box
                img = ttf.crop(img, t, l, b - t, r - l)

            if pad_to_square:
                img, padding = self._pad_to_square(img, mode="constant")
                if pad_dims is None:
                    new_x_dim, new_y_dim = img.shape[-1], img.shape[-2]
                    self._transforms["pad"] = {
                        "w_in": x_dim,
                        "w_out": new_x_dim,
                        "h_in": y_dim,
                        "h_out": new_y_dim,
                    }
                    pad_dims = new_x_dim, new_y_dim
                x_dim, y_dim = pad_dims

            if scale:
                if target_res is None:
                    if x_dim > y_dim:
                        width = self._scale
                        height = int(np.round(y_dim * self._scale / x_dim))
                    else:
                        height = self._scale
                        width = int(np.round(x_dim * self._scale / y_dim))

                    self._transforms["scale"] = {
                        "w_in": x_dim,
                        "w_out": width,
                        "h_in": y_dim,
                        "h_out": height,
                    }
                    target_res = (height, width)

                img = ttf.resize(
                    img, target_res, InterpolationMode.BILINEAR, antialias=True
                )

            img = ttf.to_pil_image(img)
            img.save(frame)

        self._write_transforms()

    def scale(self):
        target_res = None
        for frame in self._get_frame_list():
            frame_id = int(frame.parent.name.split("_")[-1])
            logger.info(f"Scaling frame: {frame_id}")

            for filename in (
                Video2DatasetConverter.IMAGE_FILE_NAME,
                Video2DatasetConverter.PARSING_FILE_NAME,
                Video2DatasetConverter.SEG_FILE_NAME,
            ):
                path = frame.parent / filename
                if not path.exists():
                    logger.warning(f"{path} does not exist")
                    continue

                img = Image.open(path)
                x_dim, y_dim = img.size

                if target_res is None:
                    if x_dim > y_dim:
                        width = self._scale
                        height = int(np.round(y_dim * self._scale / x_dim))
                    else:
                        height = self._scale
                        width = int(np.round(x_dim * self._scale / y_dim))

                    self._transforms["scale"] = {
                        "w_in": x_dim,
                        "w_out": width,
                        "h_in": y_dim,
                        "h_out": height,
                    }
                    self._write_transforms()

                    target_res = (width, height)

                img = img.resize(target_res, Image.BICUBIC)
                img.save(path)

    def _annotate_facial_landmarks(self):
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D, flip_input=True, device="cpu"
        )
        frames = self._get_frame_list()
        landmarks = {}
        bboxes = {}

        for frame in frames:
            frame_id = int(frame.parent.name.split("_")[-1])
            logger.info(f"Annotate facial landmarks for frame: {frame_id}")
            img = np.array(Image.open(frame))
            bbox = fa.face_detector.detect_from_image(img)

            if len(bbox) == 0:
                raise RuntimeError(f"Error: No bounding box found for {frame}!")

            else:
                if len(bbox) > 1:
                    bbox = [bbox[np.argmax(np.array(bbox)[:, -1])]]

                lmks = fa.get_landmarks_from_image(img, detected_faces=bbox)[0]

            landmarks[frame_id] = lmks
            bboxes[frame_id] = bbox[0]

        return landmarks, bboxes

    def _annotate_iris_landmarks(self):
        detect_faces = FaceDetection()
        detect_face_landmarks = FaceLandmark()
        detect_iris_landmarks = IrisLandmark()

        frames = self._get_frame_list()
        landmarks = {}

        for frame in frames:
            frame_id = int(frame.parent.name.split("_")[-1])
            logger.info(f"Annotate iris landmarks for frame: {frame_id}")

            img = Image.open(frame)

            width, height = img.size
            img_size = (width, height)
            lmks = self._no_iris_landmarks

            face_detections = detect_faces(img)
            if len(face_detections) != 1:
                logger.error("Empty iris landmarks")
            else:
                for face_detection in face_detections:
                    try:
                        face_roi = face_detection_to_roi(face_detection, img_size)
                    except ValueError:
                        logger.error("Empty iris landmarks")
                        break

                    face_landmarks = detect_face_landmarks(img, face_roi)
                    if len(face_landmarks) == 0:
                        logger.error("Empty iris landmarks")
                        break

                    iris_rois = iris_roi_from_face_landmarks(face_landmarks, img_size)

                    if len(iris_rois) != 2:
                        logger.error("Empty iris landmarks")
                        break

                    lmks = []
                    for iris_roi in iris_rois[::-1]:
                        try:
                            iris_landmarks = detect_iris_landmarks(img, iris_roi).iris[
                                0:1
                            ]
                        except np.linalg.LinAlgError:
                            logger.error("Failed to get iris landmarks")
                            break

                        for landmark in iris_landmarks:
                            lmks.append(landmark.x * width)
                            lmks.append(landmark.y * height)
                            lmks.append(1.0)

                landmarks[frame_id] = np.array(lmks)

        return landmarks

    def _iris_consistency(self, lm_iris, lm_eye):
        lm_iris = np.array(lm_iris).reshape(1, 3)[:, :2]
        lm_eye = np.array(lm_eye).reshape((-1, 3))[:, :2]

        polygon_eye = mpltPath.Path(lm_eye)
        valid = polygon_eye.contains_points(lm_iris)

        return valid[0]

    def annotate_landmarks(self, add_iris=True):
        lmks_face, bboxes_faces = self._annotate_facial_landmarks()

        if add_iris:
            lmks_iris = self._annotate_iris_landmarks()

            for k in lmks_face.keys():

                lmks_face_i = lmks_face[k].flatten().tolist()
                lmks_iris_i = lmks_iris[k]

                left_face = lmks_face_i[36 * 3 : 42 * 3]
                right_face = lmks_face_i[42 * 3 : 48 * 3]

                right_iris = lmks_iris_i[:3]
                left_iris = lmks_iris_i[3:]

                if not (
                    self._iris_consistency(left_iris, left_face)
                    and self._iris_consistency(right_iris, right_face)
                ):
                    logger.warning(f"Inconsistent iris landmarks for frame {k}")
                    lmks_iris[k] = np.array(self._no_iris_landmarks)

        for k in lmks_face.keys():
            lmk_dict = {}
            lmk_dict["bounding_box"] = bboxes_faces[k].tolist()
            lmk_dict["face_keypoints_2d"] = lmks_face[k].flatten().tolist()

            if add_iris:
                lmk_dict["iris_keypoints_2d"] = lmks_iris[k].flatten().tolist()

            json_dict = {"origin": "face-alignment", "people": [lmk_dict]}
            out_path = (
                self._data_path
                / f"frame_{k:04d}"
                / Video2DatasetConverter.LMK_FILE_NAME
            )

            with open(out_path, "w") as f:
                json.dump(json_dict, f)

    def _correct_eye_labels(self, parsing, lmks):
        parsing = parsing.copy()
        right_lmk_centroid = lmks[36:42].mean(axis=0)
        left_lmk_centroid = lmks[42:48].mean(axis=0)

        eye_mask = (parsing == CLASS_IDCS["l_eye"]) | (parsing == CLASS_IDCS["r_eye"])
        out = cv2.connectedComponentsWithStats(eye_mask.astype(np.uint8), 4, cv2.CV_32S)
        num_comps, comps, _, centroids = out

        for i, centroid in enumerate(centroids):
            if i == 0:
                continue

            correction_mask = comps == i

            if np.linalg.norm(centroid - right_lmk_centroid) < np.linalg.norm(
                centroid - left_lmk_centroid
            ):
                parsing[correction_mask] = CLASS_IDCS["r_eye"]
            else:
                parsing[correction_mask] = CLASS_IDCS["l_eye"]

        return parsing

    def _read_lmks(self, frame):
        lmk_path = frame.parent / Video2DatasetConverter.LMK_FILE_NAME
        with open(lmk_path, "r") as f:
            lmks = json.load(f)["people"][0]["face_keypoints_2d"]
            lmks = np.array(lmks).reshape(-1, 3)
        return lmks

    @staticmethod
    def _pad_to_square(img_tensor, mode="replicate"):
        y_dim, x_dim = img_tensor.shape[-2:]
        if y_dim < x_dim:
            diff = x_dim - y_dim
            top = diff // 2
            bottom = diff - top
            padding = (0, 0, top, bottom)
        elif x_dim < y_dim:
            diff = y_dim - x_dim
            left = diff // 2
            right = diff - left
            padding = (left, right, 0, 0)
        else:
            return img_tensor, (0, 0, 0, 0)
        return (
            torch.nn.functional.pad(img_tensor[None], padding, mode=mode)[0],
            padding,
        )

    @staticmethod
    def _remove_padding(img_tensor, padding):
        left, right, top, bottom = padding
        right = -right if right > 0 else None
        bottom = -bottom if bottom > 0 else None

        return img_tensor[..., top:bottom, left:right]

    def annotate_segmentation(self):
        for frame in self._get_frame_list():
            frame_id = int(frame.parent.name.split("_")[-1])
            parsing_path = frame.parent / Video2DatasetConverter.PARSING_FILE_NAME
            seg_path = frame.parent / Video2DatasetConverter.SEG_FILE_NAME
            logger.info(f"Annotate segmentation for frame: {frame_id}")
            parsing = np.array(Image.open(parsing_path))
            seg = (parsing != CLASS_IDCS["background"]) & (parsing != CLASS_IDCS["cloth"])
            seg = seg.astype(np.uint8) * SEGMENTATION_LABELS["head"]
            cv2.imwrite(str(seg_path), seg)

    def annotate_parsing(self):
        n_classes = 19
        model = BiSeNet(n_classes=n_classes)
        model.load_state_dict(torch.load(PARSING_MODEL_PATH, map_location="cpu"))
        model.eval()

        normalize_img = Compose(
            [
                Normalize(-1, 2),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1)

        matting_model = MattingNetwork("mobilenetv3").eval()
        matting_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location="cpu"))

        rec = [None] * 4
        downsample_ratio = None

        with torch.no_grad():
            for frame in self._get_frame_list():
                frame_id = int(frame.parent.name.split("_")[-1])
                logger.info(f"Annotate parsing for frame: {frame_id}")
                img = ttf.to_tensor(Image.open(frame))[None]
                if downsample_ratio is None:
                    downsample_ratio = min(512 / max(img.shape[-2:]), 1)
                _, pha, *rec = matting_model(img, *rec, downsample_ratio=downsample_ratio)

                seg = pha

                img = img * seg + bgr * (1 - seg)

                img, padding = Video2DatasetConverter._pad_to_square(img, mode="constant")
                img = normalize_img(img)
                padded_img_size = img.shape[-2:]
                img = ttf.resize(img, 512)

                with torch.no_grad():
                    seg_scores = model(img)[0]
                    seg_labels = seg_scores.argmax(1, keepdim=True).int()

                    parsing = seg_labels[0]
                    parsing = ttf.resize(
                        parsing, padded_img_size, InterpolationMode.NEAREST
                    )
                    parsing = Video2DatasetConverter._remove_padding(parsing, padding)[0]
                    parsing = parsing.cpu().numpy()

                    lmks_path = frame.parent / Video2DatasetConverter.LMK_FILE_NAME
                    if lmks_path.exists():
                        lmks = self._read_lmks(frame)
                        parsing = self._correct_eye_labels(parsing, lmks[:, :2])

                    parsing[parsing == CLASS_IDCS["headwear"]] = CLASS_IDCS["hair"]
                    parsing[seg[0,0].cpu().numpy() == 0] = CLASS_IDCS["background"]

                    parsing_img = Image.fromarray(parsing)
                    path = frame.parent / Video2DatasetConverter.PARSING_FILE_NAME
                    parsing_img.save(path)

    @staticmethod
    def get_face_bbox(lmks, img_size):
        umin = np.min(lmks[:, 0])
        umax = np.max(lmks[:, 0])
        vmin = np.min(lmks[:, 1])
        vmax = np.max(lmks[:, 1])

        umean = np.mean((umin, umax))
        vmean = np.mean((vmin, vmax))

        l = round(1.2 * np.max((umax - umin, vmax - vmin)))

        if l > np.min(img_size):
            l = np.min(img_size)

        us = round(np.max((0, umean - float(l) / 2)))
        ue = us + l

        vs = round(np.max((0, vmean - float(l) / 2)))
        ve = vs + l

        if ue > img_size[1]:
            ue = img_size[1]
            us = img_size[1] - l

        if ve > img_size[0]:
            ve = img_size[0]
            vs = img_size[0] - l

        us = int(us)
        ue = int(ue)

        vs = int(vs)
        ve = int(ve)

        return vs, ve, us, ue

    def annotate_face_normals(self):
        model = ResNetUNet(n_class=3)
        model.load_state_dict(torch.load(NORMAL_MODEL_PATH, map_location="cpu"))
        model.eval()

        for frame in self._get_frame_list():
            frame_id = int(frame.parent.name.split("_")[-1])
            logger.info(f"Annotate normals for frame: {frame_id}")

            img = Image.open(frame)
            img = ttf.to_tensor(img)
            img_size = img.shape[-2:]

            lmks = self._read_lmks(frame)
            seg_path = frame.parent / Video2DatasetConverter.SEG_FILE_NAME
            seg = ttf.to_tensor(Image.open(seg_path))

            t, b, l, r = Video2DatasetConverter.get_face_bbox(lmks, img_size)
            crop = img[:, t:b, l:r]
            crop = ttf.resize(crop, 256, InterpolationMode.BICUBIC)
            crop = crop.clamp(-1, 1) * 0.5 + 0.5

            normals = model(crop[None])[0]

            normals = normals / torch.sqrt(torch.sum(normals ** 2, dim=1, keepdim=True))

            rescaled_normals = ttf.resize(
                normals[0], (b - t, r - l), InterpolationMode.BILINEAR
            )

            masked_normals = torch.zeros_like(img)
            masked_normals[:, t:b, l:r] = rescaled_normals.cpu()
            masked_normals = masked_normals * (seg > 0).float()

            normal_img = ttf.to_pil_image(masked_normals * 0.5 + 0.5)
            path = frame.parent / Video2DatasetConverter.NORMAL_FILE_NAME
            normal_img.save(path)

def make_dataset_video(out_path):
    out_path = Path(out_path)
    tmp_path = out_path / "video_tmp"
    tmp_path.mkdir(exist_ok=True)

    data_path = out_path

    data = RealDataset(
        data_path,
        load_lmk=True,
        load_seg=True,
        load_normal=True,
        load_parsing=True,
    )

    N = len(data)
    for sample in data:
        frame_id = sample["frame"]
        logger.info(f"Saving dataset video. Frame: {frame_id} of {N}")
        rgb = sample["rgb"]
        seg = sample["seg"].float()[0]
        lmks = torch.cat([sample["lmk2d"], sample["lmk2d_iris"]], 0)
        lmks = lmks.numpy()
        parsing = sample["parsing"][0]
        normals = sample["normal"] * 0.5 + 0.5
        normals = ttf.to_pil_image(normals)

        _, axes = plt.subplots(1, 5, figsize=(16, 3))
        img = ttf.to_pil_image(rgb * 0.5 + 0.5)
        axes[0].imshow(img)
        axes[1].imshow(seg.numpy())
        axes[2].imshow(parsing.numpy(), vmax=20)
        axes[3].imshow(img)
        lmks_mask = lmks[:, :2] < 0
        lmks_mask = lmks_mask.sum(axis=1) == 0
        axes[3].scatter(lmks[lmks_mask, 0], lmks[lmks_mask, 1], alpha=1, s=3)
        axes[4].imshow(normals)

        plt.savefig(tmp_path / f"frame_{frame_id:04d}.png")
        plt.close()

    subprocess.run(
        [
            "ffmpeg",
            "-pattern_type",
            "glob",
            "-i",
            f"{tmp_path}/*.png",
            "-r",
            "25",
            "-y",
            str(out_path / "dataset.mp4"),
        ]
    )

    shutil.rmtree(tmp_path)

def create_dataset(args):
    converter = Video2DatasetConverter(
        args.video,
        args.out_path,
        args.scale,
        args.force_square,
        args.keep_original_frames,
    )
    converter.extract_frames()
    converter.apply_transforms(crop_seg=False, pad_to_square=False)
    converter.annotate_landmarks()
    converter.annotate_parsing()
    converter.annotate_segmentation()
    converter.annotate_face_normals()
    make_dataset_video(args.out_path)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--scale", type=int, default=512)
    parser.add_argument("--force_square", action="store_true")
    parser.add_argument("--keep_original_frames", action="store_true")
    args = parser.parse_args()

    create_dataset(args)
