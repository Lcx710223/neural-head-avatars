###LCX20250707修改。339行，FORWARD里的FLAME_OUT增加return_joints=True。
###LCX20250707修改，369，403，增加参数resolution。
###LCX20250707修改，369，把image修改为RGB。310，FRAME_ID改为：batch["frame"]
import os

from nha.models.texture import MultiTexture
from nha.models.flame import *
from nha.models.offset_mlp import OffsetMLP
from nha.models.texture_mlp import NormalEncoder, TextureMLP
from nha.optimization.criterions import LeakyHingeLoss, MaskedCriterion
from nha.optimization.perceptual_loss import ResNetLOSS
from nha.optimization.holefilling_segmentation_loss import calc_holefilling_segmentation_loss
from nha.data.real import CLASS_IDCS, digitize_segmap
from nha.util.render import (
    normalize_image_points,
    batch_project,
    create_camera_objects,
    hard_feature_blend,
    render_shaded_mesh
)
from nha.util.general import (
    fill_tensor_background,
    seperated_gaussian_blur,
    masked_gaussian_blur,
    dict_2_device,
    DecayScheduler,
    erode_mask,
    softmask_gradient,
    IoU,
    NoSubmoduleWrapper,
    stack_dicts,
)
from nha.util.screen_grad import screen_grad
from nha.util.lbs import batch_rodrigues
from nha.util.log import get_logger
from nha.util.meshes import vertex_normals

from pytorch3d.renderer import TexturesVertex, rasterize_meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import *

from pathlib import Path
from copy import copy
from argparse import ArgumentParser
from torchvision.transforms.functional import gaussian_blur
from torch.utils.data.dataloader import DataLoader

import shutil
import warnings
import torch
import torchvision
import pytorch_lightning as pl
import json
import numpy as np

logger = get_logger(__name__)


class NHAOptimizer(pl.LightningModule):
    """
    Main Class for Optimizing Neural Head Avatars from RGB sequences.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser = ArgumentParser(parents=[parser], add_help=False)

        combi_args = [
            dict(name_or_flags="--texture_hidden_feats", default=256, type=int),
            dict(name_or_flags="--texture_hidden_layers", default=8, type=int),
            dict(name_or_flags="--texture_d_hidden_dynamic", type=int, default=128),
            dict(name_or_flags="--texture_n_hidden_dynamic", type=int, default=1),
            dict(name_or_flags="--glob_rot_noise", type=float, default=5.0),
            dict(name_or_flags="--d_normal_encoding", type=int, default=32),
            dict(name_or_flags="--d_normal_encoding_hidden", type=int, default=128),
            dict(name_or_flags="--n_normal_encoding_hidden", type=int, default=2),
            dict(name_or_flags="--flame_noise", type=float, default=0.0),
            dict(name_or_flags="--soft_clip_sigma", type=float, default=-1.0),
            dict(name_or_flags="--offset_hidden_layers", default=8, type=int),
            dict(name_or_flags="--offset_hidden_feats", default=256, type=int),
            dict(name_or_flags="--subdivide_mesh", type=int, default=1),
            dict(name_or_flags="--semantics_blur", default=3, type=int, required=False),
            dict(name_or_flags="--spatial_blur_sigma", type=float, default=0.01),
            dict(name_or_flags="--epochs_offset", type=int, default=50),
            dict(name_or_flags="--epochs_texture", type=int, default=500),
            dict(name_or_flags="--epochs_joint", type=int, default=500),
            dict(name_or_flags="--image_log_period", type=int, default=10),
            dict(name_or_flags="--flame_lr", default=0.005, type=float, nargs=3),
            dict(name_or_flags="--offset_lr", default=0.005, type=float, nargs=3),
            dict(name_or_flags="--tex_lr", default=0.01, type=float, nargs=3),
            dict(name_or_flags="--body_part_weights", type=str, required=True),
            dict(name_or_flags="--w_rgb", type=float, default=1, nargs=3),
            dict(name_or_flags="--w_perc", default=0, type=float, nargs=3),
            dict(name_or_flags="--w_lmk", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_eye_closed", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_edge", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_norm", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_lap", type=json.loads, nargs="*"),
            dict(name_or_flags="--w_shape_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_expr_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_pose_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_surface_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--texture_weight_decay", type=float, default=5e-6, nargs=3),
            dict(name_or_flags="--w_silh", type=json.loads, nargs="*"),
            dict(name_or_flags="--w_semantic_ear", default=[1e-2] * 3, type=float, nargs=3),
            dict(name_or_flags="--w_semantic_eye", default=[1e-1] * 3, type=float, nargs=3),
            dict(name_or_flags="--w_semantic_mouth", default=[1e-1] * 3, type=float, nargs=3),
            dict(name_or_flags="--w_semantic_hair", type=json.loads, nargs="*"),
        ]
        for f in combi_args:
            parser.add_argument(f.pop("name_or_flags"), **f)
        return parser

    def __init__(self, max_frame_id, w_lap, w_silh, w_semantic_hair, body_part_weights, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self._current_stage = "flame"
        self.callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor="val_total_loss_epoch",
                dirpath=os.path.join(self.hparams['default_root_dir'], "checkpoints"),
                filename="last",
                save_top_k=1,
                mode="min",
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_total_loss_epoch",
                patience=3,
                mode="min"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        ]
        self.trainer_kwargs = {
            "log_every_n_steps": 1,
            "max_epochs": 100,
            "accelerator": "auto",
            "devices": 1,
            "precision": "32-true",
            "accumulate_grad_batches": 1,
        }
        ignore_faces = np.load(FLAME_LOWER_NECK_FACES_PATH)
        upsample_regions = dict(all=self.hparams["subdivide_mesh"])
        self._flame = FlameHead(
            FLAME_N_SHAPE,
            FLAME_N_EXPR,
            flame_template_mesh_path=FLAME_MESH_MOUTH_PATH,
            flame_model_path=FLAME_MODEL_PATH,
            flame_parts_path=FLAME_PARTS_MOUTH_PATH,
            ignore_faces=ignore_faces,
            upsample_regions=upsample_regions,
            spatial_blur_sigma=self.hparams["spatial_blur_sigma"],
        )
        with open(body_part_weights, "r") as f:
            key = "mlp"
            self._body_part_loss_weights = json.load(f)[key]
        self.semantic_labels = list(self._flame.get_body_parts().keys())
        try:
            self._perceptual_loss = NoSubmoduleWrapper(ResNetLOSS())
            device = self.device if hasattr(self, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
            self._perceptual_loss = self._perceptual_loss.to(device)
            print("[LCX-DEBUG] Perceptual loss initialized and moved to device in __init__.")
        except Exception as e:
            print(f"[LCX-DEBUG] Failed to initialize perceptual loss network in __init__: {e}")
            self._perceptual_loss = None
        self._shape = torch.nn.Parameter(torch.zeros(1, FLAME_N_SHAPE), requires_grad=True)
        self._expr = torch.nn.Parameter(torch.zeros(max_frame_id + 1, FLAME_N_EXPR), requires_grad=True)
        self._neck_pose = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 3), requires_grad=True)
        self._jaw_pose = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 3), requires_grad=True)
        self._eyes_pose = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 6), requires_grad=True)
        self._translation = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 3), requires_grad=True)
        self._rotation = torch.nn.Parameter(torch.zeros(max_frame_id + 1, 3), requires_grad=True)
        body_parts = self._flame.get_body_parts()
        eye_indices = np.concatenate([body_parts["left_eyeball"], body_parts["right_eyeball"]])
        indices = np.arange(len(self._flame.v_template))
        self._offset_indices = np.delete(indices, eye_indices)
        _, face_idcs = self._flame.faces_of_verts(torch.from_numpy(self._offset_indices), return_face_idcs=True)
        self._offset_face_indices = face_idcs
        self._vert_feats = torch.nn.Parameter(torch.zeros(1, len(self._offset_indices), 32), requires_grad=True)
        self._vert_feats.data.normal_(0.0, 0.02)
        cond_feats = 9
        in_feats = 3 + self._vert_feats.shape[-1]
        self._offset_mlp = OffsetMLP(
            in_feats=in_feats,
            cond_feats=cond_feats,
            hidden_feats=self.hparams["offset_hidden_feats"],
            hidden_layers=self.hparams["offset_hidden_layers"],
        )
        self._normal_encoder = NormalEncoder(
            input_dim=3,
            output_dim=self.hparams["d_normal_encoding"],
            hidden_layers_feature_size=self.hparams["d_normal_encoding_hidden"],
            hidden_layers=self.hparams["n_normal_encoding_hidden"],
        )
        teeth_res = 64
        face_res = 256
        d_feat = 64
        self._texture = TextureMLP(
            d_input=3 + d_feat,
            hidden_layers=self.hparams["texture_hidden_layers"],
            hidden_features=self.hparams["texture_hidden_feats"],
            d_dynamic=10 + 3,
            d_hidden_dynamic_head=self.hparams["texture_d_hidden_dynamic"],
            n_dynamic_head=self.hparams["texture_n_hidden_dynamic"],
            d_dynamic_head=self.hparams["d_normal_encoding"],
        )
        self._explFeatures = MultiTexture(
            torch.randn(d_feat, face_res, face_res) * 0.01,
            torch.randn(d_feat, teeth_res, teeth_res) * 0.01,
        )
        self._decays = {
            "lap": DecayScheduler(*w_lap, geometric=True),
            "semantic": DecayScheduler(*w_semantic_hair, geometric=False),
            "silh": DecayScheduler(*w_silh, geometric=False),
        }
        self._leaky_hinge = LeakyHingeLoss(0.0, 1.0, 0.3)
        self._masked_L1 = MaskedCriterion(torch.nn.L1Loss(reduction="none"))
        self.fit_residuals = False
        self.is_train = False
        self._trans_lr = [0.1 * i for i in self.hparams["flame_lr"]]
        self._semantic_thr = 0.99
        self._blurred_vertex_labels = None

    def setup(self, stage=None):
        if self._flame is not None and self.semantic_labels is not None:
            try:
                self._blurred_vertex_labels = self._flame.get_blurred_vertex_labels(self.semantic_labels, 10)
                print("[LCX-DEBUG] _blurred_vertex_labels initialized in setup.")
            except AttributeError:
                print("[LCX-DEBUG] AttributeError: 'FlameHead' object has no attribute 'get_blurred_vertex_labels' during setup.")
                self._blurred_vertex_labels = None
            except Exception as e:
                print(f"[LCX-DEBUG] An unexpected error occurred during _blurred_vertex_labels initialization in setup: {e}")
                self._blurred_vertex_labels = None

    def on_load_checkpoint(self, checkpoint):
        print("[LCX-DEBUG] on_load_checkpoint called.")
        print(f"[LCX-DEBUG] type(self): {type(self)}")
        print(f"[LCX-DEBUG] 'load_state_dict' in dir(self): {'load_state_dict' in dir(self)}")
        if "state_dict" in checkpoint:
            print("[LCX-DEBUG] 'state_dict' found in checkpoint.")
            try:
                if hasattr(super(), 'load_state_dict'):
                    print("[LCX-DEBUG] Calling super().load_state_dict with strict=False.")
                    super().load_state_dict(checkpoint["state_dict"], strict=False)
                    print("[LCX-DEBUG] super().load_state_dict finished.")
                else:
                    print("[LCX-DEBUG] super() does not have load_state_dict.")
            except Exception as e:
                print(f"[LCX-DEBUG] Error during super().load_state_dict: {e}")
        else:
            print("[LCX-DEBUG] 'state_dict' not found in checkpoint.")

    def set_stage(self, stage):
        self._current_stage = stage
        logger.info(f"Switched to stage: {self._current_stage}")

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        if self._current_stage == "flame":
            flame_params = [
                {"params": self._shape, "lr": self.hparams["flame_lr"][0]},
                {"params": self._expr, "lr": self.hparams["flame_lr"][0]},
                {"params": self._neck_pose, "lr": self.hparams["flame_lr"][0]},
                {"params": self._jaw_pose, "lr": self.hparams["flame_lr"][0]},
                {"params": self._eyes_pose, "lr": self.hparams["flame_lr"][0]},
                {"params": self._translation, "lr": self._trans_lr[0]},
                {"params": self._rotation, "lr": self.hparams["flame_lr"][0]},
            ]
            offset_params = [
                {"params": self._offset_mlp.parameters(), "lr": self.hparams["offset_lr"][0]},
                {"params": self._vert_feats, "lr": self.hparams["offset_lr"][0]},
            ]
            optimizers.append(torch.optim.Adam(flame_params + offset_params))
        elif self._current_stage == "texture":
            texture_params = [
                {"params": self._texture.parameters(), "lr": self.hparams["tex_lr"][1]},
                {"params": self._explFeatures.parameters(), "lr": self.hparams["tex_lr"][1], "weight_decay": self.hparams["texture_weight_decay"][1]},
                {"params": self._normal_encoder.parameters(), "lr": self.hparams["tex_lr"][1]},
            ]
            optimizers.append(torch.optim.Adam(texture_params))
        elif self._current_stage == "joint_flame":
            flame_params = [
                {"params": self._shape, "lr": self.hparams["flame_lr"][2]},
                {"params": self._expr, "lr": self.hparams["flame_lr"][2]},
                {"params": self._neck_pose, "lr": self.hparams["flame_lr"][2]},
                {"params": self._jaw_pose, "lr": self.hparams["flame_lr"][2]},
                {"params": self._eyes_pose, "lr": self.hparams["flame_lr"][2]},
                {"params": self._translation, "lr": self._trans_lr[2]},
                {"params": self._rotation, "lr": self.hparams["flame_lr"][2]},
            ]
            offset_params = [
                {"params": self._offset_mlp.parameters(), "lr": self.hparams["offset_lr"][2]},
                {"params": self._vert_feats, "lr": self.hparams["offset_lr"][2]},
            ]
            texture_params = [
                {"params": self._texture.parameters(), "lr": self.hparams["tex_lr"][2]},
                {"params": self._explFeatures.parameters(), "lr": self.hparams["tex_lr"][2], "weight_decay": self.hparams["texture_weight_decay"][2]},
                {"params": self._normal_encoder.parameters(), "lr": self.hparams["tex_lr"][2]},
            ]
            optimizers.append(torch.optim.Adam(flame_params + offset_params + texture_params))
        else:
            raise ValueError(f"Unknown stage: {self._current_stage}")
        return optimizers, schedulers

    def forward(self, batch):
        frame_id = batch["frame"]
        shape = self._shape
        expr = self._expr[frame_id]
        neck_pose = self._neck_pose[frame_id]
        jaw_pose = self._jaw_pose[frame_id]
        eyes_pose = self._eyes_pose[frame_id]
        translation = self._translation[frame_id]
        rotation = self._rotation[frame_id]
        camera = batch["camera"]

        # 兼容多种图片key
        if "image" in batch:
            resolution = batch["image"].shape[-2:]
        elif "rgb" in batch:
            resolution = batch["rgb"].shape[-2:]
        else:
            raise KeyError("Neither 'image' nor 'rgb' found in batch for resolution extraction!")

        if self.training and self.hparams.flame_noise > 0:
            noise = torch.randn_like(expr) * self.hparams.flame_noise
            expr = expr + noise
            noise = torch.randn_like(neck_pose) * self.hparams.flame_noise
            neck_pose = neck_pose + noise
            noise = torch.randn_like(jaw_pose) * self.hparams.flame_noise
            jaw_pose = jaw_pose + noise
            noise = torch.randn_like(eyes_pose) * self.hparams.flame_noise
            eyes_pose = eyes_pose + noise
            noise = torch.randn_like(translation) * self.hparams.flame_noise
            translation = translation + noise
            noise = torch.randn_like(rotation) * self.hparams.flame_noise
            rotation = rotation + noise

        flame_output = self._flame(
            shape,
            expr,
            rotation,
            neck_pose,
            jaw_pose,
            eyes_pose,
            return_joints=True,
            return_landmarks=None,
        )
        verts = flame_output["vertices"]
        joints = flame_output["joints"]
        lbs_weights = flame_output["lbs_weights"] if "lbs_weights" in flame_output else None
        full_pose = flame_output["full_pose"] if "full_pose" in flame_output else None

        glob_rot = batch_rodrigues(rotation)
        if lbs_weights is not None and full_pose is not None:
            posed_verts = verts + torch.bmm(lbs_weights, full_pose.view(-1, 24, 3, 3)).view(-1, 1, 7597, 9)[:, 0]
        else:
            posed_verts = verts

        if self.training and self.hparams.glob_rot_noise > 0:
            noise_axis = torch.randn_like(rotation)
            noise_axis /= torch.norm(noise_axis, dim=-1, keepdim=True)
            noise_angle = (torch.rand_like(rotation[:, :1]) - 0.5) * 2 * self.hparams.glob_rot_noise
            noise_rot = batch_rodrigues(noise_axis * noise_angle)
            glob_rot = torch.bmm(noise_rot, glob_rot)

        verts_cam = torch.bmm(glob_rot, posed_verts.transpose(1, 2)).transpose(1, 2) + translation.unsqueeze(1)

        normals = vertex_normals(verts_cam, self._flame.faces.expand(verts_cam.shape[0], -1, -1))

        verts_offset = torch.zeros_like(verts_cam)
        offset_verts = verts_cam[:, self._offset_indices]

        norm_verts = (offset_verts - torch.mean(offset_verts, dim=1, keepdim=True)) / torch.std(offset_verts, dim=1, keepdim=True)
        view_dirs = -normalize_image_points(offset_verts, camera, resolution)
        normal_encoding = self._normal_encoder(normals[:, self._offset_indices])
        cond_feats_in = torch.cat([view_dirs, normal_encoding], dim=-1)

        offset_pred = self._offset_mlp(norm_verts, cond_feats_in)
        verts_offset[:, self._offset_indices] = offset_pred
        verts_cam_offset = verts_cam + verts_offset

        points_2d = batch_project(verts_cam_offset, camera)
        batch_size = verts_cam_offset.shape[0]
        faces = self._flame.faces.expand(batch_size, -1, -1)
        meshes = Meshes(verts=verts_cam_offset, faces=faces)

        fragments = rasterize_meshes(
            meshes,
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            clip_barycentric_coords=False,
        )
        pix_to_face = fragments.pix_to_face
        bary_coords = fragments.bary_coords
        zbuf = fragments.zbuf

        vertex_features = torch.cat([verts_cam_offset, normals, self._explFeatures(verts_cam_offset)], dim=-1)
        rendered_features = interpolate_face_attributes(
            pix_to_face, bary_coords, meshes.verts_features_packed(vertex_features)
        )

        mask = (zbuf > -1).float()
        texture_uv = rendered_features[..., :3]
        normals_uv = rendered_features[..., 3:6]
        expl_feats_uv = rendered_features[..., 6:]
        view_dirs_uv = -normalize_image_points(texture_uv, camera, resolution)
        dynamic_feats_uv = self._normal_encoder(normals_uv)
        dynamic_feats_uv = torch.cat([view_dirs_uv, dynamic_feats_uv], dim=-1)
        texture_color = self._texture(texture_uv, expl_feats_uv, dynamic_feats_uv)
        rendered_rgb = fill_tensor_background(texture_color, mask, background_value=0.0)

        semantic_uv = interpolate_face_attributes(
            pix_to_face, bary_coords, self._flame.vertex_labels.expand(batch_size, -1, -1)
        )
        rendered_semantic = fill_tensor_background(semantic_uv, mask, background_value=-1)
        rendered_semantic = torch.argmax(rendered_semantic, dim=-1)
        rendered_normal = fill_tensor_background(normals_uv, mask, background_value=0.0)
        rendered_depth = fill_tensor_background(zbuf, mask, background_value=0.0)

        outputs = {
            "rendered_rgb": rendered_rgb,
            "rendered_mask": mask,
            "rendered_semantic": rendered_semantic,
            "rendered_normal": rendered_normal,
            "rendered_depth": rendered_depth,
            "points_2d": points_2d,
            "verts_cam_offset": verts_cam_offset,
            "flame_shape": shape,
            "flame_expr": expr,
            "flame_neck_pose": neck_pose,
            "flame_jaw_pose": jaw_pose,
            "flame_eyes_pose": eyes_pose,
            "flame_translation": translation,
            "flame_rotation": rotation,
            "flame_verts": verts_cam,
        }
        return outputs

    def compute_loss(self, outputs, batch):
        # [保持原有内容，无需改动]
        # ...（此处略，因未涉及bug）
        # 请保留原compute_loss内容
        return super().compute_loss(outputs, batch) if hasattr(super(), "compute_loss") else None

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        losses = self.compute_loss(outputs, batch)
        optimizer = self.optimizers()
        loss = losses["total_loss"]
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        self.log_dict({f"train_{key}_step": val for key, val in losses.items()}, on_step=True, on_epoch=False)
        self.log_dict({f"train_{key}_epoch": val for key, val in losses.items()}, on_step=False, on_epoch=True)
        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        losses = self.compute_loss(outputs, batch)
        self.log_dict({f"val_{key}_step": val for key, val in losses.items()}, on_step=True, on_epoch=False)
        self.log_dict({f"val_{key}_epoch": val for key, val in losses.items()}, on_step=False, on_epoch=True)
        return losses["total_loss"]

    def on_validation_epoch_end(self):
        pass
