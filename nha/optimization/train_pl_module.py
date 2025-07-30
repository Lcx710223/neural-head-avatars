###JULES-20250726,修改第88行：原来的train_dataloader 修改为：train_dataloaders。
###JULES-20250727。修改38，85（新165行），93行（新175）。移除了对已弃用的add_argparse_args的调用，并相应地手动添加了参数；同时，更新了trainer.fit方法的参数，以适应新版本的 API。
###JULES-20250729，修改第20行，加入了：import pathlib torch.serialization.add_safe_globals([pathlib.PosixPath])
###JULES-20250730,修改第200行。保存了STAGE_OPTIM.CKPT之后，再覆盖LAST.CKPT。JULES确认了原文CKPT保存逻辑有缺陷。原文只保存了一个特定于阶段的CKPT，然后错误地将检查点路径更新到last.ckpt而不实际创建文件。

import json
import time
from collections import OrderedDict

from torch.utils.data import DataLoader

from nha.evaluation.eval_suite import evaluate_models
from nha.util.log import get_logger
import os

import pytorch_lightning as pl
from pathlib import Path
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from configargparse import ArgumentParser as ConfigArgumentParser
import pathlib
torch.serialization.add_safe_globals([pathlib.PosixPath])


from nha.evaluation.visualizations import generate_novel_view_folder, reconstruct_sequence

logger = get_logger(__name__)


def train_pl_module(optimizer_module, data_module, args=None):
    """
    optimizes an instance of the given optimization module on an instance of a given data_module. Takes arguments
    either from CLI or from 'args'

    :param optimizer_module:
    :param data_module:
    :param args: list similar to sys.argv to manually insert args to parse from
    :return:
    """
    # creating argument parser
    parser = ArgumentParser()
    
    # JULES-20250727中文注释：
    # 以下代码手动添加了 NHAOptimizer 和 RealDataModule 的参数。
    # 这是因为在 pytorch-lightning 1.9.5 版本中，`add_argparse_args` 方法已被弃用。
    # 我们通过直接定义这些参数来替代旧的调用方式。
    combi_args = [
        # texture settings
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

        # geometry refinement mlp settings
        dict(name_or_flags="--offset_hidden_layers", default=8, type=int),
        dict(name_or_flags="--offset_hidden_feats", default=256, type=int),

        # FLAME settings
        dict(name_or_flags="--subdivide_mesh", type=int, default=1),
        dict(name_or_flags="--semantics_blur", default=3, type=int, required=False),
        dict(name_or_flags="--spatial_blur_sigma", type=float, default=0.01),

        # training timeline settings
        dict(name_or_flags="--epochs_offset", type=int, default=50,
             help="Until which epoch to train flame parameters and offsets jointly"),
        dict(name_or_flags="--epochs_texture", type=int, default=500,
             help="Until which epoch to train texture while keeping model fixed"),
        dict(name_or_flags="--epochs_joint", type=int, default=500,
             help="Until which epoch to train model jointly while keeping model fixed"),
        dict(name_or_flags="--image_log_period", type=int, default=10),

        # lr settings
        dict(name_or_flags="--flame_lr", default=0.005, type=float, nargs=3),
        dict(name_or_flags="--offset_lr", default=0.005, type=float, nargs=3),
        dict(name_or_flags="--tex_lr", default=0.01, type=float, nargs=3),

        # loss weights
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

    # Manually adding arguments from RealDataModule
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
    
    parser = pl.Trainer.add_argparse_args(parser)

    parser = ConfigArgumentParser(parents=[parser], add_help=False)
    parser.add_argument('--config', required=True, is_config_file=True)
    parser.add_argument("--checkpoint_file", type=str, required=False, default="",
                        help="checkpoint to load model from")

    args = parser.parse_args() if args is None else parser.parse_args(args)
    args_dict = vars(args)

    print(f"Start Model training with the following configuration: \n {parser.format_values()}")

    # init datamodule
    while True:
        try:
            data = data_module(**args_dict)
            data.setup()
            break
        except FileNotFoundError as e1:
            print(e1)
            print("Retry data loading after 2 minutes ... zZZZZ")
            time.sleep(2 * 60)

    # init optimizer
    args_dict['max_frame_id'] = data.max_frame_id

    # JULES-20250730-14:38 中文注释：
    # 动态查找最新的检查点文件，以避免在新版本目录创建时出现 FileNotFoundError。
    log_root = Path(args_dict["default_root_dir"]) / "lightning_logs"
    if log_root.exists():
        versions = sorted([d.name for d in log_root.iterdir() if d.is_dir() and d.name.startswith("version_")],
                          key=lambda x: int(x.split('_')[1]), reverse=True)
        if versions:
            latest_version_dir = log_root / versions[0]
            latest_ckpt = latest_version_dir / "checkpoints" / "last.ckpt"
            if latest_ckpt.exists() and not args.checkpoint_file:
                logger.info(f"Found latest checkpoint: {latest_ckpt}")
                args_dict["checkpoint_file"] = str(latest_ckpt)

    if args.checkpoint_file:
        model = optimizer_module.load_from_checkpoint(args.checkpoint_file, strict=True, **args_dict)
    else:
        model = optimizer_module(**args_dict)

    stages = ["offset", "texture", "joint"]
    stage_jumps = [args_dict["epochs_offset"], args_dict["epochs_offset"] + args_dict["epochs_texture"],
                   args_dict["epochs_offset"] + args_dict["epochs_texture"] + args_dict["epochs_joint"]]

    experiment_logger = TensorBoardLogger(args_dict["default_root_dir"],
                                          name="lightning_logs")
    log_dir = Path(experiment_logger.log_dir)

    for i, stage in enumerate(stages):
        current_epoch = torch.load(args_dict["checkpoint_file"])["epoch"] if args_dict["checkpoint_file"] else 0
        if current_epoch < stage_jumps[i]:
            logger.info(f"Running the {stage}-optimization stage.")

            ckpt_file = args_dict["checkpoint_file"] if args_dict["checkpoint_file"] else None
            trainer = pl.Trainer.from_argparse_args(args, callbacks=model.callbacks,
                                                    max_epochs=stage_jumps[i],
                                                    logger=experiment_logger)

            # JULES-20250727中文注释：
            # 在 pytorch-lightning 1.9.5 版本中，`resume_from_checkpoint` 参数已从 `Trainer` 的构造函数中移除，
            # 并移至 `fit` 方法的 `ckpt_path` 参数。同时，`train_dataloader` 和 `val_dataloader`
            # 已分别重命名为 `train_dataloaders` 和 `val_dataloaders`。
            trainer.fit(model,
                        train_dataloaders=data.train_dataloader(batch_size=data._train_batch[i]),
                        val_dataloaders=data.val_dataloader(batch_size=data._val_batch[i]),
                        ckpt_path=ckpt_file)

            stage_ckpt_path = Path(trainer.log_dir) / "checkpoints" / (stage + "_optim.ckpt")
            trainer.save_checkpoint(stage_ckpt_path)
            
            # JULES-20250730-14:59 中文注释：
            # 将特定阶段的检查点重命名为 last.ckpt，以确保下一阶段可以正确加载。
            last_ckpt_path = Path(trainer.log_dir) / "checkpoints" / "last.ckpt"
            # 使用 os.replace 来原子地重命名文件，如果 last.ckpt 已存在，则覆盖它。
            os.replace(stage_ckpt_path, last_ckpt_path)
            
            args_dict["checkpoint_file"] = str(last_ckpt_path)

    # visualizations and evaluations
    proc_id = int(os.environ.get("SLURM_PROCID", 0))
    if proc_id == 0:
        model_name = log_dir.name
        logger.info("Producing Visualizations")
        vis_path = log_dir / "NovelViewSynthesisResults"
        model = optimizer_module.load_from_checkpoint(args_dict["checkpoint_file"],
                                                      strict=True, **args_dict).eval().cuda()
        generate_novel_view_folder(model, data, angles=[[0, 0], [-30, 0], [-60, 0]],
                                   outdir=vis_path, center_novel_views=True)
        # os.system("module load FFmpeg")
        os.system(
            f"for split in train val; do for angle in 0_0 -30_0 -60_0; do ffmpeg -pattern_type glob -i {vis_path}/$split/$angle/'*.png' {vis_path}/{vis_path.parent.name}-$split-$angle.mp4;done;done")

        # freeing up space
        try:
            del trainer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass

        # quantitative evaluation of val dataset
        bs = max(args_dict["validation_batch_size"])
        dataloader = DataLoader(data._val_set, batch_size=bs,
                                num_workers=bs, shuffle=False)
        model_dict = {model_name: args_dict["checkpoint_file"]}
        model_dict = OrderedDict(model_dict)
        eval_path = log_dir / f"QuantitativeEvaluation-{model_name}.json"

        eval_dict = evaluate_models(models=model_dict, dataloader=dataloader)
        with open(eval_path, "w") as f:
            json.dump(eval_dict, f)

        # scene reconstruction
        reconstruct_sequence(model_dict, dataset=data._val_set, batch_size=bs,
                             savepath=str(log_dir / f"SceneReconstruction{model_name}-val.mp4"))
        reconstruct_sequence(model_dict, dataset=data._train_set, batch_size=bs,
                             savepath=str(log_dir / f"SceneReconstruction-{model_name}-train.mp4"))
