### LCX 20250621 大修改。把CKPTS的路径统一到:default_root_dir/checkpoints/last.ckpt里面去。default_root_dir，在ini文件里定义，目前我设置的是LCX-ME01。
### 保存、resume、评估都用LCX-ME01/checkpoints/last.ckpt。不再用Lightning自动拼出的version_x/checkpoints路径。只需保证dirpath和resume_from_checkpoint一致，且都用绝对路径。
### LCX20250622修改了反序列化的代码，主要是加载CKPTS时的信任与安全问题。LCX20250623IMPORT JSON。
### LCX20250623修改了：import json ###LCX20250630修改107行，切换训练场景。

import time
from collections import OrderedDict
import json
from torch.utils.data import DataLoader

from nha.evaluation.eval_suite import evaluate_models
from nha.util.log import get_logger
import os

import pytorch_lightning as pl
from pathlib import Path
import torch
import pathlib  ### LCX:新增
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from configargparse import ArgumentParser as ConfigArgumentParser

from nha.evaluation.visualizations import generate_novel_view_folder, reconstruct_sequence

# 允许 PosixPath 反序列化。LCX:20250622
torch.serialization.add_safe_globals([pathlib.PosixPath])

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
    parser = optimizer_module.add_argparse_args(parser)
    parser = data_module.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = ConfigArgumentParser(parents=[parser], add_help=False)
    parser.add_argument('--config', required=True, is_config_file=True)
    parser.add_argument("--checkpoint_file", type=str, required=False, default="",
                        help="(Not used, always resume from default_root_dir/checkpoints/last.ckpt)")

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

    # 统一的 checkpoint 路径
    args_dict['max_frame_id'] = data.max_frame_id
    ckpt_path = os.path.join(args_dict["default_root_dir"], "checkpoints", "last.ckpt")

    # init optimizer ###LCX20250704 STRICT改为FALSE。
    if os.path.isfile(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        model = optimizer_module.load_from_checkpoint(ckpt_path, strict=False, **args_dict)
    else:
        print(f"No checkpoint found at {ckpt_path}, training from scratch.")
        model = optimizer_module(**args_dict)

    stages = ["offset", "texture", "joint"]
    stage_jumps = [
        args_dict["epochs_offset"],
        args_dict["epochs_offset"] + args_dict["epochs_texture"],
        args_dict["epochs_offset"] + args_dict["epochs_texture"] + args_dict["epochs_joint"]
    ]

    # 对应每个stage，指定NHAOptimizer的stage名称（重要！）
    stage_names = ["flame", "texture", "joint_flame"]

    experiment_logger = TensorBoardLogger(args_dict["default_root_dir"], name="lightning_logs")
    log_dir = Path(experiment_logger.log_dir)

    for i, stage in enumerate(stages):
        # 判断当前 checkpoint 是否存在并可读取 epoch
        if os.path.isfile(ckpt_path):
            try:
                current_epoch = torch.load(ckpt_path)["epoch"]
            except Exception as e:
                print(f"Warning: failed to read current epoch from checkpoint: {e}")
                current_epoch = 0
        else:
            current_epoch = 0

        if current_epoch < stage_jumps[i]:
            logger.info(f"Running the {stage}-optimization stage.")

            # 关键：明确切换NHAOptimizer的current_stage ###LCX20250630修改训练场景的切换。
            if hasattr(model, "set_stage"):
                model.set_stage(stage_names[i])
                logger.info(f"Set model.current_stage to {stage_names[i]}.")

            trainer = pl.Trainer.from_argparse_args(
                args,
                callbacks=model.callbacks,
                resume_from_checkpoint=ckpt_path if os.path.isfile(ckpt_path) else None,
                max_epochs=stage_jumps[i],
                logger=experiment_logger
            )

            trainer.fit(
                model,
                train_dataloaders=data.train_dataloader(batch_size=data._train_batch[i]),
                val_dataloaders=data.val_dataloader(batch_size=data._val_batch[i])
            )

            # 每次都保存到同一个ckpt
            trainer.save_checkpoint(ckpt_path)

    # visualizations and evaluations
    proc_id = int(os.environ.get("SLURM_PROCID", 0))
    if proc_id == 0:
        model_name = log_dir.name
        logger.info("Producing Visualizations")
        vis_path = log_dir / "NovelViewSynthesisResults"
        if os.path.isfile(ckpt_path):
            model = optimizer_module.load_from_checkpoint(ckpt_path, strict=True, **args_dict).eval().cuda()
            generate_novel_view_folder(model, data, angles=[[0, 0], [-30, 0], [-60, 0]],
                                    outdir=vis_path, center_novel_views=True)
            os.system(
                f"for split in train val; do for angle in 0_0 -30_0 -60_0; do ffmpeg -pattern_type glob -i {vis_path}/$split/$angle/'*.png' {vis_path}/{vis_path.parent.name}-$split-$angle.mp4;done;done"
            )
        else:
            print(f"Warning: checkpoint file '{ckpt_path}' not found for evaluation/visualization.")

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
        model_dict = {model_name: ckpt_path}
        model_dict = OrderedDict(model_dict)
        eval_path = log_dir / f"QuantitativeEvaluation-{model_name}.json"

        if os.path.isfile(ckpt_path):
            eval_dict = evaluate_models(models=model_dict, dataloader=dataloader)
            with open(eval_path, "w") as f:
                json.dump(eval_dict, f)  ###需要在顶部加入IMPORT.20250623LCX.

            # scene reconstruction
            reconstruct_sequence(model_dict, dataset=data._val_set, batch_size=bs,
                                savepath=str(log_dir / f"SceneReconstruction{model_name}-val.mp4"))
            reconstruct_sequence(model_dict, dataset=data._train_set, batch_size=bs,
                                savepath=str(log_dir / f"SceneReconstruction-{model_name}-train.mp4"))
        else:
            print(f"Warning: checkpoint file '{ckpt_path}' not found. Evaluation and reconstruction skipped.")
