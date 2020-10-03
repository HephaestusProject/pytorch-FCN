from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import time

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule

from src.model.net import FCN, ShortChunkCNN_Res
from src.task.runner import AutotaggingRunner


def get_audio(mp3_path):
    waveform, sr = torchaudio.load(mp3_path)
    downsample_resample = torchaudio.transforms.Resample(sr, 16000)
    audio_tensor = downsample_resample(waveform)
    return audio_tensor, audio_tensor.shape[1]


def crop_audio(audio_tensor, input_length=464000):
    random_idx = int(
        np.floor(np.random.random(1) * (audio_tensor.shape[1] - input_length))
    )
    audio_tensor = audio_tensor[:1, random_idx : random_idx + input_length]
    return audio_tensor


def get_config(args: Namespace) -> DictConfig:
    parent_config_dir = Path("conf")
    child_config_dir = parent_config_dir / args.dataset
    model_config_dir = child_config_dir / "model"
    pipeline_config_dir = child_config_dir / "pipeline"
    runner_config_dir = child_config_dir / "runner"

    config = OmegaConf.create()
    model_config = OmegaConf.load(model_config_dir / f"{args.model}.yaml")
    pipeline_config = OmegaConf.load(pipeline_config_dir / f"{args.pipeline}.yaml")
    runner_config = OmegaConf.load(runner_config_dir / f"{args.runner}.yaml")
    config.update(model=model_config, pipeline=pipeline_config, runner=runner_config)
    return config


def main(args) -> None:
    config = get_config(args)

    # load model
    if args.model == "ShortChunkCNN_Res":
        input_length = 59049
        model = ShortChunkCNN_Res(**config.model.params)
        checkpoint_path = f"exp/mtat/ShortChunkCNN_Res/rv01/epoch=27-roc_auc=0.8948-pr_auc=0.4039.ckpt"
    elif args.model == "FCN":
        input_length = 464000
        model = FCN(**config.model.params)
        checkpoint_path = (
            f"exp/mtat/FCN/rv00/epoch=48-roc_auc=0.9025-pr_auc=0.4342.ckpt"
        )
    runner = AutotaggingRunner(model, config.runner)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    runner.load_state_dict(state_dict.get("state_dict"))

    # Load Audio
    audio, audio_length = get_audio(args.audio_path)
    audio = crop_audio(audio, audio_length)
    labels = np.load("dataset/mtat/split/tags.npy")

    # # predict
    runner.eval()
    runner.freeze()
    prediction = runner(audio).squeeze().numpy()

    result = {}
    for pred_index, pred_val in enumerate(prediction):
        if pred_val > args.threshold:
            result[labels[pred_index]] = pred_val
    print(result)

    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", default="FCN", type=str, choices=["ShortChunkCNN_Res", "FCN"]
    )
    parser.add_argument("--dataset", default="mtat", type=str)
    parser.add_argument("--runner", default="rv00", type=str)
    parser.add_argument("--threshold", default=0.4, type=float)
    parser.add_argument("--pipeline", default="pv00", type=str)
    parser.add_argument(
        "--audio_path",
        default="dataset/mtat/test_mp3/sample1.mp3",
        type=str,
        choices=["dataset/test_mp3/sample1.mp3", "dataset/test_mp3/sample2.mp3"],
    )
    args = parser.parse_args()
    main(args)
