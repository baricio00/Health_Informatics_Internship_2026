import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_FEATURE_SIZE = 48
_PRETRAINED_SWIN_ENCODER = None


def as_plain_list(values):
    if values is None:
        return None
    if isinstance(values, str):
        values = values.strip().strip("[]()").split(",")
    return [int(value) for value in values]


def as_plain_tuple(values):
    if values is None:
        return None
    if isinstance(values, tuple):
        return values
    return tuple(values)


def with_swin_defaults(model_cfg, roi_size):
    cfg = dict(model_cfg or {})
    cfg.setdefault("spatial_dims", 3)
    cfg.setdefault("in_channels", 1)
    cfg.setdefault("out_channels", 2)
    cfg.setdefault("feature_size", DEFAULT_FEATURE_SIZE)
    cfg.setdefault("use_checkpoint", True)
    cfg.setdefault("img_size", as_plain_list(roi_size))
    return cfg


def swin_kwargs_for_signature(model_cfg, signature_keys):
    cfg = dict(model_cfg or {})
    sequence_keys = {"img_size", "depths", "num_heads"}
    kwargs = {}

    for key in signature_keys:
        if key in {"self", "args", "kwargs"} or key not in cfg:
            continue
        value = cfg[key]
        if key in sequence_keys and value is not None:
            value = as_plain_tuple(value)
        kwargs[key] = value

    return kwargs


def torch_load_weights(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_pretrained_swin_encoder(model, weights_path):
    weights = torch_load_weights(weights_path)
    if hasattr(model, "load_from"):
        model.load_from(weights=weights)
        return

    state_dict = (
        weights.get("state_dict", weights) if isinstance(weights, dict) else weights
    )
    model.load_state_dict(state_dict, strict=False)


def build_swin_unetr(**model_cfg):
    from monai.networks.nets import SwinUNETR

    signature_keys = inspect.signature(SwinUNETR).parameters.keys()
    kwargs = swin_kwargs_for_signature(model_cfg, signature_keys)
    model = SwinUNETR(**kwargs)

    if _PRETRAINED_SWIN_ENCODER:
        load_pretrained_swin_encoder(model, _PRETRAINED_SWIN_ENCODER)
        if int(os.environ.get("RANK", 0)) == 0:
            print(
                f"Loaded pretrained Swin UNETR encoder: {_PRETRAINED_SWIN_ENCODER}",
                flush=True,
            )

    return model


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train MONAI Swin UNETR with the same data, Azure, checkpoint, "
            "and OmegaConf override contract used by the project training jobs."
        )
    )
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--split_csv", type=str, default="cv_splits.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for local runs. Azure/torchrun DDP still uses LOCAL_RANK when CUDA is available.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help=(
            "Record recoverable trial failures such as CUDA OOM in summary.json "
            "and exit 0 so sweep pipelines can continue."
        ),
    )
    parser.add_argument(
        "--pretrained_swin_encoder",
        type=str,
        default=None,
        help=(
            "Optional path to MONAI's self-supervised Swin UNETR encoder weights "
            "(for example model_swinvit.pt). This is separate from --checkpoint, "
            "which is for full project checkpoints."
        ),
    )
    return parser


def load_config(config_path, dotlist_overrides):
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.load(config_path)
    cli_conf = OmegaConf.from_dotlist(dotlist_overrides)
    cfg = OmegaConf.merge(cfg, cli_conf)

    model_cfg = OmegaConf.to_container(cfg.get("model", {}), resolve=True)
    roi_size = OmegaConf.to_container(cfg.transforms.roi_size, resolve=True)
    swin_cfg = with_swin_defaults(model_cfg, roi_size)

    with open_dict(cfg):
        cfg.model = swin_cfg

    return cfg


def run(args, unknown_args):
    global _PRETRAINED_SWIN_ENCODER

    _PRETRAINED_SWIN_ENCODER = args.pretrained_swin_encoder
    cfg = load_config(args.config, unknown_args)

    train_job = importlib.import_module("scripts.train_job_only_dice")
    train_job.get_segresnet = build_swin_unetr

    try:
        train_job.main(args, cfg)
    except Exception as exc:
        if args.continue_on_error and train_job.is_cuda_oom(exc):
            print(
                "CUDA OOM recorded as a failed sweep trial. "
                "Exiting 0 because --continue_on_error was set.",
                flush=True,
            )
            train_job.record_training_failure(args, cfg, exc, "cuda_oom")
            sys.exit(0)
        raise


def main():
    parser = build_arg_parser()
    args, unknown_args = parser.parse_known_args()
    run(args, unknown_args)


if __name__ == "__main__":
    main()
