import argparse
import importlib
import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_UNETR_CONFIG = {
    "feature_size": 16,
    "hidden_size": 768,
    "mlp_dim": 3072,
    "num_heads": 12,
    "proj_type": "perceptron",
    "norm_name": "instance",
    "res_block": True,
    "dropout_rate": 0.0,
}


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


def with_unetr_defaults(model_cfg, roi_size):
    cfg = dict(model_cfg or {})
    cfg.setdefault("spatial_dims", 3)
    cfg.setdefault("in_channels", 1)
    cfg.setdefault("out_channels", 2)
    cfg.setdefault("img_size", as_plain_list(roi_size))

    for key, value in DEFAULT_UNETR_CONFIG.items():
        cfg.setdefault(key, value)

    return cfg


def unetr_kwargs_for_signature(model_cfg, signature_keys):
    cfg = dict(model_cfg or {})
    sequence_keys = {"img_size"}
    kwargs = {}

    for key in signature_keys:
        if key in {"self", "args", "kwargs"} or key not in cfg:
            continue
        value = cfg[key]
        if key in sequence_keys and value is not None:
            value = as_plain_tuple(value)
        kwargs[key] = value

    return kwargs


def build_unetr(**model_cfg):
    from monai.networks.nets import UNETR

    signature_keys = inspect.signature(UNETR).parameters.keys()
    kwargs = unetr_kwargs_for_signature(model_cfg, signature_keys)
    return UNETR(**kwargs)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train MONAI UNETR with the same data, Azure, checkpoint, "
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
    return parser


def load_config(config_path, dotlist_overrides):
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.load(config_path)
    cli_conf = OmegaConf.from_dotlist(dotlist_overrides)
    cfg = OmegaConf.merge(cfg, cli_conf)

    model_cfg = OmegaConf.to_container(cfg.get("model", {}), resolve=True)
    roi_size = OmegaConf.to_container(cfg.transforms.roi_size, resolve=True)
    unetr_cfg = with_unetr_defaults(model_cfg, roi_size)

    with open_dict(cfg):
        cfg.model = unetr_cfg

    return cfg


def run(args, unknown_args):
    cfg = load_config(args.config, unknown_args)

    train_job = importlib.import_module("scripts.train_job_only_dice")
    train_job.get_segresnet = build_unetr

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
