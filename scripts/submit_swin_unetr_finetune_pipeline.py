import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import submit_swin_unetr_finetune_queue as queue


DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_AZURE_COMPUTE = "azureml:clusterprdwe-g-t2-vzhst6"
DEFAULT_AZURE_ENVIRONMENT = "azureml:media-env:8"
DEFAULT_AZURE_INPUT_DATA = "azureml:LEMS-CT-NIfTI:1"
DEFAULT_AZURE_TENANT_ID = "ebbb4fc3-587b-477c-a1e8-ee47d8c02546"
DEFAULT_AZURE_SUBSCRIPTION_ID = "ab211f7b-463f-4833-9605-d260e596a35a"
DEFAULT_AZURE_RESOURCE_GROUP = "73da10b4-5dff-54e2-db0d-3a1fab882485"
DEFAULT_AZURE_WORKSPACE_NAME = "73da10b45dff54e2db0d3a1fab882485"
DEFAULT_EXPERIMENT_NAME = "myocardium_swin_unetr_finetune_pipeline"
DEFAULT_OUTPUT_DIR = "outputs/swin_unetr_finetune_pipeline"
DEFAULT_CONFIG = "config/train_config.yaml"
DEFAULT_SPLIT_CSV = "data/cv_splits.csv"
DEFAULT_PRETRAINED_SWIN_ENCODER = "weights/model_swinvit.pt"
DEFAULT_OVERRIDES = [
    "training.train_batch_size=1",
    "training.accumulation_steps=4",
    "inference.sw_batch_size=1",
]


@dataclass(frozen=True)
class FoldFineTuneSpec:
    fold: int
    checkpoint_path: str
    name: str
    display_name: str
    command: str


def azure_compute_name(compute_ref):
    compute_ref = str(compute_ref).strip()
    return compute_ref.removeprefix("azureml:")


def safe_azure_name(value, max_length=240):
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe or "job")[:max_length]


def checkpoint_path_for_fold(args, fold, index):
    if args.checkpoint_job_names:
        if len(args.checkpoint_job_names) != len(args.folds):
            raise ValueError(
                "Provide one checkpoint job name per fold when using "
                "--checkpoint_job_names."
            )
        return queue.checkpoint_path_from_job_name(args.checkpoint_job_names[index], fold)

    if args.checkpoint_job_name:
        return queue.checkpoint_path_from_job_name(args.checkpoint_job_name, fold)

    return queue.checkpoint_for_fold(args.checkpoint_path, fold)


def fold_command(args, fold):
    parts = [
        "python scripts/swin_UNETR.py",
        "--input_data ${{inputs.input_data}}",
        "--output_model ${{outputs.output_model}}",
        f"--split_csv {args.split_csv}",
        f"--fold {fold}",
        f"--config {args.config}",
        "--device cuda",
        "--checkpoint ${{inputs.checkpoint_file}}",
        f"--pretrained_swin_encoder {args.pretrained_swin_encoder}",
        *args.extra_overrides,
    ]
    return " ".join(parts)


def build_fold_specs(args):
    specs = []
    for index, fold in enumerate(args.folds):
        checkpoint_path = checkpoint_path_for_fold(args, fold, index)
        name = safe_azure_name(f"swin_unetr_finetune_fold_{fold}")
        specs.append(
            FoldFineTuneSpec(
                fold=fold,
                checkpoint_path=checkpoint_path,
                name=name,
                display_name=f"Myocardium Swin UNETR Fine-tuning Fold {fold}",
                command=fold_command(args, fold),
            )
        )
    return specs


def make_pipeline_command_component(spec, args, has_previous):
    from azure.ai.ml import Input, Output, command

    command_text = spec.command
    if has_previous:
        command_text = (
            "test -f ${{inputs.previous_marker}}/done.txt && " + command_text
        )
    command_text = (
        command_text
        + " && mkdir -p ${{outputs.completion_marker}}"
        + " && printf done > ${{outputs.completion_marker}}/done.txt"
    )

    inputs = {
        "input_data": Input(type="uri_folder"),
        "checkpoint_file": Input(type="uri_file"),
    }
    if has_previous:
        inputs["previous_marker"] = Input(type="uri_folder")

    return command(
        name=spec.name,
        display_name=spec.display_name,
        description=f"Swin UNETR fine-tuning for fold {spec.fold}",
        code=str(PROJECT_ROOT),
        command=command_text,
        environment=args.azure_environment,
        compute=azure_compute_name(args.azure_compute),
        inputs=inputs,
        outputs={
            "output_model": Output(type="uri_folder"),
            "completion_marker": Output(type="uri_folder"),
        },
        instance_count=args.azure_instance_count,
        is_deterministic=False,
    )


def build_azure_pipeline(args):
    from azure.ai.ml import Input
    from azure.ai.ml.dsl import pipeline

    specs = build_fold_specs(args)
    compute_name = azure_compute_name(args.azure_compute)

    @pipeline(
        display_name=args.pipeline_display_name,
        description=(
            "Swin UNETR fine-tuning across all requested folds. Child jobs are "
            "created up front and run sequentially through marker dependencies."
        ),
        experiment_name=args.azure_experiment_name,
        default_compute=compute_name,
    )
    def swin_finetune_pipeline(input_data):
        previous_marker = None
        last_marker = None

        for spec in specs:
            component = make_pipeline_command_component(
                spec,
                args,
                has_previous=previous_marker is not None,
            )
            node_inputs = {
                "input_data": input_data,
                "checkpoint_file": Input(
                    type="uri_file",
                    path=spec.checkpoint_path,
                    mode="download",
                ),
            }
            if previous_marker is not None:
                node_inputs["previous_marker"] = previous_marker

            node = component(**node_inputs)
            node.name = spec.name
            node.display_name = spec.display_name
            node.tags = {
                "fold": str(spec.fold),
                "checkpoint_path": spec.checkpoint_path,
                "model": "swin_unetr",
                "stage": "fine_tuning",
            }
            previous_marker = node.outputs.completion_marker
            last_marker = previous_marker

        return {"pipeline_done": last_marker}

    return swin_finetune_pipeline(
        input_data=Input(
            type="uri_folder",
            path=args.azure_input_data,
            mode="ro_mount",
        )
    )


def get_azure_ml_client(args):
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

    if args.azure_auth_mode == "browser":
        credential = InteractiveBrowserCredential(tenant_id=args.azure_tenant_id)
    else:
        credential = DefaultAzureCredential()

    return MLClient(
        credential=credential,
        subscription_id=args.azure_subscription_id,
        resource_group_name=args.azure_resource_group,
        workspace_name=args.azure_workspace_name,
    )


def serialize_azure_job(returned_job):
    return {
        "name": returned_job.name,
        "id": returned_job.id,
        "status": getattr(returned_job, "status", ""),
        "studio_url": getattr(returned_job, "studio_url", ""),
    }


def submit_pipeline(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_job = build_azure_pipeline(args)
    pipeline_yaml = out_dir / "swin_unetr_finetune_pipeline_job.yml"
    pipeline_job.dump(pipeline_yaml)
    print(f"Azure pipeline YAML: {pipeline_yaml}", flush=True)

    if args.dry_run:
        print("Dry run only. Azure pipeline not submitted.", flush=True)
        return pipeline_yaml

    ml_client = get_azure_ml_client(args)
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    payload = serialize_azure_job(returned_job)
    submission_json = out_dir / "swin_unetr_finetune_pipeline_submission.json"
    submission_json.write_text(json.dumps(payload, indent=2))

    print(f"Submitted pipeline: {payload['name']}", flush=True)
    if payload["studio_url"]:
        print(f"Studio URL: {payload['studio_url']}", flush=True)
    return submission_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Submit one Azure ML parent pipeline containing Swin UNETR "
            "fine-tuning child jobs for all requested folds."
        )
    )
    checkpoint_source = parser.add_mutually_exclusive_group(required=True)
    checkpoint_source.add_argument(
        "--checkpoint_path",
        help=(
            "Checkpoint URI template. Use {fold} for per-fold paths, for "
            "example azureml://.../fold_{fold}/best_metric_model.pth."
        ),
    )
    checkpoint_source.add_argument(
        "--checkpoint_job_name",
        help=(
            "One completed Azure ML training job whose output_model contains "
            "fold_<fold>/best_metric_model.pth for every requested fold."
        ),
    )
    checkpoint_source.add_argument(
        "--checkpoint_job_names",
        nargs="+",
        help=(
            "Completed Azure ML training job names in the same order as --folds. "
            "Use this when each fold checkpoint came from a separate source job."
        ),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(DEFAULT_FOLDS))
    parser.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pretrained_swin_encoder",
        default=DEFAULT_PRETRAINED_SWIN_ENCODER,
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--pipeline_display_name",
        default="Myocardium Swin UNETR Fine-tuning Pipeline",
    )
    parser.add_argument(
        "--azure_auth_mode",
        choices=["default", "browser"],
        default="default",
    )
    parser.add_argument("--azure_compute", default=DEFAULT_AZURE_COMPUTE)
    parser.add_argument("--azure_environment", default=DEFAULT_AZURE_ENVIRONMENT)
    parser.add_argument("--azure_input_data", default=DEFAULT_AZURE_INPUT_DATA)
    parser.add_argument("--azure_tenant_id", default=DEFAULT_AZURE_TENANT_ID)
    parser.add_argument("--azure_subscription_id", default=DEFAULT_AZURE_SUBSCRIPTION_ID)
    parser.add_argument("--azure_resource_group", default=DEFAULT_AZURE_RESOURCE_GROUP)
    parser.add_argument("--azure_workspace_name", default=DEFAULT_AZURE_WORKSPACE_NAME)
    parser.add_argument("--azure_experiment_name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--azure_instance_count", type=int, default=1)

    args, extra_overrides = parser.parse_known_args(argv)
    args.extra_overrides = [*DEFAULT_OVERRIDES, *extra_overrides]

    if not args.folds:
        parser.error("--folds must contain at least one fold")
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds cannot contain duplicate fold values")

    return args


def main(argv=None):
    args = parse_args(argv)
    submit_pipeline(args)


if __name__ == "__main__":
    main(sys.argv[1:])
