import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_AZURE_COMPUTE = "azureml:clusterprdwe-g-t2-vzhst6"
DEFAULT_AZURE_ENVIRONMENT = "azureml:media-env:8"
DEFAULT_AZURE_INPUT_DATA = "azureml:LEMS-CT-NIfTI:1"
DEFAULT_AZURE_TENANT_ID = "ebbb4fc3-587b-477c-a1e8-ee47d8c02546"
DEFAULT_AZURE_SUBSCRIPTION_ID = "ab211f7b-463f-4833-9605-d260e596a35a"
DEFAULT_AZURE_RESOURCE_GROUP = "73da10b4-5dff-54e2-db0d-3a1fab882485"
DEFAULT_AZURE_WORKSPACE_NAME = "73da10b45dff54e2db0d3a1fab882485"
DEFAULT_EXPERIMENT_NAME = "myocardium_segresnet_cv_pipeline"
DEFAULT_OUTPUT_DIR = "outputs/segresnet_cv_pipeline"
DEFAULT_CONFIG = "config/train_config.yaml"
DEFAULT_SPLIT_CSV = "data/cv_splits_qc.csv"


@dataclass(frozen=True)
class FoldTrainingSpec:
    fold: int
    name: str
    display_name: str
    command: str


def azure_compute_name(compute_ref):
    return str(compute_ref).strip().removeprefix("azureml:")


def safe_azure_name(value, max_length=240):
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe or "job")[:max_length]


def _project_relative_path(path):
    candidate = (PROJECT_ROOT / path).resolve()
    try:
        return candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must be inside the project: {path}") from exc


def _copy_tree(src, dst):
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        dirs_exist_ok=True,
    )


def prepare_code_bundle(args):
    code_dir = Path(args.output_dir) / "azure_code"
    if code_dir.exists():
        shutil.rmtree(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)

    for directory in ("scripts", "jobs", "lems_ct"):
        _copy_tree(PROJECT_ROOT / directory, code_dir / directory)

    for file_path in (args.config, args.split_csv):
        relative_path = _project_relative_path(file_path)
        destination = code_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative_path, destination)

    return code_dir


def training_command(args, fold):
    parts = [
        "python scripts/train_job.py",
        "--input_data ${{inputs.input_data}}",
        "--output_model ${{outputs.output_model}}",
        f"--split_csv {args.split_csv}",
        f"--fold {fold}",
        f"--config {args.config}",
        *args.extra_overrides,
    ]
    return " ".join(parts)


def inference_command(args):
    parts = [
        "python jobs/inference_job.py",
        "--input_data ${{inputs.input_data}}",
        "--output_dir ${{outputs.output_dir}}",
        f"--fold_count {len(args.folds)}",
        f"--split_csv {args.split_csv}",
        f"--test_fold {args.test_fold}",
        f"--config {args.config}",
    ]
    parts.extend(
        f"--w{index} ${{{{inputs.w{index}}}}}" for index, _ in enumerate(args.folds)
    )
    return " ".join(parts)


def build_training_specs(args):
    return [
        FoldTrainingSpec(
            fold=fold,
            name=safe_azure_name(f"segresnet_train_fold_{fold}"),
            display_name=f"Myocardium SegResNet Training Fold {fold}",
            command=training_command(args, fold),
        )
        for fold in args.folds
    ]


def make_training_component(spec, args, has_previous):
    from azure.ai.ml import Input, Output, command

    command_text = spec.command
    inputs = {"input_data": Input(type="uri_folder")}
    if has_previous:
        inputs["previous_marker"] = Input(type="uri_folder")
        command_text = "test -f ${{inputs.previous_marker}}/done.txt && " + command_text
    command_text = (
        command_text
        + " && mkdir -p ${{outputs.completion_marker}}"
        + " && printf done > ${{outputs.completion_marker}}/done.txt"
    )

    return command(
        name=spec.name,
        display_name=spec.display_name,
        description=f"SegResNet training for QC CV fold {spec.fold}",
        code=str(args.code_dir),
        command=command_text,
        environment=args.azure_environment,
        environment_variables={
            "PYTHONPATH": ".",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        compute=azure_compute_name(args.azure_compute),
        inputs=inputs,
        outputs={
            "output_model": Output(type="uri_folder"),
            "completion_marker": Output(type="uri_folder"),
        },
        distribution={
            "type": "pytorch",
            "process_count_per_instance": args.process_count_per_instance,
        },
        instance_count=args.azure_instance_count,
        shm_size=args.shm_size,
        is_deterministic=False,
    )


def make_inference_component(args):
    from azure.ai.ml import Input, Output, command

    inputs = {
        "input_data": Input(type="uri_folder"),
        "final_marker": Input(type="uri_folder"),
    }
    inputs.update(
        {f"w{index}": Input(type="uri_folder") for index, _ in enumerate(args.folds)}
    )
    command_text = (
        "test -f ${{inputs.final_marker}}/done.txt && " + inference_command(args)
    )

    return command(
        name=safe_azure_name("segresnet_cv_ensemble_inference"),
        display_name="Myocardium SegResNet CV Ensemble Inference",
        description="Ensemble held-out QC test inference from the five CV fold outputs.",
        code=str(args.code_dir),
        command=command_text,
        environment=args.azure_environment,
        environment_variables={"PYTHONPATH": "."},
        compute=azure_compute_name(args.azure_compute),
        inputs=inputs,
        outputs={"output_dir": Output(type="uri_folder")},
        instance_count=1,
        shm_size=args.shm_size,
        is_deterministic=False,
    )


def build_azure_pipeline(args):
    from azure.ai.ml import Input
    from azure.ai.ml.dsl import pipeline

    specs = build_training_specs(args)
    compute_name = azure_compute_name(args.azure_compute)

    @pipeline(
        display_name=args.pipeline_display_name,
        description=(
            "Full SegResNet 5-fold CV training on folds 0-4 from "
            "cv_splits_qc.csv, followed by ensemble inference on fold -1."
        ),
        experiment_name=args.azure_experiment_name,
        default_compute=compute_name,
    )
    def segresnet_cv_pipeline(input_data):
        fold_nodes = []
        previous_marker = None
        for spec in specs:
            component = make_training_component(
                spec,
                args,
                has_previous=previous_marker is not None,
            )
            node_inputs = {"input_data": input_data}
            if previous_marker is not None:
                node_inputs["previous_marker"] = previous_marker

            node = component(**node_inputs)
            node.name = spec.name
            node.display_name = spec.display_name
            node.tags = {
                "fold": str(spec.fold),
                "model": "segresnet",
                "split_csv": args.split_csv,
                "stage": "training",
            }
            fold_nodes.append(node)
            previous_marker = node.outputs.completion_marker

        inference_component = make_inference_component(args)
        inference_inputs = {
            "input_data": input_data,
            "final_marker": previous_marker,
        }
        inference_inputs.update(
            {
                f"w{index}": node.outputs.output_model
                for index, node in enumerate(fold_nodes)
            }
        )
        inference_node = inference_component(**inference_inputs)
        inference_node.name = safe_azure_name("segresnet_cv_ensemble_inference")
        inference_node.tags = {
            "model": "segresnet",
            "split_csv": args.split_csv,
            "test_fold": str(args.test_fold),
            "stage": "ensemble_inference",
        }
        return {"ensemble_output": inference_node.outputs.output_dir}

    return segresnet_cv_pipeline(
        input_data=Input(type="uri_folder", path=args.azure_input_data, mode="ro_mount")
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
    args.code_dir = prepare_code_bundle(args)

    pipeline_job = build_azure_pipeline(args)
    pipeline_yaml = out_dir / "segresnet_cv_pipeline_job.yml"
    pipeline_job.dump(pipeline_yaml)
    print(f"Azure pipeline YAML: {pipeline_yaml}", flush=True)

    if args.dry_run:
        print("Dry run only. Azure pipeline not submitted.", flush=True)
        return pipeline_yaml

    ml_client = get_azure_ml_client(args)
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    payload = serialize_azure_job(returned_job)
    submission_json = out_dir / "segresnet_cv_pipeline_submission.json"
    submission_json.write_text(json.dumps(payload, indent=2))

    print(f"Submitted pipeline: {payload['name']}", flush=True)
    if payload["studio_url"]:
        print(f"Studio URL: {payload['studio_url']}", flush=True)
    return submission_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Submit one Azure ML parent pipeline containing five SegResNet CV "
            "training jobs plus held-out ensemble inference."
        )
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(DEFAULT_FOLDS))
    parser.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--test_fold", type=int, default=-1)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--pipeline_display_name",
        default="Myocardium SegResNet Full CV Pipeline",
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
    parser.add_argument("--process_count_per_instance", type=int, default=1)
    parser.add_argument("--shm_size", default="8g")

    args, extra_overrides = parser.parse_known_args(argv)
    args.extra_overrides = extra_overrides

    if not args.folds:
        parser.error("--folds must contain at least one fold")
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds cannot contain duplicate fold values")
    if args.test_fold in args.folds:
        parser.error("--test_fold must not be one of the training CV folds")
    if len(args.folds) != 5:
        parser.error("The current ensemble inference job expects exactly five folds")
    if args.folds != DEFAULT_FOLDS:
        parser.error(
            "The full CV ensemble pipeline requires folds in order: 0 1 2 3 4"
        )

    return args


def main(argv=None):
    args = parse_args(argv)
    submit_pipeline(args)


if __name__ == "__main__":
    main(sys.argv[1:])
