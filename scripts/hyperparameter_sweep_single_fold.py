import argparse
import csv
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lems_ct.src.utils.data import resolve_local_path


DEFAULT_LRS = ["3e-5", "1e-4", "3e-4"]
DEFAULT_WEIGHT_DECAYS = ["1e-5", "1e-4", "1e-3", "1e-2"]
DEFAULT_ROI_SIZES = ["96"]
DEFAULT_LOSSES = ["dice", "dice_focal", "hausdorff", "tversky"]
DEFAULT_TARGET_SPACINGS = ["1.0"]
OOM_SAFE_CUDA_ALLOC_CONF = "expandable_segments:True"
DEFAULT_AZURE_COMPUTE = "azureml:vmprdwe8-g-t2-vzhst6"
DEFAULT_AZURE_ENVIRONMENT = "azureml:media-env:7"
DEFAULT_AZURE_INPUT_DATA = "azureml:LEMS-CT-NIfTI:1"
DEFAULT_AZURE_TENANT_ID = "ebbb4fc3-587b-477c-a1e8-ee47d8c02546"
DEFAULT_AZURE_SUBSCRIPTION_ID = "ab211f7b-463f-4833-9605-d260e596a35a"
DEFAULT_AZURE_RESOURCE_GROUP = "73da10b4-5dff-54e2-db0d-3a1fab882485"
DEFAULT_AZURE_WORKSPACE_NAME = "73da10b45dff54e2db0d3a1fab882485"
AZURE_TERMINAL_STATUSES = {
    "Completed",
    "Failed",
    "Canceled",
    "Cancelled",
    "NotResponding",
}


def parse_float_values(values):
    return [float(value) for value in values]


def parse_roi_sizes(values):
    roi_sizes = []
    for value in values:
        normalized = str(value).lower().replace("^3", "")
        roi_sizes.append(int(normalized))
    return roi_sizes


def parse_spacing(value):
    normalized = str(value).strip().strip("[]()")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) == 1:
        spacing = float(parts[0])
        return (spacing, spacing, spacing)
    if len(parts) == 3:
        return tuple(float(part) for part in parts)
    raise argparse.ArgumentTypeError(
        f"Invalid spacing '{value}'. Use '0.5' or '0.5,0.5,0.5'."
    )


def parse_spacings(values):
    return [parse_spacing(value) for value in values]


def format_float(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def format_vector(values):
    return "x".join(format_float(value) for value in values)


def yaml_scalar(value):
    return json.dumps(str(value))


def azure_compute_name(compute_ref):
    compute_ref = str(compute_ref).strip()
    if compute_ref.startswith("azureml:"):
        return compute_ref.removeprefix("azureml:")

    match = re.search(r"/computes/([^/]+)$", compute_ref)
    if match:
        return match.group(1)

    return compute_ref


def safe_azure_name(value, max_length=240):
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "job"
    return safe[:max_length]


def run_id_for(fold, lr, weight_decay, roi_size, loss_name, spacing):
    return (
        f"fold{fold}"
        f"_loss-{loss_name}"
        f"_lr-{format_float(lr)}"
        f"_wd-{format_float(weight_decay)}"
        f"_roi-{roi_size}"
        f"_spacing-{format_vector(spacing)}"
    )


def canonical_sweep_loss_name(loss_name):
    normalized = str(loss_name).lower().replace("-", "_")
    aliases = {
        "dice": "dice",
        "dice_ce": "dice_ce",
        "dicece": "dice_ce",
        "dice_focal": "dice_focal",
        "dicefocal": "dice_focal",
        "hausdorff": "hausdorff",
        "haussdorf": "hausdorff",
        "hausdorff_dt": "hausdorff",
        "tversky": "tversky",
    }
    return aliases.get(normalized, normalized)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    if not path.exists():
        return {}
    with path.open() as file_obj:
        return json.load(file_obj)


def metric_sort_value(row):
    try:
        return float(row.get("best_val_dice", "nan"))
    except (TypeError, ValueError):
        return float("-inf")


def write_results(path, rows):
    ranked_rows = sorted(rows, key=metric_sort_value, reverse=True)
    write_csv(path, ranked_rows)


def build_jobs(args, sweep_dir):
    lrs = parse_float_values(args.lrs)
    weight_decays = parse_float_values(args.weight_decays)
    roi_sizes = parse_roi_sizes(args.roi_sizes)
    spacings = parse_spacings(args.target_spacings)
    max_iterations = int(args.max_iterations)

    if max_iterations > 8000:
        raise SystemExit("ERROR: --max_iterations must be <= 8000 for this sweep.")

    jobs = []
    combinations = itertools.product(
        lrs,
        roi_sizes,
        weight_decays,
        args.losses,
        spacings,
    )

    for lr, roi_size, weight_decay, loss_name, spacing in combinations:
        canonical_loss = canonical_sweep_loss_name(loss_name)
        run_id = run_id_for(args.fold, lr, weight_decay, roi_size, loss_name, spacing)
        run_root = sweep_dir / run_id
        resource_overrides = []
        if canonical_loss == "hausdorff" and not args.disable_hausdorff_memory_overrides:
            resource_overrides.extend(
                [
                    f"training.train_batch_size={args.hausdorff_train_batch_size}",
                    f"training.accumulation_steps={args.hausdorff_accumulation_steps}",
                    f"training.hausdorff_downsample={args.hausdorff_downsample}",
                    f"training.hausdorff_validation_loss={args.hausdorff_validation_loss}",
                    f"inference.sw_batch_size={args.hausdorff_sw_batch_size}",
                ]
            )
            if args.hausdorff_disable_ema:
                resource_overrides.append("training.use_ema=false")

        training_overrides = [
            *args.extra_overrides,
            f"training.loss={loss_name}",
            f"training.lr={lr}",
            f"training.weight_decay={weight_decay}",
            f"training.max_iterations={max_iterations}",
            f"transforms.roi_size=[{roi_size},{roi_size},{roi_size}]",
            f"transforms.target_spacing=[{spacing[0]},{spacing[1]},{spacing[2]}]",
            *resource_overrides,
        ]
        command = [
            args.python_executable,
            str(resolve_local_path(args.train_script)),
            "--input_data",
            str(resolve_local_path(args.input_data)),
            "--output_model",
            str(run_root),
            "--split_csv",
            args.split_csv,
            "--fold",
            str(args.fold),
            "--config",
            args.config,
            "--device",
            args.device,
        ]
        if not args.stop_on_trial_error:
            command.append("--continue_on_error")
        command.extend(training_overrides)

        azure_overrides = " ".join(shlex.quote(override) for override in training_overrides)
        if azure_overrides:
            azure_overrides = f" {azure_overrides}"
        azure_continue = "" if args.stop_on_trial_error else "--continue_on_error "
        azure_command = (
            f"PYTORCH_CUDA_ALLOC_CONF={shlex.quote(OOM_SAFE_CUDA_ALLOC_CONF)} "
            "python scripts/train_job_only_dice.py "
            "--input_data ${{inputs.input_data}} "
            "--output_model ${{outputs.output_model}} "
            f"--split_csv {shlex.quote(args.split_csv)} "
            f"--fold {args.fold} "
            f"--config {shlex.quote(args.config)} "
            f"--device {shlex.quote(args.device)} "
            f"{azure_continue}"
            f"{azure_overrides}"
        )
        jobs.append(
            {
                "run_id": run_id,
                "fold": args.fold,
                "loss": loss_name,
                "base_lr": lr,
                "weight_decay": weight_decay,
                "roi_size": f"{roi_size}x{roi_size}x{roi_size}",
                "target_spacing": format_vector(spacing),
                "max_iterations": max_iterations,
                "resource_overrides": " ".join(resource_overrides),
                "continue_on_error": not args.stop_on_trial_error,
                "run_root": str(run_root),
                "fold_output": str(run_root / f"fold_{args.fold}"),
                "log_path": str(run_root / "train.log"),
                "command": command,
                "command_text": shlex.join(command),
                "azure_command": azure_command,
            }
        )

    if args.limit:
        jobs = jobs[: args.limit]

    return jobs


def manifest_rows(jobs):
    rows = []
    for job in jobs:
        rows.append(
            {
                "run_id": job["run_id"],
                "fold": job["fold"],
                "loss": job["loss"],
                "base_lr": job["base_lr"],
                "weight_decay": job["weight_decay"],
                "roi_size": job["roi_size"],
                "target_spacing": job["target_spacing"],
                "max_iterations": job["max_iterations"],
                "resource_overrides": job["resource_overrides"],
                "continue_on_error": job["continue_on_error"],
                "run_root": job["run_root"],
                "fold_output": job["fold_output"],
                "log_path": job["log_path"],
                "command": job["command_text"],
            }
        )
    return rows


def azure_job_yaml(job, args):
    display_name = f"Myocardium HP Sweep {job['run_id']}"
    description = (
        "Single-fold SegResNet hyperparameter sweep run. "
        f"loss={job['loss']}, lr={job['base_lr']}, "
        f"weight_decay={job['weight_decay']}, roi={job['roi_size']}, "
        f"spacing={job['target_spacing']}."
    )
    code_path = resolve_local_path(args.azure_code)
    distribution_block = ""
    if args.azure_instance_count > 1 or args.azure_process_count_per_instance > 1:
        distribution_block = f"""
distribution:
  type: pytorch
  process_count_per_instance: {args.azure_process_count_per_instance}
"""

    return f"""$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
display_name: {yaml_scalar(display_name)}
description: {yaml_scalar(description)}
experiment_name: {yaml_scalar(args.azure_experiment_name)}
compute: {args.azure_compute}
environment: {args.azure_environment}

environment_variables:
  PYTHONPATH: "."

inputs:
  input_data:
    type: uri_folder
    path: {args.azure_input_data}
    mode: ro_mount

outputs:
  output_model:
    type: uri_folder

code: {yaml_scalar(code_path)}

command: >-
  {job["azure_command"]}

resources:
  instance_count: {args.azure_instance_count}
{distribution_block}
"""


def write_azure_job_files(jobs, args, sweep_dir):
    azure_jobs_dir = sweep_dir / "azure_job_yamls"
    azure_jobs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for job in jobs:
        job_file = azure_jobs_dir / f"{job['run_id']}.yml"
        job_file.write_text(azure_job_yaml(job, args))
        job["azure_job_file"] = str(job_file)
        rows.append(
            {
                "run_id": job["run_id"],
                "fold": job["fold"],
                "loss": job["loss"],
                "base_lr": job["base_lr"],
                "weight_decay": job["weight_decay"],
                "roi_size": job["roi_size"],
                "target_spacing": job["target_spacing"],
                "max_iterations": job["max_iterations"],
                "resource_overrides": job["resource_overrides"],
                "continue_on_error": job["continue_on_error"],
                "azure_job_file": str(job_file),
                "azure_command": job["azure_command"],
            }
        )

    azure_manifest_path = sweep_dir / "azure_job_manifest.csv"
    write_csv(azure_manifest_path, rows)
    return azure_manifest_path


def azure_submission_row(job, status, returncode=None, stdout="", stderr=""):
    job_name = ""
    studio_url = ""
    try:
        payload = json.loads(stdout) if stdout else {}
        job_name = payload.get("name", "")
        studio_url = payload.get("studio_url", "")
    except json.JSONDecodeError:
        pass

    return {
        "run_id": job["run_id"],
        "status": status,
        "returncode": returncode,
        "azure_job_name": job_name,
        "studio_url": studio_url,
        "fold": job["fold"],
        "loss": job["loss"],
        "base_lr": job["base_lr"],
        "weight_decay": job["weight_decay"],
        "roi_size": job["roi_size"],
        "target_spacing": job["target_spacing"],
        "max_iterations": job["max_iterations"],
        "azure_job_file": job.get("azure_job_file", ""),
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }


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


def azure_job_status(ml_client, job_name):
    returned_job = ml_client.jobs.get(job_name)
    return getattr(returned_job, "status", "")


def submit_azure_sweep_sdk(jobs, args, sweep_dir):
    from azure.ai.ml import load_job

    submissions_path = sweep_dir / "azure_submissions.csv"
    rows = []
    pending = list(jobs)
    running = []
    consecutive_failures = 0
    ml_client = get_azure_ml_client(args)

    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            submitted_path = Path(job["run_root"]) / "azure_submission.json"
            if args.skip_completed and submitted_path.exists():
                print(f"Skipping submitted Azure job: {job['run_id']}", flush=True)
                stdout = submitted_path.read_text()
                rows.append(azure_submission_row(job, "skipped_submitted", 0, stdout, ""))
                write_csv(submissions_path, rows)
                consecutive_failures = 0
                continue

            Path(job["run_root"]).mkdir(parents=True, exist_ok=True)
            print(f"Submitting Azure job via SDK: {job['run_id']}", flush=True)
            try:
                azure_job = load_job(job["azure_job_file"])
                returned_job = ml_client.jobs.create_or_update(azure_job)
                payload = serialize_azure_job(returned_job)
                stdout = json.dumps(payload)
                submitted_path.write_text(stdout)
                rows.append(azure_submission_row(job, "submitted", 0, stdout, ""))
                running.append(
                    {
                        "job": job,
                        "azure_job_name": payload["name"],
                        "studio_url": payload["studio_url"],
                    }
                )
                print(f"submitted: {job['run_id']} ({payload['name']})", flush=True)
            except Exception as exc:
                consecutive_failures += 1
                stderr = f"{type(exc).__name__}: {exc}"
                rows.append(azure_submission_row(job, "submission_failed", 1, "", stderr))
                print(f"submission_failed: {job['run_id']} ({stderr})", flush=True)

            write_csv(submissions_path, rows)

            if args.azure_submit_interval_seconds > 0:
                time.sleep(args.azure_submit_interval_seconds)

        for item in list(running):
            status = azure_job_status(ml_client, item["azure_job_name"])
            if status not in AZURE_TERMINAL_STATUSES:
                continue

            running.remove(item)
            job = item["job"]
            payload = {
                "name": item["azure_job_name"],
                "status": status,
                "studio_url": item["studio_url"],
            }
            row_status = "completed" if status == "Completed" else "failed"
            rows.append(azure_submission_row(job, row_status, 0, json.dumps(payload), ""))
            write_csv(submissions_path, rows)
            print(f"{row_status}: {job['run_id']} ({status})", flush=True)

            if status == "Completed":
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        if (
            args.max_consecutive_failures
            and consecutive_failures >= args.max_consecutive_failures
        ):
            print(
                "Stopping Azure submissions after "
                f"{consecutive_failures} consecutive failures.",
                flush=True,
            )
            break

        if pending or running:
            time.sleep(args.poll_seconds)

    return submissions_path


def make_pipeline_command_component(job, args, has_previous):
    from azure.ai.ml import Input, Output, command

    command_text = job["azure_command"]
    inputs = {"input_data": Input(type="uri_folder")}
    if has_previous:
        inputs["previous_output"] = Input(type="uri_folder")
        command_text = f"test -d ${{{{inputs.previous_output}}}} && {command_text}"

    return command(
        name=safe_azure_name(f"sweep_train_{job['run_id']}"),
        display_name=job["run_id"],
        description=(
            f"loss={job['loss']}, lr={job['base_lr']}, "
            f"weight_decay={job['weight_decay']}, roi={job['roi_size']}, "
            f"spacing={job['target_spacing']}"
        ),
        code=str(resolve_local_path(args.azure_code)),
        command=command_text,
        environment=args.azure_environment,
        inputs=inputs,
        outputs={"output_model": Output(type="uri_folder")},
        is_deterministic=False,
    )


def build_sequential_azure_pipeline(jobs, args):
    from azure.ai.ml import Input
    from azure.ai.ml.dsl import pipeline

    pipeline_display_name = f"Myocardium HP Sweep Queue {args.sweep_name or 'single_fold'}"
    compute_name = azure_compute_name(args.azure_compute)

    @pipeline(
        display_name=pipeline_display_name,
        description="Sequential hyperparameter sweep. All child jobs are created up front and run one by one.",
        experiment_name=args.azure_experiment_name,
        default_compute=compute_name,
    )
    def sweep_pipeline(input_data):
        previous_output = None
        last_output = None

        for idx, job in enumerate(jobs):
            train_component = make_pipeline_command_component(
                job,
                args,
                has_previous=previous_output is not None,
            )
            if previous_output is None:
                node = train_component(input_data=input_data)
            else:
                node = train_component(
                    input_data=input_data,
                    previous_output=previous_output,
                )

            node.name = f"train_{idx:04d}"
            node.display_name = job["run_id"]
            node.tags = {
                "run_id": job["run_id"],
                "fold": str(job["fold"]),
                "loss": str(job["loss"]),
                "lr": str(job["base_lr"]),
                "weight_decay": str(job["weight_decay"]),
                "roi_size": str(job["roi_size"]),
                "target_spacing": str(job["target_spacing"]),
            }
            previous_output = node.outputs.output_model
            last_output = previous_output

        return {"last_output": last_output}

    return sweep_pipeline(
        input_data=Input(
            type="uri_folder",
            path=args.azure_input_data,
            mode="ro_mount",
        )
    )


def submit_azure_pipeline(jobs, args, sweep_dir):
    submissions_path = sweep_dir / "azure_submissions.csv"
    pipeline_job = build_sequential_azure_pipeline(jobs, args)
    pipeline_yaml_path = sweep_dir / "azure_pipeline_job.yml"
    pipeline_job.dump(pipeline_yaml_path)

    if args.dry_run:
        print(f"Azure pipeline YAML: {pipeline_yaml_path}", flush=True)
        print("Dry run only. Azure pipeline not submitted.", flush=True)
        return submissions_path

    ml_client = get_azure_ml_client(args)
    print(f"Submitting Azure pipeline with {len(jobs)} queued child jobs...", flush=True)
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    payload = serialize_azure_job(returned_job)

    submitted_path = sweep_dir / "azure_pipeline_submission.json"
    submitted_path.write_text(json.dumps(payload, indent=2))

    rows = [
        {
            "run_id": "pipeline",
            "status": "submitted",
            "returncode": 0,
            "azure_job_name": payload["name"],
            "studio_url": payload["studio_url"],
            "fold": args.fold,
            "loss": "",
            "base_lr": "",
            "weight_decay": "",
            "roi_size": "",
            "target_spacing": "",
            "max_iterations": args.max_iterations,
            "azure_job_file": str(pipeline_yaml_path),
            "stdout": json.dumps(payload),
            "stderr": "",
        }
    ]
    write_csv(submissions_path, rows)
    print(f"submitted pipeline: {payload['name']}", flush=True)
    print(f"Studio URL: {payload['studio_url']}", flush=True)
    return submissions_path


def submit_azure_sweep(jobs, args, sweep_dir):
    submissions_path = sweep_dir / "azure_submissions.csv"
    rows = []
    consecutive_failures = 0

    for job in jobs:
        submitted_path = Path(job["run_root"]) / "azure_submission.json"
        if args.skip_completed and submitted_path.exists():
            print(f"Skipping submitted Azure job: {job['run_id']}", flush=True)
            stdout = submitted_path.read_text()
            rows.append(azure_submission_row(job, "skipped_submitted", 0, stdout, ""))
            write_csv(submissions_path, rows)
            consecutive_failures = 0
            continue

        Path(job["run_root"]).mkdir(parents=True, exist_ok=True)
        command = [
            args.az_executable,
            "ml",
            "job",
            "create",
            "--file",
            job["azure_job_file"],
            "--output",
            "json",
            "--only-show-errors",
        ]
        if args.azure_subscription_id:
            command.extend(["--subscription", args.azure_subscription_id])
        if args.azure_resource_group:
            command.extend(["--resource-group", args.azure_resource_group])
        if args.azure_workspace_name:
            command.extend(["--workspace-name", args.azure_workspace_name])
        print(f"Submitting Azure job: {job['run_id']}", flush=True)
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        status = "submitted" if result.returncode == 0 else "submission_failed"
        if result.returncode == 0:
            submitted_path.write_text(result.stdout)
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        rows.append(
            azure_submission_row(
                job,
                status,
                result.returncode,
                result.stdout,
                result.stderr,
            )
        )
        write_csv(submissions_path, rows)
        print(f"{status}: {job['run_id']} (returncode={result.returncode})", flush=True)

        if (
            args.max_consecutive_failures
            and consecutive_failures >= args.max_consecutive_failures
        ):
            print(
                "Stopping Azure submissions after "
                f"{consecutive_failures} consecutive failures.",
                flush=True,
            )
            break

        if args.azure_submit_interval_seconds > 0:
            time.sleep(args.azure_submit_interval_seconds)

    return submissions_path


def result_row(job, status, returncode=None):
    summary = load_json(Path(job["fold_output"]) / "summary.json")
    trial_status = summary.get("status", "")
    if trial_status:
        status = trial_status
    row = {
        "run_id": job["run_id"],
        "status": status,
        "returncode": returncode,
        "fold": job["fold"],
        "loss": job["loss"],
        "base_lr": job["base_lr"],
        "weight_decay": job["weight_decay"],
        "roi_size": job["roi_size"],
        "target_spacing": job["target_spacing"],
        "max_iterations": job["max_iterations"],
        "resource_overrides": job["resource_overrides"],
        "continue_on_error": job["continue_on_error"],
        "trial_status": summary.get("status", ""),
        "failure_type": summary.get("failure_type", ""),
        "error_type": summary.get("error_type", ""),
        "error_message": summary.get("error_message", ""),
        "best_val_dice": summary.get("best_val_dice", ""),
        "last_val_dice": summary.get("val_dice", ""),
        "last_val_loss": summary.get("val_loss", ""),
        "last_train_loss": summary.get("train_loss", ""),
        "last_step": summary.get("step", ""),
        "best_checkpoint": summary.get("best_checkpoint", ""),
        "metrics_csv": str(Path(job["fold_output"]) / "metrics.csv"),
        "log_path": job["log_path"],
    }
    return row


def run_sweep(jobs, args, sweep_dir):
    results_path = sweep_dir / "sweep_results.csv"
    pending = list(jobs)
    running = []
    results = []
    consecutive_failures = 0

    try:
        while pending or running:
            if (
                args.max_consecutive_failures
                and consecutive_failures >= args.max_consecutive_failures
            ):
                print(
                    "Stopping sweep after "
                    f"{consecutive_failures} consecutive failed jobs. "
                    "Check the latest train.log before resuming.",
                    flush=True,
                )
                break

            while pending and len(running) < args.max_parallel:
                job = pending.pop(0)
                summary_path = Path(job["fold_output"]) / "summary.json"
                if args.skip_completed and summary_path.exists():
                    print(f"Skipping completed run: {job['run_id']}", flush=True)
                    results.append(result_row(job, "skipped_completed", 0))
                    write_results(results_path, results)
                    consecutive_failures = 0
                    continue

                run_root = Path(job["run_root"])
                run_root.mkdir(parents=True, exist_ok=True)
                with (run_root / "command.txt").open("w") as command_file:
                    command_file.write(job["command_text"] + "\n")

                log_file = Path(job["log_path"]).open("w")
                print(f"Launching {job['run_id']}", flush=True)
                process_env = os.environ.copy()
                process_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", OOM_SAFE_CUDA_ALLOC_CONF)
                process = subprocess.Popen(
                    job["command"],
                    cwd=PROJECT_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=process_env,
                )
                running.append({"job": job, "process": process, "log_file": log_file})

            time.sleep(args.poll_seconds)

            for item in list(running):
                returncode = item["process"].poll()
                if returncode is None:
                    continue

                item["log_file"].close()
                running.remove(item)
                status = "completed" if returncode == 0 else "failed"
                print(f"{status}: {item['job']['run_id']} (returncode={returncode})", flush=True)
                results.append(result_row(item["job"], status, returncode))
                write_results(results_path, results)
                if returncode == 0:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

    except KeyboardInterrupt:
        print("Interrupted. Terminating running jobs...", flush=True)
        for item in running:
            item["process"].terminate()
            item["log_file"].close()
        write_results(results_path, results)
        return results_path

    return results_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a single-fold hyperparameter sweep using train_job_only_dice.py."
    )
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_model", type=str, default="./sweep_outputs")
    parser.add_argument("--split_csv", type=str, default="data/cv_splits.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--train_script", type=str, default="scripts/train_job_only_dice.py")
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--sweep_name", type=str, default=None)
    parser.add_argument(
        "--backend",
        choices=["local", "azure"],
        default="local",
        help="local runs subprocesses here; azure submits Azure ML command jobs.",
    )
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument(
        "--max_consecutive_failures",
        type=int,
        default=3,
        help="Stop after this many consecutive failed jobs. Use 0 to disable.",
    )
    parser.add_argument("--poll_seconds", type=float, default=10.0)
    parser.add_argument("--max_iterations", type=int, default=8000)
    parser.add_argument("--lrs", nargs="+", default=DEFAULT_LRS)
    parser.add_argument("--weight_decays", nargs="+", default=DEFAULT_WEIGHT_DECAYS)
    parser.add_argument("--roi_sizes", nargs="+", default=DEFAULT_ROI_SIZES)
    parser.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    parser.add_argument("--target_spacings", nargs="+", default=DEFAULT_TARGET_SPACINGS)
    parser.add_argument(
        "--stop_on_trial_error",
        action="store_true",
        help="Let a failed child trial stop the local sweep or Azure pipeline.",
    )
    parser.add_argument(
        "--disable_hausdorff_memory_overrides",
        action="store_true",
        help="Do not apply smaller batch/accumulation defaults for Hausdorff trials.",
    )
    parser.add_argument("--hausdorff_train_batch_size", type=int, default=1)
    parser.add_argument("--hausdorff_accumulation_steps", type=int, default=4)
    parser.add_argument("--hausdorff_downsample", type=int, default=2)
    parser.add_argument("--hausdorff_validation_loss", type=str, default="dice")
    parser.add_argument("--hausdorff_sw_batch_size", type=int, default=1)
    parser.add_argument(
        "--hausdorff_disable_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable EMA for Hausdorff trials to reduce GPU memory.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit jobs for smoke testing.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument(
        "--azure_submitter",
        choices=["pipeline", "sdk", "cli"],
        default="pipeline",
        help=(
            "pipeline creates all child jobs up front with sequential dependencies; "
            "sdk submits/waits from this process; cli requires the az ml extension."
        ),
    )
    parser.add_argument(
        "--azure_auth_mode",
        choices=["default", "browser"],
        default="browser",
        help="browser avoids relying on a local Azure CLI install; default uses DefaultAzureCredential.",
    )
    parser.add_argument("--az_executable", type=str, default="az")
    parser.add_argument("--azure_compute", type=str, default=DEFAULT_AZURE_COMPUTE)
    parser.add_argument("--azure_environment", type=str, default=DEFAULT_AZURE_ENVIRONMENT)
    parser.add_argument("--azure_input_data", type=str, default=DEFAULT_AZURE_INPUT_DATA)
    parser.add_argument("--azure_code", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--azure_tenant_id", type=str, default=DEFAULT_AZURE_TENANT_ID)
    parser.add_argument("--azure_subscription_id", type=str, default=DEFAULT_AZURE_SUBSCRIPTION_ID)
    parser.add_argument("--azure_resource_group", type=str, default=DEFAULT_AZURE_RESOURCE_GROUP)
    parser.add_argument("--azure_workspace_name", type=str, default=DEFAULT_AZURE_WORKSPACE_NAME)
    parser.add_argument(
        "--azure_experiment_name",
        type=str,
        default="myocardium_hparam_sweep_single_fold",
    )
    parser.add_argument("--azure_instance_count", type=int, default=1)
    parser.add_argument("--azure_process_count_per_instance", type=int, default=1)
    parser.add_argument("--azure_submit_interval_seconds", type=float, default=0.0)

    args, extra_overrides = parser.parse_known_args()
    args.extra_overrides = extra_overrides

    if args.max_parallel < 1:
        parser.error("--max_parallel must be >= 1")

    return args


def main():
    args = parse_args()
    sweep_name = args.sweep_name or datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    sweep_dir = resolve_local_path(args.output_model) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, sweep_dir)
    manifest_path = sweep_dir / "sweep_manifest.csv"
    write_csv(manifest_path, manifest_rows(jobs))

    print(f"Sweep directory: {sweep_dir}", flush=True)
    print(f"Jobs: {len(jobs)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    if args.backend == "azure":
        azure_manifest_path = write_azure_job_files(jobs, args, sweep_dir)
        print(f"Azure job manifest: {azure_manifest_path}", flush=True)

        if args.azure_submitter == "pipeline":
            submissions_path = submit_azure_pipeline(jobs, args, sweep_dir)
        elif args.azure_submitter == "sdk":
            if args.dry_run:
                print("Dry run only. Azure YAML files generated, no jobs submitted.", flush=True)
                return
            submissions_path = submit_azure_sweep_sdk(jobs, args, sweep_dir)
        else:
            if args.dry_run:
                print("Dry run only. Azure YAML files generated, no jobs submitted.", flush=True)
                return
            submissions_path = submit_azure_sweep(jobs, args, sweep_dir)
        print(f"Azure submissions: {submissions_path}", flush=True)
        return

    if args.dry_run:
        print("Dry run only. No training jobs launched.", flush=True)
        return

    results_path = run_sweep(jobs, args, sweep_dir)
    print(f"Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
