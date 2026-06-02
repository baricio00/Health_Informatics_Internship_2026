import argparse
import json
import re
import subprocess
import sys
import time


DEFAULT_JOB_FILE = "jobs/unetr_finetune_job.yml"
DEFAULT_COMPUTE = "azureml:clusterprdwe-vzhst6"
DEFAULT_POLL_SECONDS = 60
DEFAULT_CHECKPOINT_DATASTORE = "workspaceblobstore"
DEFAULT_CHECKPOINT_OUTPUT = "output_model"
DEFAULT_CHECKPOINT_FILE = "best_metric_model.pth"
UNRESOLVED_PLACEHOLDER_RE = re.compile(r"<[^>]+>|\{[^{}]+\}|\bpath/to\b")
TERMINAL_STATUSES = {
    "Completed",
    "Failed",
    "Canceled",
    "Cancelled",
    "NotResponding",
}


def normalize_compute(value):
    value = str(value or "").strip()
    return value.removeprefix("azureml:")


def active_jobs_on_compute(jobs, compute):
    compute_name = normalize_compute(compute)
    active = []
    for job in jobs:
        status = str(job.get("status", ""))
        if status in TERMINAL_STATUSES:
            continue
        if normalize_compute(job.get("compute", "")) == compute_name:
            active.append(job)
    return active


def checkpoint_for_fold(checkpoint_template, fold):
    try:
        checkpoint_path = str(checkpoint_template).format(fold=fold)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            "Invalid checkpoint_path template. Use only the supported {fold} "
            "placeholder, or pass --checkpoint_job_name."
        ) from exc

    validate_checkpoint_path(checkpoint_path)
    return checkpoint_path


def checkpoint_path_from_job_name(job_name, fold):
    job_name = str(job_name or "").strip()
    if not job_name:
        raise ValueError("checkpoint_job_name cannot be empty.")
    if UNRESOLVED_PLACEHOLDER_RE.search(job_name):
        raise ValueError(
            f"checkpoint_job_name still contains a placeholder: {job_name}"
        )

    return (
        f"azureml://datastores/{DEFAULT_CHECKPOINT_DATASTORE}/paths/azureml/"
        f"{job_name}/{DEFAULT_CHECKPOINT_OUTPUT}/fold_{fold}/"
        f"{DEFAULT_CHECKPOINT_FILE}"
    )


def checkpoint_path_for_args(args, fold):
    if args.checkpoint_job_name:
        return checkpoint_path_from_job_name(args.checkpoint_job_name, fold)
    return checkpoint_for_fold(args.checkpoint_path, fold)


def validate_checkpoint_path(checkpoint_path):
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        raise ValueError("checkpoint_path cannot be empty.")

    match = UNRESOLVED_PLACEHOLDER_RE.search(checkpoint_path)
    if match:
        raise ValueError(
            "checkpoint_path still contains an unresolved placeholder "
            f"({match.group(0)}). Replace it with a real Azure ML output URI "
            "or pass --checkpoint_job_name <completed_training_job_name>."
        )


def build_job_create_command(
    job_file,
    fold,
    checkpoint_path,
    az_executable="az",
    output="json",
):
    return [
        az_executable,
        "ml",
        "job",
        "create",
        "--file",
        job_file,
        "--set",
        f"inputs.fold={fold}",
        "--set",
        f"inputs.checkpoint_file.path={checkpoint_path}",
        "--output",
        output,
        "--only-show-errors",
    ]


def run_json(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or result.stdout.strip())
    if not result.stdout.strip():
        return {}
    return json.loads(result.stdout)


def list_jobs(az_executable, max_results):
    return run_json(
        [
            az_executable,
            "ml",
            "job",
            "list",
            "--max-results",
            str(max_results),
            "--output",
            "json",
            "--only-show-errors",
        ]
    )


def job_status(az_executable, job_name):
    payload = run_json(
        [
            az_executable,
            "ml",
            "job",
            "show",
            "--name",
            job_name,
            "--query",
            "{name:name,status:status,studio_url:services.Studio.endpoint}",
            "--output",
            "json",
            "--only-show-errors",
        ]
    )
    return payload


def wait_for_idle_compute(args):
    while True:
        active = active_jobs_on_compute(
            list_jobs(args.az_executable, args.max_results),
            args.compute,
        )
        if not active:
            return

        names = ", ".join(f"{job.get('name')}:{job.get('status')}" for job in active)
        print(
            f"Waiting for active jobs on {args.compute}: {names}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


def wait_for_job(args, job_name):
    while True:
        payload = job_status(args.az_executable, job_name)
        status = payload.get("status", "")
        print(f"{job_name}: {status}", flush=True)
        if status in TERMINAL_STATUSES:
            if status != "Completed":
                raise SystemExit(f"Azure job {job_name} ended with status {status}")
            return payload
        time.sleep(args.poll_seconds)


def submit_one(args, fold):
    try:
        checkpoint_path = checkpoint_path_for_args(args, fold)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None

    wait_for_idle_compute(args)

    command = build_job_create_command(
        job_file=args.job_file,
        fold=fold,
        checkpoint_path=checkpoint_path,
        az_executable=args.az_executable,
    )
    print(f"Submitting fold {fold}: {checkpoint_path}", flush=True)
    payload = run_json(command)
    job_name = payload["name"]
    studio_url = payload.get("studio_url") or payload.get("services", {}).get("Studio", {}).get("endpoint", "")
    print(f"Submitted {job_name}", flush=True)
    if studio_url:
        print(f"Studio URL: {studio_url}", flush=True)

    return wait_for_job(args, job_name)


def parse_args(
    argv=None,
    default_job_file=DEFAULT_JOB_FILE,
    default_compute=DEFAULT_COMPUTE,
    description=None,
):
    parser = argparse.ArgumentParser(
        description=description
        or (
            "Submit UNETR fine-tuning jobs sequentially. The script waits until "
            "the target Azure ML compute has no active jobs, submits one job, "
            "waits for it to finish, then moves to the next fold."
        )
    )
    checkpoint_source = parser.add_mutually_exclusive_group(required=True)
    checkpoint_source.add_argument(
        "--checkpoint_path",
        help=(
            "Checkpoint URI. Use {fold} in the URI to format per-fold paths, "
            "for example azureml://.../fold_{fold}/best_metric_model.pth."
        ),
    )
    checkpoint_source.add_argument(
        "--checkpoint_job_name",
        help=(
            "Completed Azure ML training job name whose output_model contains "
            "fold_<fold>/best_metric_model.pth. This builds the workspaceblobstore "
            "URI automatically."
        ),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--job_file", default=default_job_file)
    parser.add_argument("--compute", default=default_compute)
    parser.add_argument("--poll_seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max_results", type=int, default=100)
    parser.add_argument("--az_executable", default="az")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for fold in args.folds:
        submit_one(args, fold)
    print("All queued fine-tuning jobs completed.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
