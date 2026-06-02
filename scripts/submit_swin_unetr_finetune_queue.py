import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import submit_unetr_finetune_queue as queue


DEFAULT_JOB_FILE = "jobs/swin_unetr_finetune_job.yml"
DEFAULT_COMPUTE = "azureml:clusterprdwe-g-t2-vzhst6"

normalize_compute = queue.normalize_compute
active_jobs_on_compute = queue.active_jobs_on_compute
checkpoint_for_fold = queue.checkpoint_for_fold
checkpoint_path_from_job_name = queue.checkpoint_path_from_job_name
checkpoint_path_for_args = queue.checkpoint_path_for_args
validate_checkpoint_path = queue.validate_checkpoint_path
build_job_create_command = queue.build_job_create_command
run_json = queue.run_json
list_jobs = queue.list_jobs
job_status = queue.job_status
wait_for_idle_compute = queue.wait_for_idle_compute
wait_for_job = queue.wait_for_job
submit_one = queue.submit_one


def parse_args(argv=None):
    return queue.parse_args(
        argv,
        default_job_file=DEFAULT_JOB_FILE,
        default_compute=DEFAULT_COMPUTE,
        description=(
            "Submit Swin UNETR fine-tuning jobs sequentially. The script waits "
            "until the target Azure ML compute has no active jobs, submits one "
            "job, waits for it to finish, then moves to the next fold."
        ),
    )


def main(argv=None):
    args = parse_args(argv)
    for fold in args.folds:
        submit_one(args, fold)
    print("All queued Swin UNETR fine-tuning jobs completed.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
