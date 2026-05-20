#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOB_FILE="${PROJECT_ROOT}/jobs/train_job.yml"

for i in {0..4}
do
  echo "Submitting Fold $i to Azure ML..."
  
  az ml job create --file "$JOB_FILE" \
    --set display_name="Myocardium SegResNet Training - fold $i" \
    --set inputs.fold=$i
    
  echo "Fold $i submitted successfully!"
  echo "--------------------------------"
done
