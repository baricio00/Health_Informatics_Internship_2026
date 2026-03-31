# azureml-project-blueprint

End-to-end Azure Machine Learning reference project demonstrating **Azure ML CLI v2** workflows with **YAML-defined** data assets, environments, jobs, component-based pipelines, MLflow tracking, and model registration.

---

## Overview

`azureml-project-blueprint` is a reference implementation showing how to structure and manage an Azure Machine Learning project using **Azure ML CLI v2** and **declarative YAML configurations**.

The repository follows a clear separation between:

- **Pipelines (orchestration layer)** — a multi-step pipeline wired via YAML  
- **Components (reusable computational steps)** — each in its own folder with YAML + Python  
- **Assets and environments (infrastructure layer)** — data and environment definitions  

A synthetic clinical dataset is included to demonstrate preprocessing, training, MLflow tracking, and model registration in a reproducible way.

---

## What This Project Demonstrates

| Capability | Where |
|---|---|
| Data asset registration and versioning via YAML | `data/` |
| Environment definition and reproducibility | `environments/` |
| Standalone command job | `jobs/` |
| Reusable CLI components | `pipelines/training_pipeline/*/` |
| Pipeline orchestration via YAML | `pipelines/training_pipeline/training_pipeline.yml` |
| MLflow experiment tracking | Inside every training / evaluation script |
| Model registration in Azure ML Model Registry | `pipelines/training_pipeline/register_model/` |

---

## Project Structure

```
azureml-project-blueprint/
├── data/
│   ├── generate_synthetic_data.py      # Generate synthetic clinical CSV
│   ├── clinical_readmission.csv        # Generated dataset (500 rows)
│   └── clinical_readmission.yml        # Data asset YAML (uri_file)
│
├── environments/
│   ├── conda_dependencies.yaml         # Conda env spec (scikit-learn, MLflow)
│   └── environment.yml                 # Azure ML environment YAML
│
├── jobs/
│   ├── train_job.py                    # Standalone training script
│   └── train_job.yml                   # Command job YAML
│
├── pipelines/
│   └── training_pipeline/
│       ├── training_pipeline.yml       # Pipeline YAML (orchestration)
│       ├── data_prep/
│       │   ├── data_prep.py            # Stratified train/test split
│       │   └── data_prep.yml           # Component YAML
│       ├── train/
│       │   ├── train.py                # Logistic Regression + CV
│       │   └── train.yml               # Component YAML
│       ├── evaluate/
│       │   ├── evaluate.py             # Test-set evaluation & metrics
│       │   └── evaluate.yml            # Component YAML
│       └── register_model/
│           ├── register_model.py       # Model registration
│           └── register_model.yml      # Component YAML
│
└── README.md
```

---

## Prerequisites

- Azure CLI with the `ml` extension (`az extension add -n ml`)
- An Azure ML workspace
- A compute cluster — **must be set** in every YAML (`azureml:<YOUR-COMPUTE-CLUSTER>`)

---

## Workflow – Recommended Setup (Team Standard)

The steps below describe the recommended workflow, from connecting to the VM all the way to running jobs and pipelines.

---

### Step 1 — Connect to the Compute Instance with VS Code

1. Go to **Azure ML Studio → Compute → Compute instances**.
2. Select your compute instance.
3. If it is stopped, click **Start**.
4. In the **Applications** column:
   - Click **VS Code (Web)**, or  
   - Click the three dots `...` → **VS Code (Desktop)** if the desktop app is installed.

VS Code opens automatically and connects to the remote compute instance.

The integrated terminal runs directly on the VM (`azureuser@<vm-name>`), and the working directory  
`~/cloudfiles/code/Users/<username>/` is persistent and shared across sessions.

---

### Step 2 — Authenticate with Azure CLI

Open the integrated terminal in VS Code and run:

```bash
# Interactive login (opens the browser or uses device code flow)
az login

# Check the active subscription
az account show --output table

# If needed, select the correct subscription
az account set --subscription "<subscription-id>"
```

Verify that the `ml` extension is installed:

```bash
az extension show -n ml --query version -o tsv
# If missing:
az extension add -n ml
```

---

### Step 3 — Run a Standalone Job or a Pipeline

#### Option A — Standalone command job

A standalone job is useful for quick tests or one-off runs. Everything is defined in a single YAML that specifies inputs, outputs, environment, and command.

```bash
# 1. Register the environment (first time only, or when it changes)
az ml environment create --file environments/environment.yml

# 2. Register the data asset (first time only, or for new versions)
az ml data create --file data/clinical_readmission.yml

# 3. Submit the job
az ml job create --file jobs/train_job.yml
```

Monitor execution:

```bash
# Check status from the CLI
az ml job show --name <job-name> --query status -o tsv

# Stream logs in real time
az ml job stream --name <job-name>
```

The job also appears in **Azure ML Studio → Experiments → blueprint_standalone_job**.

---

#### Option B — Pipeline (recommended for structured workflows)

The pipeline splits the work into **reusable components**, each in its own folder with separate code and YAML. This enables **automatic caching**: Azure ML only re-runs components whose inputs or code have changed.

```bash
# 1. Register environment and data (if not already done)
az ml environment create --file environments/environment.yml
az ml data create --file data/clinical_readmission.yml

# 2. Submit the pipeline
az ml job create --file pipelines/training_pipeline/training_pipeline.yml
```

The pipeline runs 4 steps in sequence:

| Step | Component | What it does |
|------|-----------|--------------|
| 1 | `data_prep` | Stratified train/test split (70/30) |
| 2 | `train` | Logistic Regression with cross-validation |
| 3 | `evaluate` | Test-set metrics (accuracy, F1, ROC-AUC) |
| 4 | `register_model` | Registers the model in the Model Registry |

Monitor the pipeline:

```bash
# Pipeline status
az ml job show --name <pipeline-job-name> --query status -o tsv

# Stream logs (includes child step logs)
az ml job stream --name <pipeline-job-name>
```

The pipeline graph is visible in **Azure ML Studio → Jobs → blueprint_training_pipeline**.

---

### Common Commands Reference

| Action | Command |
|--------|---------|
| Login | `az login` |
| Set defaults | `az configure --defaults group=<rg> workspace=<ws>` |
| Create environment | `az ml environment create --file environments/environment.yml` |
| Register data asset | `az ml data create --file data/clinical_readmission.yml` |
| Run standalone job | `az ml job create --file jobs/train_job.yml` |
| Run pipeline | `az ml job create --file pipelines/training_pipeline/training_pipeline.yml` |
| Job status | `az ml job show --name <name> --query status` |
| Stream logs | `az ml job stream --name <name>` |
| List registered models | `az ml model list --query "[].{name:name, version:version}" -o table` |

---

## Quick Start

```bash
# Authenticate and set defaults
az login
az configure --defaults group=<resource-group> workspace=<workspace-name>

# One-time setup
az ml environment create --file environments/environment.yml
az ml data create --file data/clinical_readmission.yml

# Run the pipeline
az ml job create --file pipelines/training_pipeline/training_pipeline.yml
```


## Key Design Decisions

| Decision | Rationale |
|---|---|
| **One folder per component** with separate `code: ./` | Enables Azure ML **component-level caching** — only re-runs when that component's code or inputs change |
| **`uri_folder` outputs** | Standard way to pass data between pipeline steps; compatible with data asset registration |
| **MLflow logging in every script** | Unified experiment tracking across standalone jobs and pipeline components |
| **Model registration as pipeline step** | Demonstrates the full lifecycle: train → evaluate → register |

---

## Customisation

- **Compute**: set `azureml:<YOUR-COMPUTE-CLUSTER>` in `jobs/train_job.yml` and `training_pipeline.yml` — this is workspace-specific and has no default
- **Environment version**: bump the `version` field in `environments/environment.yml` and each component YAML
- **Data**: swap `clinical_readmission.csv` with your own dataset; adapt `data_prep.py` accordingly
- **Model**: replace Logistic Regression with any scikit-learn estimator in `train/train.py`


python LEMS-CT-project/scripts/train.py --out_dir ./output/models