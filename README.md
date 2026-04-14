# LEMS-CT: Late Enhancement Myocardium CT Segmentation

Automatic segmentation of heart myocardium from CT angiography scans of TAVI (Transcatheter Aortic Valve Implantation) patients. Built with PyTorch and [MONAI](https://monai.io/), trained and tracked on Azure Machine Learning with 5-fold cross-validation.

## Project Structure

```
lems_ct/
├── src/
│   ├── models/        # SegResNet model wrapper
│   ├── metrics/       # Surface distance metrics (ASD, HD95) and Dice
│   └── utils/         # Data loading, transforms, and training utilities
scripts/               # Training and inference entry points
config/                # Hyperparameters and model configuration
jobs/                  # Azure ML job definitions
notebooks/             # Data exploration and setup
data/                  # Cross-validation splits and data asset metadata
environments/          # Conda environment specifications
```

## Setup

### Environment

```bash
conda env create -f environments/conda_dependencies.yaml
```

The Azure ML environment uses the base image `mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04` with the same conda dependencies.

### Data

Patient CT scans are stored in NIfTI format on an Azure Datastore (`LEMS-CT-NIfTI:1`). Each patient folder contains:

- `CT_LATE.nii.gz` — CT angiography image
- `registration_mask.nii.gz` — Ground truth myocardial segmentation mask

Cross-validation fold assignments are defined in `data/cv_splits.csv`.

## Model

**Architecture**: [SegResNet](https://docs.monai.io/en/stable/networks.html#segresnet) (3D residual segmentation network from MONAI)

| Parameter      | Value          |
|----------------|----------------|
| Spatial dims   | 3              |
| In channels    | 1 (grayscale)  |
| Out channels   | 2 (background + myocardium) |
| Init filters   | 16             |
| Blocks down    | [1, 2, 2, 4]  |
| Blocks up      | [1, 1, 1]     |

**Loss**: DiceFocalLoss (Dice weight 0.5, Focal weight 0.5, gamma 2.0, alpha 0.75)

## Training

### Key settings (from `config/train_config.yaml`)

- Learning rate: 1e-4 with exponential warmup (500 steps) + polynomial decay
- Max iterations: 30,000
- Batch size: 2 (with 2 gradient accumulation steps → effective batch size 4)
- Validation every 500 iterations
- Exponential Moving Average (EMA) with decay 0.999
- Automatic Mixed Precision (AMP)
- Gradient clipping (max norm 1.0)

### Preprocessing

1. Reorient to RAS
2. Crop to foreground
3. Resample to 1.0 × 1.0 × 1.0 mm spacing
4. Clamp HU values to [-100, 400], scale to [0, 1]
5. Random 96×96×96 patch extraction (training only)
6. Random flips and 90° rotations (training only)

Validation uses sliding window inference with 50% overlap.

### Run locally

```bash
python scripts/train_job.py \
  --input_data /path/to/mounted/data \
  --output_model ./outputs \
  --fold 0 \
  --split_csv data/cv_splits.csv \
  --config config/train_config.yaml
```

### Resume from checkpoint

```bash
python scripts/train_job.py \
  --input_data /path/to/data \
  --output_model ./outputs \
  --fold 0 \
  --checkpoint /path/to/checkpoint.pth \
  --resume
```

### Submit all folds to Azure ML

```bash
bash scripts/submit_train_folds.sh
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Dice** | Overlap between prediction and ground truth (higher is better) |
| **HD95** | 95th percentile Hausdorff distance in mm (lower is better) |
| **ASD** | Average surface distance in mm (lower is better) |

Metrics are logged per fold to MLflow (e.g. `fold_0_val_dice`, `fold_0_val_hd95`, `fold_0_val_ASD`).
