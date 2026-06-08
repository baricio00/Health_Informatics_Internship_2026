"""Largest-connected-component helpers for binary myocardium segmentation masks.

Use LCC only after prediction discretization. The training loss must continue to
receive untouched logits. For ensemble inference, average probabilities first,
then discretize and apply LCC once to the final mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label
from monai.transforms import KeepLargestConnectedComponent

_KEEP_LARGEST_CC = KeepLargestConnectedComponent(
    applied_labels=[1],
    is_onehot=False,
    independent=True,
    connectivity=1,
    num_components=1,
)


def keep_largest_cc_after_argmax(pred_mask: torch.Tensor) -> torch.Tensor:
    """Keep one foreground component in each item of a [B, D, H, W] label map."""
    if pred_mask.ndim != 4:
        raise ValueError(f"Expected [B, D, H, W], received {tuple(pred_mask.shape)}")
    cleaned = [
        _KEEP_LARGEST_CC(mask.unsqueeze(0).to(torch.uint8))[0]
        for mask in pred_mask
    ]
    return torch.stack(cleaned, dim=0)


def discretize_clean_ensemble_probs(ensemble_probs: torch.Tensor) -> torch.Tensor:
    """Discretize final averaged ensemble probabilities and apply one LCC pass."""
    if ensemble_probs.ndim != 5:
        raise ValueError(
            f"Expected [B, C, D, H, W], received {tuple(ensemble_probs.shape)}"
        )
    return keep_largest_cc_after_argmax(torch.argmax(ensemble_probs, dim=1))


def keep_largest_cc_numpy(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Return a uint8 binary mask containing only the largest 3D component."""
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, received shape {mask.shape}")

    foreground = mask.astype(bool)
    if not foreground.any():
        return np.zeros_like(mask, dtype=np.uint8)

    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    components, component_count = label(foreground, structure=structure)
    if component_count == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    counts = np.bincount(components.ravel())
    counts[0] = 0  # Ignore background.
    return (components == counts.argmax()).astype(np.uint8)


def _iter_nifti_files(output_dir: Path) -> Iterable[Path]:
    for path in sorted(output_dir.rglob("*")):
        name = path.name.lower()
        if path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz")):
            yield path


def _as_binary_3d(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """Return a 3D binary view and its original shape, or None for non-mask files."""
    array = np.asarray(data)
    original_shape = array.shape

    if array.ndim == 4 and 1 in (array.shape[0], array.shape[-1]):
        array = np.squeeze(array)
    if array.ndim != 3 or not np.isfinite(array).all():
        return None

    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-6):
        return None
    unique_values = set(np.unique(rounded).astype(int).tolist())
    if not unique_values.issubset({0, 1}):
        return None

    return rounded.astype(np.uint8), original_shape


def _restore_shape(mask: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    if mask.shape == original_shape:
        return mask
    if len(original_shape) == 4 and original_shape[0] == 1:
        return mask[None, ...]
    if len(original_shape) == 4 and original_shape[-1] == 1:
        return mask[..., None]
    raise ValueError(f"Cannot restore cleaned mask shape {mask.shape} to {original_shape}")


def postprocess_exported_nifti_masks(output_dir: str | Path) -> list[Path]:
    """Apply LCC in place to binary NIfTI masks exported under ``output_dir``.

    Non-binary NIfTI files are skipped deliberately. This lets the Azure ML
    launcher post-process exported masks without accidentally touching CT
    volumes, probability maps, or unrelated artifacts.
    """
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Inference output directory does not exist: {root}")

    cleaned_paths: list[Path] = []
    skipped_paths: list[Path] = []

    for path in _iter_nifti_files(root):
        image = nib.load(str(path))
        parsed = _as_binary_3d(np.asanyarray(image.dataobj))
        if parsed is None:
            skipped_paths.append(path)
            continue

        mask, original_shape = parsed
        cleaned = _restore_shape(keep_largest_cc_numpy(mask), original_shape)

        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        output = nib.Nifti1Image(cleaned.astype(np.uint8), image.affine, header=header)
        output.set_qform(image.get_qform(), int(image.header["qform_code"]))
        output.set_sform(image.get_sform(), int(image.header["sform_code"]))
        nib.save(output, str(path))
        cleaned_paths.append(path)

    print(
        f"LCC post-processing complete: cleaned {len(cleaned_paths)} binary NIfTI mask(s); "
        f"skipped {len(skipped_paths)} non-binary NIfTI file(s).",
        flush=True,
    )
    for path in cleaned_paths:
        print(f"  cleaned: {path}", flush=True)

    return cleaned_paths
