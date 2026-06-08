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


def keep_largest_cc_after_argmax(pred_mask: torch.Tensor) -> torch.Tensor:
    """Keep one foreground component in each item of a [B, D, H, W] label map."""
    if pred_mask.ndim != 4:
        raise ValueError(f"Expected [B, D, H, W], received {tuple(pred_mask.shape)}")
    device = pred_mask.device
    cleaned = []
    for mask in pred_mask.detach().cpu().numpy():
        cleaned.append(
            torch.as_tensor(
                keep_largest_cc_numpy(mask),
                dtype=torch.uint8,
                device=device,
            )
        )
    return torch.stack(cleaned, dim=0)


def lcc_label_map_after_argmax(logits_or_probs: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Argmax a [B, C, D, H, W] tensor, then CPU-clean each foreground class."""
    if logits_or_probs.ndim != 5:
        raise ValueError(
            f"Expected [B, C, D, H, W], received {tuple(logits_or_probs.shape)}"
        )
    label_map = torch.argmax(logits_or_probs, dim=1)
    device = logits_or_probs.device
    cleaned_batches = []

    for labels in label_map.detach().cpu().numpy():
        cleaned = np.zeros_like(labels, dtype=np.uint8)
        for class_idx in range(1, int(out_channels)):
            class_mask = keep_largest_cc_numpy(labels == class_idx)
            cleaned[class_mask.astype(bool)] = class_idx
        cleaned_batches.append(torch.as_tensor(cleaned, dtype=torch.uint8, device=device))

    return torch.stack(cleaned_batches, dim=0)


def one_hot_label_map(label_map: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Convert [B, D, H, W] label maps to [B, C, D, H, W] uint8 one-hot masks."""
    if label_map.ndim != 4:
        raise ValueError(f"Expected [B, D, H, W], received {tuple(label_map.shape)}")
    one_hot = torch.nn.functional.one_hot(
        label_map.long(), num_classes=int(out_channels)
    )
    return one_hot.movedim(-1, 1).to(torch.uint8)


def lcc_one_hot_after_argmax(logits_or_probs: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Argmax and CPU-clean foreground components, returning one-hot masks."""
    return one_hot_label_map(
        lcc_label_map_after_argmax(logits_or_probs, out_channels),
        out_channels,
    )


def discretize_clean_ensemble_probs(ensemble_probs: torch.Tensor) -> torch.Tensor:
    """Discretize final averaged ensemble probabilities and apply one LCC pass."""
    if ensemble_probs.ndim != 5:
        raise ValueError(
            f"Expected [B, C, D, H, W], received {tuple(ensemble_probs.shape)}"
        )
    return lcc_label_map_after_argmax(ensemble_probs, ensemble_probs.shape[1])


def keep_largest_cc_numpy(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Return a uint8 binary mask containing only the largest 3D component."""
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, received shape {mask.shape}")

    foreground = mask.astype(bool)
    if not foreground.any():
        return np.zeros_like(mask, dtype=np.uint8)

    try:
        from scipy.ndimage import generate_binary_structure, label
    except Exception:
        return _keep_largest_cc_numpy_fallback(foreground, connectivity)

    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    components, component_count = label(foreground, structure=structure)
    if component_count == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    counts = np.bincount(components.ravel())
    counts[0] = 0  # Ignore background.
    return (components == counts.argmax()).astype(np.uint8)


def _neighbor_offsets(connectivity: int) -> list[tuple[int, int, int]]:
    connectivity = int(connectivity)
    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                if abs(dz) + abs(dy) + abs(dx) <= connectivity:
                    offsets.append((dz, dy, dx))
    return offsets


def _keep_largest_cc_numpy_fallback(foreground: np.ndarray, connectivity: int) -> np.ndarray:
    """Small pure-NumPy fallback used only when SciPy is unavailable."""
    foreground = np.asarray(foreground, dtype=bool)
    visited = np.zeros_like(foreground, dtype=bool)
    largest_component: list[tuple[int, int, int]] = []
    offsets = _neighbor_offsets(connectivity)
    shape = foreground.shape

    for start in zip(*np.nonzero(foreground)):
        start = tuple(int(value) for value in start)
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []

        while stack:
            point = stack.pop()
            component.append(point)
            for offset in offsets:
                neighbor = tuple(point[axis] + offset[axis] for axis in range(3))
                if any(neighbor[axis] < 0 or neighbor[axis] >= shape[axis] for axis in range(3)):
                    continue
                if foreground[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        if len(component) > len(largest_component):
            largest_component = component

    output = np.zeros_like(foreground, dtype=np.uint8)
    if largest_component:
        z, y, x = zip(*largest_component)
        output[z, y, x] = 1
    return output


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
