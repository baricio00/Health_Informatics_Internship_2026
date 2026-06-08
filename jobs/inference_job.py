"""Azure ML entrypoint for ensemble inference with final NIfTI LCC cleaning."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference_job import main
from scripts.lcc_postprocessing import postprocess_exported_nifti_masks


def _output_dir_from_cli() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_dir", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args.output_dir


if __name__ == "__main__":
    output_dir = _output_dir_from_cli()
    main()
    postprocess_exported_nifti_masks(output_dir)
