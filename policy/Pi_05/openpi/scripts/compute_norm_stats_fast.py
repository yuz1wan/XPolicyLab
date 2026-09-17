"""Fast scalar-only statistics for supported YAM task configurations (no model imports)."""

import argparse
from pathlib import Path

from openpi.training.fast_yam_norm import run
from openpi.training.yam_tasks import TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", choices=TASKS, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, help="Default: one thread per dimension, capped by CPU count")
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(
        args.config_name,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        workers=args.workers,
        include_tail=args.include_tail,
    )


if __name__ == "__main__":
    main()
