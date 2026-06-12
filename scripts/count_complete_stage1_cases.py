"""Count completed stage1 cases for a case manifest.

For each manifest row, checks
``{root}/{prefix}__{pretrain_method}/stage1/{image}__{candidate_id}/joint/metrics.json``
for ``sweep_case_complete == true``. Counting is manifest-driven on purpose:
output roots can contain stray case directories (e.g. reference runs), so
directory globs over-count.

Prints ``done=<n> total=<m> missing=<global indices>`` and exits 0 when all
cases are complete, 1 otherwise (usable as a loop condition).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def case_metrics_path(root: Path, prefix: str, row: dict[str, str]) -> Path:
    return (
        root
        / f"{prefix}__{row['pretrain_method']}"
        / "stage1"
        / f"{row['image']}__{row['candidate_id']}"
        / "joint"
        / "metrics.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing the {prefix}__{method} case directories",
    )
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--max-missing",
        type=int,
        default=20,
        help="Maximum number of missing global indices to print",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.case_manifest.open()))
    done = 0
    missing: list[str] = []
    for row in rows:
        path = case_metrics_path(args.root, args.prefix, row)
        complete = False
        if path.is_file():
            try:
                complete = json.loads(path.read_text()).get("sweep_case_complete") is True
            except Exception:
                complete = False
        if complete:
            done += 1
        else:
            missing.append(row.get("global_case_index", row.get("case_index", "?")))

    shown = ",".join(missing[: args.max_missing])
    if len(missing) > args.max_missing:
        shown += ",..."
    print(f"done={done} total={len(rows)} missing=[{shown}]")
    return 0 if done == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
