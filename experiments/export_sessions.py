"""Export a trusted local ElenchiX database to the paper's binary cohort CSV schema."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from elenchix.kt.data import REQUIRED_COLUMNS, load_binary_events
from elenchix.kt.online import OnlineStateStore


def _folds(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    result = {}
    for row in manifest["learners"]:
        learner = str(row["source_user_id"])
        fold = row["fold"]
        if type(fold) is not int or not 0 <= fold <= 4:
            raise ValueError("fold manifest values must be integers in [0, 4]")
        if learner in result:
            raise ValueError("fold manifest contains a duplicate learner")
        result[learner] = fold
    return result


def export_sessions(
    state_path: Path,
    output: Path,
    *,
    learner_ids: list[str] | None = None,
    fold_manifest: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Export committed training events, retaining their scores and chronological order.

    The database is a trusted local pickle snapshot. Its BinaryEvent ledger is
    the same input used by cumulative AKT fitting. Pending/unscored rounds add
    no rows. Anonymous IDs derive from the stored learner order, so selecting
    a subset or adding learners does not renumber existing learners.
    """
    state_path, output = Path(state_path).resolve(), Path(output).resolve()
    protected = [state_path]
    if fold_manifest is not None:
        protected.append(Path(fold_manifest).resolve())
    if any(output == path or (output.exists() and output.samefile(path)) for path in protected):
        raise ValueError("CSV output must not overwrite an input database or manifest")
    if output.suffix.lower() != ".csv":
        raise ValueError("output must use the .csv extension")
    if output.exists() and not overwrite:
        raise FileExistsError("Output already exists; choose another filename or use --overwrite.")
    revision, state = OnlineStateStore.read_only(state_path)
    if state is None:
        raise ValueError("The database has no completed assessment records.")
    if state["tracker"]["schema_version"] != 1:
        raise ValueError("unsupported online state schema")
    selected = set(learner_ids) if learner_ids is not None else None
    if selected is not None:
        known = set(state["history"]) | {active.learner_id for active in state["pending"].values()}
        unknown = selected - known
        if unknown:
            raise ValueError(f"Unknown learners: {sorted(unknown)}")
    completed = {
        (learner, record.evidence.round_index, record.evidence.case_id)
        for learner, records in state["history"].items()
        for record in records
    }
    events = sorted(
        (
            event
            for event in state["tracker"]["events"]
            if selected is None or event.learner_id in selected
        ),
        key=lambda event: (event.learner_order, event.round_index, event.order_in_round),
    )
    if not events:
        raise ValueError("Selected learners have no completed scores; pending or unscored encounters add no rows.")
    assignments = _folds(fold_manifest)
    learners = {event.learner_id for event in events}
    if assignments is not None and not learners <= assignments.keys():
        raise ValueError("fold manifest must cover every exported learner")
    rows = []
    for event in events:
        if (event.learner_id, event.round_index, event.case_id) not in completed:
            raise ValueError("database contains a score without a completed encounter")
        row = asdict(event)
        row["learner_id"] = f"learner_{event.learner_order + 1:016x}"
        if assignments is not None:
            row["fold"] = assignments[event.learner_id]
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the complete CSV before publishing it. A failed export leaves
    # existing files intact; hard-link publication refuses concurrent overwrites.
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=".export-",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        load_binary_events(temporary, expected_learners=len(learners))
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "revision": revision,
        "events": len(rows),
        "learners": len(learners),
        "fold_source": "manifest" if assignments is not None else "database",
        "folds": sorted({row["fold"] for row in rows}),
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the local database to the paper binary CSV schema.")
    parser.add_argument("--state", type=Path, default=Path("artifacts/live_test/state.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("data/private/live_binary.csv"))
    parser.add_argument("--learner-id", action="append", help="select a learner; may be repeated")
    parser.add_argument("--fold-manifest", type=Path, help="optional JSON learner-to-fold assignment for experiments")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing CSV export")
    args = parser.parse_args(argv)
    try:
        result = export_sessions(
            args.state,
            args.output,
            learner_ids=args.learner_id,
            fold_manifest=args.fold_manifest,
            overwrite=args.overwrite,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Exported {result['learners']} learners and {result['events']} scores: {result['output']}")
    if result["fold_source"] == "database":
        print("Folds retained from the database. For five-fold experiments, supply --fold-manifest; online records default to fold 0.")
    else:
        print("Folds assigned from the manifest. The source database is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
