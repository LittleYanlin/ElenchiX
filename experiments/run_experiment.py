"""Run sequential KT -> nested graph adaptation -> paper-format result tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from elenchix.config import load_config
from elenchix.graph import create_graph_store
from elenchix.kt.data import load_binary_events
from elenchix.kt.evaluation import DEFAULT_BOOTSTRAP_SEED, write_results
from elenchix.kt.preparation import cohort_rows, load_nested_predictions, prepare_rows
from elenchix.kt.pykt_akt import AKTTrainConfig, train_akt_oof
from elenchix.kt.reliable_transfer import fully_nested_predictions


def run_experiment(
    data: Path,
    config: Path,
    output: Path,
    *,
    backbone: str = "AKT",
    backbone_predictions: Path | None = None,
    training: AKTTrainConfig | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    events = load_binary_events(data)
    source_rows = cohort_rows(events)
    targets = [row for row in source_rows if row["round_index"] > 1]
    if {row["fold"] for row in targets} != set(range(5)):
        raise ValueError("Each of the five learner folds (0-4) needs KP targets after encounter 1.")
    if backbone_predictions is None and backbone.upper() != "AKT":
        raise ValueError(
            "Other KT backbones require --backbone-predictions with nested OOF values."
        )
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    settings = load_config(config)
    graph = create_graph_store(settings.graph)
    try:
        known = {node.id for node in graph.list_knowledge()}
        observed = {row["target_id"] for row in source_rows}
        # The frozen gold graph covers a subset of the assessment vocabulary.
        # Unmapped targets still belong to the KT benchmark; their empty graph
        # slots make the selective adapter preserve the backbone prediction.
        graph_coverage = {
            "knowledge_ids_in_data": len(observed),
            "knowledge_ids_in_graph": len(known),
            "knowledge_ids_without_graph_nodes": len(observed - known),
            "target_rows_without_graph_nodes": sum(
                row["target_id"] not in known for row in targets
            ),
        }
    finally:
        graph.close()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory to keep experiment runs separate.")
    generated = backbone_predictions is None
    if generated:
        training = training or AKTTrainConfig(workers=1)
        print(
            f"Training AKT: five outer folds, four inner folds, {training.epochs} epochs "
            f"and {len(training.seed_offsets)} seeds...",
            flush=True,
        )
        train_akt_oof(events, output / "backbone", training)
        backbone_predictions = output / "backbone/akt_nested_predictions.csv"
    predictions = load_nested_predictions(backbone_predictions, events)
    print("Fitting source expectations and the nested graph adapter...", flush=True)
    rows = prepare_rows(source_rows, targets, predictions, config)
    predicted, fitted = fully_nested_predictions(rows)
    print(
        f"Exporting paper tables and {bootstrap_replicates} paired bootstrap replicates...",
        flush=True,
    )
    metrics = write_results(
        output,
        predicted,
        fitted,
        backbone=backbone,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    inputs = {"data": data, "graph": settings.graph.json_path, "predictions": backbone_predictions}
    manifest = {
        "backbone": backbone,
        "prediction_source": "trained" if generated else "external",
        "learners": len({event.learner_id for event in events}),
        "events": len(events),
        "outer_folds": 5,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "graph_config": settings.graph.model_dump(mode="json"),
        "graph_coverage": graph_coverage,
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in inputs.items()
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/private.testing.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default="AKT", help="name used in result tables")
    parser.add_argument(
        "--backbone-predictions",
        type=Path,
        help="replace AKT with another KT model's nested prediction CSV",
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel AKT training processes")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        run_experiment(
            args.data,
            args.config,
            args.output_dir,
            backbone=args.backbone,
            backbone_predictions=args.backbone_predictions,
            training=AKTTrainConfig(workers=args.workers, device=args.device),
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")
    print(f"Results: {args.output_dir.resolve() / 'paper_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
