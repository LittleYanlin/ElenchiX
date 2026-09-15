"""Prepare a private JSON test graph from the frozen G0 CSVs and full case context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from elenchix.schemas import GraphEdge, GraphNode


def prepare_graph(gold_dir: Path, normalized_dir: Path, output: Path) -> dict:
    inputs = []

    def csv_rows(name):
        path = gold_dir / name
        inputs.append(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def json_rows(name):
        path = normalized_dir / name
        inputs.append(path)
        with path.open(encoding="utf-8-sig") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    cases = csv_rows("case_nodes.csv")
    context = {str(row["case_id"]): row for row in json_rows("cases.jsonl")}
    nodes = []
    for row in cases:
        case_id = str(row["case_id"])
        full = context[case_id]
        body = full.get("raw_text") or full.get("summary")
        if not body:
            raise ValueError(f"missing full context for case {case_id}")
        attributes = {
            key: full[key]
            for key in (
                "summary",
                "chief_complaint",
                "department",
                "main_diagnosis",
                "differential_diagnoses",
                "stage_focus",
            )
            if key in full
        }
        attributes.update(vignette=body, difficulty=int(float(row.get("difficulty") or 3)))
        nodes.append(GraphNode(id=case_id, type="case", name=row["title"], attributes=attributes))
    for row in csv_rows("knowledge_points_189.csv"):
        nodes.append(GraphNode(id=row["kp_name"], type="knowledge", name=row["kp_name"]))
    for row in csv_rows("assessment_points_20.csv"):
        nodes.append(
            GraphNode(
                id=row["ability_code"],
                type="assessment_point",
                name=row["name"],
                attributes={key: row[key] for key in ("module", "description") if key in row},
            )
        )
    edges = []

    def edge(source, target, relation, weight=1.0):
        edges.append(
            GraphEdge(source=str(source), target=str(target), type=relation, weight=float(weight))
        )

    for row in csv_rows("case_kp_membership.csv"):
        edge(row["case_id"], row["kp_name"], "G0_CASE_KP")
    for row in csv_rows("case_assessment_membership.csv"):
        edge(row["case_id"], row["ability_code"], "G0_CASE_ASSESSMENT")
    for row in csv_rows("kp_kp_edges.csv"):
        edge(row["source"], row["target"], row["relation_type"], row["weight"])
    for row in csv_rows("assessment_kp_edges.csv"):
        edge(row["assessment_point"], row["kp"], "G0_REQUIRES_KNOWLEDGE", row["weight"])
    case_ids = {str(row["case_id"]) for row in cases}
    for row in json_rows("case_case_similarity_edges.jsonl"):
        if {str(row["left_case_id"]), str(row["right_case_id"])} <= case_ids:
            edge(row["left_case_id"], row["right_case_id"], "SIMILAR_CASE", row["similarity_score"])
    for row in json_rows("case_case_progression_edges.jsonl"):
        if {str(row["from_case_id"]), str(row["to_case_id"])} <= case_ids:
            edge(row["from_case_id"], row["to_case_id"], "NEXT_CASE", row["progression_score"])
    ids = {node.id for node in nodes}
    if len(ids) != len(nodes) or any(e.source not in ids or e.target not in ids for e in edges):
        raise ValueError("duplicate graph identity or missing edge endpoint")
    keys = [(e.source, e.target, e.type) for e in edges]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate graph edge")
    directed = set(keys)
    if any(
        (target, source, relation) not in directed
        for source, target, relation in keys
        if relation in {"DIFFERENTIAL_WITH", "ASSESSMENT_VARIANT_OF"}
    ):
        raise ValueError("a symmetric experimental relation is missing its reverse edge")
    manifest = {
        "scope": "G0 experimental subgraph with complete normalized case context",
        "nodes": dict(Counter(node.type for node in nodes)),
        "relations": dict(Counter(e.type for e in edges)),
        "input_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }
    payload = {
        "schema_version": "1.0",
        "provenance": manifest,
        "nodes": [n.model_dump() for n in sorted(nodes, key=lambda n: n.id)],
        "edges": [
            e.model_dump() for e in sorted(edges, key=lambda e: (e.type, e.source, e.target))
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("private_graph/g0_live.json"))
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_graph(args.gold_dir, args.normalized_dir, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
