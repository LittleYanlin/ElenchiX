# ElenchiX

Code for **ElenchiX: A Multi-Agent Clinical Reasoning Tutoring System with Cross-Case Learner Modeling and a Clinical-Graph Adapter**. This repository contains the interactive tutoring workflow and the paper's sequential KT + graph-adapter prediction pipeline.

**Implementation note.** This repository is a streamlined public version of the study implementation. It omits the frontend, backend service infrastructure, and other application-specific components used during the original experiments. The authors have verified that the core tutoring workflow and KT–graph-adapter methodology remain consistent with the original implementation; the released prediction pipeline reproduces the paper's results.

## 1. Run ElenchiX

Use Python 3.11 or 3.12 and install dependencies with `uv sync --locked --group kt`. For a new installation, copy `.env.example` to `.env` and `configs/testing.example.yaml` to `configs/private.testing.yaml`; fill in the three agents' model names, API credentials/endpoints, and JSON graph path.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m elenchix.testing
```

Enter a learner ID and topic. The terminal shows planning reasons, executed tools, tutor replies, assessment, and cumulative AKT/graph-adapter refitting. `/finish` ends the dialogue immediately and starts assessment/refitting; `/next` starts another case; `/quit` exits. Dialogue and model state are saved in `artifacts/live_test/state.sqlite3` (`kt.online_state_path`).

## 2. Export the database to CSV

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/export_sessions.py --state artifacts/live_test/state.sqlite3 --output data/private/cohort.csv
```

The export contains committed assessment events with anonymous learner IDs and the experiment schema:

```text
learner_id,learner_order,group,fold,round_index,order_in_round,case_id,target_type,target_id,response,score_01
```

`response = int(score > 0)` and `score_01 = (score + 1) / 2`. Add `--overwrite` to refresh an existing export. The database remains unchanged.

For five-fold experiments, append `--fold-manifest data/private/folds.json`. Its format is `{"learners": [{"source_user_id": "student_01", "fold": 0}]}`; include every exported learner and assign learner-disjoint folds 0–4. Each fold needs scored KP targets after encounter 1. Without a manifest, the export retains the database folds (online records default to 0). Reuse the study's original fold assignments when reproducing its results.

## 3. Run the prediction experiment

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/run_experiment.py --data data/private/cohort.csv --output-dir artifacts/experiment_akt
```

This runs AKT training, nested graph-adapter fitting, and evaluation. Default values are: five learner folds, four inner folds, 20 epochs, batch size 64, maximum sequence length 200, and five seeds. Inner splits group identical case trajectories; each outer training set needs at least four groups with targets after encounter 1. Add `--workers N` to train independent models in parallel. Use a new output directory for each run.
To replace sequential KT, supply another model's nested out-of-fold probabilities:

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/run_experiment.py --data data/private/cohort.csv --backbone DKT --backbone-predictions data/private/dkt_nested_predictions.csv --output-dir artifacts/experiment_dkt
```

AKT training is built in; other KT models supply a CSV with `outer_context,target_key,prediction`. Use `BinaryEvent.target_key` from `elenchix.kt.data.load_binary_events` as the target identifier.

Each run writes:

| File | Contents |
| --- | --- |
| `paper_table.csv` | Backbone and adapted all-KP, all-cold, and reachable-cold AUC; all-KP Log loss and Brier score; target counts |
| `paired_bootstrap.csv` | Adapter-minus-backbone changes and 95% percentile intervals from 2,000 paired learner-clustered samples; AUC changes in percentage points |
| `metrics.json` | AUC, Log loss, and Brier score for all three slices |
| `predictions.csv` | Pooled OOF target labels, both probabilities, and adaptation eligibility |
| `fit_manifests.json` | Relation reliabilities and residual readout parameters per outer fold |
| `run_manifest.json` | Input hashes, graph configuration, backbone name, and bootstrap settings |

## 4. Load graph data from JSON

Set `graph.json_path` in `configs/private.testing.yaml` to a JSON file (relative paths resolve from the YAML directory). Use the node types `case`, `knowledge`, and `assessment_point`:

```json
{
  "nodes": [
    {"id": "case_1", "type": "case", "name": "Case 1", "attributes": {"topic": "Pneumonia", "vignette": "Complete case context"}},
    {"id": "kp_1", "type": "knowledge", "name": "History taking"},
    {"id": "kp_2", "type": "knowledge", "name": "Evidence interpretation"},
    {"id": "A1", "type": "assessment_point", "name": "Reasoning ability"}
  ],
  "edges": [
    {"source": "case_1", "target": "kp_2", "type": "G0_CASE_KP", "weight": 1.0},
    {"source": "case_1", "target": "A1", "type": "G0_CASE_ASSESSMENT", "weight": 1.0},
    {"source": "A1", "target": "kp_1", "type": "G0_REQUIRES_KNOWLEDGE", "weight": 1.0},
    {"source": "kp_1", "target": "kp_2", "type": "EVIDENCE_FOR", "weight": 0.9}
  ]
}
```

Match relation names to the mappings in `configs/testing.example.yaml`. Use unique node IDs, valid edge endpoints, and extraction-confidence weights in [0, 1]. Full live testing requires all 20 rubric abilities with Ability-to-KP links and complete case context. `transfer_relations` contains directed KP-to-KP relations; an incoming edge carries source evidence into the target KP.
