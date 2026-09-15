"""Paper-faithful official pyKT AKT training and causal inference."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import BinaryEvent
from .pykt_compat import load_pykt_components

PAPER_SEED_OFFSETS = (0, 1009, 2027, 3037, 4051)


@dataclass(frozen=True, slots=True)
class AKTTrainConfig:
    epochs: int = 20
    learning_rate: float = 1e-4
    seed: int = 20260723
    seed_offsets: tuple[int, ...] = PAPER_SEED_OFFSETS
    maxlen: int = 200
    d_model: int = 128
    n_blocks: int = 1
    dropout: float = 0.1
    d_ff: int = 256
    final_fc_dim: int = 512
    num_attn_heads: int = 8
    l2: float = 1e-5
    device: str = "cpu"
    batch_size: int = 64
    workers: int = 1
    threads_per_worker: int = 1


@dataclass(frozen=True, slots=True)
class _AKTJob:
    name: str
    model_fold: int
    seed_offset: int
    fit_events: tuple[BinaryEvent, ...]
    valid_events: tuple[BinaryEvent, ...]
    vocab: dict[str, dict[str, int]]
    checkpoint: str
    config: AKTTrainConfig


@dataclass(frozen=True, slots=True)
class _AKTJobResult:
    name: str
    model_fold: int
    seed_offset: int
    actual_seed: int
    checkpoint: str
    predictions: list[dict[str, Any]]


def build_vocab(events: Sequence[BinaryEvent]) -> dict[str, dict[str, int]]:
    tokens = sorted({event.item_token for event in events})
    concepts = {"<PAD_OR_UNK>": 0}
    concepts.update({token: index for index, token in enumerate(tokens, start=1)})
    return {"concepts": concepts, "questions": dict(concepts)}


def _configure_torch_threads(count: int) -> None:
    import torch

    torch.set_num_threads(max(1, int(count)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _seed_everything(seed: int) -> None:
    load_pykt_components().set_seed(int(seed))


def _consume_dataloader_base_seed() -> int:
    """Advance torch RNG exactly as a new pyKT DataLoader iterator does."""

    import torch

    return int(torch.empty((), dtype=torch.int64).random_().item())


def _new_model(vocab: dict[str, dict[str, int]], config: AKTTrainConfig):
    import sys

    import torch

    components = load_pykt_components()
    model_class = components.model_classes["akt"]
    module = sys.modules[model_class.__module__]
    module.device = torch.device(config.device)
    model = model_class(
        n_question=len(vocab["concepts"]),
        n_pid=len(vocab["questions"]),
        d_model=config.d_model,
        n_blocks=config.n_blocks,
        dropout=config.dropout,
        d_ff=config.d_ff,
        final_fc_dim=config.final_fc_dim,
        num_attn_heads=config.num_attn_heads,
        l2=config.l2,
        emb_type="qid",
    )
    return model.to(torch.device(config.device))


def _prefix_sequences(
    events: Sequence[BinaryEvent], *, maxlen: int
) -> list[list[BinaryEvent]]:
    """Reproduce the paper's round-prefix-all-targets expansion."""

    grouped: dict[str, list[BinaryEvent]] = defaultdict(list)
    learner_order: dict[str, int] = {}
    for event in events:
        grouped[event.learner_id].append(event)
        learner_order[event.learner_id] = event.learner_order
    output: list[list[BinaryEvent]] = []
    for learner_id in sorted(grouped, key=lambda item: learner_order[item]):
        ordered = sorted(
            grouped[learner_id],
            key=lambda item: (
                item.round_index,
                item.order_in_round,
                item.question_token,
                item.item_token,
            ),
        )
        for target in ordered:
            if target.round_index <= 1:
                continue
            history = [item for item in ordered if item.round_index < target.round_index]
            history = history[-max(1, int(maxlen) - 1) :]
            if history:
                output.append([*history, target])
    return output


def _official_batch(
    sequences: Sequence[Sequence[BinaryEvent]],
    vocab: dict[str, dict[str, int]],
    config: AKTTrainConfig,
) -> tuple[Any, Any, Any, Any, Any]:
    """Build the tensors produced by pyKT KTDataset + model_forward for AKT."""

    import torch

    if not sequences:
        raise ValueError("AKT batch is empty")
    if any(len(sequence) < 2 or len(sequence) > config.maxlen for sequence in sequences):
        raise ValueError("AKT sequence length is outside [2, maxlen]")

    def padded(values: list[int], fill: int) -> list[int]:
        return values + [fill] * (config.maxlen - len(values))

    concepts = []
    questions = []
    responses = []
    select_masks = []
    for sequence in sequences:
        concepts.append(
            padded([vocab["concepts"].get(event.item_token, 0) for event in sequence], -1)
        )
        questions.append(
            padded(
                [vocab["questions"].get(event.question_token, 0) for event in sequence],
                -1,
            )
        )
        responses.append(padded([int(event.response) for event in sequence], -1))
        select_masks.append(padded([-1] * (len(sequence) - 1) + [1], -1))

    device = torch.device(config.device)
    c_full = torch.tensor(concepts, dtype=torch.long, device=device)
    q_full = torch.tensor(questions, dtype=torch.long, device=device)
    r_full = torch.tensor(responses, dtype=torch.float, device=device)
    s_full = torch.tensor(select_masks, dtype=torch.long, device=device)
    masks = (c_full[:, :-1] != -1) & (c_full[:, 1:] != -1)
    c = c_full[:, :-1] * masks
    q = q_full[:, :-1] * masks
    r = r_full[:, :-1] * masks
    c_shift = c_full[:, 1:] * masks
    q_shift = q_full[:, 1:] * masks
    r_shift = r_full[:, 1:] * masks
    selected = s_full[:, 1:] != -1
    cc = torch.cat((c[:, 0:1], c_shift), dim=1)
    cq = torch.cat((q[:, 0:1], q_shift), dim=1)
    cr = torch.cat((r[:, 0:1], r_shift), dim=1)
    return cc, cr, cq, r_shift, selected


def _fit_model(
    model: Any,
    sequences: Sequence[Sequence[BinaryEvent]],
    vocab: dict[str, dict[str, int]],
    config: AKTTrainConfig,
) -> None:
    import torch
    from torch.nn.functional import binary_cross_entropy

    if not sequences:
        raise ValueError("AKT training requires at least one round-prefix sequence")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _epoch in range(config.epochs):
        # pyKT trains through a torch DataLoader.  Creating its iterator consumes
        # one int64 base seed even with shuffle=False and num_workers=0.  The
        # value is otherwise unused, but advancing the RNG here is essential to
        # reproduce the dropout masks (and therefore the reported checkpoints).
        _consume_dataloader_base_seed()
        model.train()
        for start in range(0, len(sequences), config.batch_size):
            batch = sequences[start : start + config.batch_size]
            concept, response, question, response_shift, selected = _official_batch(
                batch, vocab, config
            )
            prediction, regularization = model(
                concept.long(), response.long(), question.long()
            )
            chosen = torch.masked_select(prediction[:, 1:], selected)
            target = torch.masked_select(response_shift, selected)
            loss = binary_cross_entropy(chosen.double(), target.double()) + regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _predict_targets(
    model: Any,
    sequences: Sequence[Sequence[BinaryEvent]],
    vocab: dict[str, dict[str, int]],
    config: AKTTrainConfig,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), config.batch_size):
            batch = sequences[start : start + config.batch_size]
            concept, response, question, _response_shift, selected = _official_batch(
                batch, vocab, config
            )
            prediction, _regularization = model(
                concept.long(), response.long(), question.long()
            )
            probabilities = torch.masked_select(prediction[:, 1:], selected).cpu().tolist()
            for sequence, probability in zip(batch, probabilities, strict=True):
                event = sequence[-1]
                rows.append(
                    {
                        "target_key": event.target_key,
                        "learner_id": event.learner_id,
                        "learner_order": event.learner_order,
                        "fold": event.fold,
                        "round_index": event.round_index,
                        "order_in_round": event.order_in_round,
                        "case_id": event.case_id,
                        "target_type": event.target_type,
                        "target_id": event.target_id,
                        "target": event.response,
                        "prediction": float(probability),
                    }
                )
    return rows


def _run_job(job: _AKTJob) -> _AKTJobResult:
    import torch

    _configure_torch_threads(job.config.threads_per_worker)
    actual_seed = job.config.seed + job.seed_offset + 31 * job.model_fold
    _seed_everything(actual_seed)
    model = _new_model(job.vocab, job.config)
    _fit_model(
        model,
        _prefix_sequences(job.fit_events, maxlen=job.config.maxlen),
        job.vocab,
        job.config,
    )
    checkpoint = Path(job.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    valid_sequences = _prefix_sequences(job.valid_events, maxlen=job.config.maxlen)
    predictions = (
        _predict_targets(model, valid_sequences, job.vocab, job.config)
        if valid_sequences
        else []
    )
    return _AKTJobResult(
        name=job.name,
        model_fold=job.model_fold,
        seed_offset=job.seed_offset,
        actual_seed=actual_seed,
        checkpoint=str(checkpoint),
        predictions=predictions,
    )


def _execute_jobs(jobs: Sequence[_AKTJob], workers: int) -> list[_AKTJobResult]:
    if workers <= 1:
        return [_run_job(job) for job in jobs]
    output: list[_AKTJobResult] = []
    with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
        futures = {executor.submit(_run_job, job): job.name for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            output.append(result)
            print(
                f"[AKT complete] {result.name} seed_offset={result.seed_offset} "
                f"actual_seed={result.actual_seed}",
                flush=True,
            )
    return output


def _average_members(members: Sequence[_AKTJobResult]) -> list[dict[str, Any]]:
    if not members:
        raise ValueError("cannot ensemble an empty AKT member list")
    by_member = [
        {str(row["target_key"]): row for row in member.predictions} for member in members
    ]
    keys = set(by_member[0])
    if any(set(current) != keys for current in by_member[1:]):
        raise AssertionError("AKT ensemble members have different target coverage")
    output = []
    for key in keys:
        base = dict(by_member[0][key])
        base["prediction"] = float(
            np.mean([float(current[key]["prediction"]) for current in by_member])
        )
        base["ensemble_members"] = len(members)
        output.append(base)
    output.sort(
        key=lambda row: (
            int(row["learner_order"]),
            int(row["round_index"]),
            int(row["order_in_round"]),
        )
    )
    return output


def _sequence_fold_assignment(
    events: Sequence[BinaryEvent], *, folds: int, seed: int
) -> dict[str, int]:
    """Rebuild G29's outcome-free case-sequence clustered inner folds."""

    by_learner: dict[str, list[BinaryEvent]] = defaultdict(list)
    for event in events:
        by_learner[event.learner_id].append(event)
    learner_rows = []
    for learner_id, learner_events in by_learner.items():
        cases_by_round: dict[int, set[str]] = defaultdict(set)
        for event in learner_events:
            cases_by_round[event.round_index].add(event.case_id)
        if any(len(values) != 1 for values in cases_by_round.values()):
            raise ValueError(f"learner {learner_id} has multiple cases in one round")
        case_ids = [next(iter(cases_by_round[index])) for index in sorted(cases_by_round)]
        sequence_id = hashlib.sha1("\x1f".join(case_ids).encode("utf-8")).hexdigest()[:20]
        eligible = sum(event.round_index > 1 for event in learner_events)
        # Single-encounter learners still supply training history. Assign them
        # after target-bearing clusters rather than dropping their fold entry.
        learner_rows.append((learner_id, sequence_id, eligible))
    by_sequence: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for learner_id, sequence_id, eligible in learner_rows:
        by_sequence[sequence_id].append((learner_id, eligible))
    clusters = [
        {
            "sequence_id": sequence_id,
            "learners": [item[0] for item in rows],
            "learner_count": len(rows),
            "eligible": sum(item[1] for item in rows),
        }
        for sequence_id, rows in by_sequence.items()
    ]
    clusters.sort(
        key=lambda row: (
            -int(row["eligible"]),
            hashlib.sha1(f"{seed}|{row['sequence_id']}".encode()).hexdigest(),
        )
    )
    fold_targets = [0] * folds
    fold_learners = [0] * folds
    result: dict[str, int] = {}
    for cluster in clusters:
        fold = min(
            range(folds),
            key=lambda index: (fold_targets[index], fold_learners[index], index),
        )
        for learner_id in cluster["learners"]:
            result[str(learner_id)] = fold
        fold_targets[fold] += int(cluster["eligible"])
        fold_learners[fold] += int(cluster["learner_count"])
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty prediction file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_akt_oof(
    events: Sequence[BinaryEvent],
    output_dir: str | Path,
    config: AKTTrainConfig | None = None,
    *,
    nested: bool = True,
) -> list[dict[str, Any]]:
    """Run the reported five-seed outer and fully nested official-AKT protocol."""

    config = config or AKTTrainConfig()
    if config.epochs < 1 or config.maxlen < 2 or not config.seed_offsets:
        raise ValueError("invalid AKT training configuration")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = sorted({event.fold for event in events})
    if nested and folds != [0, 1, 2, 3, 4]:
        raise ValueError(
            "the frozen nested protocol requires learner folds [0, 1, 2, 3, 4]; "
            f"received {folds}"
        )
    global_vocab = build_vocab(events)
    (output / "vocab.json").write_text(
        json.dumps(global_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "train_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    outer_jobs = []
    for fold in folds:
        for seed_offset in config.seed_offsets:
            outer_jobs.append(
                _AKTJob(
                    name=f"outer_fold_{fold}",
                    model_fold=fold,
                    seed_offset=seed_offset,
                    fit_events=tuple(event for event in events if event.fold != fold),
                    valid_events=tuple(event for event in events if event.fold == fold),
                    vocab=global_vocab,
                    checkpoint=str(
                        output / "outer" / f"fold_{fold}" / f"seed_{seed_offset}.pt"
                    ),
                    config=config,
                )
            )
    nested_manifest: dict[str, Any] = {}
    nested_jobs: list[_AKTJob] = []
    if nested:
        for outer_context in folds:
            remaining = [event for event in events if event.fold != outer_context]
            assignment = _sequence_fold_assignment(
                remaining, folds=4, seed=20260803 + outer_context
            )
            populated = {
                assignment[event.learner_id] for event in remaining if event.round_index > 1
            }
            if populated != set(range(4)):
                raise ValueError(
                    f"outer fold {outer_context}: four inner folds need target-bearing "
                    "case-sequence clusters; collect more distinct learner trajectories"
                )
            context_vocab = build_vocab(remaining)
            vocab_path = output / "nested" / f"outer_{outer_context}" / "vocab.json"
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            vocab_path.write_text(
                json.dumps(context_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            nested_manifest[str(outer_context)] = {
                "inner_fold_seed": 20260803 + outer_context,
                "inner_folds": 4,
                "learners": len(assignment),
                "vocab_size": len(context_vocab["concepts"]) - 1,
                "fold_target_rows": {
                    str(inner_fold): sum(
                        event.round_index > 1
                        and assignment[event.learner_id] == inner_fold
                        for event in remaining
                    )
                    for inner_fold in range(4)
                },
            }
            for inner_fold in range(4):
                fit_events = tuple(
                    event
                    for event in remaining
                    if assignment[event.learner_id] != inner_fold
                )
                held_events = tuple(
                    event
                    for event in remaining
                    if assignment[event.learner_id] == inner_fold
                )
                for seed_offset in config.seed_offsets:
                    nested_jobs.append(
                        _AKTJob(
                            name=f"outer_{outer_context}_inner_{inner_fold}",
                            model_fold=inner_fold,
                            seed_offset=seed_offset,
                            fit_events=fit_events,
                            valid_events=held_events,
                            vocab=context_vocab,
                            checkpoint=str(
                                output
                                / "nested"
                                / f"outer_{outer_context}"
                                / f"fold_{inner_fold}"
                                / f"seed_{seed_offset}.pt"
                            ),
                            config=config,
                        )
                    )

    full_jobs = [
        _AKTJob(
            name="full",
            model_fold=0,
            seed_offset=seed_offset,
            fit_events=tuple(events),
            valid_events=(),
            vocab=global_vocab,
            checkpoint=str(output / "full" / f"seed_{seed_offset}.pt"),
            config=config,
        )
        for seed_offset in config.seed_offsets
    ]

    # Every model is independent.  Submit outer, nested, and full-data jobs to
    # one pool so high-core servers can stay busy without changing the paper's
    # one-thread-per-model numerical protocol.
    all_results = _execute_jobs(
        [*outer_jobs, *nested_jobs, *full_jobs], config.workers
    )
    outer_results = [
        result for result in all_results if result.name.startswith("outer_fold_")
    ]
    nested_results = [
        result
        for result in all_results
        if result.name.startswith("outer_") and "_inner_" in result.name
    ]
    full_results = [result for result in all_results if result.name == "full"]

    predictions: list[dict[str, Any]] = []
    for fold in folds:
        members = [result for result in outer_results if result.model_fold == fold]
        predictions.extend(_average_members(members))
    predictions.sort(key=lambda row: str(row["target_key"]))
    _write_csv(output / "akt_oof_predictions.csv", predictions)

    nested_rows: list[dict[str, Any]] = []
    if nested:
        oof_by_key = {str(row["target_key"]): row for row in predictions}
        for outer_context in folds:
            for event in events:
                if event.fold == outer_context and event.round_index > 1:
                    nested_rows.append(
                        {
                            "outer_context": outer_context,
                            "target_key": event.target_key,
                            "prediction": float(oof_by_key[event.target_key]["prediction"]),
                            "ensemble_members": len(config.seed_offsets),
                        }
                    )
            for inner_fold in range(4):
                name = f"outer_{outer_context}_inner_{inner_fold}"
                members = [result for result in nested_results if result.name == name]
                for row in _average_members(members):
                    nested_rows.append(
                        {
                            "outer_context": outer_context,
                            "target_key": row["target_key"],
                            "prediction": row["prediction"],
                            "ensemble_members": row["ensemble_members"],
                        }
                    )
        expected = sum(event.round_index > 1 for event in events) * len(folds)
        if len(nested_rows) != expected:
            raise AssertionError(
                f"nested AKT prediction coverage mismatch: {len(nested_rows)} != {expected}"
            )
        nested_rows.sort(key=lambda row: (row["outer_context"], row["target_key"]))
        _write_csv(output / "akt_nested_predictions.csv", nested_rows)
        (output / "nested_manifest.json").write_text(
            json.dumps(nested_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    seed_zero = min(full_results, key=lambda item: item.seed_offset)
    shutil.copyfile(seed_zero.checkpoint, output / "akt_full.pt")
    job_manifest = [
        {
            "name": result.name,
            "model_fold": result.model_fold,
            "seed_offset": result.seed_offset,
            "actual_seed": result.actual_seed,
            "checkpoint": str(Path(result.checkpoint).relative_to(output)),
            "prediction_rows": len(result.predictions),
        }
        for result in sorted(
            [*outer_results, *nested_results, *full_results],
            key=lambda item: (item.name, item.model_fold, item.seed_offset),
        )
    ]
    (output / "job_manifest.json").write_text(
        json.dumps(job_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return predictions


class AKTPredictor:
    """Causal next-event inference over one or more official AKT checkpoints."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        vocab_path: str | Path,
        config: AKTTrainConfig,
    ) -> None:
        import torch

        self.config = config
        self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        source = Path(checkpoint)
        checkpoints = sorted(source.glob("seed_*.pt")) if source.is_dir() else [source]
        if not checkpoints:
            raise ValueError(f"no AKT checkpoints found at {source}")
        self.models = []
        for path in checkpoints:
            model = _new_model(self.vocab, config)
            state = torch.load(
                path, map_location=torch.device(config.device), weights_only=True
            )
            model.load_state_dict(state, strict=True)
            model.eval()
            self.models.append(model)

    @classmethod
    def untrained_demo(
        cls,
        *,
        target_ids: Sequence[str],
        question_tokens: Sequence[str],
        config: AKTTrainConfig,
    ) -> AKTPredictor:
        instance = cls.__new__(cls)
        instance.config = config
        tokens = sorted(set(target_ids) | set(question_tokens))
        concepts = {"<PAD_OR_UNK>": 0}
        concepts.update({token: index for index, token in enumerate(tokens, start=1)})
        instance.vocab = {"concepts": concepts, "questions": dict(concepts)}
        instance.models = [_new_model(instance.vocab, config)]
        instance.models[0].eval()
        return instance

    def predict(
        self,
        target_ids: Sequence[str],
        responses: Sequence[int],
        target_id: str,
        *,
        history_question_tokens: Sequence[str] | None = None,
        question_token: str | None = None,
    ) -> float:
        import torch

        if len(target_ids) != len(responses):
            raise ValueError("AKT history target and response lengths differ")
        if history_question_tokens is not None and len(history_question_tokens) != len(
            target_ids
        ):
            raise ValueError("AKT history question and target lengths differ")
        history_questions = history_question_tokens or list(target_ids)
        history_limit = max(1, self.config.maxlen - 1)
        target_ids = target_ids[-history_limit:]
        responses = responses[-history_limit:]
        history_questions = history_questions[-history_limit:]
        concepts = [
            self.vocab["concepts"].get(item, 0) for item in [*target_ids, target_id]
        ]
        questions = [self.vocab["questions"].get(item, 0) for item in history_questions]
        questions.append(self.vocab["questions"].get(question_token or target_id, 0))
        response_values = [int(item) for item in responses] + [0]
        device = torch.device(self.config.device)
        values = []
        with torch.no_grad():
            for model in self.models:
                prediction, _regularization = model(
                    torch.tensor([concepts], dtype=torch.long, device=device),
                    torch.tensor([response_values], dtype=torch.long, device=device),
                    torch.tensor([questions], dtype=torch.long, device=device),
                )
                values.append(float(prediction[0, -1].detach().cpu()))
        return float(np.mean(values))

    def export_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "vocab": self.vocab,
            "states": [
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                for model in self.models
            ],
        }

    @classmethod
    def initialize_online(
        cls, target_tokens: Sequence[str], config: AKTTrainConfig
    ) -> AKTPredictor:
        """Initialize the shared ensemble reproducibly before any training outcomes."""
        members = []
        for offset in config.seed_offsets:
            _seed_everything(config.seed + offset)
            member = cls.untrained_demo(
                target_ids=target_tokens, question_tokens=target_tokens, config=config
            )
            members.append(member)
        if not members:
            raise ValueError("online initialization requires at least one seed")
        instance = members[0]
        instance.models = [model for member in members for model in member.models]
        return instance

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> AKTPredictor:
        instance = cls.__new__(cls)
        instance.config = AKTTrainConfig(**state["config"])
        instance.vocab = state["vocab"]
        instance.models = []
        for weights in state["states"]:
            model = _new_model(instance.vocab, instance.config)
            model.load_state_dict(weights, strict=True)
            model.eval()
            instance.models.append(model)
        return instance

    @classmethod
    def fit_cumulative(
        cls,
        events: Sequence[BinaryEvent],
        config: AKTTrainConfig,
        *,
        target_tokens: Sequence[str] = (),
    ) -> AKTPredictor:
        """Refit full-cohort models with the frozen AKT loss/batches/seeds, without CV jobs.

        The offline benchmark functions above are unchanged. Online vocabulary includes
        the known clinical graph so a newly assessed node does not collapse to PAD/UNK.
        """
        import torch

        sequences = _prefix_sequences(events, maxlen=config.maxlen)
        if not sequences:
            raise ValueError("no strict-prior sequences for online AKT fitting")
        tokens = sorted({event.item_token for event in events} | set(target_tokens))
        concepts = {"<PAD_OR_UNK>": 0, **{token: i for i, token in enumerate(tokens, 1)}}
        instance = cls.__new__(cls)
        instance.config = config
        instance.vocab = {"concepts": concepts, "questions": dict(concepts)}
        instance.models = []
        _configure_torch_threads(config.threads_per_worker)
        for offset in config.seed_offsets:
            _seed_everything(config.seed + offset)
            model = _new_model(instance.vocab, config)
            _fit_model(model, sequences, instance.vocab, config)
            if not all(torch.isfinite(value).all().item() for value in model.parameters()):
                raise RuntimeError("online AKT fitting produced non-finite parameters")
            model.eval()
            instance.models.append(model)
        return instance
