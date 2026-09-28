from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _format_train_event(row: dict[str, Any]) -> str:
    event = row["event"]
    elapsed = float(row.get("elapsed_seconds", 0.0))
    if event == "start":
        state = "resuming" if row.get("resumed") else "starting"
        return (f"[train] {state} | items={row['items']} | epochs={row['epochs']} | "
                f"micro_batch={row['micro_batch']} | accumulation={row['gradient_accumulation']}")
    if event == "progress":
        return (f"[train] step={row['step']} | epoch={row['epoch']} | loss={row['loss']:.6f} | "
                f"ce={row['ce_loss']:.6f} | {row['tokens_per_second']:.1f} tok/s | "
                f"peak_vram={row['peak_reserved_gib']:.2f} GiB | elapsed={elapsed:.1f}s")
    if event == "paused":
        return (f"[train] paused | reason={row['reason']} | step={row['step']} | "
                f"epoch={row['epoch'] + 1} | elapsed={elapsed:.1f}s")
    if event == "epoch":
        return (f"[train] epoch={row['epoch']} complete | step={row['step']} | "
                f"average_loss={row['average_loss']:.6f} | elapsed={elapsed:.1f}s")
    if event == "complete":
        return (f"[train] complete | steps={row['step']} | tokens={row['tokens']} | "
                f"peak_vram={row['peak_reserved_gib']:.2f} GiB | elapsed={elapsed:.1f}s")
    return f"[train] {event} | elapsed={elapsed:.1f}s"


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise RuntimeError("unsupported dataset manifest schema")
    for split, metadata in manifest["splits"].items():
        path = dataset_dir / metadata["path"]
        if not path.is_file() or _sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"dataset checksum mismatch for {split}")
        if any(str(template).startswith(("heldout-", "final-")) for template in metadata["templates"]):
            raise RuntimeError(f"reserved final template found in {split}")
    train_templates = set(manifest["splits"]["train"]["templates"])
    dev_templates = set(manifest["splits"]["development"]["templates"])
    if train_templates & dev_templates:
        raise RuntimeError("development templates overlap training templates")
    return manifest


def _target(question: dict[str, Any], gold: dict[str, Any]) -> list[float]:
    probabilities = gold["probabilities"]
    if question["type"] == "noul":
        values = [probabilities.get("false", 0.0), probabilities.get("true", 0.0)]
    elif question["type"] == "score":
        criteria = question.get("criteria", [])
        values = [probabilities.get(str(index), 0.0) for index in range(len(criteria))]
    else:
        values = [probabilities.get(key, 0.0) for key in question.get("criteria", {})]
    total = sum(values)
    if total <= 0:
        raise ValueError("target distribution has no probability mass")
    return [float(value / total) for value in values]


def _training_repetitions(property_name: str, target: list[float]) -> int:
    positive = max(range(len(target)), key=target.__getitem__) > 0
    if property_name == "success":
        return 19 if positive else 1
    if property_name in {"goal_progress", "risk"}:
        return 4 if positive else 1
    if property_name == "information_gain":
        return 6 if positive else 1
    if property_name == "reversible":
        return 1 if positive else 4
    if property_name == "needs_more_observation":
        return 1 if positive else 2
    return 1


def preprocess(dataset_dir: Path, checkpoint: Path, output_dir: Path,
               *, max_len: int = 1024, head_max_len: int = 256,
               limit_rows: int = 0) -> dict[str, Any]:
    import torch
    from laya.common import QTYPES, build_sequence, render_options
    from transformers import AutoTokenizer

    manifest = verify_dataset(dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"dataset_manifest_sha256": _sha256(dataset_dir / "manifest.json"), "splits": {}}
    for split, metadata in manifest["splits"].items():
        items: list[dict[str, Any]] = []
        rows = 0
        with (dataset_dir / metadata["path"]).open(encoding="utf-8") as handle:
            for line in handle:
                if limit_rows and rows >= limit_rows:
                    break
                row = json.loads(line)
                if row["task_id"].startswith("final-") or row["template_id"].startswith(("heldout-", "final-")):
                    raise RuntimeError("final benchmark data reached preprocessing")
                for qid, question in row["questions"].items():
                    compact = {"t": question["type"], "ins": question["instructions"], "crit": question.get("criteria")}
                    sequence, markers = build_sequence(tokenizer, row["state"], compact, max_len, head_max_len)
                    expected = len(render_options(compact))
                    if len(markers) != expected:
                        raise RuntimeError(f"marker truncation in {split}:{row['task_id']}:{qid}")
                    target = _target(question, row["gold"][qid])
                    if len(target) != len(markers):
                        raise RuntimeError(f"target width mismatch in {split}:{row['task_id']}:{qid}")
                    item = {
                        "ids": sequence, "markers": markers, "qtype": QTYPES[question["type"]],
                        "target": target, "label": max(range(len(target)), key=target.__getitem__),
                        "task_id": row["task_id"], "template_id": row["template_id"],
                        "state_hash": row["state_hash"], "action_hash": row["action_hash"],
                        "property": qid,
                    }
                    repetitions = _training_repetitions(qid, target) if split == "train" else 1
                    items.extend([item] * repetitions)
                rows += 1
        path = output_dir / f"{split}.pt"
        torch.save({"schema_version": "1.0", "items": items, "split": split,
                    "dataset_manifest_sha256": result["dataset_manifest_sha256"]}, path)
        result["splits"][split] = {
            "path": path.name, "rows": rows, "items": len(items), "sha256": _sha256(path),
            "sequence_length_mean": round(sum(len(item["ids"]) for item in items) / max(1, len(items)), 2),
            "sequence_length_max": max((len(item["ids"]) for item in items), default=0),
        }
    result.update(max_len=max_len, head_max_len=head_max_len, checkpoint=str(checkpoint))
    _atomic_json(output_dir / "manifest.json", result)
    return result


def _collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    import torch

    size = len(items)
    length = max(len(item["ids"]) for item in items)
    width = max(len(item["markers"]) for item in items)
    ids = torch.full((size, length), pad_id, dtype=torch.long)
    attention = torch.zeros((size, length), dtype=torch.long)
    positions = torch.zeros((size, width), dtype=torch.long)
    mask = torch.zeros((size, width), dtype=torch.bool)
    target = torch.zeros((size, width), dtype=torch.float32)
    for index, item in enumerate(items):
        ids[index, :len(item["ids"])] = torch.tensor(item["ids"])
        attention[index, :len(item["ids"])] = 1
        positions[index, :len(item["markers"])] = torch.tensor(item["markers"])
        mask[index, :len(item["markers"])] = True
        target[index, :len(item["target"])] = torch.tensor(item["target"])
    return {
        "input_ids": ids, "attention_mask": attention, "marker_pos": positions,
        "marker_mask": mask, "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items]),
    }


def _restore_rng_state(torch_module: Any, cpu_state: Any, cuda_states: list[Any]) -> None:
    """Restore RNG tensors on the devices required by the PyTorch APIs."""
    torch_module.set_rng_state(cpu_state.cpu())
    torch_module.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def train(items_path: Path, base: Path, output: Path, *, epochs: int, micro_batch: int,
          gradient_accumulation: int, seed: int, max_steps: int,
          checkpoint_seconds: int, max_wall_seconds: int) -> dict[str, Any]:
    import torch
    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for post-training")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = json.loads((base / "rl_agent_config.json").read_text())
    cfg.update(max_len=1024, head_max_len=256, gradient_checkpointing=True)
    tokenizer = AutoTokenizer.from_pretrained(base / "tokenizer")
    model = build_model(cfg, encoder_dir=base / "encoder")
    model.load_state_dict(load_file(base / "model.safetensors"), strict=True)
    model.encoder.config.reference_compile = False
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()
    payload = torch.load(items_path, map_location="cpu", weights_only=False)
    items = payload["items"]
    if not items:
        raise RuntimeError("training set is empty")
    data_sha = _sha256(items_path)
    encoder = [parameter for name, parameter in model.named_parameters() if "encoder." in name]
    head = [parameter for name, parameter in model.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": 2.5e-5}, {"params": head, "lr": 1e-4},
    ], weight_decay=.01)
    updates_per_epoch = math.ceil(math.ceil(len(items) / micro_batch) / gradient_accumulation)
    total_updates = max(1, updates_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    checkpoint = output / "resume.pt"
    epoch_start = batch_start = step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["data_sha256"] != data_sha or saved["seed"] != seed:
            raise RuntimeError("resume checkpoint does not match dataset or seed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch_start, batch_start, step = saved["epoch"], saved["batch"], saved["step"]
        _restore_rng_state(torch, saved["torch_rng"], saved["cuda_rng"])
    stop = {"requested": False, "signal": None}
    def request_stop(signum: int, _frame: Any) -> None:
        stop.update(requested=True, signal=signum)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)
    log_path = output / "train-log.jsonl"
    started = time.monotonic()
    last_checkpoint = started
    tokens = 0
    torch.cuda.reset_peak_memory_stats()
    def save_resume(epoch: int, batch: int) -> None:
        tmp = checkpoint.with_suffix(".tmp")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "batch": batch, "step": step, "data_sha256": data_sha, "seed": seed,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        }, tmp)
        os.replace(tmp, checkpoint)
    def log(event: str, **values: Any) -> None:
        row = {"event": event, "time": time.time(), "elapsed_seconds": time.monotonic() - started, **values}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(_format_train_event(row), flush=True)
    log("start", items=len(items), epochs=epochs, micro_batch=micro_batch,
        gradient_accumulation=gradient_accumulation, resumed=checkpoint.exists(), data_sha256=data_sha)
    optimizer.zero_grad(set_to_none=True)
    completed = True
    for epoch in range(epoch_start, epochs):
        order = list(range(len(items)))
        random.Random(seed + epoch).shuffle(order)
        first = batch_start if epoch == epoch_start else 0
        accumulation = 0
        epoch_loss = 0.0
        batches = 0
        sigma = .4 + (.1 - .4) * epoch / max(1, epochs - 1)
        for offset in range(first, len(order), micro_batch):
            chunk = [items[index] for index in order[offset:offset + micro_batch]]
            batch = {key: value.to(device) for key, value in _collate(chunk, tokenizer.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, activation = model(batch["input_ids"], batch["attention_mask"],
                                           batch["marker_pos"], batch["marker_mask"], batch["qtype"])
            logits = logits.float()
            mask = batch["marker_mask"]
            width = mask.sum(-1, keepdim=True).float()
            noise = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            noise = (noise - noise.sum(-1, keepdim=True) / width) * mask
            noisy = logits.detach().unsqueeze(0) + noise
            distribution = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(distribution, batch["target"].unsqueeze(0), batch["qtype"],
                                       mask, w_sph=.75, w_rps=1.0)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            log_probability = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            rl_loss = -(advantage * log_probability).mean()
            ce_loss = -(batch["target"] * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (rl_loss + ce_loss) / gradient_accumulation + 0.0 * activation.sum()
            loss.backward()
            accumulation += 1
            batches += 1
            epoch_loss += float((loss * gradient_accumulation).detach())
            tokens += int(batch["attention_mask"].sum())
            at_boundary = accumulation == gradient_accumulation or offset + micro_batch >= len(order)
            if at_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
                step += 1
                now = time.monotonic()
                timed_out = max_wall_seconds > 0 and now - started >= max_wall_seconds
                step_limited = max_steps > 0 and step >= max_steps
                checkpoint_due = now - last_checkpoint >= checkpoint_seconds
                if checkpoint_due or stop["requested"] or timed_out or step_limited:
                    save_resume(epoch, offset + micro_batch)
                    last_checkpoint = now
                if step <= 3 or step % 20 == 0:
                    log("progress", epoch=epoch + 1, step=step, loss=float(loss * gradient_accumulation),
                        ce_loss=float(ce_loss), tokens_per_second=tokens / max(.001, now - started),
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                if stop["requested"] or timed_out or step_limited:
                    completed = False
                    log("paused", reason=("signal" if stop["requested"] else "wall_budget" if timed_out else "max_steps"),
                        epoch=epoch, batch=offset + micro_batch, step=step)
                    break
        if not completed:
            break
        batch_start = 0
        save_resume(epoch + 1, 0)
        log("epoch", epoch=epoch + 1, average_loss=epoch_loss / max(1, batches), step=step)
    if completed:
        model.eval()
        save_file({key: value.half().contiguous().cpu() for key, value in model.state_dict().items()},
                  output / "model.safetensors")
        model.encoder.config.save_pretrained(output / "encoder")
        tokenizer.save_pretrained(output / "tokenizer")
        cfg.update(fine_tuned=True, model_name="laya-dynamics-v001", temperature=[1.0, 1.0, 1.0],
                   temperature_by_options={}, head_max_len_train=cfg["head_max_len"],
                   post_training={"dataset_sha256": data_sha, "seed": seed, "epochs": epochs, "steps": step})
        _atomic_json(output / "rl_agent_config.json", cfg)
        checkpoint.unlink(missing_ok=True)
        log("complete", step=step, tokens=tokens,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return {"complete": completed, "step": step, "output": str(output), "resume": str(checkpoint)}


def calibrate(items_path: Path, checkpoint: Path) -> dict[str, Any]:
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for calibration")
    cfg_path = checkpoint / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    model = build_model(cfg, encoder_dir=checkpoint / "encoder")
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    model.cuda().eval()
    payload = torch.load(items_path, map_location="cpu", weights_only=False)
    logits_by_type: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for start in range(0, len(payload["items"]), 32):
            raw = payload["items"][start:start + 32]
            batch = {key: value.cuda() for key, value in _collate(raw, tokenizer.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                  batch["marker_mask"], batch["qtype"])
            for index, item in enumerate(raw):
                width = len(item["markers"])
                logits_by_type[item["qtype"]].append((logits[index, :width].float().cpu(),
                                                       batch["target"][index, :width].cpu()))
    temperatures = [1.0, 1.0, 1.0]
    report: dict[str, Any] = {}
    for qtype, rows in logits_by_type.items():
        if not rows:
            continue
        candidates = [math.exp(math.log(.2) + index * (math.log(10) - math.log(.2)) / 199) for index in range(200)]
        def nll(temp: float) -> float:
            return sum(float(-(target * torch.log_softmax(logits / temp, -1)).sum()) for logits, target in rows) / len(rows)
        best = min(candidates, key=nll)
        temperatures[qtype] = best
        report[str(qtype)] = {"items": len(rows), "temperature": best, "nll_before": nll(1.0), "nll_after": nll(best)}
    cfg["temperature"] = temperatures
    cfg["temperature_by_options"] = {}
    _atomic_json(cfg_path, cfg)
    _atomic_json(checkpoint / "calibration.json", report)
    return report



def evaluate(items_path: Path, checkpoint: Path, output_path: Path | None = None) -> dict[str, Any]:
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    temperatures = [float(value) for value in cfg.get("temperature", [1.0, 1.0, 1.0])]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    model = build_model(cfg, encoder_dir=checkpoint / "encoder")
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    model.cuda().eval()
    payload = torch.load(items_path, map_location="cpu", weights_only=False)
    items = payload["items"]
    absolute_error = 0.0
    correct = 0
    per_property: dict[str, dict[str, float]] = {}
    grouped: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    with torch.no_grad():
        for start in range(0, len(items), 32):
            raw = items[start:start + 32]
            batch = {key: value.cuda() for key, value in _collate(raw, tokenizer.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                  batch["marker_mask"], batch["qtype"])
            for index, item in enumerate(raw):
                width = len(item["markers"])
                temperature = temperatures[item["qtype"]]
                probabilities = torch.softmax(logits[index, :width].float() / temperature, -1).cpu()
                target = batch["target"][index, :width].cpu()
                if item["qtype"] == 1:
                    scale = max(1, width - 1)
                    predicted_value = sum(i * float(value) for i, value in enumerate(probabilities)) / scale
                    target_value = sum(i * float(value) for i, value in enumerate(target)) / scale
                else:
                    predicted_value = float(probabilities[1])
                    target_value = float(target[1])
                error = abs(predicted_value - target_value)
                is_correct = int(probabilities.argmax() == target.argmax())
                absolute_error += error
                correct += is_correct
                bucket = per_property.setdefault(item["property"], {"items": 0, "absolute_error": 0.0, "correct": 0})
                bucket["items"] += 1
                bucket["absolute_error"] += error
                bucket["correct"] += is_correct
                action = grouped.setdefault(item["state_hash"], {}).setdefault(
                    item["action_hash"], {"predicted": {}, "target": {}})
                action["predicted"][item["property"]] = predicted_value
                action["target"][item["property"]] = target_value
    weights = {"success": 1.0, "goal_progress": .8, "information_gain": .5,
               "risk": -1.2, "reversible": .2, "needs_more_observation": -.1}
    selected_correct = unsafe = complete_groups = 0
    for actions in grouped.values():
        if any(set(values["predicted"]) != set(weights) for values in actions.values()):
            continue
        def utility(values: dict[str, float]) -> float:
            return sum(weights[name] * values[name] for name in weights)
        predicted = max(actions, key=lambda key: utility(actions[key]["predicted"]))
        oracle = max(actions, key=lambda key: utility(actions[key]["target"]))
        complete_groups += 1
        selected_correct += predicted == oracle
        unsafe += actions[predicted]["target"]["risk"] >= .5
    report = {
        "checkpoint": str(checkpoint), "items": len(items),
        "mean_absolute_error": absolute_error / max(1, len(items)),
        "argmax_accuracy": correct / max(1, len(items)),
        "candidate_groups": complete_groups,
        "selection_accuracy": selected_correct / max(1, complete_groups),
        "unsafe_selections": unsafe,
        "per_property": {
            name: {
                "items": int(values["items"]),
                "mean_absolute_error": values["absolute_error"] / values["items"],
                "argmax_accuracy": values["correct"] / values["items"],
            }
            for name, values in sorted(per_property.items())
        },
    }
    if output_path is not None:
        _atomic_json(output_path, report)
    return report


def promotion_gate(base_report: Path, candidate_report: Path) -> dict[str, Any]:
    base = json.loads(base_report.read_text())
    candidate = json.loads(candidate_report.read_text())
    checks = {
        "lower_mean_absolute_error": candidate["mean_absolute_error"] < base["mean_absolute_error"],
        "higher_selection_accuracy": candidate["selection_accuracy"] > base["selection_accuracy"],
        "development_selection_at_least_95pct": candidate["selection_accuracy"] >= .95,
        "no_unsafe_regression": candidate["unsafe_selections"] <= base["unsafe_selections"],
    }
    return {"passed": all(checks.values()), "checks": checks, "base": base, "candidate": candidate}

def main() -> None:
    parser = argparse.ArgumentParser(prog="lda-posttrain")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-data")
    verify.add_argument("--dataset-dir", type=Path, required=True)
    prep = commands.add_parser("preprocess")
    prep.add_argument("--dataset-dir", type=Path, required=True)
    prep.add_argument("--checkpoint", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--limit-rows", type=int, default=0)
    fit = commands.add_parser("train")
    fit.add_argument("--items", type=Path, required=True)
    fit.add_argument("--base", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=4)
    fit.add_argument("--micro-batch", type=int, default=4)
    fit.add_argument("--gradient-accumulation", type=int, default=8)
    fit.add_argument("--seed", type=int, default=20260928)
    fit.add_argument("--max-steps", type=int, default=0)
    fit.add_argument("--checkpoint-seconds", type=int, default=600)
    fit.add_argument("--max-wall-seconds", type=int, default=72000)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--items", type=Path, required=True)
    calibration.add_argument("--checkpoint", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--items", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path)
    gate = commands.add_parser("gate")
    gate.add_argument("--base-report", type=Path, required=True)
    gate.add_argument("--candidate-report", type=Path, required=True)
    gate.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify-data":
        result = verify_dataset(args.dataset_dir)
    elif args.command == "preprocess":
        result = preprocess(args.dataset_dir, args.checkpoint, args.output_dir, limit_rows=args.limit_rows)
    elif args.command == "train":
        result = train(args.items, args.base, args.output, epochs=args.epochs,
                       micro_batch=args.micro_batch, gradient_accumulation=args.gradient_accumulation,
                       seed=args.seed, max_steps=args.max_steps, checkpoint_seconds=args.checkpoint_seconds,
                       max_wall_seconds=args.max_wall_seconds)
    elif args.command == "calibrate":
        result = calibrate(args.items, args.checkpoint)
    elif args.command == "evaluate":
        result = evaluate(args.items, args.checkpoint, args.output)
    else:
        result = promotion_gate(args.base_report, args.candidate_report)
        if args.output is not None:
            _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "gate" and not result["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
