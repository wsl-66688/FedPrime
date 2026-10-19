"""Audit, launch and summarize the released Food101 Table 1 implementation.

Python 3.8 compatible. This does not replace the released data selection protocol.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark/RAW_DATA/FOOD101"
METHODS = ("Fed-Prime", "FedAvg_P", "FedMSplit_P", "Fed-Intra", "Fed-Inter")
SCENARIOS = ("both", "text", "image")
COLUMNS = ("same_as_train", "miss_both", "full_modal", "text_only", "image_only")
PROTOCOL = "released_data_subset_paper_training_parameters"


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def task_name(scenario):
    ratio = "0.5" if scenario == "both" else "0.0"
    return ("food101_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_"
            "missing_type_{0}_{0}_both_ratio_{1}".format(scenario, ratio))


def inspect_inputs(runtime=False, gpu=0):
    import pyarrow as pa

    report = {"protocol": PROTOCOL, "errors": [], "warnings": [], "arrows": {}, "tasks": {}}
    report["warnings"].append(
        "Released code selects Arrow prefixes; exact author class IDs, 80:20 split, "
        "repeat seeds and model-selection rule are not established."
    )
    source = (ROOT / "benchmark/food101/base_dataset.py").read_text(encoding="utf-8")
    divisors = re.findall(r"end_index\s*=\s*total_rows\s*//\s*([0-9.]+)", source)
    if len(divisors) != 2:
        raise RuntimeError("Data selection code changed; review the audit before using it")
    for split, divisor in zip(("train", "test"), divisors):
        path = DATA / "generate_arrows" / ("food101_" + split + ".arrow")
        try:
            with pa.memory_map(str(path), "r") as stream:
                reader = pa.ipc.RecordBatchFileReader(stream)
                sizes = [reader.get_batch(i).num_rows for i in range(reader.num_record_batches)]
                total = sum(sizes)
                effective = int(total // float(divisor))
                labels = []
                for i in range(reader.num_record_batches):
                    if len(labels) >= effective:
                        break
                    batch = reader.get_batch(i)
                    labels.extend(batch.column(batch.schema.get_field_index("label"))
                                  .slice(0, effective - len(labels)).to_pylist())
                report["arrows"][split] = {
                    "bytes": path.stat().st_size, "rows": total,
                    "effective_rows": effective, "record_batches": reader.num_record_batches,
                    "labels": dict(Counter(labels)),
                    "validation": "IPC structure and selected labels; image payloads not reread",
                }
                if effective == 0:
                    report["errors"].append(split + ": empty effective dataset")
        except Exception as exc:
            report["errors"].append(str(path.relative_to(ROOT)) + ": " + str(exc))
    if len(report["arrows"]) == 2:
        label_sets = [set(report["arrows"][s]["labels"]) for s in ("train", "test")]
        if len(label_sets[0]) != 8 or label_sets[0] != label_sets[1]:
            report["errors"].append("Expected the same 8 labels in the released train/test subset")

    resource_root = ROOT / "benchmark/pretrained_model_weight"
    expected = {"vilt_200k_mlm_itm.ckpt": 1628097027,
                "bert-base-uncased/vocab.txt": 231508,
                "bert-base-uncased/config.json": 433}
    report["resources"] = {}
    for relative, size in expected.items():
        path = resource_root / relative
        actual = path.stat().st_size if path.exists() else None
        report["resources"][relative] = {"bytes": actual, "expected_bytes": size}
        if actual != size:
            report["errors"].append("Missing or different-size resource: " + relative)

    for scenario in SCENARIOS:
        path = ROOT / "fedtask" / task_name(scenario) / "data.json"
        if not path.exists():
            report["tasks"][scenario] = {"exists": False}
            report["warnings"].append("Task not generated: " + scenario)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            clients = data["client_names"]
            ids = [i for client in clients for i in data[client]["dtrain"]]
            ntrain = report["arrows"]["train"]["effective_rows"]
            ntest = report["arrows"]["test"]["effective_rows"]
            if len(clients) != 20 or sorted(ids) != list(range(ntrain)):
                raise ValueError("Client count, overlap or train coverage mismatch")
            if sorted(data["dtest"]) != list(range(ntest)):
                raise ValueError("Test coverage mismatch")
            src = data["datasrc"]
            if (src["missing_type"] != {"train": scenario, "test": scenario}
                    or src["missing_ratio"] != {"train": 0.7, "test": 0.7}
                    or src["both_ratio"] != (0.5 if scenario == "both" else 0.0)
                    or src.get("simulate_missing", False)):
                raise ValueError("Task missing-modality settings mismatch")
            report["tasks"][scenario] = {
                "exists": True, "sha256": digest(path), "clients": len(clients),
                "train": len(ids), "test": len(data["dtest"]),
                "client_sizes": [len(data[c]["dtrain"]) for c in clients],
            }
        except Exception as exc:
            report["errors"].append("Task " + scenario + ": " + str(exc))

    if runtime:
        try:
            for module in ("torch", "transformers", "timm", "wandb", "ujson", "scipy", "sklearn"):
                importlib.import_module(module)
            import torch
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(str(resource_root / "bert-base-uncased"),
                                                      local_files_only=True)
            if tokenizer.vocab_size != 30522:
                raise ValueError("Unexpected tokenizer vocabulary size")
            if gpu < 0 or not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
                raise RuntimeError("Selected CUDA GPU is unavailable: " + str(gpu))
            report["runtime"] = {"python": sys.version, "executable": sys.executable,
                                 "torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(gpu),
                                 "gpu_total_gib": torch.cuda.get_device_properties(gpu).total_memory / 1024 ** 3}
            for path in (DATA / "missing_tables_8_classes").glob("*.pt"):
                split = "train" if path.name.startswith("food101_train_") else "test"
                mask = torch.load(str(path), map_location="cpu")
                expected_rows = report["arrows"][split]["effective_rows"]
                if (mask.ndim != 1 or len(mask) != expected_rows
                        or not ((mask == 0) | (mask == 1) | (mask == 2)).all().item()):
                    raise ValueError("Invalid missing-modality cache: " + str(path))
        except Exception as exc:
            report["errors"].append("Runtime: " + str(exc))
    return report


def training_command(method, scenario, seed, rounds, gpu):
    top_k = 10 if method in ("Fed-Intra", "Fed-Inter") else 5
    return [sys.executable, "main.py", "--task", task_name(scenario),
            "--model", method, "--algorithm", "multimodal.food101." + method,
            "--sample", "full", "--aggregate", "other", "--num_rounds", str(rounds),
            "--proportion", "1.0", "--optimizer", "SGD", "--learning_rate", "0.05",
            "--num_epochs", "1", "--batch_size", "512", "--test_batch_size", "512",
            "--lr_scheduler", "0", "--learning_rate_decay", "1.0", "--seed", str(seed),
            "--pool_size", "20", "--top_k", str(top_k), "--num_outer_loops", "5",
            "--max_text_len", "40", "--gpu", str(gpu), "--logger", "table1_logger",
            "--log_file"]


def capture(command, filename):
    try:
        result = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        Path(filename).write_text(result.stdout, encoding="utf-8")
    except OSError as exc:
        Path(filename).write_text(str(exc), encoding="utf-8")


def snapshot_data(target):
    paths = list((DATA / "missing_tables_8_classes").glob("*.pt"))
    paths += list((DATA / "missing_tables_other_tests").glob("*.pt"))
    paths += list((ROOT / "fedtask").glob("food101_cnum20_dist0_skew0_seed0_*/data.json"))
    write_json(target, {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)})


def used_data_fingerprints(scenario):
    task = ROOT / "fedtask" / task_name(scenario) / "data.json"
    src = json.loads(task.read_text(encoding="utf-8"))["datasrc"]
    paths = [task]
    for split in ("train", "test"):
        ratio = str(src["missing_ratio"][split]).replace(".", "")
        both = str(src["both_ratio"]).replace(".", "")
        name = "food101_{}_missing_{}_{}_{}.pt".format(split, scenario, ratio, both)
        paths.append(ROOT / src["missing_table_root"] / name)
    other = DATA / "missing_tables_other_tests"
    paths.extend(other / name for name in (
        "food101_test_missing_both_0_0.pt", "food101_test_missing_text_1_0.pt",
        "food101_test_missing_image_1_0.pt"))
    for kind in SCENARIOS:
        if kind != scenario:
            paths.append(other / ("food101_test_missing_{}_07_{}.pt".format(
                kind, "05" if kind == "both" else "0")))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


class ReleasedRunContext:
    protocol = PROTOCOL
    results_root = ROOT / "experiments/table1"
    source_files = []

    def inspect_inputs(self, runtime=False, gpu=0):
        return inspect_inputs(runtime, gpu)

    def command(self, method, scenario, seed, rounds, gpu):
        return training_command(method, scenario, seed, rounds, gpu)

    def prepare(self, scenarios, audit):
        for path in (DATA / "missing_tables_8_classes", DATA / "missing_tables_other_tests", ROOT / "fedtask"):
            path.mkdir(parents=True, exist_ok=True)
        for scenario in scenarios:
            if not audit["tasks"][scenario]["exists"]:
                script = ROOT / "scripts/food101/main_exp_20clients" / (
                    {"both": "miss_both", "text": "miss_text", "image": "miss_img"}[scenario]) / "20cl.sh"
                tokens = shlex.split(script.read_text(encoding="utf-8").replace("\\\n", " "))
                subprocess.run([sys.executable] + tokens[1:], cwd=str(ROOT), check=True)

    def snapshot(self, target):
        snapshot_data(target)

    def fingerprints(self, scenario):
        return used_data_fingerprints(scenario)


def run_matrix(args, context=None):
    context = context or ReleasedRunContext()
    methods = METHODS if args.stage == "all" else ("Fed-Prime",)
    scenarios = ("both",) if args.stage == "smoke" else SCENARIOS
    rounds = 1 if args.stage == "smoke" else 250
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seed list contains duplicates")
    commands = [(m, s, seed, context.command(m, s, seed, rounds, args.gpu))
                for s in scenarios for m in methods for seed in args.seeds]
    print("Protocol:", context.protocol, "| Runs:", len(commands), "| Rounds per run:", rounds)
    print("Model selection: final round. Author selection rule remains unconfirmed.")
    print("Outer loops=5 and constant LR follow released scripts; not all are explicit paper values.")
    if args.dry_run:
        for _, _, _, command in commands:
            print(shlex.join(command))
        return

    audit = context.inspect_inputs(runtime=True, gpu=args.gpu)
    if audit["errors"]:
        raise RuntimeError(json.dumps(audit["errors"], ensure_ascii=False))
    context.prepare(scenarios, audit)
    runtime_info = audit.get("runtime", {})
    audit = context.inspect_inputs()
    audit["runtime"] = runtime_info
    if audit["errors"]:
        raise RuntimeError(json.dumps(audit["errors"], ensure_ascii=False))

    batch = context.results_root / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + args.stage + "_" + uuid.uuid4().hex[:8])
    batch.mkdir(parents=True)
    print("Results:", batch, flush=True)
    write_json(batch / "data-audit.json", audit)
    capture([sys.executable, "-m", "pip", "freeze"], batch / "pip-freeze.txt")
    capture(["nvidia-smi"], batch / "gpu.txt")
    capture(["git", "rev-parse", "HEAD"], batch / "git-head.txt")
    capture(["git", "diff"], batch / "code-changes.patch")
    sources = [ROOT / "main.py", ROOT / "generate_fedtask.py", Path(__file__),
               ROOT / "benchmark/toolkits.py", ROOT / "benchmark/__init__.py"] + context.source_files
    for folder in ("algorithm", "benchmark/food101", "utils"):
        sources.extend((ROOT / folder).rglob("*.py"))
    write_json(batch / "source-sha256.json",
               {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))})
    with zipfile.ZipFile(str(batch / "source.zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(sources)):
            archive.write(str(path), str(path.relative_to(ROOT)))
    context.snapshot(batch / "data-files-before.json")

    for method, scenario, seed, command in commands:
        run_dir = batch / (scenario + "_" + method + "_seed" + str(seed))
        run_dir.mkdir()
        write_json(run_dir / "command.json", command)
        (run_dir / "command.sh").write_text(shlex.join(command) + "\n", encoding="utf-8")
        status = {"state": "running", "stage": args.stage, "protocol": context.protocol,
                  "method": method, "scenario": scenario, "seed": seed, "rounds": rounds,
                  "selection": "final", "task_sha256": audit["tasks"][scenario]["sha256"]}
        write_json(run_dir / "status.json", status)
        env = dict(os.environ, FEDPRIME_TABLE1_RUN_DIR=str(run_dir),
                   PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace")
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            if code:
                raise RuntimeError("Training exited with code " + str(code))
            final_scores(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")), scenario,
                         expected_evaluations=rounds)
            status["data_sha256"] = context.fingerprints(scenario)
            status["state"] = "complete"
        except BaseException as exc:
            status["state"] = "failed"
            status["error"] = str(exc)
            raise
        finally:
            write_json(run_dir / "status.json", status)
            context.snapshot(batch / "data-files-after.json")
    print("Completed:", batch)
    entrypoint = getattr(context, "entrypoint", "notebook/food101_table1.py")
    print("Summarize: python " + entrypoint + " summary --results " + shlex.quote(str(batch)))


def final_scores(record, scenario, expected_evaluations=None):
    keys = ["test_acc", "test_acc" if scenario == "both" else "test_miss_both_acc",
            "test_full_modal_acc", "test_text_only_acc", "test_image_only_acc"]
    scores = {}
    length = None
    for column, key in zip(COLUMNS, keys):
        values = record[key]
        if not isinstance(values, list) or not values:
            raise ValueError("Missing evaluation series: " + key)
        if length is None:
            length = len(values)
        if len(values) != length or (expected_evaluations is not None and len(values) != expected_evaluations):
            raise ValueError("Incomplete or inconsistent evaluation series: " + key)
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Invalid accuracy values: " + key)
        scores[column] = values[-1] * 100
    return scores


def summarize(results):
    rows, groups, seen = [], defaultdict(list), set()
    scenario_data = defaultdict(set)
    for path in sorted(Path(results).glob("*/status.json")):
        status = json.loads(path.read_text(encoding="utf-8"))
        if status["state"] != "complete" or status["rounds"] != 250:
            continue
        identity = (status["scenario"], status["method"], status["seed"])
        if identity in seen:
            raise ValueError("Duplicate seed run; select one experiment batch")
        seen.add(identity)
        record = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        scores = final_scores(record, status["scenario"], expected_evaluations=250)
        row = {"scenario": status["scenario"], "method": status["method"], "seed": status["seed"], **scores}
        rows.append(row)
        fingerprint = json.dumps(status["data_sha256"], sort_keys=True)
        groups[identity[:2]].append((row, fingerprint))
        scenario_data[status["scenario"]].add(fingerprint)
    if not rows:
        raise ValueError("No complete 250-round runs; smoke/failed runs are excluded")
    if any(len(hashes) != 1 for hashes in scenario_data.values()):
        raise ValueError("Partitions or missing masks differ across methods/seeds in a scenario")
    output = Path(results)
    with (output / "table1-runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario", "method", "seed"] + list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "table1-summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", "method", "n", "seeds", "test_condition", "mean_percent", "sample_sd_percent"])
        for (scenario, method), group in sorted(groups.items()):
            for column in COLUMNS:
                values = [row[column] for row, _ in group]
                writer.writerow([scenario, method, len(values), " ".join(str(row["seed"]) for row, _ in group),
                                 column, statistics.mean(values), statistics.stdev(values) if len(values) > 1 else ""])
    print("Saved:", output / "table1-runs.csv", "and", output / "table1-summary.csv")
    print("Final-round accuracy; sample SD, not a confirmed reconstruction of the paper's ± rule.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--runtime", action="store_true")
    check.add_argument("--gpu", type=int, default=0)
    run = commands.add_parser("run")
    run.add_argument("--stage", choices=("smoke", "fedprime", "all"), default="smoke")
    run.add_argument("--seeds", nargs="+", type=int, default=[1234])
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("--dry-run", action="store_true")
    summary = commands.add_parser("summary")
    summary.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        report = inspect_inputs(args.runtime, args.gpu)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["errors"] else 0
    if args.command == "run":
        run_matrix(args)
    else:
        summarize(args.results)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, ImportError, FileNotFoundError, KeyError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
