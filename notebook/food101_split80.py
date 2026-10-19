"""Prepare and run the isolated, custom Food101 80:20 baseline (Python 3.8)."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import food101_protocol as protocol
import food101_table1 as table1


class Split80Context:
    protocol = protocol.PROTOCOL
    entrypoint = "notebook/food101_split80.py"

    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.data = self.root / "benchmark/RAW_DATA/FOOD101"
        self.relative = Path("benchmark/RAW_DATA/FOOD101/protocols") / self.protocol
        self.folder = self.root / self.relative
        self.manifest_file = self.root / "data_protocols" / (self.protocol + ".json")
        self.results_root = self.root / "experiments/table1_split80" / self.protocol
        self.source_files = [Path(__file__).resolve(), self.manifest_file]

    def command(self, method, scenario, seed, rounds, gpu):
        command = table1.training_command(method, scenario, seed, rounds, gpu)
        command[command.index("--task") + 1] = protocol.task_name(scenario)
        return command

    def prepare(self, scenarios, audit=None):
        manifest = protocol.load_prepared_manifest(self.folder / "generate_arrows")
        for scenario in scenarios:
            target = self.root / "fedtask" / protocol.task_name(scenario)
            expected = protocol.make_task(manifest, scenario, self.relative)
            if (target / "data.json").exists():
                if protocol.read_json(target / "data.json") != expected:
                    raise ValueError("Existing versioned task differs; refusing to overwrite: " + str(target))
            else:
                target.mkdir(parents=True, exist_ok=True)
                for child in ("record", "log"):
                    (target / child).mkdir(exist_ok=True)
                protocol.write_json(target / "data.json", expected)

    def inspect_inputs(self, runtime=False, gpu=0):
        import pyarrow as pa
        report = {"protocol": self.protocol, "errors": [], "warnings": [
            "Custom stratified split of the released eight classes; not the confirmed author split."
        ], "arrows": {}, "tasks": {}}
        try:
            manifest = protocol.load_prepared_manifest(self.folder / "generate_arrows")
            if protocol.read_json(self.manifest_file) != manifest:
                raise ValueError("Portable and prepared manifests differ")
            report["manifest_sha256"] = protocol.canonical_hash(manifest)
            expected_files = protocol.read_json(self.folder / "prepared-files.json")
            tables = {}
            for split in ("train", "test"):
                path = self.folder / "generate_arrows" / ("food101_" + split + ".arrow")
                if table1.digest(path) != expected_files[path.name]:
                    raise ValueError("Prepared Arrow checksum changed: " + path.name)
                with pa.memory_map(str(path), "r") as stream:
                    table = pa.ipc.RecordBatchFileReader(stream).read_all()
                    protocol.validate_table(table, path.parent, split)
                    tables[split] = table
                    report["arrows"][split] = {"rows": table.num_rows, "effective_rows": table.num_rows,
                                                "bytes": path.stat().st_size,
                                                "class_counts": manifest["class_counts"][split]}
            for folder in ("missing_tables", "other_tests"):
                for path in (self.folder / folder).glob("*.pt.json"):
                    item = protocol.read_json(path)
                    if protocol.fixed_mask(tables[item["split"]], str(path)[:-5]) is None:
                        raise ValueError("Fixed mask marker missing")
            # Require every mask used by every scenario, including all five test conditions.
            for scenario in protocol.SCENARIOS:
                for path in self._mask_paths(scenario):
                    if not path.is_file():
                        raise ValueError("Required fixed mask missing: " + str(path))
                task_path = self.root / "fedtask" / protocol.task_name(scenario) / "data.json"
                if not task_path.exists():
                    report["tasks"][scenario] = {"exists": False}
                    report["warnings"].append("Task not prepared: " + scenario)
                else:
                    task = protocol.read_json(task_path)
                    if task != protocol.make_task(manifest, scenario, self.relative):
                        raise ValueError("Task indices/configuration differ: " + scenario)
                    report["tasks"][scenario] = {
                        "exists": True, "sha256": table1.digest(task_path), "clients": 20,
                        "train": len(manifest["train"]), "test": len(manifest["test"]),
                        "client_sizes": [len(task[c]["dtrain"]) for c in task["client_names"]]}
        except Exception as exc:
            report["errors"].append("Protocol data: " + str(exc))
        weights = self.root / "benchmark/pretrained_model_weight"
        report["resources"] = {}
        for name, size in (("vilt_200k_mlm_itm.ckpt", 1628097027),
                           ("bert-base-uncased/vocab.txt", 231508),
                           ("bert-base-uncased/config.json", 433)):
            path = weights / name
            actual = path.stat().st_size if path.exists() else None
            report["resources"][name] = {"bytes": actual, "expected_bytes": size}
            if actual != size:
                report["errors"].append("Missing/different-size resource: " + name)
        if runtime:
            try:
                for module in ("torch", "transformers", "timm", "wandb", "ujson", "scipy", "sklearn"):
                    importlib.import_module(module)
                import torch
                from transformers import BertTokenizer
                if BertTokenizer.from_pretrained(str(weights / "bert-base-uncased"),
                                                 local_files_only=True).vocab_size != 30522:
                    raise ValueError("Tokenizer vocabulary differs")
                if gpu < 0 or not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
                    raise ValueError("Requested CUDA GPU unavailable")
                report["runtime"] = {"python": sys.version, "executable": sys.executable,
                                     "torch": torch.__version__, "cuda": torch.version.cuda,
                                     "gpu": torch.cuda.get_device_name(gpu)}
            except Exception as exc:
                report["errors"].append("Runtime: " + str(exc))
        return report

    def _mask_paths(self, scenario):
        both = 0.5 if scenario == "both" else 0.0
        paths = [self.folder / "missing_tables" / protocol.mask_name(s, scenario, 0.7, both)
                 for s in ("train", "test")]
        other_specs = [("both", 0, 0), ("text", 1, 0), ("image", 1, 0)]
        other_specs += [(s, 0.7, 0.5 if s == "both" else 0) for s in protocol.SCENARIOS if s != scenario]
        return paths + [self.folder / "other_tests" / protocol.mask_name("test", s, r, b)
                        for s, r, b in other_specs]

    def fingerprints(self, scenario):
        paths = [self.folder / "split.json", self.folder / "prepared-files.json",
                 self.root / "fedtask" / protocol.task_name(scenario) / "data.json"]
        paths.extend(self.folder / "generate_arrows" / ("food101_" + s + ".arrow")
                     for s in ("train", "test"))
        paths.extend(self._mask_paths(scenario))
        return {p.relative_to(self.root).as_posix(): table1.digest(p) for p in paths}

    def snapshot(self, target):
        protocol.write_json(target, {s: self.fingerprints(s) for s in protocol.SCENARIOS})


def materialize(context):
    import pyarrow as pa
    manifest = protocol.make_manifest(context.data)
    context.manifest_file.parent.mkdir(parents=True, exist_ok=True)
    if context.manifest_file.exists():
        if protocol.read_json(context.manifest_file) != manifest:
            raise ValueError("Saved split differs; refusing to replace the versioned manifest")
    else:
        protocol.write_json(context.manifest_file, manifest)
    if context.folder.exists():
        if protocol.load_prepared_manifest(context.folder / "generate_arrows") != manifest:
            raise ValueError("Existing prepared dataset differs")
        context.prepare(protocol.SCENARIOS)
        report = context.inspect_inputs()
        if report["errors"]:
            raise ValueError(str(report["errors"]))
        print("Already prepared; all versioned inputs validated:", context.folder)
        return report

    parent = context.folder.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / (".prepare_" + context.protocol + "_" + uuid.uuid4().hex[:8])
    # Only publish a complete new directory inside this protocol's parent.
    staging.resolve().relative_to(parent.resolve())
    context.folder.resolve().relative_to(parent.resolve())
    staging.mkdir()
    print("Preparing in:", staging, flush=True)
    protocol.write_json(staging / "split.json", manifest)
    pieces = []
    for split in ("train", "test"):
        wanted = set(manifest["source_ids"][split])
        found = set()
        path = context.data / "generate_arrows" / ("food101_" + split + ".arrow")
        with pa.memory_map(str(path), "r") as stream:
            reader = pa.ipc.RecordBatchFileReader(stream)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                ids = batch.column(batch.schema.get_field_index("image_id")).to_pylist()
                indices = [j for j, sid in enumerate(ids) if sid in wanted]
                if not indices:
                    continue
                selected_ids = [ids[j] for j in indices]
                if len(set(selected_ids)) != len(selected_ids) or found.intersection(selected_ids):
                    raise ValueError("Duplicate source IDs in Arrow")
                found.update(selected_ids)
                piece = pa.Table.from_batches([batch]).take(pa.array(indices, type=pa.int64()))
                pieces.append(piece.replace_schema_metadata(None))
        if found != wanted:
            raise ValueError("Source Arrow does not contain the selected original IDs")
    pool = pa.concat_tables(pieces)
    lookup = {sid: i for i, sid in enumerate(pool["image_id"].to_pylist())}
    if len(lookup) != 6727:
        raise ValueError("Pooled Arrow coverage differs")
    output_dir = staging / "generate_arrows"
    output_dir.mkdir()
    checksums = {}
    for split in ("train", "test"):
        path = output_dir / ("food101_" + split + ".arrow")
        with pa.OSFile(str(path), "wb") as sink:
            with pa.RecordBatchFileWriter(sink, pool.schema) as writer:
                ids = manifest[split]
                for start in range(0, len(ids), 128):
                    take_ids = ids[start:start + 128]
                    rows = pool.take(pa.array([lookup[s] for s in take_ids], type=pa.int64()))
                    rows = rows.set_column(rows.schema.get_field_index("split"), "split",
                                           pa.array([split] * len(take_ids), type=pa.string()))
                    writer.write_table(rows)
        checksums[path.name] = table1.digest(path)
        print("Saved", split, len(manifest[split]), "rows,", path.stat().st_size, "bytes", flush=True)
    protocol.write_json(output_dir / "protocol.json", {
        "protocol_id": context.protocol, "manifest_sha256": protocol.canonical_hash(manifest)})
    for split in ("train", "test"):
        with pa.memory_map(str(output_dir / ("food101_" + split + ".arrow")), "r") as stream:
            protocol.validate_table(pa.ipc.RecordBatchFileReader(stream).read_all(), output_dir, split)
    protocol.write_masks(staging, manifest)
    protocol.write_json(staging / "prepared-files.json", checksums)
    # Existing legacy data/tasks are never renamed, deleted or overwritten.
    staging.rename(context.folder)
    context.prepare(protocol.SCENARIOS)
    report = context.inspect_inputs()
    if report["errors"]:
        raise ValueError(str(report["errors"]))
    protocol.write_json(context.folder / "preparation-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    check = commands.add_parser("check")
    check.add_argument("--runtime", action="store_true")
    check.add_argument("--gpu", type=int, default=0)
    run = commands.add_parser("run")
    run.add_argument("--stage", choices=("smoke", "fedprime", "all"), default="smoke")
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("--seeds", nargs="+", type=int, default=[1234])
    run.add_argument("--dry-run", action="store_true")
    summary = commands.add_parser("summary")
    summary.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    context = Split80Context()
    if args.command == "prepare":
        report = materialize(context)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "check":
        report = context.inspect_inputs(args.runtime, args.gpu)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["errors"] else 0
    elif args.command == "run":
        table1.run_matrix(args, context=context)
    else:
        table1.summarize(args.results)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, ImportError, FileNotFoundError, KeyError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
