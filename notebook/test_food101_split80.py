"""Exercise split integrity and actual data/task adapters without training torch."""

import ast
from collections import Counter
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pyarrow as pa
import food101_split80 as cli
from utils import food101_protocol as protocol


def metadata(root):
    data = Path(root) / "benchmark/RAW_DATA/FOOD101"
    data.mkdir(parents=True)
    mapping = dict(zip(protocol.CLASSES, range(8)))
    train_sizes = (628, 626, 629, 625, 639, 585, 580, 592)
    all_sizes = (861, 859, 863, 857, 876, 803, 796, 812)
    parts = {"train": [], "test": [], "val": []}
    for c, n, total in zip(protocol.CLASSES, train_sizes, all_sizes):
        ids = [c + "_" + str(i) + ".jpg" for i in range(total)]
        parts["train"].extend(ids[:n])
        parts["test"].extend(ids[n:])
    texts = {s: "caption:" + s for ids in parts.values() for s in ids}
    for name, value in (("class_idx", mapping), ("split", parts), ("text", texts)):
        protocol.write_json(data / (name + ".json"), value)
    return data, parts


def arrow_table(ids, split):
    return pa.table({"image": [s.encode() for s in ids], "text": [["caption:" + s] for s in ids],
                     "label": [protocol.CLASSES.index(s.rsplit("_", 1)[0]) for s in ids],
                     "image_id": ids, "split": [split] * len(ids)})


def write_arrow(path, table):
    with pa.OSFile(str(path), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            for start in range(0, len(table), 128):
                writer.write_table(table.slice(start, 128))


def base_class():
    # Execute the real BaseDataset class while substituting only its unused torch base/transforms.
    tree = ast.parse((cli.ROOT / "benchmark/food101/base_dataset.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseDataset")
    namespace = {"torch": types.SimpleNamespace(utils=types.SimpleNamespace(
        data=types.SimpleNamespace(Dataset=object))), "pa": pa, "os": os, "io": io,
        "random": random, "Counter": Counter,
        "keys_to_transforms": lambda *a, **kw: []}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "base_dataset.py", "exec"), namespace)
    return namespace["BaseDataset"]


class Split80Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        data, parts = metadata(cls.root)
        cls.manifest = protocol.make_manifest(data)
        source = data / "generate_arrows"
        source.mkdir()
        for split in ("train", "test"):
            write_arrow(source / ("food101_" + split + ".arrow"), arrow_table(parts[split], split))
        cls.context = cli.Split80Context(cls.root)
        with patch.object(cls.context, "inspect_inputs", return_value={"errors": []}):
            with contextlib.redirect_stdout(io.StringIO()):
                cli.materialize(cls.context)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_split_is_deterministic_stratified_and_complete(self):
        m = self.manifest
        self.assertEqual(m, protocol.make_manifest(self.context.data))
        self.assertEqual((len(m["train"]), len(m["test"])), (5381, 1346))
        self.assertFalse(set(m["train"]) & set(m["test"]))
        self.assertEqual(set(m["train"] + m["test"]),
                         set(m["source_ids"]["train"] + m["source_ids"]["test"]))
        for c in protocol.CLASSES:
            n = sum(m["class_counts"][s][c] for s in ("train", "test"))
            self.assertLess(abs(m["class_counts"]["test"][c] - n * 0.2), 1)

    def test_materialized_payloads_and_labels_unchanged(self):
        for split in ("train", "test"):
            path = self.context.folder / "generate_arrows" / ("food101_" + split + ".arrow")
            with pa.memory_map(str(path), "r") as source:
                table = pa.ipc.RecordBatchFileReader(source).read_all()
                self.assertTrue(table.equals(arrow_table(self.manifest[split], split)))
        before = {p.name: cli.table1.digest(p) for p in self.context.data.joinpath("generate_arrows").glob("*.arrow")}
        with patch.object(self.context, "inspect_inputs", return_value={"errors": []}):
            with contextlib.redirect_stdout(io.StringIO()):
                cli.materialize(self.context)
        after = {p.name: cli.table1.digest(p) for p in self.context.data.joinpath("generate_arrows").glob("*.arrow")}
        self.assertEqual(before, after)

    def test_masks_fixed_and_reject_order_or_value_changes(self):
        table = arrow_table(self.manifest["train"], "train")
        path = self.context.folder / "missing_tables" / protocol.mask_name("train", "both", 0.7, 0.5)
        mask = protocol.fixed_mask(table, str(path)[:-5])
        self.assertEqual(Counter(mask), {0: 1615, 1: 1883, 2: 1883})
        with self.assertRaises(ValueError):
            protocol.fixed_mask(table.take(pa.array(list(reversed(range(len(table)))))), str(path)[:-5])
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / path.name
            protocol.write_json(Path(directory) / "protocol.json", protocol.read_json(path.parent / "protocol.json"))
            item = protocol.read_json(path)
            item["values"][0] = (item["values"][0] + 1) % 3
            protocol.write_json(copied, item)
            with self.assertRaises(ValueError):
                protocol.fixed_mask(table, str(copied)[:-5])
        self.assertIsNone(protocol.fixed_mask(table, self.root / "legacy/nonexistent.pt"))

    def test_tasks_use_one_balanced_partition_and_isolated_paths(self):
        tasks = [protocol.make_task(self.manifest, s, self.context.relative) for s in protocol.SCENARIOS]
        partitions = [[t[c]["dtrain"] for c in t["client_names"]] for t in tasks]
        self.assertEqual(partitions[0], partitions[1])
        self.assertEqual(partitions[1], partitions[2])
        self.assertEqual(sorted(i for row in partitions[0] for i in row), list(range(5381)))
        self.assertEqual(sorted(map(len, partitions[0])), [269] * 19 + [270])
        for t in tasks:
            self.assertIn(protocol.PROTOCOL, t["datasrc"]["data_dir"])
            self.assertIn(protocol.PROTOCOL, t["datasrc"]["missing_table_root"])

    def test_actual_base_loader_preserves_new_rows_and_legacy_behavior(self):
        Base = base_class()
        args = {"transform_keys": ["pixelbert"], "image_size": 384, "names": ["food101_train"],
                "text_column_name": "text", "split": "train", "remove_duplicate": False}
        with contextlib.redirect_stdout(io.StringIO()):
            new = Base(str(self.context.folder / "generate_arrows"), **args)
            old = Base(str(self.context.data / "generate_arrows"), **args)
        self.assertEqual(len(new), 5381)
        self.assertEqual(new.table["image_id"].to_pylist(), self.manifest["train"])
        self.assertEqual(len(old), int(4904 // 13.86))
        self.assertNotEqual(new.data_dir, old.data_dir)

    def test_actual_missing_dataset_uses_fixed_json_without_legacy_pt(self):
        tree = ast.parse((cli.ROOT / "benchmark/food101/dataset.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FOOD101Dataset")
        def unexpected(*args, **kwargs):
            raise AssertionError("Versioned data must not load/write legacy pt masks")
        fake_torch = types.SimpleNamespace(tensor=lambda values, dtype: values, float32="float32",
                                          load=unexpected, save=unexpected)
        namespace = {"BaseDataset": base_class(), "os": os, "torch": fake_torch, "random": random}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "dataset.py", "exec"), namespace)
        for scenario in protocol.SCENARIOS:
            info = {"ratio": {"train": 0.7, "test": 0.7},
                    "type": {"train": scenario, "test": scenario},
                    "both_ratio": 0.5 if scenario == "both" else 0.0,
                    "missing_table_root": str(self.context.folder / "missing_tables"),
                    "simulate_missing": False}
            with contextlib.redirect_stdout(io.StringIO()):
                dataset = namespace["FOOD101Dataset"](
                    str(self.context.folder / "generate_arrows"), ["pixelbert"],
                    image_size=384, split="train", missing_info=info)
            self.assertEqual(len(dataset), 5381)
            self.assertEqual(dataset.missing_table, protocol.make_mask(self.manifest["train"], scenario, 0.7,
                                                                     info["both_ratio"]))

    def test_actual_taskpipe_routes_all_test_conditions_to_new_data(self):
        Base = base_class()
        calls = []

        class Dataset(Base):
            def __init__(self, data_dir, transforms, split, missing_info, **kwargs):
                super().__init__(data_dir, transforms, names=["food101_" + split], split=split,
                                 text_column_name="text", **kwargs)
                filename = protocol.mask_name(split, missing_info["type"][split],
                                              missing_info["ratio"][split], missing_info["both_ratio"])
                mask_path = Path(missing_info["missing_table_root"]) / filename
                self.missing_table = protocol.fixed_mask(self.table, str(mask_path)[:-5])
                calls.append((data_dir, str(mask_path)))

        class Subset:
            def __init__(self, dataset, indices):
                self.dataset, self.indices = dataset, indices

            def __len__(self):
                return len(self.indices)

        tree = ast.parse((cli.ROOT / "benchmark/food101/core.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TaskPipe")
        namespace = {
            "IDXTaskPipe": type("IDXTaskPipe", (), {"TaskDataset": Subset}), "os": os, "ujson": json,
            "importlib": types.SimpleNamespace(import_module=lambda name: types.SimpleNamespace(FOOD101Dataset=Dataset)),
            "FOOD101Dataset": Dataset, "DataCollatorForLanguageModeling": object,
            "BertTokenizer": types.SimpleNamespace(from_pretrained=lambda path: object()),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), "core.py", "exec"), namespace)
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            for s in protocol.SCENARIOS:
                with contextlib.redirect_stdout(io.StringIO()):
                    train, tests, clients = namespace["TaskPipe"].load_task(
                        str(self.root / "fedtask" / protocol.task_name(s)))
                self.assertEqual(sum(map(len, train)), 5381)
                self.assertEqual(len(clients), 20)
                primary, (others, names) = tests
                self.assertEqual(len(primary), 1346)
                self.assertTrue(all(len(x) == 1346 for x in others))
                self.assertTrue({"full_modal", "text_only", "image_only"}.issubset(names))
        finally:
            os.chdir(previous)
        self.assertTrue(calls)
        self.assertTrue(all(protocol.PROTOCOL in data_dir and protocol.PROTOCOL in mask for data_dir, mask in calls))


if __name__ == "__main__":
    unittest.main()
