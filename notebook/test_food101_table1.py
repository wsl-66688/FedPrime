"""Checks for evaluation selection and independent result storage; no torch needed."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import food101_table1 as table1


def record(final, rounds=250):
    # An earlier higher score must not become the final-round result.
    return {key: [0.99] * (rounds - 1) + [final] for key in (
        "test_acc", "test_full_modal_acc", "test_text_only_acc", "test_image_only_acc")}


class Table1Tests(unittest.TestCase):
    def test_final_scores_alias_and_incomplete_evaluation(self):
        scores = table1.final_scores(record(0.8), "both", 250)
        self.assertEqual(scores["same_as_train"], 80)
        self.assertEqual(scores["miss_both"], 80)
        with self.assertRaises(ValueError):
            table1.final_scores(record(0.8, 249), "both", 250)
        with self.assertRaises(KeyError):
            table1.final_scores(record(0.8), "text", 250)

    def test_summary_ignores_smoke_and_reports_sample_sd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, score, rounds, state in (
                    (1, 0.8, 250, "complete"), (2, 0.9, 250, "complete"),
                    (3, 1.0, 1, "complete"), (4, 1.0, 250, "failed")):
                run = root / str(seed)
                run.mkdir()
                table1.write_json(run / "status.json", {
                    "state": state, "rounds": rounds, "scenario": "both", "method": "Fed-Prime",
                    "seed": seed, "data_sha256": {"task": "same", "mask": "same"}})
                table1.write_json(run / "metrics.json", record(score, rounds))
            table1.summarize(root)
            with (root / "table1-summary.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["n"], "2")
            self.assertEqual(float(rows[0]["mean_percent"]), 85)
            self.assertAlmostEqual(float(rows[0]["sample_sd_percent"]), 7.0710678118654755)
            status_file = root / "2/status.json"
            status = json.loads(status_file.read_text())
            status["data_sha256"]["mask"] = "different"
            table1.write_json(status_file, status)
            with self.assertRaises(ValueError):
                table1.summarize(root)

    def test_logger_writes_to_two_independent_run_directories(self):
        # Load the actual lightweight base logger without requiring W&B installation locally.
        fake_wandb = types.ModuleType("wandb")
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            sys.path.insert(0, str(table1.ROOT))
            try:
                from utils.logger.table1_logger import Logger
                meta = {"no_overwrite": False, "no_log_console": True,
                        "log_file": False, "early_stop": -1}
                with tempfile.TemporaryDirectory() as directory:
                    for name, value in (("a", 0.8), ("b", 0.9)):
                        target = Path(directory) / name
                        with patch.dict("os.environ", {"FEDPRIME_TABLE1_RUN_DIR": str(target)}):
                            logger = Logger(meta, name=name)
                            logger.output["test_acc"].append(value)
                            logger.save_output_as_json()
                        self.assertEqual(json.loads((target / "metrics.json").read_text())["test_acc"], [value])
                    self.assertEqual(json.loads((Path(directory) / "a/metrics.json").read_text())["test_acc"], [0.8])
            finally:
                sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
