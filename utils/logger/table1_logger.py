"""Keep each Table 1 run separate without changing evaluation or training."""

import os
from pathlib import Path

from .basic_logger import Logger as BasicLogger


class Logger(BasicLogger):
    def __init__(self, meta, *args, **kwargs):
        target = os.environ.get("FEDPRIME_TABLE1_RUN_DIR")
        if not target:
            raise RuntimeError("Use notebook/food101_table1.py to select a run directory")
        self.run_dir = Path(target).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(meta, *args, **kwargs)

    def get_output_path(self):
        return str(self.run_dir)

    def get_log_path(self):
        return str(self.run_dir)

    def get_output_name(self, suffix=".json", prefix_log_filename=None):
        return "metrics" + suffix
