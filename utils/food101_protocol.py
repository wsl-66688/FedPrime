"""Portable, explicitly versioned Food101 split; no torch dependency.

This protocol keeps the released eight classes and original train/test sample
union. It deliberately does not claim to recover the authors' unpublished split.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = "released8_stratified80_seed0_v1"
CLASSES = ("frozen_yogurt", "tacos", "gnocchi", "ramen", "sushi",
           "spaghetti_carbonara", "foie_gras", "club_sandwich")
SCENARIOS = ("both", "text", "image")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def rank(sample_id, purpose, seed=0):
    return hashlib.sha256(("fedprime80-v1|{}|{}|{}".format(seed, purpose, sample_id))
                          .encode("utf-8")).digest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_manifest(metadata_root):
    metadata_root = Path(metadata_root)
    classes = read_json(metadata_root / "class_idx.json")
    parts = read_json(metadata_root / "split.json")
    texts = read_json(metadata_root / "text.json")
    if any(classes.get(name) != i for i, name in enumerate(CLASSES)):
        raise ValueError("Released class mapping differs from the expected eight classes")
    sources = {split: [s for s in parts[split] if s in texts and s.rsplit("_", 1)[0] in CLASSES]
               for split in ("train", "test")}
    pool = sources["train"] + sources["test"]
    if len(pool) != 6727 or len(set(pool)) != len(pool):
        raise ValueError("Expected 6,727 unique released train/test sample IDs")
    groups = {c: sorted(s for s in pool if s.rsplit("_", 1)[0] == c) for c in CLASSES}
    target_test = math.ceil(len(pool) / 5)
    allocation = {c: len(ids) // 5 for c, ids in groups.items()}
    remainder_order = sorted(CLASSES, key=lambda c: (-(len(groups[c]) % 5), classes[c]))
    for c in remainder_order[:target_test - sum(allocation.values())]:
        allocation[c] += 1
    test = {s for c, ids in groups.items()
            for s in sorted(ids, key=lambda x: rank(x, "test-selection"))[:allocation[c]]}
    train_ids = sorted(set(pool) - test, key=lambda x: rank(x, "train-order"))
    test_ids = sorted(test, key=lambda x: rank(x, "test-order"))
    manifest = {
        "protocol_id": PROTOCOL, "format_version": 1, "split_seed": 0,
        "selection_algorithm": "sha256-rank-v1; largest-remainder class test allocation; ceil(N/5)",
        "description": "Released 8 classes and original train/test union; validation excluded; custom 80:20 split",
        "author_split_confirmed": False, "class_to_idx": {c: classes[c] for c in CLASSES},
        "source_ids": sources, "source_ids_sha256": canonical_hash(sources),
        "excluded_validation_count": sum(s in texts and s.rsplit("_", 1)[0] in CLASSES
                                         for s in parts.get("val", [])),
        "train": train_ids, "test": test_ids,
        "class_counts": {split: dict(Counter(s.rsplit("_", 1)[0] for s in ids))
                         for split, ids in (("train", train_ids), ("test", test_ids))},
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest.get("protocol_id") != PROTOCOL or manifest.get("split_seed") != 0:
        raise ValueError("Unsupported Food101 protocol")
    if manifest.get("class_to_idx") != dict(zip(CLASSES, range(8))):
        raise ValueError("Released eight-class label mapping changed")
    train, test = manifest["train"], manifest["test"]
    if len(train) != 5381 or len(test) != 1346:
        raise ValueError("Expected 5,381 train / 1,346 test IDs")
    source = manifest["source_ids"]["train"] + manifest["source_ids"]["test"]
    if (len(set(source)) != 6727 or len(set(train + test)) != 6727
            or set(train + test) != set(source)):
        raise ValueError("Split overlap, duplicate IDs or sample coverage mismatch")
    if canonical_hash(manifest["source_ids"]) != manifest["source_ids_sha256"]:
        raise ValueError("Source sample IDs changed")
    for split in ("train", "test"):
        counts = Counter(s.rsplit("_", 1)[0] for s in manifest[split])
        if set(counts) != set(CLASSES) or dict(counts) != manifest["class_counts"][split]:
            raise ValueError("Class counts or class identities changed")
    for c in CLASSES:
        total = manifest["class_counts"]["train"][c] + manifest["class_counts"]["test"][c]
        if abs(manifest["class_counts"]["test"][c] - total / 5) >= 1:
            raise ValueError("Test split is not stratified")


def load_prepared_manifest(data_dir, expected_hash=None):
    data_dir = Path(data_dir)
    marker = read_json(data_dir / "protocol.json")
    manifest = read_json(data_dir.parent / "split.json")
    validate_manifest(manifest)
    signature = canonical_hash(manifest)
    if (marker.get("protocol_id") != PROTOCOL or marker.get("manifest_sha256") != signature
            or (expected_hash is not None and signature != expected_hash)):
        raise ValueError("Prepared dataset does not match its protocol manifest")
    return manifest


def validate_table(table, data_dir, split):
    manifest = load_prepared_manifest(data_dir)
    ids = table["image_id"].to_pylist()
    if ids != manifest[split]:
        raise ValueError("Arrow sample IDs/order do not match the versioned split")
    labels = table["label"].to_pylist()
    expected = [manifest["class_to_idx"][s.rsplit("_", 1)[0]] for s in ids]
    if labels != expected or table["split"].to_pylist() != [split] * len(ids):
        raise ValueError("Arrow labels or split column do not match the manifest")
    return table


def mask_name(split, kind, ratio, both_ratio):
    return "food101_{}_missing_{}_{}_{}.pt.json".format(
        split, kind, str(ratio).replace(".", ""), str(both_ratio).replace(".", ""))


def make_mask(ids, kind, ratio, both_ratio, seed=0):
    n_missing = int(len(ids) * ratio)
    missing = sorted(range(len(ids)), key=lambda i: rank(ids[i], "missing", seed))[:n_missing]
    values = [0] * len(ids)
    for i in missing:
        values[i] = 2 if kind == "image" else 1
    if kind == "both":
        for i in sorted(missing, key=lambda i: rank(ids[i], "missing-image", seed))[:int(n_missing * both_ratio)]:
            values[i] = 2
    return values


def fixed_mask(table, pt_path):
    """Return a versioned mask, or None for an unchanged legacy mask directory."""
    path = Path(str(pt_path) + ".json")
    marker_path = path.parent / "protocol.json"
    if not marker_path.exists():
        return None
    marker = read_json(marker_path)
    item = read_json(path)  # Missing fixed masks must fail rather than be silently randomized.
    ids = table["image_id"].to_pylist()
    if (marker["protocol_id"] != PROTOCOL or item["protocol_id"] != PROTOCOL
            or marker["manifest_sha256"] != item["manifest_sha256"]
            or canonical_hash(ids) != item["sample_ids_sha256"] or item["seed"] != 0
            or path.name != mask_name(item["split"], item["kind"], item["ratio"], item["both_ratio"])):
        raise ValueError("Missing-mask identity does not match dataset sample order")
    expected = make_mask(ids, item["kind"], item["ratio"], item["both_ratio"], item["seed"])
    if item["values"] != expected:
        raise ValueError("Fixed missing-modality mask contents changed")
    return expected


def write_masks(root, manifest):
    signature = canonical_hash(manifest)
    marker = {"protocol_id": PROTOCOL, "manifest_sha256": signature}
    for folder in ("missing_tables", "other_tests"):
        (Path(root) / folder).mkdir()
        write_json(Path(root) / folder / "protocol.json", marker)
    specs = [("missing_tables", split, kind, 0.7, 0.5 if kind == "both" else 0.0)
             for split in ("train", "test") for kind in SCENARIOS]
    specs += [("other_tests", "test", kind, ratio, both)
              for kind, ratio, both in (("both", 0, 0), ("text", 1, 0), ("image", 1, 0),
                                        ("both", 0.7, 0.5), ("text", 0.7, 0), ("image", 0.7, 0))]
    for folder, split, kind, ratio, both in specs:
        ids = manifest[split]
        item = dict(marker, split=split, kind=kind, ratio=ratio, both_ratio=both, seed=0,
                    sample_ids_sha256=canonical_hash(ids), values=make_mask(ids, kind, ratio, both))
        write_json(Path(root) / folder / mask_name(split, kind, ratio, both), item)


def task_name(scenario):
    both = "0.5" if scenario == "both" else "0.0"
    return ("food101_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_"
            "missing_type_{0}_{0}_both_ratio_{1}_protocol_{2}".format(scenario, both, PROTOCOL))


def make_task(manifest, scenario, relative_root):
    """Same balanced client partition for every missing scenario and every method."""
    relative_root = Path(relative_root).as_posix()
    order = sorted(range(len(manifest["train"])),
                   key=lambda i: rank(manifest["train"][i], "client-partition"))
    clients = ["Client{:02d}".format(i) for i in range(20)]
    source = {"class_path": "benchmark.food101.dataset", "class_name": "FOOD101Dataset",
              "data_protocol": PROTOCOL, "manifest_sha256": canonical_hash(manifest),
              "data_dir": relative_root + "/generate_arrows",
              "missing_table_root": relative_root + "/missing_tables",
              "other_missing_table_root": relative_root + "/other_tests",
              "missing_ratio": {"train": 0.7, "test": 0.7},
              "missing_type": {"train": scenario, "test": scenario},
              "both_ratio": 0.5 if scenario == "both" else 0.0, "simulate_missing": False}
    task = {"store": "IDX", "client_names": clients, "dtest": list(range(len(manifest["test"]))),
            "datasrc": source}
    for i, name in enumerate(clients):
        task[name] = {"dtrain": order[i::20]}
    return task
