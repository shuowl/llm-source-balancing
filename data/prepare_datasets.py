#!/usr/bin/env python
"""
prepare_datasets.py  – Run ONCE to materialise all datasets listed in
configs/datasets_config.yaml.  Results are stored under data/processed_datasets
and a manifest data/prepared_datasets.yaml is produced for quick lookup.

Usage
-----
python data/prepare_datasets.py

This script will:
1. Read configs/datasets_config.yaml to get dataset definitions
2. Download and process each dataset according to its configuration
3. Save processed datasets to data/processed_datasets/
4. Generate data/prepared_datasets.yaml manifest for quick lookup

Example output structure:
├── data/
│   ├── processed_datasets/
│   │   ├── csqa_default_split/
│   │   │   ├── train.jsonl
│   │   │   ├── dev.jsonl
│   │   │   └── test.jsonl
│   │   └── gsm8k_default_split/
│   │       ├── train.jsonl
│   │       ├── dev.jsonl
│   │       ├── test.jsonl
│   │       └── gold_reasoning.jsonl
│   └── prepared_datasets.yaml  # manifest file

The manifest file contains paths and metadata for all processed datasets,
which is used by other scripts in the pipeline to locate dataset files.
"""
import pathlib, yaml, importlib, sys
from types import SimpleNamespace
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))                      # so we can import your helpers
# -----------------------------------------------------------
# helper so cfg.foo works (instead of cfg["foo"])
class AttrDict(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.__dict__ = self

datasets_cfg = yaml.safe_load(
    open(ROOT / "configs/datasets_config.yaml")
)["datasets"]

manifest = {}
for ref, cfg in tqdm(datasets_cfg.items()):
    prep_fn = {
        "csqa_custom_split":   "data.preparers.data_preparer_csqa::prepare_csqa_dataset",
        "gsm8k_mc_custom_split":"data.preparers.data_preparer_gsm8k_mc::prepare_gsm8k_mc_dataset",
        "codemmlu_sample": "data.preparers.data_preparer_codemmlu::prepare_codemmlu_dataset",
    }[cfg["type"]]
    module_name, fn_name = prep_fn.split("::")
    fn = getattr(importlib.import_module(module_name), fn_name)

    # Determine the correct manifest key
    # For gsm8k_mc_custom_split, always use "gsm8k_default_split" as the key
    # For other types, use the key from the config (ref)
    manifest_key = "gsm8k_default_split" if cfg["type"] == "gsm8k_mc_custom_split" else ref
    
    manifest[manifest_key] = fn(AttrDict(cfg), manifest_key)

out_path = ROOT / "data" / "prepared_datasets.yaml"
out_path.write_text(yaml.dump(manifest))
print(f"\n[✓] Dataset manifest written to {out_path}")
