#!/usr/bin/env python
"""
data_preparer_csqa.py
---------------------
Handles the preparation of the CommonsenseQA dataset according to specific
splitting requirements for experiments.
"""
import json
import pathlib
import random
from datasets import load_dataset
from tqdm import tqdm

def prepare_csqa_dataset(dataset_config, dataset_ref_name):
    """
    Prepares the CommonsenseQA dataset.

    - The official 'train' split is divided into a new 'train' (for LoRA) and 'dev' set.
    - The official 'validation' split is used as the 'test' set for evaluation.
    - Checks if data already processed, skips download/processing if so.

    Args:
        dataset_config (object): Configuration object for this dataset from datasets_config.yaml.
        dataset_ref_name (str): The reference name of the dataset (e.g., "csqa_default_split"),
                                used for structuring the output directory under out_dir_base.

    Returns:
        dict: Paths to the processed train, dev, and test files.
    """
    # dataset_name = dataset_config.hf_path.split('/')[-1] # Old way
    # The output directory is now based on dataset_ref_name to ensure uniqueness for this defined dataset config
    processed_data_dir = pathlib.Path(dataset_config.out_dir_base).expanduser() / dataset_ref_name
    
    train_file = processed_data_dir / "train.jsonl"
    dev_file = processed_data_dir / "dev.jsonl"
    test_file = processed_data_dir / "test.jsonl" # from official validation

    if train_file.exists() and dev_file.exists() and test_file.exists():
        print(f"Dataset '{dataset_ref_name}' already processed at {processed_data_dir}. Skipping.")
        return {
            "train_file": str(train_file),
            "dev_file": str(dev_file),
            "test_file": str(test_file),
        }

    processed_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputting processed CSQA data for '{dataset_ref_name}' to: {processed_data_dir}")

    # 1. Load official splits
    print(f"Loading {dataset_config.hf_path} from Hugging Face...")
    ds_official_train = load_dataset(dataset_config.hf_path, split="train")
    ds_official_valid = load_dataset(dataset_config.hf_path, split="validation") # This will be our test set

    # 2. Find the concept key (for splitting the training set)
    concept_key = (
        "question_concept" if "question_concept" in ds_official_train.column_names
        else "source_concept"
    )

    # 3. Group official TRAIN rows by concept
    concept2rows = {}
    for row in tqdm(ds_official_train, desc="Grouping train rows by concept"):
        concept = row[concept_key]
        concept2rows.setdefault(concept, []).append(row)

    # 4. Deterministic shuffle of concepts
    concepts = sorted(concept2rows.keys())
    random.Random(dataset_config.seed).shuffle(concepts)

    # 5. Split concepts for our new train/dev sets from official train set
    cutoff = int(len(concepts) * dataset_config.train_ratio)
    our_train_concepts, our_dev_concepts = concepts[:cutoff], concepts[cutoff:]

    def dump(rows, file_path):
        with open(file_path, "w") as f:
            for r in tqdm(rows, desc=f"Writing {file_path.name}"):
                f.write(json.dumps(r) + "\n")

    # 6. Write our new train.jsonl (for LoRA)
    our_train_rows = [r for c in our_train_concepts for r in concept2rows[c]]
    dump(our_train_rows, train_file)

    # 7. Write our new dev.jsonl (for LoRA)
    our_dev_rows = [r for c in our_dev_concepts for r in concept2rows[c]]
    dump(our_dev_rows, dev_file)

    # 8. Write test.jsonl (from official validation split)
    dump(list(ds_official_valid), test_file)

    print(f"CSQA Preparation Complete for '{dataset_ref_name}':")
    print(f"  Our Train (for LoRA) concepts: {len(our_train_concepts)}, Rows: {len(our_train_rows)}")
    print(f"  Our Dev (for LoRA)   concepts: {len(our_dev_concepts)}, Rows: {len(our_dev_rows)}")
    print(f"  Test (official validation) rows: {len(ds_official_valid)}")

    return {
        "train_file": str(train_file),
        "dev_file": str(dev_file),
        "test_file": str(test_file),
    }

if __name__ == "__main__":
    # Example of how to run this script standalone for debugging
    class DummyConfig:
        def __init__(self, hf_path, train_ratio, seed, out_dir_base):
            self.hf_path = hf_path
            self.train_ratio = train_ratio
            self.seed = seed
            self.out_dir_base = out_dir_base
            self.type = "csqa_custom_split" # Matches config

    dummy_config = DummyConfig(
        hf_path="tau/commonsense_qa",
        train_ratio=0.7,
        seed=42,
        out_dir_base="data/temp_csqa_prep" # Standalone debug output base
    )
    prepare_csqa_dataset(dummy_config, "csqa_debug_standalone") # Use a descriptive name for the debug run
