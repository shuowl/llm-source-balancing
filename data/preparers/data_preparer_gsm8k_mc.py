#!/usr/bin/env python
"""
data_preparer_gsm8k_mc.py
-------------------------
Handles the preparation of the GSM8K-MC dataset.
"""
import json
import pathlib
import random
import hashlib
from datasets import load_dataset
from tqdm import tqdm

def prepare_gsm8k_mc_dataset(dataset_config, dataset_ref_name):
    """
    Prepares the GSM8K-MC dataset in CSQA-compatible format.

    - Official 'train' split is divided into new 'train' (for LoRA) and 'dev'.
    - Official 'test' split is used as the 'test' set for evaluation.
    - Fetches 'openai/gsm8k' for gold reasoning to potentially augment LoRA training data.
    - Formats data to match CSQA structure with id, question, choices, answerKey fields.
    - Checks if data already processed, skips download/processing if so.

    Args:
        dataset_config (object): Config object from datasets_config.yaml.
        dataset_ref_name (str): The reference name of the dataset (e.g., "gsm8k_default_split"),
                                used for structuring the output directory.

    Returns:
        dict: Paths to processed train, dev, and test files.
    """
    # Always use "gsm8k_default_split" as the canonical name for GSM8K
    canonical_name = "gsm8k_default_split"
    processed_data_dir = pathlib.Path(dataset_config.out_dir_base).expanduser() / canonical_name

    train_file = processed_data_dir / "train.jsonl"
    dev_file = processed_data_dir / "dev.jsonl"
    test_file = processed_data_dir / "test.jsonl"
    gold_reasoning_file = processed_data_dir / "gold_reasoning.jsonl"

    if (train_file.exists() and dev_file.exists() and test_file.exists()
            and gold_reasoning_file.exists()):
        print(f"Dataset '{canonical_name}' already processed at {processed_data_dir}. Skipping.")
        return {
            "train_file": str(train_file),
            "dev_file": str(dev_file),
            "test_file": str(test_file),
            "gold_reasoning_file": str(gold_reasoning_file),
        }

    processed_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputting processed GSM8K-MC data for '{canonical_name}' to: {processed_data_dir}")

    # 1. Load GSM8K-MC dataset
    print(f"Loading {dataset_config.hf_path} from Hugging Face...")
    ds_gsm8k_mc_train = load_dataset(dataset_config.hf_path, split="train")
    ds_gsm8k_mc_test = load_dataset(dataset_config.hf_path, split="test") # This is our final test set

    # 2. Load original GSM8K for gold reasoning (optional, for LoRA prompt enrichment)
    gold_reasoning_path   = dataset_config.aux_hf_paths.get("gold_reasoning_dataset")
    gold_reasoning_config = dataset_config.aux_hf_paths.get("gold_reasoning_config", "main")
    reasoning_map = {}
    if gold_reasoning_path:
        print(
            f"Loading gold reasoning dataset {gold_reasoning_path}/{gold_reasoning_config} (train)…"
        )
        ds_gold_reasoning = load_dataset(
            gold_reasoning_path, gold_reasoning_config, split="train"
        )

        # Build fast lookup: canonicalised question → reasoning string
        def canon(s) -> str | None:
            if not isinstance(s, str):
                return None
            return " ".join(s.strip().split())  # collapse whitespace

        reasoning_map = {
            canon(r["question"]): r["answer"] for r in ds_gold_reasoning
        }

        # Save the raw gold-reasoning set just in case
        with open(gold_reasoning_file, "w") as f:
            for r in ds_gold_reasoning:
                f.write(json.dumps(r) + "\n")
        print(f"Gold reasoning dataset saved to {gold_reasoning_file}")

    # ------------------------------------------------------------------ #
    # Helper to convert GSM8K-MC format to CSQA-compatible format
    # ------------------------------------------------------------------ #
    def convert_to_csqa_format(dataset):
        rows = []
        for r in dataset:
            # Get question (handle both field name variants)
            question_text = r.get("question") or r.get("Question", "")
            question_hash = hashlib.md5(question_text.encode()).hexdigest()

            # Extract choice columns (single-letter keys) and sort
            choice_cols = sorted(
                [k for k in r.keys()
                 if len(k) == 1 and k.isalpha()
                    and k not in ("",)],  # Only exclude empty string
                key=lambda x: x
            )
            labels = choice_cols
            texts = [r.get(col, "") for col in choice_cols]
            # Answer comes from the "Answer" column
            answer_key = r.get("Answer") or r.get("answerKey")

            # Convert to CSQA format
            csqa_row = {
                "id": question_hash,
                "question": question_text,
                "choices": {
                    "label": labels,
                    "text": texts
                },
                "answerKey": answer_key
            }

            # Add gold reasoning if available
            q_key = canon(question_text)
            gold_reasoning = reasoning_map.get(q_key) if q_key else None
            if gold_reasoning:
                csqa_row["gold_reasoning"] = gold_reasoning

            rows.append(csqa_row)
        return rows

    gsm8k_mc_train_list = convert_to_csqa_format(ds_gsm8k_mc_train)

    # 3. Split the official 'train' set of GSM8K-MC into our new train/dev
    # GSM8K-MC doesn't have inherent concepts like CSQA, so we do a random split of rows.
    
    # Convert to list for shuffling
    random.Random(dataset_config.seed).shuffle(gsm8k_mc_train_list)

    cutoff = int(len(gsm8k_mc_train_list) * dataset_config.train_ratio)
    our_train_rows = gsm8k_mc_train_list[:cutoff]
    our_dev_rows = gsm8k_mc_train_list[cutoff:]

    def dump(rows, file_path):
        with open(file_path, "w") as f:
            for r in tqdm(rows, desc=f"Writing {file_path.name}"):
                f.write(json.dumps(r) + "\n")

    # 4. Write our new train.jsonl (for LoRA)
    dump(our_train_rows, train_file)

    # 5. Write our new dev.jsonl (for LoRA)
    dump(our_dev_rows, dev_file)

    # 6. Write test.jsonl (from official GSM8K-MC test split)
    dump(convert_to_csqa_format(ds_gsm8k_mc_test), test_file)

    print(f"GSM8K-MC Preparation Complete for '{canonical_name}':")
    print(f"  Our Train (for LoRA) rows: {len(our_train_rows)}")
    print(f"  Our Dev (for LoRA)   rows: {len(our_dev_rows)}")
    print(f"  Test (official test) rows: {len(ds_gsm8k_mc_test)}")

    return {
        "train_file": str(train_file),
        "dev_file": str(dev_file),
        "test_file": str(test_file),
        "gold_reasoning_file": str(gold_reasoning_file),
    }

if __name__ == "__main__":
    class DummyConfig:
        def __init__(self, hf_path, aux_hf_paths, train_ratio, seed, out_dir_base):
            self.hf_path = hf_path
            self.aux_hf_paths = aux_hf_paths
            self.train_ratio = train_ratio
            self.seed = seed
            self.out_dir_base = out_dir_base
            self.type = "gsm8k_mc_custom_split"

    dummy_config_gsm8k = DummyConfig(
        hf_path="guipenedo/gsm8k-mc",
        aux_hf_paths={"gold_reasoning_dataset": "openai/gsm8k"},
        train_ratio=0.7,
        seed=42,
        out_dir_base="data/temp_gsm8k_mc_prep" # Standalone debug output base
    )
    prepare_gsm8k_mc_dataset(dummy_config_gsm8k, "gsm8k_default_split")
