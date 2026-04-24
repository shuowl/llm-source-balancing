#!/usr/bin/env python3
"""
Compute forgetting vs robustness gains between base and SFT models.

Usage:
    python compute_forgetting.py <base_results_dir> <sft_results_dir> [<sft2_results_dir> ...]

Example:
    python compute_forgetting.py ./results/mmlupro/qwen3_8b_base \
        ./results/mmlupro/qwen3_8b_sft_gsm8k \
        ./results/mmlupro/qwen3_8b_sft_csqa
"""

import json
import os
import sys
from pathlib import Path


def find_samples_file(directory, task_pattern="samples_"):
    """Find the samples JSONL file in a results directory."""
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.startswith(task_pattern) and f.endswith('.jsonl'):
                return os.path.join(root, f)
    return None


def load_samples(path):
    """Load per-sample results from a JSONL file."""
    samples = {}
    with open(path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            doc_id = sample['doc_id']
            samples[doc_id] = {
                'acc': sample['acc'],
                'record_id': sample['doc'].get('Record ID', doc_id),
                'target': sample.get('target', ''),
            }
    return samples


def compare_models(base, sft, sft_name):
    """
    Compare base vs SFT model results.

    Categories:
    - Both correct (robust)
    - Base wrong -> SFT correct (gain/robustness improvement)
    - Base correct -> SFT wrong (forgetting)
    - Both wrong (no change)
    """
    both_correct = 0
    gain = 0  # base wrong -> sft correct
    forget = 0  # base correct -> sft wrong
    both_wrong = 0

    gain_ids = []
    forget_ids = []

    for doc_id in base:
        base_acc = base[doc_id]['acc']
        sft_acc = sft[doc_id]['acc']

        if base_acc == 1 and sft_acc == 1:
            both_correct += 1
        elif base_acc == 0 and sft_acc == 1:
            gain += 1
            gain_ids.append(doc_id)
        elif base_acc == 1 and sft_acc == 0:
            forget += 1
            forget_ids.append(doc_id)
        else:
            both_wrong += 1

    total = len(base)
    print(f"\n{sft_name}:")
    print(f"  Both correct (robust):     {both_correct:3d} ({both_correct/total*100:5.1f}%)")
    print(f"  Base wrong -> SFT correct: {gain:3d} ({gain/total*100:5.1f}%) [GAIN]")
    print(f"  Base correct -> SFT wrong: {forget:3d} ({forget/total*100:5.1f}%) [FORGET]")
    print(f"  Both wrong:                {both_wrong:3d} ({both_wrong/total*100:5.1f}%)")
    print(f"  -----------------------------------------")
    print(f"  Net change:                {gain - forget:+3d} ({(gain-forget)/total*100:+5.1f}%)")
    print(f"  Base accuracy:             {(both_correct + forget)/total*100:5.1f}%")
    print(f"  SFT accuracy:              {(both_correct + gain)/total*100:5.1f}%")

    return {
        'name': sft_name,
        'both_correct': both_correct,
        'gain': gain,
        'forget': forget,
        'both_wrong': both_wrong,
        'gain_ids': gain_ids,
        'forget_ids': forget_ids,
        'total': total,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    base_dir = sys.argv[1]
    sft_dirs = sys.argv[2:]

    # Find and load base samples
    base_file = find_samples_file(base_dir)
    if not base_file:
        print(f"Error: Could not find samples file in {base_dir}")
        sys.exit(1)

    print(f"Base file: {base_file}")
    base_samples = load_samples(base_file)
    print(f"Loaded {len(base_samples)} samples from base model")

    # Compare with each SFT model
    results = []
    for sft_dir in sft_dirs:
        sft_file = find_samples_file(sft_dir)
        if not sft_file:
            print(f"Warning: Could not find samples file in {sft_dir}, skipping")
            continue

        print(f"SFT file: {sft_file}")
        sft_samples = load_samples(sft_file)

        # Extract model name from directory
        sft_name = Path(sft_dir).name.replace('results_', '')

        print("\n" + "="*70)
        print(f"BASE vs {sft_name}")
        print("="*70)

        result = compare_models(base_samples, sft_samples, sft_name)
        results.append(result)

    # Print summary table
    if len(results) > 1:
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)

        # Header
        header = f"{'Metric':<35}"
        for r in results:
            header += f" {r['name'][:15]:>15}"
        print(header)
        print("-" * (35 + 16 * len(results)))

        # Rows
        metrics = [
            ('Both correct', 'both_correct'),
            ('Gain (base wrong->SFT right)', 'gain'),
            ('Forget (base right->SFT wrong)', 'forget'),
            ('Both wrong', 'both_wrong'),
        ]

        for label, key in metrics:
            row = f"{label:<35}"
            for r in results:
                row += f" {r[key]:>15}"
            print(row)

        print("-" * (35 + 16 * len(results)))

        # Net change
        row = f"{'Net change':<35}"
        for r in results:
            net = r['gain'] - r['forget']
            row += f" {net:>+15}"
        print(row)

    # Save results to JSON
    output_file = "forgetting_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'base_file': base_file,
            'results': [{k: v for k, v in r.items()} for r in results]
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
