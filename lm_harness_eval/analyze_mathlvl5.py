#!/usr/bin/env python3
"""
Analyze Math Level 5 accuracy for model families and compute forgetting/robustness.

This script analyzes results for:
- Qwen3-8B family: base, SFT GSM8K, SFT CSQA
- Llama3-8B-Instruct family: base, SFT GSM8K, SFT CSQA

Usage:
    python analyze_mathlvl5.py [--results-dir DIR] [--output FILE]

Example:
    python analyze_mathlvl5.py --results-dir ./results/mathlvl5 --output mathlvl5_analysis.md
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def find_samples_files(directory, task_prefix="samples_leaderboard_math"):
    """Find all Math Level 5 samples JSONL files in a results directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if f.startswith(task_prefix) and f.endswith('.jsonl'):
                files.append(os.path.join(root, f))
    return sorted(files)


def load_samples(paths):
    """Load per-sample results from JSONL files, aggregating across subtasks."""
    samples = {}
    for path in paths:
        # Extract subtask from filename like "samples_leaderboard_math_algebra_hard_..."
        filename = Path(path).stem
        parts = filename.split('_')
        # Find subtask name after "leaderboard_math_"
        subtask = "unknown"
        for i, p in enumerate(parts):
            if p == 'math' and i + 1 < len(parts):
                # Collect subtask parts until we hit a hash or timestamp
                subtask_parts = []
                for j in range(i + 1, len(parts)):
                    if len(parts[j]) > 10 or parts[j].isdigit():
                        break
                    subtask_parts.append(parts[j])
                subtask = '_'.join(subtask_parts)
                break

        with open(path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                unique_id = f"{subtask}_{sample['doc_id']}"
                # Handle different key names for accuracy
                acc = sample.get('acc', sample.get('exact_match', 0))
                samples[unique_id] = {
                    'acc': acc,
                    'subtask': subtask,
                    'doc_id': sample['doc_id'],
                }
    return samples


def load_results_json(directory):
    """Load the results JSON file to get accuracy metrics."""
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.startswith('results_') and f.endswith('.json'):
                with open(os.path.join(root, f), 'r') as fp:
                    return json.load(fp)
    return None


def get_accuracy_from_results(results_json):
    """Extract overall accuracy from results JSON."""
    if not results_json or 'results' not in results_json:
        return 0.0

    metrics = results_json['results'].get('leaderboard_math_hard', {})
    for key in ['exact_match,none', 'exact_match,custom-extract', 'acc,none']:
        if key in metrics:
            return metrics[key]

    return 0.0


def compare_models(base_samples, sft_samples):
    """Compare base vs SFT model results per sample."""
    both_correct = 0
    gain = 0  # base wrong -> sft correct
    forget = 0  # base correct -> sft wrong
    both_wrong = 0

    common_ids = set(base_samples.keys()) & set(sft_samples.keys())

    for uid in common_ids:
        base_acc = base_samples[uid]['acc']
        sft_acc = sft_samples[uid]['acc']

        if base_acc == 1 and sft_acc == 1:
            both_correct += 1
        elif base_acc == 0 and sft_acc == 1:
            gain += 1
        elif base_acc == 1 and sft_acc == 0:
            forget += 1
        else:
            both_wrong += 1

    total = len(common_ids)
    return {
        'both_correct': both_correct,
        'gain': gain,
        'forget': forget,
        'both_wrong': both_wrong,
        'total': total,
        'base_acc': (both_correct + forget) / total * 100 if total > 0 else 0,
        'sft_acc': (both_correct + gain) / total * 100 if total > 0 else 0,
        'net_change': gain - forget,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze Math Level 5 results')
    parser.add_argument('--results-dir', default='./results/mathlvl5',
                        help='Directory containing Math Level 5 results')
    parser.add_argument('--output', default='mathlvl5_analysis.md',
                        help='Output markdown file')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Define model families
    families = {
        'Qwen3-8B': {
            'base': 'qwen3_8b_base',
            'sft_gsm8k': 'qwen3_8b_sft_gsm8k',
            'sft_csqa': 'qwen3_8b_sft_csqa',
        },
        'Llama3-8B-Instruct': {
            'base': 'llama3_8b_base',
            'sft_gsm8k': 'llama3_8b_sft_gsm8k',
            'sft_csqa': 'llama3_8b_sft_csqa',
        },
    }

    # Collect all data
    all_data = {}

    for family_name, models in families.items():
        all_data[family_name] = {}

        for model_type, model_dir in models.items():
            model_path = results_dir / model_dir
            if not model_path.exists():
                print(f"Warning: {model_path} does not exist, skipping")
                continue

            # Load samples
            sample_files = find_samples_files(model_path)
            if sample_files:
                samples = load_samples(sample_files)
                all_data[family_name][model_type] = {
                    'samples': samples,
                    'n_samples': len(samples),
                }

            # Load accuracy from results JSON
            results_json = load_results_json(model_path)
            if results_json:
                overall_acc = get_accuracy_from_results(results_json)
                if model_type not in all_data[family_name]:
                    all_data[family_name][model_type] = {}
                all_data[family_name][model_type]['overall_acc'] = overall_acc

    # Generate markdown report
    md_lines = []
    md_lines.append("# Math Level 5 Analysis Report\n")
    md_lines.append("## Overview\n")
    md_lines.append("Comparison of base models vs SFT models on Math Level 5 (leaderboard_math_hard) benchmark.\n")
    md_lines.append("This benchmark contains Level 5 (hardest) problems from the MATH dataset.\n")

    # Overall Accuracy Table
    md_lines.append("## Overall Accuracy\n")
    md_lines.append("| Model Family | Model | Accuracy |")
    md_lines.append("|--------------|-------|----------|")

    for family_name, models in all_data.items():
        for model_type in ['base', 'sft_gsm8k', 'sft_csqa']:
            if model_type in models and 'overall_acc' in models[model_type]:
                acc = models[model_type]['overall_acc'] * 100
                model_label = {
                    'base': 'Base',
                    'sft_gsm8k': 'SFT GSM8K',
                    'sft_csqa': 'SFT CSQA'
                }[model_type]
                md_lines.append(f"| {family_name} | {model_label} | {acc:.2f}% |")

    md_lines.append("")

    # Forgetting/Robustness Analysis
    md_lines.append("## Forgetting & Robustness Analysis\n")
    md_lines.append("Per-sample comparison between base and SFT models.\n")
    md_lines.append("- **Gain**: Questions base got wrong but SFT got right (improvement)")
    md_lines.append("- **Forget**: Questions base got right but SFT got wrong (regression)")
    md_lines.append("- **Net Change**: Gain - Forget\n")

    for family_name, models in all_data.items():
        md_lines.append(f"### {family_name}\n")

        if 'base' not in models or 'samples' not in models.get('base', {}):
            md_lines.append("*Base model results not available*\n")
            continue

        base_samples = models['base']['samples']

        md_lines.append("| Metric | SFT GSM8K | SFT CSQA |")
        md_lines.append("|--------|-----------|----------|")

        comparisons = {}
        for sft_type in ['sft_gsm8k', 'sft_csqa']:
            if sft_type in models and 'samples' in models[sft_type]:
                comparisons[sft_type] = compare_models(base_samples, models[sft_type]['samples'])

        if comparisons:
            metrics = [
                ('Total Samples', 'total', ''),
                ('Both Correct', 'both_correct', ''),
                ('Gain (base wrong→SFT right)', 'gain', ''),
                ('Forget (base right→SFT wrong)', 'forget', ''),
                ('Both Wrong', 'both_wrong', ''),
                ('Base Accuracy', 'base_acc', '%'),
                ('SFT Accuracy', 'sft_acc', '%'),
                ('Net Change', 'net_change', ''),
            ]

            for label, key, suffix in metrics:
                row = f"| {label} |"
                for sft_type in ['sft_gsm8k', 'sft_csqa']:
                    if sft_type in comparisons:
                        val = comparisons[sft_type][key]
                        if key == 'net_change':
                            row += f" {val:+d} |"
                        elif suffix == '%':
                            row += f" {val:.2f}% |"
                        else:
                            row += f" {val} |"
                    else:
                        row += " N/A |"
                md_lines.append(row)

        md_lines.append("")

    # Write to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"Analysis saved to {output_path}")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Print summary to console
    for family_name, models in all_data.items():
        print(f"\n{family_name}:")
        for model_type in ['base', 'sft_gsm8k', 'sft_csqa']:
            if model_type in models and 'overall_acc' in models[model_type]:
                acc = models[model_type]['overall_acc'] * 100
                label = {'base': 'Base', 'sft_gsm8k': 'SFT GSM8K', 'sft_csqa': 'SFT CSQA'}[model_type]
                print(f"  {label}: {acc:.2f}%")


if __name__ == "__main__":
    main()
