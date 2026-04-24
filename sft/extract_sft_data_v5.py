#!/usr/bin/env python3
"""Extract SFT training data with full control over all variants, with separate tier handling.

This script implements tier-separated training data extraction where:
- t1: Gets bare and specific variants from tier1 sources only
- t2: Gets bare and specific variants from tier2 sources only
- t1t2: Gets both sets (including duplicate bare from both tiers)

For t1t2, the distribution is:
- Tier 1:
  - bare from d1nu1nin
  - upos, dpos from d1nu1nin (single-source positive)
  - uneg, dneg from d1nu1nin (single-source negative)
  - dpup, dpun, dnup, dnun from d1nu1nin (doc-first double-source)
  - updp, updn, undp, undn from u1nd1nin (user-first double-source)
- Tier 2:
  - bare from d2nu2nin (duplicate of tier1 bare)
  - upos, dpos from d2nu2nin (single-source positive)
  - uneg, dneg from d2nu2nin (single-source negative)
  - dpup, dpun, dnup, dnun from d2nu2nin (doc-first double-source)
  - updp, updn, undp, undn from u2nd2nin (user-first double-source)

Balanced Strategy Formats:
1. Full format: bare_A_upos_B_uneg_C_dpos_D_dneg_E_dpup_F_updp_G_dnun_H_undn_I_dpun_J_dnup_K_updn_L_undp_M
   where A-M are relative weights (can be 0) that will be normalized to sum to 100%

2. Friendly names (shortcuts for common configurations):
   - bare_100: 100% bare
   - no_bare_ss: No bare, single-source only
   - half_bare_ss: 50% bare, 50% single-source
   - all_variants: All 13 variants included
   - agreement_ds: Agreement double-source variants (trust + skepticism)

Supported datasets: csqa, gsm8k
Supported models: qwen3_8b, llama3_8b_instruct

Example Usage:
# For tier strategy t1, use only tier1 sources
python extract_sft_data_v5.py csqa qwen3_8b \
    --tier-strategy t1 \
    --balanced-strategy half_bare_ss \
    --num-train 2000

# For tier strategy t1t2, use both tier1 and tier2 sources (duplicate bare)
python sft/extract_sft_data_v5.py csqa qwen3_8b \
    --tier-strategy t1t2 \
    --balanced-strategy all_variants \
    --num-train all  # Use all available data

# Fair comparison with fixed total size
# Ensures consistent total dataset size for fair comparison between strategies.
# For t1 or t2: Total examples = num_train
# For t1t2: Total examples = num_train × 2 (each tier gets num_train)
python sft/extract_sft_data_v5.py csqa qwen3_8b \
    --tier-strategy t1 \
    --balanced-strategy bare_100 \
    --num-train 2000 \
    --fixed-total-size  # Results in exactly 2000 examples (2000 bare)
"""

import json
import os
import argparse
import random
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.exp_name import parse_experiment_name

# Anchor paths to the repo root (parent of sft/) so paths work regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / 'results'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'sft' / 'results'

# Define friendly name mappings
FRIENDLY_STRATEGIES = {
    'bare_100': 'bare_100_upos_0_uneg_0_dpos_0_dneg_0_dpup_0_updp_0_dnun_0_undn_0_dpun_0_dnup_0_updn_0_undp_0',
    'no_bare_ss': 'bare_0_upos_30_uneg_20_dpos_30_dneg_20_dpup_0_updp_0_dnun_0_undn_0_dpun_0_dnup_0_updn_0_undp_0',
    'half_bare_ss': 'bare_50_upos_15_uneg_10_dpos_15_dneg_10_dpup_0_updp_0_dnun_0_undn_0_dpun_0_dnup_0_updn_0_undp_0',
    'all_variants': 'bare_30_upos_10_uneg_5_dpos_10_dneg_5_dpup_5_updp_5_dnun_5_undn_5_dpun_5_dnup_5_updn_5_undp_5',
    'agreement_ds': 'bare_30_upos_15_uneg_10_dpos_15_dneg_10_dpup_5_updp_5_dnun_5_undn_5_dpun_0_dnup_0_updn_0_undp_0',
}

ALL_VARIANTS = ['bare', 'upos', 'uneg', 'dpos', 'dneg', 'dpup', 'updp',
                'dnun', 'undn', 'dpun', 'dnup', 'updn', 'undp']

def load_merged_results(exp_name: str) -> List[Dict[str, Any]]:
    """Load merged results for an experiment."""
    results_path = str(RESULTS_DIR / exp_name / 'merged_results.jsonl')

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")

    results = []
    with open(results_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    return results

def resolve_strategy_name(strategy: str) -> str:
    """Resolve friendly name to full strategy string if applicable."""
    if strategy in FRIENDLY_STRATEGIES:
        print(f"Using friendly strategy '{strategy}'")
        return FRIENDLY_STRATEGIES[strategy]
    return strategy

def parse_balanced_strategy(strategy: str) -> Tuple[Dict[str, float], bool]:
    """Parse balanced strategy string to extract and normalize percentages.

    Returns:
        Tuple of (normalized percentages dict, is_valid)

    Format: bare_A_upos_B_uneg_C_dpos_D_dneg_E_dpup_F_updp_G_dnun_H_undn_I_dpun_J_dnup_K_updn_L_undp_M
    The values are treated as relative weights and will be normalized to sum to 100%.
    """
    # Resolve friendly name if applicable
    strategy = resolve_strategy_name(strategy)

    # Define the expected order of components
    components = ['bare', 'upos', 'uneg', 'dpos', 'dneg', 'dpup', 'updp', 'dnun', 'undn', 'dpun', 'dnup', 'updn', 'undp']

    # Build pattern for parsing
    pattern_parts = []
    for comp in components:
        pattern_parts.append(f'{comp}_(\\d+(?:\\.\\d+)?)')
    pattern = '^' + '_'.join(pattern_parts) + '$'

    match = re.match(pattern, strategy)
    if not match:
        return {}, False

    # Extract raw values
    raw_values = {}
    for i, comp in enumerate(components):
        raw_values[comp] = float(match.group(i + 1))

    # Calculate total for normalization
    total = sum(raw_values.values())
    if total == 0:
        print(f"Error: All percentages are zero")
        return {}, False

    # Normalize to sum to 100%
    percentages = {}
    for comp, value in raw_values.items():
        percentages[comp] = (value / total) * 100.0

    # Print normalization info if total wasn't 100
    if abs(total - 100.0) > 0.1:
        print(f"Note: Normalizing percentages from total {total:.1f} to 100%")

    return percentages, True

def derive_experiment_names(dataset: str, model: str, tier_strategy: str) -> List[str]:
    """Derive all experiment names needed based on tier strategy.

    For v5:
    - t1: Only tier 1 experiments (d1nu1nin, u1nd1nin)
    - t2: Only tier 2 experiments (d2nu2nin, u2nd2nin)
    - t1t2: Both tier 1 and tier 2 experiments
    """
    exp_names = []

    if tier_strategy == 't1':
        exp_names.append(f"{dataset}__{model}__d1nu1nin__nocot")
        exp_names.append(f"{dataset}__{model}__u1nd1nin__nocot")

    elif tier_strategy == 't2':
        exp_names.append(f"{dataset}__{model}__d2nu2nin__nocot")
        exp_names.append(f"{dataset}__{model}__u2nd2nin__nocot")

    elif tier_strategy == 't1t2':
        exp_names.append(f"{dataset}__{model}__d1nu1nin__nocot")
        exp_names.append(f"{dataset}__{model}__u1nd1nin__nocot")
        exp_names.append(f"{dataset}__{model}__d2nu2nin__nocot")
        exp_names.append(f"{dataset}__{model}__u2nd2nin__nocot")

    elif tier_strategy == 't0':
        # For t0, just use d1nu1nin as default
        exp_names.append(f"{dataset}__{model}__d1nu1nin__nocot")

    return exp_names

def verify_required_probes_exist(exp_names: List[str], percentages: Dict[str, float]):
    """Verify all required experiments and probe variants exist."""
    missing_experiments = []
    missing_probes = []

    doc_first_variants = ['bare', 'upos', 'dpos', 'uneg', 'dneg', 'dpup', 'dnun', 'dpun', 'dnup']
    user_first_variants = ['updp', 'undn', 'updn', 'undp']

    for exp_name in exp_names:
        results_path = str(RESULTS_DIR / exp_name / 'merged_results.jsonl')
        if not os.path.exists(results_path):
            missing_experiments.append(exp_name)
            continue

        results = load_merged_results(exp_name)
        available_variants = set(item['probe_variant'] for item in results)

        if 'd1nu1nin' in exp_name or 'd2nu2nin' in exp_name:
            expected_variants = doc_first_variants
        elif 'u1nd1nin' in exp_name or 'u2nd2nin' in exp_name:
            expected_variants = user_first_variants
        else:
            expected_variants = [v for v in percentages.keys() if percentages[v] > 0]

        for variant in expected_variants:
            if percentages.get(variant, 0) > 0 and variant not in available_variants:
                missing_probes.append(f"{exp_name}/{variant}")

    if missing_experiments or missing_probes:
        print("ERROR: Missing required resources:")
        if missing_experiments:
            print("\nMissing experiments:")
            for exp in missing_experiments:
                print(f"  - {exp}")
        if missing_probes:
            print("\nMissing probe variants:")
            for probe in missing_probes:
                print(f"  - {probe}")
        raise FileNotFoundError(f"Missing {len(missing_experiments)} experiments and {len(missing_probes)} probe variants")

def load_and_merge_results(exp_names: List[str], tier_strategy: str, num_train: str, seed: int = 42) -> List[Dict[str, Any]]:
    """Load and merge results from multiple experiments with optional initial QID sampling.

    Args:
        exp_names: List of experiment names to load
        tier_strategy: The tier strategy being used
        num_train: Number of initial QIDs to sample ('all' or a number)
        seed: Random seed for QID sampling
    """
    # Pick a doc-first experiment to enumerate the QID universe
    base_exp = None
    for exp_name in exp_names:
        if 'd1nu1nin' in exp_name or 'd2nu2nin' in exp_name:
            base_exp = exp_name
            break
    if base_exp is None:
        base_exp = exp_names[0]

    results_path = str(RESULTS_DIR / base_exp / 'merged_results.jsonl')
    all_qids = set()
    with open(results_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['probe_variant'] == 'bare':
                all_qids.add(item['qid'])

    total_available = len(all_qids)
    print(f"\nDataset analysis:")
    print(f"  Total QIDs available: {total_available}")

    if num_train == 'all':
        num_samples = total_available
        print(f"  Requested: all {num_samples} samples")
    else:
        num_samples = int(num_train)
        print(f"  Requested: {num_samples} samples")

    selected_qids = None
    if num_train != 'all':
        print(f"\nSelecting {num_samples} QIDs randomly using seed {seed}...")
        sorted_qids = sorted(list(all_qids))
        if len(sorted_qids) > num_samples:
            rng = random.Random(seed)
            selected_qids = set(rng.sample(sorted_qids, num_samples))
            print(f"  Selected {num_samples} QIDs from {len(sorted_qids)} total")
        else:
            selected_qids = set(sorted_qids)
            print(f"  Using all {len(sorted_qids)} available QIDs (less than requested {num_samples})")

    print(f"\nLoading experiment data...")
    all_results = []

    for exp_name in exp_names:
        if selected_qids is not None:
            results = []
            with open(RESULTS_DIR / exp_name / 'merged_results.jsonl', 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if item['qid'] in selected_qids:
                        results.append(item)
        else:
            results = load_merged_results(exp_name)

        if 'd1nu1nin' in exp_name:
            tier_config = 'd1nu1nin'
            tier = 'tier1'
        elif 'u1nd1nin' in exp_name:
            tier_config = 'u1nd1nin'
            tier = 'tier1'
        elif 'd2nu2nin' in exp_name:
            tier_config = 'd2nu2nin'
            tier = 'tier2'
        elif 'u2nd2nin' in exp_name:
            tier_config = 'u2nd2nin'
            tier = 'tier2'
        else:
            tier_config = exp_name.split('__')[2]
            tier = 'unknown'

        for item in results:
            item['sft_strategy'] = f'tier_{tier_strategy}'
            item['tier_config'] = tier_config
            item['tier'] = tier
            item['source_experiment'] = exp_name

        all_results.extend(results)

        if selected_qids is not None:
            print(f"  {exp_name}: loaded {len(results)} items (filtered from {num_samples} QIDs)")
        else:
            print(f"  {exp_name}: loaded {len(results)} items")

    return all_results

def apply_balanced_strategy_v5(all_results: List[Dict[str, Any]], strategy: str, seed: int,
                               tier_strategy: str = None,
                               fixed_total_size: bool = False, num_train: str = None) -> List[Dict[str, Any]]:
    """Apply balanced strategy with v5 tier separation logic.

    For v5:
    - t1: Sample from tier1 sources only
    - t2: Sample from tier2 sources only
    - t1t2: Sample from both tiers separately, then combine (includes duplicate bare)
    """
    percentages, is_valid = parse_balanced_strategy(strategy)
    if not is_valid:
        raise ValueError(f"Invalid balanced strategy: {strategy}")

    print(f"\nUsing seed: {seed} (deterministic, no hash offset)")

    print(f"\nApplying strategy with percentages:")
    for variant, pct in percentages.items():
        if pct > 0:
            print(f"  {variant}: {pct:.1f}%")

    if tier_strategy == 't1':
        candidates = sample_tier(all_results, percentages, 'tier1', 'd1nu1nin', 'u1nd1nin',
                                seed, fixed_total_size, num_train)

    elif tier_strategy == 't2':
        candidates = sample_tier(all_results, percentages, 'tier2', 'd2nu2nin', 'u2nd2nin',
                                seed, fixed_total_size, num_train)

    elif tier_strategy == 't1t2':
        print("\n=== Sampling from Tier 1 ===")
        # Each tier gets the full num_train amount; total = num_train * 2
        tier1_candidates = sample_tier(all_results, percentages, 'tier1', 'd1nu1nin', 'u1nd1nin',
                                      seed, fixed_total_size, num_train)

        print("\n=== Sampling from Tier 2 ===")
        # Use SAME seed for tier2 to ensure same QIDs are selected
        tier2_candidates = sample_tier(all_results, percentages, 'tier2', 'd2nu2nin', 'u2nd2nin',
                                      seed, fixed_total_size, num_train)

        candidates = tier1_candidates + tier2_candidates

        print(f"\n=== Combined Results ===")
        print(f"Tier 1 contributed: {len(tier1_candidates)} samples")
        print(f"Tier 2 contributed: {len(tier2_candidates)} samples")

    else:
        # t0 or other - use all results
        candidates = sample_tier(all_results, percentages, None, None, None,
                                seed, fixed_total_size, num_train)

    # Shuffle the final dataset
    random.seed(seed + 100)
    random.shuffle(candidates)

    # Update metadata
    for item in candidates:
        item['balanced_strategy'] = strategy
        item['sampling_seed'] = seed

    # Print final distribution
    print(f"\nFinal distribution ({len(candidates)} total):")
    variant_counts = {}
    tier_counts = {}
    detailed_breakdown = {}

    for item in candidates:
        variant = item['probe_variant']
        tier_config = item.get('tier_config', 'unknown')

        variant_counts[variant] = variant_counts.get(variant, 0) + 1
        tier_counts[tier_config] = tier_counts.get(tier_config, 0) + 1

        if variant not in detailed_breakdown:
            detailed_breakdown[variant] = {}
        if tier_config not in detailed_breakdown[variant]:
            detailed_breakdown[variant][tier_config] = 0
        detailed_breakdown[variant][tier_config] += 1

    for variant in ALL_VARIANTS:
        if variant in variant_counts:
            count = variant_counts[variant]
            pct = (count / len(candidates)) * 100
            print(f"  {variant}: {count} ({pct:.1f}%)")

    print(f"\nBy tier configuration:")
    for config, count in sorted(tier_counts.items()):
        print(f"  {config}: {count}")

    print(f"\nDetailed breakdown (variant -> source):")
    for variant in ALL_VARIANTS:
        if variant in detailed_breakdown:
            print(f"  {variant}:")
            for tier_config, count in sorted(detailed_breakdown[variant].items()):
                print(f"    from {tier_config}: {count}")

    return candidates

def sample_tier(all_results: List[Dict[str, Any]], percentages: Dict[str, float],
                tier: str, doc_first_config: str, user_first_config: str,
                seed: int, fixed_total_size: bool = False,
                num_train: str = None) -> List[Dict[str, Any]]:
    """Sample from a specific tier.

    For each tier:
    - bare: from doc-first config (d1nu1nin or d2nu2nin)
    - upos, dpos: from doc-first config (single-source positive)
    - uneg, dneg: from doc-first config (single-source negative)
    - dpup, dpun, dnup, dnun: from doc-first config (doc-first double-source)
    - updp, updn, undp, undn: from user-first config (user-first double-source)
    """
    # Build variant pools
    variant_pools = {}

    if tier is not None:
        tier_results = [r for r in all_results if r.get('tier') == tier]

        # Bare and single-source from doc-first
        for variant in ['bare', 'upos', 'dpos', 'uneg', 'dneg']:
            variant_pools[variant] = [item for item in tier_results
                                     if item['probe_variant'] == variant
                                     and item['tier_config'] == doc_first_config]

        # Doc-first double-source
        for variant in ['dpup', 'dnun', 'dpun', 'dnup']:
            variant_pools[variant] = [item for item in tier_results
                                     if item['probe_variant'] == variant
                                     and item['tier_config'] == doc_first_config]

        # User-first double-source
        for variant in ['updp', 'undn', 'updn', 'undp']:
            variant_pools[variant] = [item for item in tier_results
                                     if item['probe_variant'] == variant
                                     and item['tier_config'] == user_first_config]
    else:
        # For t0, use all results
        for variant in ALL_VARIANTS:
            variant_pools[variant] = [item for item in all_results
                                     if item['probe_variant'] == variant]

    # Capacity per variant
    variant_capacities = {}
    for variant, pct in percentages.items():
        if pct > 0 and variant in variant_pools:
            available = len(variant_pools[variant])
            if available > 0:
                max_dataset_size = int(available / (pct / 100))
                variant_capacities[variant] = (available, max_dataset_size)

    # Determine final dataset size
    if fixed_total_size and num_train != 'all':
        # Fixed size based on num_train: use directly as the total dataset size for this tier
        T_final = int(num_train)
        constraining_variant = 'fixed_size'
    else:
        # Most-constrained-variant mode
        min_dataset_size = float('inf')
        constraining_variant = None
        for variant, (avail, max_size) in variant_capacities.items():
            if max_size < min_dataset_size:
                min_dataset_size = max_size
                constraining_variant = variant
        T_final = min_dataset_size if min_dataset_size != float('inf') else 0

    if tier:
        print(f"\nSampling from {tier} (doc-first: {doc_first_config}, user-first: {user_first_config}):")
    else:
        print(f"\nSampling from all tiers:")

    print(f"  Variant capacities:")
    for variant, (avail, max_size) in sorted(variant_capacities.items()):
        print(f"    {variant} ({percentages.get(variant, 0)}%): available={avail}, max_dataset={max_size}")
    print(f"  Most constrained variant: {constraining_variant}")
    print(f"  Target dataset size: {T_final}")

    # Sample each variant from its sorted pool
    candidates = []
    for i, variant in enumerate(ALL_VARIANTS):
        target = int((percentages.get(variant, 0) / 100) * T_final)
        if target > 0 and variant in variant_pools:
            # seed + i: different variants get different samples,
            # but the same variant is consistent across tiers
            rng = random.Random(seed + i)
            pool_sorted = sorted(variant_pools[variant], key=lambda x: x['qid'])
            variant_sample = rng.sample(pool_sorted, min(target, len(pool_sorted)))
            candidates.extend(variant_sample)
            print(f"  {variant}: sampled {len(variant_sample)} (target {target})")

    return candidates

def save_candidates(candidates: List[Dict[str, Any]], output_path: str):
    """Save candidates to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for item in candidates:
            f.write(json.dumps(item) + '\n')

    print(f"\nSaved {len(candidates)} candidates to {output_path}")

def print_statistics(candidates: List[Dict[str, Any]], tier_strategy: str, balanced_strategy: str):
    """Print statistics about extracted candidates."""
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Tier strategy: {tier_strategy}")
    print(f"Balanced strategy: {balanced_strategy[:50]}...")
    print(f"Total candidates: {len(candidates)}")

def validate_balanced_strategy(strategy: str) -> bool:
    """Validate balanced strategy format."""
    if strategy in FRIENDLY_STRATEGIES:
        return True
    percentages, is_valid = parse_balanced_strategy(strategy)
    return is_valid

def main():
    parser = argparse.ArgumentParser(
        description='Extract balanced SFT training data (v5 with tier separation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy Formats:
  Balanced:
    1. Full format: bare_A_upos_B_uneg_C_dpos_D_dneg_E_dpup_F_updp_G_dnun_H_undn_I_dpun_J_dnup_K_updn_L_undp_M
       where A-M are relative weights (can be 0) that will be normalized to sum to 100%

    2. Friendly names:
       - bare_100: 100% bare examples
       - no_bare_ss: No bare, single-source only (30% upos, 20% uneg, 30% dpos, 20% dneg)
       - half_bare_ss: 50% bare, 50% single-source (15% upos, 10% uneg, 15% dpos, 10% dneg)
       - all_variants: All 13 variants (30% bare, then distributed across all others)
       - agreement_ds: Agreement double-source (30% bare, plus trust and skepticism variants)

Supported datasets: csqa, gsm8k
Supported models: qwen3_8b, llama3_8b_instruct

Examples:
  # For t1: uses only tier1 sources
  python extract_sft_data_v5.py csqa qwen3_8b \\
      --tier-strategy t1 \\
      --balanced-strategy half_bare_ss \\
      --num-train 2000

  # For t1t2: samples from both tiers separately (includes duplicate bare)
  python extract_sft_data_v5.py csqa qwen3_8b \\
      --tier-strategy t1t2 \\
      --balanced-strategy all_variants \\
      --num-train all
        """
    )

    parser.add_argument('dataset',
                       choices=['csqa', 'gsm8k'],
                       help='Dataset name')
    parser.add_argument('model',
                       choices=['qwen3_8b', 'llama3_8b_instruct'],
                       help='Model name')
    parser.add_argument('--tier-strategy',
                       choices=['t0', 't1', 't2', 't1t2'],
                       required=True,
                       help='Tier strategy: t0 (single), t1 (tier 1), t2 (tier 2), t1t2 (both tiers)')
    parser.add_argument('--balanced-strategy',
                       required=True,
                       help='Balanced strategy: friendly name or full format')
    parser.add_argument('--seed',
                       type=int,
                       default=42,
                       help='Random seed for sampling (default: 42, deterministic)')
    parser.add_argument('--num-train',
                       default='all',
                       help='Number of training samples to use (default: all). Use "all" or a number like 2000')
    parser.add_argument('--fixed-total-size',
                       action='store_true',
                       help='Use fixed total dataset size based on num-train (e.g., 4000 for t1 with num-train=2000) instead of adjusting based on variant availability. Ensures fair comparison between balanced strategies.')

    args = parser.parse_args()

    # Validate num-train parameter
    if args.num_train != 'all':
        try:
            num_train_val = int(args.num_train)
            if num_train_val <= 0:
                print(f"Error: --num-train must be positive, got {num_train_val}")
                sys.exit(1)
        except ValueError:
            print(f"Error: --num-train must be 'all' or a positive integer, got '{args.num_train}'")
            sys.exit(1)

    # Validate balanced strategy
    if not validate_balanced_strategy(args.balanced_strategy):
        print(f"Error: Invalid balanced strategy: {args.balanced_strategy}")
        print("Use a friendly name (bare_100, no_bare_ss, half_bare_ss, all_variants, agreement_ds)")
        print("or full format: bare_A_upos_B_uneg_C_dpos_D_dneg_E_dpup_F_updp_G_dnun_H_undn_I_dpun_J_dnup_K_updn_L_undp_M")
        sys.exit(1)

    # Resolve strategy to full format
    full_strategy = resolve_strategy_name(args.balanced_strategy)

    percentages, is_valid = parse_balanced_strategy(full_strategy)
    if not is_valid:
        print(f"Error: Strategy validation failed")
        sys.exit(1)

    try:
        exp_names = derive_experiment_names(args.dataset, args.model, args.tier_strategy)
        print(f"\nTier strategy '{args.tier_strategy}' requires:")
        for exp in exp_names:
            print(f"  - {exp}")

        print("\nVerifying all required probe variants exist...")
        verify_required_probes_exist(exp_names, percentages)
        print("All required resources found!")

        all_results = load_and_merge_results(exp_names, args.tier_strategy, args.num_train, args.seed)
        print(f"Loaded {len(all_results)} total results from {len(exp_names)} experiments")

        candidates = apply_balanced_strategy_v5(
            all_results,
            full_strategy,
            args.seed,
            args.tier_strategy,
            args.fixed_total_size,
            args.num_train
        )

        # Save results to appropriate folder
        output_dir = os.environ.get('OUTPUT_DIR', str(DEFAULT_OUTPUT_DIR))

        # Use friendly name in filename if applicable
        strategy_for_filename = args.balanced_strategy if len(args.balanced_strategy) < 50 else args.balanced_strategy[:47] + "..."

        # Include num_train in filename. If 'all', use actual dataset size.
        if args.num_train == 'all':
            dataset_sizes = {'csqa': 6958, 'gsm8k': 5227}
            num_train_str = f"n{dataset_sizes[args.dataset]}"
        else:
            num_train_str = f"n{args.num_train}"

        # 'f' tag for fixed total size
        version_str = "v5_f" if args.fixed_total_size else "v5"

        output_filename = (
            f"{args.dataset}__{args.model}__"
            f"{version_str}_{num_train_str}_{args.tier_strategy}_{strategy_for_filename}_candidates.jsonl"
        )
        output_path = f'{output_dir}/{output_filename}'
        save_candidates(candidates, output_path)

        print_statistics(candidates, args.tier_strategy, args.balanced_strategy)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
