#!/usr/bin/env python3
"""
distribution_metrics_analysis.py
================================
Compute distribution-level metrics comparing all probe variants against bare baseline:
- KL divergence from bare
- Entropy (for each probe variant)
- Pearson correlation between log-prob vectors
- -logP(correct) metrics (absolute values and deltas from bare)


Output (will be overwritten if it exists)
-----------------------------------------
results/<exp-name>/distribution_metrics.json
results/<exp-name>/distribution_metrics_detailed.csv
results/<exp-name>/neg_logp_correct_absolute.csv
results/<exp-name>/entropy_absolute.csv

Note: This script will OVERWRITE existing output files each time it runs.

Usage:
------
python analyze/distribution_metrics_analysis.py --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot
python analyze/distribution_metrics_analysis.py --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot --results-dir results_hf_test
"""

import argparse
import json
import pandas as pd
import numpy as np
from scipy.stats import entropy, pearsonr, bootstrap
import pathlib
import warnings
import sys
from typing import Dict, List, Tuple, Any

# ─── make utils import-able ───────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.exp_name import parse_experiment_name

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="BCa interval may be inaccurate")


def find_merged_results(exp_name: str, results_dir: pathlib.Path = None) -> pathlib.Path:
    """Find merged_results.jsonl for given experiment"""
    base_dir = results_dir if results_dir else ROOT / "results"
    
    p = base_dir / exp_name / "merged_results.jsonl"
    if not p.exists():
        sys.exit(f"[ERR] missing file: {p}")
    return p


def load_data(merged_path: pathlib.Path) -> pd.DataFrame:
    """Load merged results data"""
    print(f"Loading data from: {merged_path}")
    df = pd.read_json(merged_path, lines=True)
    print(f"Loaded {len(df):,} rows")
    
    # Check for required columns
    required_cols = ['qid', 'probe_variant', 'probs', 'gold']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for required probe variants
    available_probes = df['probe_variant'].unique()
    print(f"Available probe variants: {sorted(available_probes)}")
    
    if 'bare' not in available_probes:
        raise ValueError("Missing 'bare' baseline probe variant")
    
    # Load canonical wrong answers
    results_dir = merged_path.parent
    canonical_wrong_path = results_dir / "canonical_wrong.jsonl"
    if not canonical_wrong_path.exists():
        raise ValueError(f"Missing canonical_wrong.jsonl file: {canonical_wrong_path}")
    
    canonical_wrong_map = {}
    with open(canonical_wrong_path) as f:
        for line in f:
            data = json.loads(line)
            canonical_wrong_map[data["qid"]] = data["canonical_wrong"]
    
    df['canonical_wrong'] = df['qid'].map(canonical_wrong_map)
    print(f"Loaded canonical wrong answers for {len(canonical_wrong_map)} questions")
    
    return df


def extract_prob_vectors(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """Extract probability and log-probability vectors from probs field.
    
    Normalizes the probability vectors to have consistent positions:
    - Index 0: gold answer probability
    - Index 1: canonical wrong answer probability  
    - Index 2: sum of all other answer probabilities
    """
    # Parse experiment to check if it's an OpenAI model
    exp = parse_experiment_name(experiment_name)
    is_openai_model = exp.model_key in ['gpt_4o', 'gpt_4o_mini']
    
    # Filter rows with valid probs data
    df = df[df['probs'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    
    def compute_log_probs(probs_list):
        """Compute log probabilities from logits or use logp directly"""
        # Check if we have logp (OpenAI) or logit (local models)
        if is_openai_model or 'logp' in probs_list[0]:
            # OpenAI models provide log probabilities directly
            log_probs = np.array([item['logp'] for item in probs_list])
        else:
            # Local models provide logits, compute log probabilities
            logits = np.array([item['logit'] for item in probs_list])
            
            # Compute log probabilities using log-sum-exp trick for numerical stability
            max_logit = np.max(logits)
            log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
            log_probs = logits - log_sum_exp
        
        return log_probs
    
    def normalize_prob_vector(row):
        """Normalize probability vector for a single row"""
        probs_list = row['probs']
        gold_letter = row['gold']
        canonical_wrong_letter = row['canonical_wrong']
        
        # Compute log probabilities from logits
        log_probs = compute_log_probs(probs_list)
        
        # Find indices for gold and canonical wrong
        gold_idx = None
        canonical_wrong_idx = None
        other_indices = []
        
        for i, item in enumerate(probs_list):
            letter = item['letter']
            if letter == gold_letter:
                gold_idx = i
            elif letter == canonical_wrong_letter:
                canonical_wrong_idx = i
            else:
                other_indices.append(i)
        
        if gold_idx is None:
            raise ValueError(f"Gold answer '{gold_letter}' not found in probs for qid {row['qid']}")
        if canonical_wrong_idx is None:
            raise ValueError(f"Canonical wrong answer '{canonical_wrong_letter}' not found in probs for qid {row['qid']}")
        
        # Extract probabilities and log probabilities
        gold_prob = probs_list[gold_idx]['prob']
        gold_logp = log_probs[gold_idx]
        
        canonical_wrong_prob = probs_list[canonical_wrong_idx]['prob']
        canonical_wrong_logp = log_probs[canonical_wrong_idx]
        
        # Sum probabilities for all other answers
        other_prob = sum(probs_list[i]['prob'] for i in other_indices)
        # For log probs, we need to convert back to probs, sum, then take log
        if other_prob > 0:
            other_logp = np.log(other_prob)
        else:
            other_logp = -np.inf
        
        # Create normalized vectors
        prob_vector = np.array([gold_prob, canonical_wrong_prob, other_prob])
        logp_vector = np.array([gold_logp, canonical_wrong_logp, other_logp])
        
        return prob_vector, logp_vector
    
    # Vectorized processing for better performance
    print("Normalizing probability vectors...")
    
    # Use list comprehension for better performance
    results = [normalize_prob_vector(row) for _, row in df.iterrows()]
    
    df['prob_vector'] = [r[0] for r in results]
    df['logp_vector'] = [r[1] for r in results]
    
    print("Normalized probability vectors to consistent positions:")
    print("  - Index 0: gold answer")
    print("  - Index 1: canonical wrong answer")
    print("  - Index 2: sum of other answers")
    
    return df


def compute_metrics_for_pair(bare_probs: np.ndarray, bare_logps: np.ndarray,
                           probe_probs: np.ndarray, probe_logps: np.ndarray,
                           epsilon: float = 1e-9) -> Dict[str, float]:
    """Compute distribution metrics between bare and probe distributions"""
    # Add epsilon and normalize to ensure valid probability distributions
    bare_probs = bare_probs + epsilon
    bare_probs = bare_probs / bare_probs.sum()
    
    probe_probs = probe_probs + epsilon
    probe_probs = probe_probs / probe_probs.sum()
    
    # KL divergence: D_KL(probe || bare)
    kl_div = entropy(probe_probs, bare_probs, base=2)
    
    # Entropy of each distribution
    h_bare = entropy(bare_probs, base=2)
    h_probe = entropy(probe_probs, base=2)
    entropy_delta = h_probe - h_bare
    
    # Pearson correlation between log-prob vectors
    # Use computed log-probs for consistency
    logp_bare_calc = np.log2(bare_probs)
    logp_probe_calc = np.log2(probe_probs)
    
    # Only compute correlation if there's variance
    if np.std(logp_bare_calc) > 1e-6 and np.std(logp_probe_calc) > 1e-6:
        corr, _ = pearsonr(logp_bare_calc, logp_probe_calc)
    else:
        corr = np.nan
    
    # -logP(correct) metrics
    # Index 0 is the gold answer probability
    neg_logp_bare = -np.log2(bare_probs[0])
    neg_logp_probe = -np.log2(probe_probs[0])
    neg_logp_delta = neg_logp_probe - neg_logp_bare  # positive = worse, negative = better
    
    return {
        'kl_divergence': kl_div,
        'entropy_bare': h_bare,
        'entropy_probe': h_probe,
        'entropy_delta': entropy_delta,
        'pearson_correlation': corr,
        'neg_logp_bare': neg_logp_bare,
        'neg_logp_probe': neg_logp_probe,
        'neg_logp_delta': neg_logp_delta
    }


def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics for all probe variants compared to bare"""
    results = []
    absolute_neg_logp_results = []  # Store absolute -logP(correct) for all probes
    absolute_entropy_results = []  # Store absolute entropy for all probes

    # Get unique questions
    questions = df['qid'].unique()
    all_probe_variants = df['probe_variant'].unique()  # Include bare
    probe_variants = [p for p in all_probe_variants if p != 'bare']
    
    # Pre-group data by qid and probe_variant for faster access
    grouped_by_qid_probe = df.groupby(['qid', 'probe_variant'])
    
    # Process in batches for better performance
    print(f"Processing {len(questions)} questions...")
    
    # Vectorized computation for absolute metrics
    epsilon = 1e-9
    
    # Process all absolute metrics at once
    for _, row in df.iterrows():
        probe_probs = row['prob_vector'] + epsilon
        probe_probs = probe_probs / probe_probs.sum()
        
        # -logP(correct) - Index 0 is gold answer
        neg_logp = -np.log2(probe_probs[0])
        
        # Absolute entropy
        h_probe = entropy(probe_probs, base=2)
        
        absolute_neg_logp_results.append({
            'qid': row['qid'],
            'probe_variant': row['probe_variant'],
            'neg_logp_correct': neg_logp
        })
        
        absolute_entropy_results.append({
            'qid': row['qid'],
            'probe_variant': row['probe_variant'],
            'entropy': h_probe
        })
    
    # Create lookup dictionaries for faster access
    bare_data_dict = {}
    for _, row in df[df['probe_variant'] == 'bare'].iterrows():
        bare_data_dict[row['qid']] = {
            'prob_vector': row['prob_vector'],
            'logp_vector': row['logp_vector']
        }
    
    # Process comparisons
    for _, row in df[df['probe_variant'] != 'bare'].iterrows():
        qid = row['qid']
        probe = row['probe_variant']
        
        if qid not in bare_data_dict:
            print(f"Warning: No bare data for qid {qid}")
            continue
        
        bare_probs = bare_data_dict[qid]['prob_vector']
        bare_logps = bare_data_dict[qid]['logp_vector']
        probe_probs = row['prob_vector']
        probe_logps = row['logp_vector']
        
        # Compute metrics
        metrics = compute_metrics_for_pair(bare_probs, bare_logps, 
                                         probe_probs, probe_logps)
        
        results.append({
            'qid': qid,
            'probe_variant': probe,
            **metrics
        })
            
    
    results_df = pd.DataFrame(results)
    absolute_neg_logp_df = pd.DataFrame(absolute_neg_logp_results)
    absolute_entropy_df = pd.DataFrame(absolute_entropy_results)

    return results_df, absolute_neg_logp_df, absolute_entropy_df


def compute_ci(data: pd.Series, confidence_level: float = 0.95, 
               n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """
    Calculate mean and 95% confidence interval using bootstrap.
    
    This is the standard approach for NeurIPS papers when dealing with
    metrics computed across multiple data points (questions).
    
    Args:
        data: Series of metric values (e.g., KL divergences across questions)
        confidence_level: Confidence level for CI (default 0.95)
        n_bootstrap: Number of bootstrap samples (default 1000 for publication quality)
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
    # Clean data
    data = data.dropna()
    data = data[np.isfinite(data)]
    
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    
    mean_val = float(data.mean())
    
    if len(data) < 10:
        # Too few samples for meaningful CI
        return (mean_val, np.nan, np.nan)
    
    # Bootstrap resampling for CI
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    data_arr = data.values
    n = len(data_arr)
    
    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample_idx = rng.randint(0, n, size=n)
        resample_mean = data_arr[resample_idx].mean()
        bootstrap_means.append(resample_mean)
    
    # Calculate percentile CI
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(bootstrap_means, 100 * alpha)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha))
    
    return (mean_val, float(ci_lower), float(ci_upper))


def format_ci(mean: float, low: float, high: float, decimals: int = 4) -> str:
    """Format mean and CI for display"""
    if np.isnan(mean):
        return "N/A"
    if np.isnan(low) or np.isnan(high):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"


def main():
    parser = argparse.ArgumentParser(description="Compute distribution metrics with bare as baseline")
    parser.add_argument("--experiment_name", required=True, help="Experiment name")
    parser.add_argument("--results-dir", type=pathlib.Path, help="Custom results directory")
    args = parser.parse_args()
    
    # Find and load data
    merged_path = find_merged_results(args.experiment_name, args.results_dir)
    out_dir = merged_path.parent
    
    # Output file will be overwritten if it exists
    output_file = out_dir / "distribution_metrics.json"
    
    print(f"\n{'='*70}")
    print(f"Distribution Metrics Analysis")
    print(f"{'='*70}")
    print(f"Experiment: {args.experiment_name}")
    
    # Load and process data
    df = load_data(merged_path)
    df = extract_prob_vectors(df, args.experiment_name)
    
    # Calculate metrics
    print("\nCalculating distribution metrics...")
    results_df, absolute_neg_logp_df, absolute_entropy_df = calculate_all_metrics(df)
    
    if results_df.empty:
        print("No metrics could be calculated")
        return
    
    # Group by probe variant and calculate statistics
    grouped = results_df.groupby('probe_variant')
    
    # Calculate statistics with CIs for each metric
    metrics_stats = {}
    
    print("\nCalculating confidence intervals (bootstrap with 1000 resamples)...")
    # Use bootstrap CI - standard for NeurIPS
    for probe in sorted(grouped.groups.keys()):
        probe_data = grouped.get_group(probe)
        
        metrics_stats[probe] = {
            'kl_divergence': compute_ci(probe_data['kl_divergence']),
            'entropy_delta': compute_ci(probe_data['entropy_delta']),
            'pearson_correlation': compute_ci(probe_data['pearson_correlation']),
            'neg_logp_delta': compute_ci(probe_data['neg_logp_delta']),
            'n_samples': len(probe_data)
        }
    
    # Calculate absolute -logP(correct) statistics for all probes
    neg_logp_grouped = absolute_neg_logp_df.groupby('probe_variant')
    neg_logp_stats = {}
    for probe in sorted(neg_logp_grouped.groups.keys()):
        probe_data = neg_logp_grouped.get_group(probe)
        neg_logp_stats[probe] = compute_ci(probe_data['neg_logp_correct'])
    
    # Calculate absolute entropy statistics for all probes
    entropy_grouped = absolute_entropy_df.groupby('probe_variant')
    entropy_stats = {}
    for probe in sorted(entropy_grouped.groups.keys()):
        probe_data = entropy_grouped.get_group(probe)
        entropy_stats[probe] = compute_ci(probe_data['entropy'])
    
    # Display results
    print(f"\n{'─'*70}")
    print("Distribution Metrics vs Bare Baseline")
    print("(Using normalized positions: gold=0, canonical_wrong=1, others=2)")
    print(f"{'─'*70}")
    print(f"{'Probe':<10} {'KL Divergence':<30} {'Entropy Delta':<30} {'Pearson Corr':<30} {'-logP(correct) Δ':<30} {'N':<5}")
    print(f"{'─'*70}")
    
    for probe in sorted(metrics_stats.keys()):
        stats = metrics_stats[probe]
        kl_str = format_ci(*stats['kl_divergence'])
        ent_str = format_ci(*stats['entropy_delta'])
        corr_str = format_ci(*stats['pearson_correlation'])
        neg_logp_delta_str = format_ci(*stats['neg_logp_delta'])
        n = stats['n_samples']
        
        print(f"{probe:<10} {kl_str:<30} {ent_str:<30} {corr_str:<30} {neg_logp_delta_str:<30} {n:<5}")
    
    # Display absolute -logP(correct) for all probes
    print(f"\n{'─'*70}")
    print("Absolute -logP(correct) for All Probe Variants")
    print("(Lower is better - measures confidence in correct answer)")
    print(f"{'─'*70}")
    print(f"{'Probe':<10} {'-logP(correct)':<30}")
    print(f"{'─'*70}")
    
    # Sort probes to show bare first
    probe_order = ['bare'] + sorted([p for p in neg_logp_stats.keys() if p != 'bare'])
    for probe in probe_order:
        if probe in neg_logp_stats:
            neg_logp_str = format_ci(*neg_logp_stats[probe])
            print(f"{probe:<10} {neg_logp_str:<30}")
    
    # Display absolute entropy for all probes
    print(f"\n{'─'*70}")
    print("Absolute Entropy for All Probe Variants")
    print("(Higher is better - measures uncertainty/spread in distribution)")
    print(f"{'─'*70}")
    print(f"{'Probe':<10} {'Entropy':<30}")
    print(f"{'─'*70}")
    
    # Sort probes to show bare first
    for probe in probe_order:
        if probe in entropy_stats:
            entropy_str = format_ci(*entropy_stats[probe])
            print(f"{probe:<10} {entropy_str:<30}")
    
    print(f"\n{'─'*70}")
    print(f"Total comparisons: {len(results_df)}")
    
    # Prepare output data
    output_data = {
        'experiment': args.experiment_name,
        'n_questions': len(df['qid'].unique()),
        'probe_variants': sorted(metrics_stats.keys()),
        'normalization': {
            'description': 'Probability vectors normalized to consistent positions across questions',
            'positions': {
                '0': 'gold answer probability',
                '1': 'canonical wrong answer probability',
                '2': 'sum of all other answer probabilities'
            }
        },
        'probe_metrics': {},
        'neg_logp_correct_absolute': {},  # Absolute -logP(correct) for all probes
        'neg_logp_correct_delta': {},     # Delta from bare for other probes
        'entropy_absolute': {}             # Absolute entropy for all probes
    }
    
    # Add absolute -logP(correct) for all probes including bare
    for probe, stats in neg_logp_stats.items():
        output_data['neg_logp_correct_absolute'][probe] = {
            'mean': stats[0],
            'ci_low': stats[1],
            'ci_high': stats[2]
        }
    
    # Add absolute entropy for all probes including bare
    for probe, stats in entropy_stats.items():
        output_data['entropy_absolute'][probe] = {
            'mean': stats[0],
            'ci_low': stats[1],
            'ci_high': stats[2]
        }
    
    # Add delta metrics for comparison probes
    for probe, stats in metrics_stats.items():
        output_data['probe_metrics'][probe] = {
            'kl_divergence': {
                'mean': stats['kl_divergence'][0],
                'ci_low': stats['kl_divergence'][1],
                'ci_high': stats['kl_divergence'][2]
            },
            'entropy_delta': {
                'mean': stats['entropy_delta'][0],
                'ci_low': stats['entropy_delta'][1],
                'ci_high': stats['entropy_delta'][2]
            },
            'pearson_correlation': {
                'mean': stats['pearson_correlation'][0],
                'ci_low': stats['pearson_correlation'][1],
                'ci_high': stats['pearson_correlation'][2]
            },
            'neg_logp_delta': {
                'mean': stats['neg_logp_delta'][0],
                'ci_low': stats['neg_logp_delta'][1],
                'ci_high': stats['neg_logp_delta'][2]
            },
            'n_samples': stats['n_samples']
        }
        
        # Also add to neg_logp_correct_delta (same as neg_logp_delta in probe_metrics)
        output_data['neg_logp_correct_delta'][probe] = {
            'mean': stats['neg_logp_delta'][0],
            'ci_low': stats['neg_logp_delta'][1],
            'ci_high': stats['neg_logp_delta'][2]
        }
    
    # Save detailed results
    results_df.to_csv(out_dir / "distribution_metrics_detailed.csv", index=False)
    
    # Save absolute -logP(correct) values
    absolute_neg_logp_df.to_csv(out_dir / "neg_logp_correct_absolute.csv", index=False)
    
    # Save absolute entropy values
    absolute_entropy_df.to_csv(out_dir / "entropy_absolute.csv", index=False)

    # Save summary JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    
    print(f"\n✓ Saved distribution_metrics.json in {out_dir}")
    print(f"✓ Saved distribution_metrics_detailed.csv in {out_dir}")
    print(f"✓ Saved neg_logp_correct_absolute.csv in {out_dir}")
    print(f"✓ Saved entropy_absolute.csv in {out_dir}")


if __name__ == "__main__":
    main()