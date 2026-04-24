#!/usr/bin/env python3
"""
generate_core_metrics_report.py
===============================
Generate focused reports with core metrics from all experiments.

This script creates 4 CSV files:
1. all_core_metrics.csv - All metrics combined
2. logistic_core_metrics.csv - Logistic regression metrics only (including accuracy and UNKNOWN rates)
3. choice_core_metrics.csv - Choice metrics only
4. distribution_core_metrics.csv - Distribution metrics only

Confidence Intervals:
- Logistic regression: Cluster-robust SEs (clustered by qid)
- Distribution metrics: Bootstrap with 1000 resamples
- Choice metrics: No CIs (direct proportions from full dataset)

Usage:
------
python analyze/generate_core_metrics_report.py
python analyze/generate_core_metrics_report.py --output-dir analyze/exp_results --results-dir results
"""

import argparse
import json
import pandas as pd
import pathlib
import sys
from typing import Dict, Any, List

# Make imports available
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from core.exp_name import parse_experiment_name, ExpName

# Probe variants for accuracy and distribution metrics
# All 13 probe variants (bare + 12 probes)
ALL_PROBE_VARIANTS = ['bare', 'upos', 'uneg', 'dpos', 'dneg', 'dpup', 'dpun', 'dnup', 'dnun', 'updp', 'updn', 'undp', 'undn']
# Accuracy variants - now includes all 13 probes (including user-first variants)
ACCURACY_VARIANTS = ['bare', 'upos', 'uneg', 'dpos', 'dneg', 'dpup', 'dpun', 'dnup', 'dnun', 'updp', 'updn', 'undp', 'undn']
# Distribution variants - all probes except bare
DISTRIBUTION_VARIANTS = ['upos', 'uneg', 'dpos', 'dneg', 'dpup', 'dpun', 'dnup', 'dnun', 'updp', 'updn', 'undp', 'undn']
CHOICE_VARIANTS = ['upos', 'uneg', 'dpos', 'dneg']

# Odds ratio fields to keep
ODDS_RATIO_FIELDS = [
    'intercept',
    'βP_parametric_correctness', 
    'δU_user_display_effect',
    'δD_doc_display_effect',
    'βU_user_correct_boost',
    'βD_doc_correct_boost',
    'user_correct_composite',
    'doc_correct_composite',
    'user_selectivity',
    'doc_selectivity'
]


def load_json_safely(filepath: pathlib.Path) -> Dict[str, Any] | None:
    """Load JSON file safely, returning None if file doesn't exist or is invalid."""
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"Warning: Failed to load {filepath}")
        return None


def extract_exp_name_fields(exp_name: ExpName) -> Dict[str, Any]:
    """Extract all relevant fields from parsed experiment name."""
    return {
        'experiment_name': exp_name.full_name,
        'dataset': exp_name.dataset,
        'model_key': exp_name.model_key,
        'model_family': exp_name.model_family,
        'reasoning_mode': exp_name.reasoning_mode,
        'doc_strength': exp_name.doc_strength,
        'user_strength': exp_name.user_strength,
        'doc_tier': exp_name.doc_tier,
        'user_tier': exp_name.user_tier,
        'user_first': exp_name.user_first,
        'instruction': exp_name.instruction,
        'use_cot': exp_name.use_cot,
    }


def extract_logistic_metrics(logistic_data: Dict[str, Any], breakdown_data: Dict[str, Any], exp_name: ExpName = None) -> Dict[str, Any]:
    """Extract core logistic regression metrics with confidence intervals."""
    metrics = {}
    
    # Extract accuracy and UNKNOWN rate for each probe variant from breakdown
    if breakdown_data and 'conditions' in breakdown_data:
        # Map condition names to probe variants
        # For doc-first experiments (default)
        condition_to_probe_doc_first = {
            'U0D0_Ucorr0Dcorr0': 'bare',
            'U1D0_Ucorr1Dcorr0': 'upos',
            'U1D0_Ucorr0Dcorr0': 'uneg',
            'U0D1_Ucorr0Dcorr1': 'dpos',
            'U0D1_Ucorr0Dcorr0': 'dneg',
            'U1D1_Ucorr1Dcorr1': 'dpup',
            'U1D1_Ucorr1Dcorr0': 'dpun',
            'U1D1_Ucorr0Dcorr1': 'dnup',
            'U1D1_Ucorr0Dcorr0': 'dnun',
        }
        
        # For user-first experiments
        condition_to_probe_user_first = {
            'U0D0_Ucorr0Dcorr0': 'bare',
            'U1D0_Ucorr1Dcorr0': 'upos',
            'U1D0_Ucorr0Dcorr0': 'uneg',
            'U0D1_Ucorr0Dcorr1': 'dpos',
            'U0D1_Ucorr0Dcorr0': 'dneg',
            'U1D1_Ucorr1Dcorr1': 'updp',  # user-first probe names
            'U1D1_Ucorr1Dcorr0': 'updn',
            'U1D1_Ucorr0Dcorr1': 'undp',
            'U1D1_Ucorr0Dcorr0': 'undn',
        }
        
        # Choose mapping based on experiment type
        if exp_name and exp_name.user_first:
            condition_to_probe = condition_to_probe_user_first
        else:
            condition_to_probe = condition_to_probe_doc_first
        
        for condition, probe in condition_to_probe.items():
            if condition in breakdown_data['conditions']:
                cond_data = breakdown_data['conditions'][condition]
                metrics[f'{probe}_accuracy'] = cond_data.get('correct_rate')
                metrics[f'{probe}_unknown_rate'] = cond_data.get('unknown_rate')
    
    # Extract odds ratios with CIs from logistic results
    if logistic_data and 'odds_ratios' in logistic_data:
        odds = logistic_data['odds_ratios']
        for field in ODDS_RATIO_FIELDS:
            if field in odds:
                value = odds[field]
                # Handle both composite/selectivity (with CIs) and simple odds ratios
                if isinstance(value, dict):
                    # Composite or selectivity with CIs
                    if 'odds_ratio' in value:
                        metrics[f'odds_ratio_{field}'] = value['odds_ratio']
                        metrics[f'odds_ratio_{field}_ci_lower'] = value.get('ci_lower')
                        metrics[f'odds_ratio_{field}_ci_upper'] = value.get('ci_upper')
                    elif 'ratio' in value:
                        metrics[f'odds_ratio_{field}'] = value['ratio']
                        metrics[f'odds_ratio_{field}_ci_lower'] = value.get('ci_lower')
                        metrics[f'odds_ratio_{field}_ci_upper'] = value.get('ci_upper')
                else:
                    # Simple odds ratio
                    metrics[f'odds_ratio_{field}'] = value
    
    # Extract individual coefficient CIs if available
    if logistic_data and 'coefficients' in logistic_data:
        coef_mapping = {
            'const': 'intercept',
            'P_i': 'βP_parametric_correctness',
            'U_pres': 'δU_user_display_effect',
            'D_pres': 'δD_doc_display_effect',
            'U_pres_corr': 'βU_user_correct_boost',
            'D_pres_corr': 'βD_doc_correct_boost'
        }
        for coef_name, display_name in coef_mapping.items():
            if coef_name in logistic_data['coefficients']:
                coef_data = logistic_data['coefficients'][coef_name]
                if 'odds_ratio_ci_lower' in coef_data:
                    metrics[f'odds_ratio_{display_name}_ci_lower'] = coef_data['odds_ratio_ci_lower']
                    metrics[f'odds_ratio_{display_name}_ci_upper'] = coef_data['odds_ratio_ci_upper']
    
    # Add selectivity if available
    if logistic_data and 'odds_ratios' in logistic_data:
        odds = logistic_data['odds_ratios']
        if 'user_selectivity' in odds:
            sel = odds['user_selectivity']
            metrics['odds_ratio_user_selectivity'] = sel.get('ratio')
            metrics['odds_ratio_user_selectivity_ci_lower'] = sel.get('ci_lower')
            metrics['odds_ratio_user_selectivity_ci_upper'] = sel.get('ci_upper')
        if 'doc_selectivity' in odds:
            sel = odds['doc_selectivity']
            metrics['odds_ratio_doc_selectivity'] = sel.get('ratio')
            metrics['odds_ratio_doc_selectivity_ci_lower'] = sel.get('ci_lower')
            metrics['odds_ratio_doc_selectivity_ci_upper'] = sel.get('ci_upper')
    
    return metrics


def extract_choice_metrics(choice_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract core choice metrics for specific probes."""
    metrics = {}
    
    # Extract metrics for specific probes only
    for probe in CHOICE_VARIANTS:
        # User source metrics
        if 'user_source' in choice_data and probe in choice_data['user_source']:
            user_data = choice_data['user_source'][probe]
            metrics[f'user_{probe}_oar'] = user_data.get('oar')
            metrics[f'user_{probe}_car'] = user_data.get('car')
            metrics[f'user_{probe}_mr'] = user_data.get('mr')
            metrics[f'user_{probe}_prior_bias'] = user_data.get('prior_bias')
            metrics[f'user_{probe}_prior_correction'] = user_data.get('prior_correction')
            metrics[f'user_{probe}_prior_neither'] = user_data.get('prior_neither')
            metrics[f'user_{probe}_context_bias'] = user_data.get('context_bias')
            metrics[f'user_{probe}_context_robustness'] = user_data.get('context_robustness')
            metrics[f'user_{probe}_context_neither'] = user_data.get('context_neither')
        
        # Doc source metrics
        if 'doc_source' in choice_data and probe in choice_data['doc_source']:
            doc_data = choice_data['doc_source'][probe]
            metrics[f'doc_{probe}_oar'] = doc_data.get('oar')
            metrics[f'doc_{probe}_car'] = doc_data.get('car')
            metrics[f'doc_{probe}_mr'] = doc_data.get('mr')
            metrics[f'doc_{probe}_prior_bias'] = doc_data.get('prior_bias')
            metrics[f'doc_{probe}_prior_correction'] = doc_data.get('prior_correction')
            metrics[f'doc_{probe}_prior_neither'] = doc_data.get('prior_neither')
            metrics[f'doc_{probe}_context_bias'] = doc_data.get('context_bias')
            metrics[f'doc_{probe}_context_robustness'] = doc_data.get('context_robustness')
            metrics[f'doc_{probe}_context_neither'] = doc_data.get('context_neither')
    
    return metrics


def extract_distribution_metrics(dist_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract core distribution metrics with confidence intervals."""
    metrics = {}
    
    # NOTE: Susceptibility is excluded (computation disabled)
    # NOTE: Pearson correlation is excluded (not included in core metrics)
    
    # Extract differences from bare (12 probes) with CIs
    if 'probe_metrics' in dist_data:
        for probe in DISTRIBUTION_VARIANTS:
            if probe in dist_data['probe_metrics']:
                probe_data = dist_data['probe_metrics'][probe]
                
                # KL divergence difference from bare
                if 'kl_divergence' in probe_data:
                    metrics[f'{probe}_kl_diff'] = probe_data['kl_divergence'].get('mean')
                    metrics[f'{probe}_kl_diff_ci_lower'] = probe_data['kl_divergence'].get('ci_low')
                    metrics[f'{probe}_kl_diff_ci_upper'] = probe_data['kl_divergence'].get('ci_high')
                
                # -logP(correct) difference from bare
                if 'neg_logp_delta' in probe_data:
                    metrics[f'{probe}_neg_logp_diff'] = probe_data['neg_logp_delta'].get('mean')
                    metrics[f'{probe}_neg_logp_diff_ci_lower'] = probe_data['neg_logp_delta'].get('ci_low')
                    metrics[f'{probe}_neg_logp_diff_ci_upper'] = probe_data['neg_logp_delta'].get('ci_high')
                
                # Entropy delta
                if 'entropy_delta' in probe_data:
                    metrics[f'{probe}_entropy_delta'] = probe_data['entropy_delta'].get('mean')
                    metrics[f'{probe}_entropy_delta_ci_lower'] = probe_data['entropy_delta'].get('ci_low')
                    metrics[f'{probe}_entropy_delta_ci_upper'] = probe_data['entropy_delta'].get('ci_high')
    
    # Extract absolute entropy values (each experiment has 9 probes, but we check all 13 for compatibility)
    if 'entropy_absolute' in dist_data:
        # Check all possible probes (bare + 12 variants) - each experiment only has 9
        all_probes = ['bare'] + list(DISTRIBUTION_VARIANTS)
        for probe in all_probes:
            if probe in dist_data['entropy_absolute']:
                entropy_data = dist_data['entropy_absolute'][probe]
                metrics[f'{probe}_entropy'] = entropy_data.get('mean')
                metrics[f'{probe}_entropy_ci_lower'] = entropy_data.get('ci_low')
                metrics[f'{probe}_entropy_ci_upper'] = entropy_data.get('ci_high')
    
    # Extract individual -logP(correct) values (each experiment has 9 probes, but we check all 13 for compatibility)
    if 'neg_logp_correct_absolute' in dist_data:
        # Check all possible probes (bare + 12 variants) - each experiment only has 9
        all_probes = ['bare'] + list(DISTRIBUTION_VARIANTS)
        for probe in all_probes:
            if probe in dist_data['neg_logp_correct_absolute']:
                logp_data = dist_data['neg_logp_correct_absolute'][probe]
                metrics[f'{probe}_neg_logp'] = logp_data.get('mean')
                metrics[f'{probe}_neg_logp_ci_lower'] = logp_data.get('ci_low')
                metrics[f'{probe}_neg_logp_ci_upper'] = logp_data.get('ci_high')
    
    return metrics


def process_experiment(exp_dir: pathlib.Path) -> Dict[str, Any] | None:
    """Process a single experiment directory and extract core metrics."""
    exp_name_str = exp_dir.name
    results_folder = exp_dir.parent.name
    results_folder_path = exp_dir.parent

    # Parse experiment name
    try:
        exp_name = parse_experiment_name(exp_name_str)
    except ValueError as e:
        print(f"Warning: Skipping {exp_name_str} - {e}")
        return None

    # Initialize result dictionary with experiment fields
    result = extract_exp_name_fields(exp_name)
    # Add source results directory information
    result['results_dir'] = results_folder
    result['results_dir_path'] = str(results_folder_path)
    
    # Load and extract logistic metrics
    logistic_data = load_json_safely(exp_dir / "logistic_regression_results.json")
    breakdown_data = load_json_safely(exp_dir / "logistic_regression_breakdown.json")
    logistic_metrics = extract_logistic_metrics(logistic_data, breakdown_data, exp_name)
    
    # Load and extract choice metrics
    choice_data = load_json_safely(exp_dir / "choice_metrics.json")
    choice_metrics = extract_choice_metrics(choice_data) if choice_data else {}
    
    # Load and extract distribution metrics
    dist_data = load_json_safely(exp_dir / "distribution_metrics.json")
    dist_metrics = extract_distribution_metrics(dist_data) if dist_data else {}
    
    return {
        'exp_fields': result,
        'logistic': logistic_metrics,
        'choice': choice_metrics,
        'distribution': dist_metrics
    }


def create_dataframes(all_results: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create separate dataframes for each metric type and combined."""
    # Extract experiment fields and metrics
    exp_data = []
    logistic_data = []
    choice_data = []
    dist_data = []
    
    for result in all_results:
        exp_fields = result['exp_fields']
        
        # Logistic data (includes exp fields)
        logistic_row = {**exp_fields, **result['logistic']}
        logistic_data.append(logistic_row)
        
        # Choice data (includes exp fields)
        choice_row = {**exp_fields, **result['choice']}
        choice_data.append(choice_row)
        
        # Distribution data (includes exp fields)
        dist_row = {**exp_fields, **result['distribution']}
        dist_data.append(dist_row)
        
        # Combined data
        combined_row = {**exp_fields, **result['logistic'], **result['choice'], **result['distribution']}
        exp_data.append(combined_row)
    
    # Create dataframes
    df_all = pd.DataFrame(exp_data)
    df_logistic = pd.DataFrame(logistic_data)
    df_choice = pd.DataFrame(choice_data)
    df_dist = pd.DataFrame(dist_data)
    
    # Define column order - include results_dir info at the beginning
    exp_cols = ['results_dir', 'results_dir_path', 'experiment_name', 'dataset', 'model_key', 'model_family', 'reasoning_mode',
                'doc_strength', 'user_strength', 'doc_tier', 'user_tier',
                'user_first', 'instruction', 'use_cot']
    
    # Logistic columns (accuracy first, then unknown rates, then odds ratios)
    accuracy_cols = [f'{v}_accuracy' for v in ACCURACY_VARIANTS]
    unknown_cols = [f'{v}_unknown_rate' for v in ACCURACY_VARIANTS]
    odds_cols = [f'odds_ratio_{field}' for field in ODDS_RATIO_FIELDS]
    logistic_cols = accuracy_cols + unknown_cols + odds_cols
    
    # Choice columns
    choice_cols = []
    for source in ['user', 'doc']:
        for probe in CHOICE_VARIANTS:
            for metric in ['oar', 'car', 'mr', 'prior_bias', 'prior_correction', 'prior_neither', 'context_bias', 'context_robustness', 'context_neither']:
                choice_cols.append(f'{source}_{probe}_{metric}')
    
    # Distribution columns
    dist_cols = []
    
    # KL differences (12 probes)
    for probe in DISTRIBUTION_VARIANTS:
        dist_cols.append(f'{probe}_kl_diff')
    
    # -logP(correct) differences (12 probes)
    for probe in DISTRIBUTION_VARIANTS:
        dist_cols.append(f'{probe}_neg_logp_diff')
    
    # Entropy delta values (12 probes) - differences from bare
    for probe in DISTRIBUTION_VARIANTS:
        dist_cols.append(f'{probe}_entropy_delta')
    
    # Absolute entropy values (13 probes including bare - each experiment has 9)
    all_probes_with_bare = ['bare'] + list(DISTRIBUTION_VARIANTS)
    for probe in all_probes_with_bare:
        dist_cols.append(f'{probe}_entropy')
    
    # Individual -logP(correct) values (13 probes including bare - each experiment has 9)
    for probe in all_probes_with_bare:
        dist_cols.append(f'{probe}_neg_logp')
    
    # Reorder columns for each dataframe
    df_all = df_all[exp_cols + [col for col in logistic_cols + choice_cols + dist_cols if col in df_all.columns]]
    df_logistic = df_logistic[exp_cols + [col for col in logistic_cols if col in df_logistic.columns]]
    df_choice = df_choice[exp_cols + [col for col in choice_cols if col in df_choice.columns]]
    df_dist = df_dist[exp_cols + [col for col in dist_cols if col in df_dist.columns]]
    
    return df_all, df_logistic, df_choice, df_dist


def main():
    parser = argparse.ArgumentParser(description="Generate core metrics reports")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path(__file__).parent / "exp_results",
                       help="Output directory for CSV files (default: analyze/exp_results, relative to this script)")
    parser.add_argument("--results-dir", type=pathlib.Path, nargs="+",
                       default=[ROOT / "results",
                                ROOT / "results_0727_2",
                                ROOT / "results_0810_for_sft",
                                ROOT / "results_0811_system_instructions",
                                ROOT / "results_0812"],
                       help="Results directory/directories to scan (default: all 5 results directories)")
    parser.add_argument("--append", action="store_true",
                       help="Append to existing CSV files instead of overwriting them")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    results_dirs = args.results_dir
    
    # Validate all results directories exist
    for results_dir in results_dirs:
        if not results_dir.exists():
            sys.exit(f"Results directory not found: {results_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all experiment directories across all results directories
    exp_dirs = []
    for results_dir in results_dirs:
        for d in results_dir.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                exp_dirs.append(d)
    
    print(f"Found {len(exp_dirs)} experiment directories across {len(results_dirs)} results directories")
    
    # Process each experiment
    all_results = []
    for exp_dir in sorted(exp_dirs):
        print(f"Processing {exp_dir.name}...")
        result = process_experiment(exp_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("No valid results found")
        return
    
    # Create dataframes
    df_all, df_logistic, df_choice, df_dist = create_dataframes(all_results)
    
    # Save to CSV files
    files_to_save = [
        (df_all, "all_core_metrics.csv"),
        (df_logistic, "logistic_core_metrics.csv"),
        (df_choice, "choice_core_metrics.csv"),
        (df_dist, "distribution_core_metrics.csv")
    ]
    
    for df, filename in files_to_save:
        output_path = output_dir / filename
        if args.append and output_path.exists():
            # Read existing data to get experiment names
            existing_df = pd.read_csv(output_path)
            existing_experiments = set(existing_df['experiment_name'].values)
            
            # Filter new data to only include experiments not already in existing data
            new_experiments_df = df[~df['experiment_name'].isin(existing_experiments)]
            
            if len(new_experiments_df) > 0:
                # Append new rows directly to the file without modifying existing content
                # Write new experiments without header to a temporary string
                new_data_str = new_experiments_df.to_csv(index=False, header=False)
                
                # Append the new data to the existing file
                with open(output_path, 'a') as f:
                    f.write(new_data_str)
                
                print(f"Appended {len(new_experiments_df)} new experiments to {output_path} (total: {len(existing_df) + len(new_experiments_df)} experiments)")
            else:
                print(f"No new experiments to append to {output_path} (keeping {len(existing_df)} existing experiments)")
        else:
            # Overwrite mode or file doesn't exist
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} experiments to {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Mode: {'APPEND' if args.append else 'OVERWRITE'}")
    print(f"- Scanned directories: {', '.join(str(d) for d in results_dirs)}")
    print(f"- Total experiments processed: {len(df_all)}")
    print(f"- Datasets: {', '.join(df_all['dataset'].unique())}")
    print(f"- Models: {', '.join(df_all['model_key'].unique())}")
    print(f"\n{'Updated' if args.append else 'Generated'} 4 CSV files in {output_dir}:")
    print(f"  - all_core_metrics.csv ({len(df_all.columns)} columns)")
    print(f"  - logistic_core_metrics.csv ({len(df_logistic.columns)} columns)")
    print(f"  - choice_core_metrics.csv ({len(df_choice.columns)} columns)")
    print(f"  - distribution_core_metrics.csv ({len(df_dist.columns)} columns)")


if __name__ == "__main__":
    main()