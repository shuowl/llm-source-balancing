#!/usr/bin/env python3
"""
choice_metrics_analysis.py
==========================
Compute choice-level metrics (OAR, CAR, MR, prior-bias, context-bias)
using bare probe as baseline and comparing against single-source probes.

Uses merged_results.jsonl and computes metrics for:
- User source: upos, uneg
- Doc source: dpos, dneg

Where:
- OAR (Original-Answer Ratio): fraction where model sticks to bare answer
- CAR (Counter-Answer Ratio): fraction where model adopts external cue answer
- MR (Memorization Ratio): OAR / (OAR + CAR)
- Prior bias: bare wrong, external correct, but model doesn't flip
- Context bias: bare correct, external wrong, but model gets misled

Output (will be overwritten if it exists)
-----------------------------------------
results/<exp-name>/choice_metrics.json

Note: This script will OVERWRITE the existing output file each time it runs.

Usage
-----
python analyze/choice_metrics_analysis.py --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot
python analyze/choice_metrics_analysis.py --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot --results-dir results_hf_test
"""

import argparse, json, sys, pandas as pd, pathlib
from typing import Dict, List, Any, Union
import numpy as np

# ─── make utils import-able ───────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def safe_div(n: float, d: float) -> Union[float, None]:
    """Safe division handling zero denominators, returns None instead of NaN for JSON compatibility"""
    return n / d if d > 0 else None


def safe_float(value: float) -> Union[float, None]:
    """Convert float to None if NaN for JSON compatibility"""
    return None if pd.isna(value) else float(value)


def compute_metrics_for_probe(bare_df: pd.DataFrame, probe_df: pd.DataFrame, 
                             probe_name: str) -> Dict[str, Any]:
    """
    Compute OAR, CAR, MR, and bias/correction metrics for a single probe variant.
    
    Metrics include:
    - OAR/CAR/MR: Original answer retention, counter answer adoption, memorization ratio
    - Prior bias: Model wrong, source correct, but outputs prior (wrong) answer
    - Prior correction: Model wrong, source correct, and outputs source (correct) answer  
    - Prior neither: Model wrong, source correct, but outputs neither prior nor source
    - Context bias: Model correct, source wrong, and outputs source (wrong) answer
    - Context robustness: Model correct, source wrong, but outputs prior (correct) answer
    - Context neither: Model correct, source wrong, but outputs neither prior nor source
    
    Args:
        bare_df: DataFrame with bare probe results (baseline)
        probe_df: DataFrame with current probe variant results
        probe_name: Name of the probe variant (e.g., 'upos', 'uneg')
    
    Returns:
        Dictionary with computed metrics
    """
    # Merge bare and probe data on qid
    merged = bare_df[['qid', 'output_letter', 'output_correct', 'gold', 'canonical_wrong', 'model_correct']].merge(
        probe_df[['qid', 'output_letter', 'output_correct', 
                  'user_present', 'doc_present', 'user_correct', 'doc_correct']], 
        on='qid', 
        suffixes=('_bare', '_probe')
    )
    
    # Determine which external source is present and its correctness
    if 'upos' in probe_name or 'uneg' in probe_name:
        external_correct = merged['user_correct']
        external_present = merged['user_present']
        source_type = 'user'
    else:  # dpos, dneg
        external_correct = merged['doc_correct']
        external_present = merged['doc_present']
        source_type = 'doc'
    
    # Only consider rows where external source is present
    merged = merged[external_present].copy()
    
    if len(merged) == 0:
        return {
            'oar': None,
            'car': None,
            'mr': None,
            'prior_bias': None,
            'context_bias': None,
            'n_samples': 0,
            'oar_details': {},
            'car_details': {},
            'prior_bias_details': {},
            'context_bias_details': {}
        }
    
    # Determine what answer the external cue is suggesting
    is_positive = 'pos' in probe_name
    if is_positive:
        # For positive probes, external cue suggests the gold (correct) answer
        external_suggestion = merged['gold']
    else:
        # For negative probes, external cue suggests the canonical wrong answer
        external_suggestion = merged['canonical_wrong']
    
    # Identify conflict cases: where external suggestion differs from bare answer
    conflict_mask = external_suggestion != merged['output_letter_bare']
    n_conflict = conflict_mask.sum()
    
    # OAR and CAR details
    oar_details = {
        'n_conflict': int(n_conflict),
        'n_conflict_and_sticks_to_model': 0
    }
    
    car_details = {
        'n_conflict': int(n_conflict),
        'n_conflict_and_adopts_external': 0
    }
    
    # Compute OAR/CAR metrics only on conflict cases
    if n_conflict > 0:
        conflict_df = merged[conflict_mask].copy()
        
        # OAR: Model sticks to bare answer despite conflicting external cue
        oar_mask = conflict_df['output_letter_probe'] == conflict_df['output_letter_bare']
        n_oar = oar_mask.sum()
        oar = oar_mask.mean()
        
        # CAR: Model adopts external cue's suggested answer
        car_mask = conflict_df['output_letter_probe'] == external_suggestion[conflict_mask]
        n_car = car_mask.sum()
        car = car_mask.mean()
        
        # MR: Memorization Ratio
        mr = safe_div(oar, oar + car)
        
        oar_details['n_conflict_and_sticks_to_model'] = int(n_oar)
        car_details['n_conflict_and_adopts_external'] = int(n_car)
    else:
        oar = None
        car = None
        mr = None
    
    # ========== PRIOR-RELATED METRICS (Model initially wrong, source correct) ==========
    # Base mask: model wrong initially, external source is correct
    prior_base_mask = ~merged['output_correct_bare'] & external_correct
    n_prior_base = prior_base_mask.sum()
    
    if n_prior_base > 0:
        prior_subset = merged[prior_base_mask].copy()
        
        # Prior bias: outputs prior answer (which is wrong)
        prior_bias_mask = prior_subset['output_letter_probe'] == prior_subset['output_letter_bare']
        n_prior_bias = prior_bias_mask.sum()
        prior_bias = n_prior_bias / n_prior_base
        
        # Prior correction: outputs source answer (which is correct)
        # For positive probes, source suggests gold; for negative, source suggests canonical_wrong
        # But since external is correct here, it must be a positive probe suggesting gold
        if is_positive:
            source_answer = prior_subset['gold']
        else:
            # This shouldn't happen for prior correction (negative probe with correct source)
            # but handle it anyway
            source_answer = prior_subset['canonical_wrong']
        
        prior_correction_mask = prior_subset['output_letter_probe'] == source_answer
        n_prior_correction = prior_correction_mask.sum()
        prior_correction = n_prior_correction / n_prior_base
        
        # Prior neither: outputs neither prior nor source answer
        prior_neither_mask = ~prior_bias_mask & ~prior_correction_mask
        n_prior_neither = prior_neither_mask.sum()
        prior_neither = n_prior_neither / n_prior_base
    else:
        prior_bias = None
        prior_correction = None
        prior_neither = None
        n_prior_bias = 0
        n_prior_correction = 0
        n_prior_neither = 0
    
    prior_bias_details = {
        'n_model_wrong': int((~merged['output_correct_bare']).sum()),
        f'n_{source_type}_correct': int(external_correct.sum()),
        f'n_model_wrong_and_{source_type}_correct': int(n_prior_base),
        f'n_prior_bias': int(n_prior_bias),
        f'n_prior_correction': int(n_prior_correction),
        f'n_prior_neither': int(n_prior_neither)
    }
    
    # ========== CONTEXT-RELATED METRICS (Model initially correct, source wrong) ==========
    # Base mask: model correct initially, external source is wrong
    context_base_mask = merged['output_correct_bare'] & ~external_correct
    n_context_base = context_base_mask.sum()
    
    if n_context_base > 0:
        context_subset = merged[context_base_mask].copy()
        
        # Context bias: outputs source answer (which is wrong)
        # For positive probes, source suggests gold; for negative, source suggests canonical_wrong
        # But since external is wrong here, negative probe suggests canonical_wrong
        if is_positive:
            # Positive probe with wrong source shouldn't happen in standard setup
            source_answer = context_subset['gold']
        else:
            source_answer = context_subset['canonical_wrong']
        
        context_bias_mask = context_subset['output_letter_probe'] == source_answer
        n_context_bias = context_bias_mask.sum()
        context_bias = n_context_bias / n_context_base
        
        # Context robustness: outputs prior answer (which is correct)
        context_robustness_mask = context_subset['output_letter_probe'] == context_subset['output_letter_bare']
        n_context_robustness = context_robustness_mask.sum()
        context_robustness = n_context_robustness / n_context_base
        
        # Context neither: outputs neither prior nor source answer
        context_neither_mask = ~context_bias_mask & ~context_robustness_mask
        n_context_neither = context_neither_mask.sum()
        context_neither = n_context_neither / n_context_base
    else:
        context_bias = None
        context_robustness = None
        context_neither = None
        n_context_bias = 0
        n_context_robustness = 0
        n_context_neither = 0
    
    context_bias_details = {
        'n_model_correct': int(merged['output_correct_bare'].sum()),
        f'n_{source_type}_wrong': int((~external_correct).sum()),
        f'n_model_correct_and_{source_type}_wrong': int(n_context_base),
        f'n_context_bias': int(n_context_bias),
        f'n_context_robustness': int(n_context_robustness),
        f'n_context_neither': int(n_context_neither)
    }
    
    return {
        'oar': safe_float(oar),
        'car': safe_float(car),
        'mr': safe_float(mr),
        'prior_bias': safe_float(prior_bias),
        'prior_correction': safe_float(prior_correction),
        'prior_neither': safe_float(prior_neither),
        'context_bias': safe_float(context_bias),
        'context_robustness': safe_float(context_robustness),
        'context_neither': safe_float(context_neither),
        'n_samples': len(merged),
        'oar_details': oar_details,
        'car_details': car_details,
        'prior_bias_details': prior_bias_details,
        'context_bias_details': context_bias_details
    }


def analyze_experiment(merged_results_path: pathlib.Path) -> Dict[str, Any]:
    """
    Analyze merged results to compute choice-level metrics.
    
    Args:
        merged_results_path: Path to merged_results.jsonl
    
    Returns:
        Dictionary with metrics for each source and probe variant
    """
    # Load merged results
    df = pd.read_json(merged_results_path, lines=True)
    
    # Check if we have the required probe variants
    required_probes = ['bare', 'upos', 'uneg', 'dpos', 'dneg']
    available_probes = df['probe_variant'].unique()
    missing_probes = set(required_probes) - set(available_probes)
    
    if missing_probes:
        print(f"Warning: Missing probe variants: {missing_probes}")
        print(f"Available: {available_probes}")
    
    # Extract bare baseline
    bare_df = df[df['probe_variant'] == 'bare'].copy()
    
    if len(bare_df) == 0:
        raise ValueError("No 'bare' probe variant found in data")
    
    # Compute metrics for each probe variant
    results = {
        'experiment': str(merged_results_path.parent.name),
        'n_questions': len(bare_df),
        'bare_accuracy': float(bare_df['output_correct'].mean()),
        'user_source': {},
        'doc_source': {}
    }
    
    # User source probes
    for probe in ['upos', 'uneg']:
        if probe in available_probes:
            probe_df = df[df['probe_variant'] == probe].copy()
            metrics = compute_metrics_for_probe(bare_df, probe_df, probe)
            results['user_source'][probe] = metrics
    
    # Doc source probes  
    for probe in ['dpos', 'dneg']:
        if probe in available_probes:
            probe_df = df[df['probe_variant'] == probe].copy()
            metrics = compute_metrics_for_probe(bare_df, probe_df, probe)
            results['doc_source'][probe] = metrics
    
    # Compute aggregated metrics per source
    for source in ['user_source', 'doc_source']:
        if results[source]:
            # Average across positive and negative probes, handling None values
            source_metrics = list(results[source].values())
            
            def safe_nanmean(values):
                """Compute mean ignoring None values"""
                filtered = [v for v in values if v is not None]
                return float(np.mean(filtered)) if filtered else None
            
            results[f'{source}_avg'] = {
                'oar': safe_nanmean([m['oar'] for m in source_metrics]),
                'car': safe_nanmean([m['car'] for m in source_metrics]),
                'mr': safe_nanmean([m['mr'] for m in source_metrics]),
                'prior_bias': safe_nanmean([m['prior_bias'] for m in source_metrics]),
                'prior_correction': safe_nanmean([m['prior_correction'] for m in source_metrics]),
                'prior_neither': safe_nanmean([m['prior_neither'] for m in source_metrics]),
                'context_bias': safe_nanmean([m['context_bias'] for m in source_metrics]),
                'context_robustness': safe_nanmean([m['context_robustness'] for m in source_metrics]),
                'context_neither': safe_nanmean([m['context_neither'] for m in source_metrics])
            }
    
    return results


def find_merged_results(exp_name: str, results_dir: pathlib.Path = None) -> pathlib.Path:
    """Find merged_results.jsonl for given experiment"""
    base_dir = results_dir if results_dir else ROOT / "results"
    
    p = base_dir / exp_name / "merged_results.jsonl"
    if not p.exists():
        # Try regular results.jsonl as fallback
        p_alt = base_dir / exp_name / "results.jsonl"
        if p_alt.exists():
            print(f"Warning: Using results.jsonl instead of merged_results.jsonl")
            return p_alt
        sys.exit(f"[ERR] missing file: {p}")
    return p


def format_metric(value: Union[float, None]) -> str:
    """Format metric value for display"""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compute choice-level metrics from merged results")
    parser.add_argument("--experiment_name", required=True, help="Experiment name to analyze")
    parser.add_argument("--results-dir", type=pathlib.Path, help="Custom results directory")
    args = parser.parse_args()
    
    # Find data file
    merged_path = find_merged_results(args.experiment_name, args.results_dir)
    out_dir = merged_path.parent
    output_file = out_dir / "choice_metrics.json"
    
    # Output file will be overwritten if it exists
    
    print(f"\nAnalyzing: {merged_path}")
    print("="*70)
    
    # Run analysis
    results = analyze_experiment(merged_path)
    
    # Display results
    print(f"\nExperiment: {results['experiment']}")
    print(f"Questions: {results['n_questions']}")
    print(f"Bare accuracy: {results['bare_accuracy']:.3f}")
    
    print("\n" + "-"*70)
    print("User Source Metrics")
    print("-"*70)
    for probe, metrics in results['user_source'].items():
        print(f"\n{probe}:")
        print(f"  OAR: {format_metric(metrics['oar'])}")
        print(f"  CAR: {format_metric(metrics['car'])}")
        print(f"  MR:  {format_metric(metrics['mr'])}")
        
        # OAR details
        oar_d = metrics['oar_details']
        print(f"  OAR details: conflict={oar_d['n_conflict']}, sticks_to_model={oar_d['n_conflict_and_sticks_to_model']}")
        
        # CAR details
        car_d = metrics['car_details']
        print(f"  CAR details: conflict={car_d['n_conflict']}, adopts_external={car_d['n_conflict_and_adopts_external']}")
        
        # Prior-related metrics
        pb_d = metrics['prior_bias_details']
        print(f"  Prior metrics (model wrong, source correct):")
        print(f"    Prior bias: {format_metric(metrics['prior_bias'])} (outputs prior/wrong)")
        print(f"    Prior correction: {format_metric(metrics['prior_correction'])} (outputs source/correct)")
        print(f"    Prior neither: {format_metric(metrics['prior_neither'])} (outputs neither)")
        print(f"    Details: base_cases={pb_d.get('n_model_wrong_and_user_correct', 0)}, "
              f"bias={pb_d.get('n_prior_bias', 0)}, "
              f"correction={pb_d.get('n_prior_correction', 0)}, "
              f"neither={pb_d.get('n_prior_neither', 0)}")
        
        # Context-related metrics
        cb_d = metrics['context_bias_details']
        print(f"  Context metrics (model correct, source wrong):")
        print(f"    Context bias: {format_metric(metrics['context_bias'])} (outputs source/wrong)")
        print(f"    Context robustness: {format_metric(metrics['context_robustness'])} (outputs prior/correct)")
        print(f"    Context neither: {format_metric(metrics['context_neither'])} (outputs neither)")
        print(f"    Details: base_cases={cb_d.get('n_model_correct_and_user_wrong', 0)}, "
              f"bias={cb_d.get('n_context_bias', 0)}, "
              f"robustness={cb_d.get('n_context_robustness', 0)}, "
              f"neither={cb_d.get('n_context_neither', 0)}")
    
    if 'user_source_avg' in results:
        print(f"\nUser Average:")
        avg = results['user_source_avg']
        print(f"  OAR: {format_metric(avg['oar'])}")
        print(f"  CAR: {format_metric(avg['car'])}")
        print(f"  MR:  {format_metric(avg['mr'])}")
        print(f"  Prior bias: {format_metric(avg['prior_bias'])} | Correction: {format_metric(avg['prior_correction'])} | Neither: {format_metric(avg['prior_neither'])}")
        print(f"  Context bias: {format_metric(avg['context_bias'])} | Robustness: {format_metric(avg['context_robustness'])} | Neither: {format_metric(avg['context_neither'])}")
    
    print("\n" + "-"*70)
    print("Doc Source Metrics")
    print("-"*70)
    for probe, metrics in results['doc_source'].items():
        print(f"\n{probe}:")
        print(f"  OAR: {format_metric(metrics['oar'])}")
        print(f"  CAR: {format_metric(metrics['car'])}")
        print(f"  MR:  {format_metric(metrics['mr'])}")
        
        # OAR details
        oar_d = metrics['oar_details']
        print(f"  OAR details: conflict={oar_d['n_conflict']}, sticks_to_model={oar_d['n_conflict_and_sticks_to_model']}")
        
        # CAR details
        car_d = metrics['car_details']
        print(f"  CAR details: conflict={car_d['n_conflict']}, adopts_external={car_d['n_conflict_and_adopts_external']}")
        
        # Prior-related metrics
        pb_d = metrics['prior_bias_details']
        print(f"  Prior metrics (model wrong, source correct):")
        print(f"    Prior bias: {format_metric(metrics['prior_bias'])} (outputs prior/wrong)")
        print(f"    Prior correction: {format_metric(metrics['prior_correction'])} (outputs source/correct)")
        print(f"    Prior neither: {format_metric(metrics['prior_neither'])} (outputs neither)")
        print(f"    Details: base_cases={pb_d.get('n_model_wrong_and_doc_correct', 0)}, "
              f"bias={pb_d.get('n_prior_bias', 0)}, "
              f"correction={pb_d.get('n_prior_correction', 0)}, "
              f"neither={pb_d.get('n_prior_neither', 0)}")
        
        # Context-related metrics  
        cb_d = metrics['context_bias_details']
        print(f"  Context metrics (model correct, source wrong):")
        print(f"    Context bias: {format_metric(metrics['context_bias'])} (outputs source/wrong)")
        print(f"    Context robustness: {format_metric(metrics['context_robustness'])} (outputs prior/correct)")
        print(f"    Context neither: {format_metric(metrics['context_neither'])} (outputs neither)")
        print(f"    Details: base_cases={cb_d.get('n_model_correct_and_doc_wrong', 0)}, "
              f"bias={cb_d.get('n_context_bias', 0)}, "
              f"robustness={cb_d.get('n_context_robustness', 0)}, "
              f"neither={cb_d.get('n_context_neither', 0)}")
    
    if 'doc_source_avg' in results:
        print(f"\nDoc Average:")
        avg = results['doc_source_avg']
        print(f"  OAR: {format_metric(avg['oar'])}")
        print(f"  CAR: {format_metric(avg['car'])}")
        print(f"  MR:  {format_metric(avg['mr'])}")
        print(f"  Prior bias: {format_metric(avg['prior_bias'])} | Correction: {format_metric(avg['prior_correction'])} | Neither: {format_metric(avg['prior_neither'])}")
        print(f"  Context bias: {format_metric(avg['context_bias'])} | Robustness: {format_metric(avg['context_robustness'])} | Neither: {format_metric(avg['context_neither'])}")
    
    # Save results
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved choice_metrics.json in {out_dir}")


if __name__ == "__main__":
    main()