#!/usr/bin/env python
"""
logistic_regression_analysis.py
===============================
Analyze the impact of priors using a 5-parameter logistic regression model.

Uses merged_results.jsonl and performs logistic regression with:
log(p/(1-p)) = β₀ + β_P P_i + δ_U U_pres + δ_D D_pres + β_U (U_pres * U_corr) + β_D (D_pres * D_corr)

Where:
- P_i: parametric correctness (model_correct)
- U_pres/D_pres: presence of user/doc assertions
- U_corr/D_corr: correctness of user/doc assertions (when present)
- δ_U/δ_D: effect when external source is present with incorrect assertion (susceptibility to misinformation)
- β_U/β_D: additional effect when assertion switches from incorrect to correct (selectivity)

Outputs (will be overwritten if they exist)
--------------------------------------------
results/<exp-name>/logistic_regression_summary.txt     (verbatim console log)
results/<exp-name>/logistic_regression_results.json    (detailed regression results)
results/<exp-name>/logistic_regression_breakdown.json  (correct rates by condition)

Note: This script will OVERWRITE existing output files each time it runs.

Example CLI:
```
python analyze/logistic_regression_analysis.py --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot
python analyze/logistic_regression_analysis.py --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot --results-dir results_hf_test
```
"""
from __future__ import annotations
import argparse, pathlib, sys, json, numpy as np, pandas as pd, statsmodels.api as sm
import warnings
from scipy import stats

# ─── make utils import-able ───────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Suppress warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")

# ---------------------------------------------------------------------

def find_results_file(exp: str, results_dir: pathlib.Path = None) -> pathlib.Path:
    """Find merged_results.jsonl (preferred) or results.jsonl"""
    base_dir = results_dir if results_dir else ROOT / "results"
    
    p_merged = base_dir / exp / "merged_results.jsonl"
    if p_merged.exists():
        return p_merged
    
    p_regular = base_dir / exp / "results.jsonl"
    if p_regular.exists():
        return p_regular
    
    sys.exit(f"[ERR] missing file: {p_merged} or {p_regular}")

def load_data(p: pathlib.Path) -> pd.DataFrame:
    """Load and validate data"""
    df = pd.read_json(p, lines=True)
    
    # Check for required columns
    required_cols = ['user_present', 'doc_present', 'user_correct', 'doc_correct', 'model_correct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        sys.exit(f"[ERR] Missing required columns: {missing_cols}. Expected merged_results.jsonl format.")
    
    return df

def prepare_regression_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for 5-parameter logistic regression"""
    # Create binary variables
    df['U_pres'] = df['user_present'].astype(int)
    df['D_pres'] = df['doc_present'].astype(int)
    df['U_corr'] = df['user_correct'].astype(int)
    df['D_corr'] = df['doc_correct'].astype(int)
    df['P_i'] = df['model_correct'].astype(int)
    
    # Create interaction terms
    df['U_pres_corr'] = (df['U_pres'] * df['U_corr']).astype(int)
    df['D_pres_corr'] = (df['D_pres'] * df['D_corr']).astype(int)
    
    # Create correct outcome variable
    if 'output_correct' in df.columns:
        df['correct'] = df['output_correct'].astype(int)
    else:
        # Fallback to comparing output and gold
        df['correct'] = (df['output_letter'] == df['gold']).astype(int)
    
    # Prepare feature matrix
    X = df[['P_i', 'U_pres', 'D_pres', 'U_pres_corr', 'D_pres_corr']]
    X = sm.add_constant(X)
    
    return df, X

def fit_logistic_regression(df: pd.DataFrame, X: pd.DataFrame) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    """Fit the 5-parameter logistic regression model with cluster-robust standard errors"""
    # Use cluster-robust standard errors to account for repeated qids across probe variants
    return sm.Logit(df['correct'], X).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['qid']},
        disp=False
    )

def compute_breakdown_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correct rates by all prior combinations"""
    breakdown = df.groupby(['user_present', 'doc_present', 'user_correct', 'doc_correct'])['correct'].agg(['mean', 'count'])
    breakdown = breakdown.rename(columns={'mean': 'correct_rate'})
    return breakdown

def compute_unknown_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute UNKNOWN rates by all prior combinations"""
    df['is_unknown'] = (df['output_letter'] == 'UNKNOWN').astype(int)
    breakdown = df.groupby(['user_present', 'doc_present', 'user_correct', 'doc_correct'])['is_unknown'].agg(['mean', 'sum', 'count'])
    breakdown = breakdown.rename(columns={'mean': 'unknown_rate', 'sum': 'unknown_count', 'count': 'total_count'})
    return breakdown

def compute_composite_ci(model, delta_name: str, beta_name: str, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Compute confidence interval for composite effect exp(delta + beta).
    
    Args:
        model: Fitted statsmodels logit model
        delta_name: Name of the display effect coefficient (e.g., 'U_pres')
        beta_name: Name of the correctness boost coefficient (e.g., 'U_pres_corr')
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        (point_estimate, ci_lower, ci_upper) for the composite odds ratio
    """
    # Get coefficients
    delta = model.params[delta_name]
    beta = model.params[beta_name]
    
    # Get variance-covariance matrix
    cov_matrix = model.cov_params()
    
    # Get variances and covariance
    var_delta = cov_matrix.loc[delta_name, delta_name]
    var_beta = cov_matrix.loc[beta_name, beta_name]
    cov_delta_beta = cov_matrix.loc[delta_name, beta_name]
    
    # Composite on log scale
    L = delta + beta
    
    # Variance of the sum (includes covariance term)
    var_L = var_delta + var_beta + 2 * cov_delta_beta
    
    # Standard error
    se_L = np.sqrt(var_L)
    
    # CI on log scale (using normal approximation)
    z_score = stats.norm.ppf(1 - alpha/2)
    ci_lower_log = L - z_score * se_L
    ci_upper_log = L + z_score * se_L
    
    # Convert to odds ratio scale
    or_composite = np.exp(L)
    ci_lower = np.exp(ci_lower_log)
    ci_upper = np.exp(ci_upper_log)
    
    return float(or_composite), float(ci_lower), float(ci_upper)

def compute_selectivity_ci(model, delta_name: str, beta_name: str, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Compute confidence interval for selectivity ratio exp(beta - delta).
    This represents the benefit-to-harm ratio: exp(beta)/exp(delta).
    
    Args:
        model: Fitted statsmodels logit model
        delta_name: Name of the display effect coefficient (e.g., 'U_pres')
        beta_name: Name of the correctness boost coefficient (e.g., 'U_pres_corr')
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        (point_estimate, ci_lower, ci_upper) for the selectivity ratio
    """
    # Get coefficients
    delta = model.params[delta_name]
    beta = model.params[beta_name]
    
    # Get variance-covariance matrix
    cov_matrix = model.cov_params()
    
    # Get variances and covariance
    var_delta = cov_matrix.loc[delta_name, delta_name]
    var_beta = cov_matrix.loc[beta_name, beta_name]
    cov_delta_beta = cov_matrix.loc[delta_name, beta_name]
    
    # Selectivity on log scale (beta - delta)
    L = beta - delta
    
    # Variance of the difference (note the minus sign in covariance term)
    var_L = var_beta + var_delta - 2 * cov_delta_beta
    
    # Standard error
    se_L = np.sqrt(var_L)
    
    # CI on log scale (using normal approximation)
    z_score = stats.norm.ppf(1 - alpha/2)
    ci_lower_log = L - z_score * se_L
    ci_upper_log = L + z_score * se_L
    
    # Convert to odds ratio scale
    selectivity = np.exp(L)
    ci_lower = np.exp(ci_lower_log)
    ci_upper = np.exp(ci_upper_log)
    
    return float(selectivity), float(ci_lower), float(ci_upper)

def format_odds_ratios(model) -> dict:
    """Convert coefficients to odds ratios with meaningful names, including composite effects and selectivity with CIs"""
    odds = np.exp(model.params).round(3)
    
    rename_map = {
        'const': 'intercept',
        'P_i': 'βP_parametric_correctness',
        'U_pres': 'δU_user_display_effect',
        'D_pres': 'δD_doc_display_effect',
        'U_pres_corr': 'βU_user_correct_boost',
        'D_pres_corr': 'βD_doc_correct_boost'
    }
    
    odds_dict = odds.rename(rename_map).to_dict()
    
    # Add composite odds ratios with confidence intervals
    # These represent the total effect when a prior is both present AND correct
    user_or, user_ci_low, user_ci_high = compute_composite_ci(model, 'U_pres', 'U_pres_corr')
    doc_or, doc_ci_low, doc_ci_high = compute_composite_ci(model, 'D_pres', 'D_pres_corr')
    
    odds_dict['user_correct_composite'] = {
        'odds_ratio': round(user_or, 3),
        'ci_lower': round(user_ci_low, 3),
        'ci_upper': round(user_ci_high, 3)
    }
    odds_dict['doc_correct_composite'] = {
        'odds_ratio': round(doc_or, 3),
        'ci_lower': round(doc_ci_low, 3),
        'ci_upper': round(doc_ci_high, 3)
    }
    
    # Add selectivity ratios with confidence intervals
    # These represent the benefit-to-harm ratio: exp(beta)/exp(delta)
    user_sel, user_sel_ci_low, user_sel_ci_high = compute_selectivity_ci(model, 'U_pres', 'U_pres_corr')
    doc_sel, doc_sel_ci_low, doc_sel_ci_high = compute_selectivity_ci(model, 'D_pres', 'D_pres_corr')
    
    odds_dict['user_selectivity'] = {
        'ratio': round(user_sel, 3),
        'ci_lower': round(user_sel_ci_low, 3),
        'ci_upper': round(user_sel_ci_high, 3)
    }
    odds_dict['doc_selectivity'] = {
        'ratio': round(doc_sel, 3),
        'ci_lower': round(doc_sel_ci_low, 3),
        'ci_upper': round(doc_sel_ci_high, 3)
    }
    
    return odds_dict

def extract_regression_results(model, df: pd.DataFrame) -> dict:
    """Extract comprehensive regression results for JSON output"""
    results = {
        "model_info": {
            "formula": "log(p/(1-p)) = β₀ + β_P*P_i + δ_U*U_pres + δ_D*D_pres + β_U*(U_pres*U_corr) + β_D*(D_pres*D_corr)",
            "dep_variable": "correct",
            "model_type": "Logit",
            "method": "MLE",
            "standard_errors": "Cluster-robust (clustered by qid)",
            "converged": bool(model.mle_retvals['converged']),
            "n_observations": int(model.nobs),
            "n_clusters": int(len(df['qid'].unique())),
            "df_residuals": int(model.df_resid),
            "df_model": int(model.df_model)
        },
        "model_fit": {
            "pseudo_r_squared": float(model.prsquared),
            "log_likelihood": float(model.llf),
            "ll_null": float(model.llnull),
            "llr_p_value": float(model.llr_pvalue),
            "aic": float(model.aic),
            "bic": float(model.bic)
        },
        "coefficients": {},
        "odds_ratios": format_odds_ratios(model),
        "summary_stats": {
            "total_rows": int(len(df)),
            "overall_correct_rate": float(df['correct'].mean()),
            "correct_count": int(df['correct'].sum()),
            "incorrect_count": int(len(df) - df['correct'].sum())
        }
    }
    
    # Extract coefficient details
    for var_name in model.params.index:
        ci_lower = float(model.conf_int().loc[var_name, 0])
        ci_upper = float(model.conf_int().loc[var_name, 1])
        results["coefficients"][var_name] = {
            "coef": float(model.params[var_name]),
            "std_err": float(model.bse[var_name]),
            "z_score": float(model.tvalues[var_name]),
            "p_value": float(model.pvalues[var_name]),
            "conf_int_025": ci_lower,
            "conf_int_975": ci_upper,
            "odds_ratio": float(np.exp(model.params[var_name])),
            "odds_ratio_ci_lower": float(np.exp(ci_lower)),
            "odds_ratio_ci_upper": float(np.exp(ci_upper))
        }
    
    return results

def format_breakdown_json(breakdown: pd.DataFrame, unknown_breakdown: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Format breakdown stats for JSON output"""
    unknown_count = (df['output_letter'] == 'UNKNOWN').sum()
    unknown_rate = unknown_count / len(df) if len(df) > 0 else 0
    
    data = {
        "overall_correct_rate": float(df['correct'].mean()),
        "overall_unknown_rate": float(unknown_rate),
        "overall_unknown_count": int(unknown_count),
        "total_samples": len(df),
        "conditions": {}
    }
    
    # Merge correct rate and unknown rate data
    merged = breakdown.merge(unknown_breakdown, left_index=True, right_index=True, how='outer')
    
    for (user_p, doc_p, user_c, doc_c), row in merged.iterrows():
        key = f"U{int(user_p)}D{int(doc_p)}_Ucorr{int(user_c)}Dcorr{int(doc_c)}"
        data["conditions"][key] = {
            "user_present": bool(user_p),
            "doc_present": bool(doc_p),
            "user_correct": bool(user_c),
            "doc_correct": bool(doc_c),
            "correct_rate": float(row.get('correct_rate', 0)),
            "count": int(row.get('count', 0)),
            "unknown_rate": float(row.get('unknown_rate', 0)),
            "unknown_count": int(row.get('unknown_count', 0)),
            "description": f"User {'present' if user_p else 'absent'} ({'correct' if user_c else 'wrong'}), Doc {'present' if doc_p else 'absent'} ({'correct' if doc_c else 'wrong'})"
        }
    
    return data

# ─── Main execution ──────────────────────────────────────────────────
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compute Prior-Dominance Index with 5-parameter logistic regression")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment_name", help="Experiment name")
    group.add_argument("--results_jsonl", help="Direct path to results file")
    parser.add_argument("--results-dir", type=pathlib.Path, help="Custom results directory")
    args = parser.parse_args()
    
    # Find data file
    if args.experiment_name:
        res_path = find_results_file(args.experiment_name, args.results_dir)
    else:
        res_path = pathlib.Path(args.results_jsonl)
    
    out_dir = res_path.parent
    
    # Output files (will be overwritten if they exist)
    output_files = [
        out_dir / "logistic_regression_summary.txt",
        out_dir / "logistic_regression_results.json",
        out_dir / "logistic_regression_breakdown.json"
    ]
    
    # Buffer for logging
    log_buffer = []
    def log(s=""):
        print(s)
        log_buffer.append(str(s))
    
    # Load and prepare data
    log(f"\n{'='*70}")
    log(f"Logistic Regression Analysis (5-Parameter Model)")
    log(f"{'='*70}\n")
    log(f"Data file: {res_path}")
    
    df = load_data(res_path)
    df, X = prepare_regression_data(df)
    
    log(f"\nTotal samples: {len(df):,}")
    log(f"Overall correct rate: {df['correct'].mean():.3%}")
    
    # Calculate UNKNOWN rate
    unknown_count = (df['output_letter'] == 'UNKNOWN').sum()
    unknown_rate = unknown_count / len(df) if len(df) > 0 else 0
    log(f"Overall UNKNOWN rate: {unknown_rate:.3%} ({unknown_count:,} out of {len(df):,})")
    
    # Compute breakdown statistics
    breakdown = compute_breakdown_stats(df)
    
    log(f"\n{'─'*70}")
    log("Correct Rate by Prior Combinations")
    log(f"{'─'*70}")
    log("User | Doc  | User  | Doc   | Correct |")
    log("Pres | Pres | Corr  | Corr  |  Rate   | Count")
    log("-----|------|-------|-------|---------|------")
    
    for (user_p, doc_p, user_c, doc_c), row in breakdown.iterrows():
        user_p_str = "Yes" if user_p else "No "
        doc_p_str = "Yes" if doc_p else "No "
        user_c_str = "Right" if user_c else "Wrong"
        doc_c_str = "Right" if doc_c else "Wrong"
        log(f" {user_p_str} | {doc_p_str}  | {user_c_str} | {doc_c_str} | {row['correct_rate']:7.1%} | {int(row['count']):5d}")
    
    # Compute and display UNKNOWN breakdown
    unknown_breakdown = compute_unknown_breakdown(df)
    log(f"\n{'─'*70}")
    log("UNKNOWN Rate by Prior Combinations")
    log(f"{'─'*70}")
    log("User | Doc  | User  | Doc   | UNKNOWN | UNKNOWN |")
    log("Pres | Pres | Corr  | Corr  |  Rate   |  Count  | Total")
    log("-----|------|-------|-------|---------|---------|------")
    
    for (user_p, doc_p, user_c, doc_c), row in unknown_breakdown.iterrows():
        user_p_str = "Yes" if user_p else "No "
        doc_p_str = "Yes" if doc_p else "No "
        user_c_str = "Right" if user_c else "Wrong"
        doc_c_str = "Right" if doc_c else "Wrong"
        log(f" {user_p_str} | {doc_p_str}  | {user_c_str} | {doc_c_str} | {row['unknown_rate']:7.1%} | {int(row['unknown_count']):7d} | {int(row['total_count']):5d}")
    
    # Fit logistic regression
    log(f"\n{'─'*70}")
    log("5-Parameter Logistic Regression (with Cluster-Robust SEs)")
    log(f"{'─'*70}")
    
    model = fit_logistic_regression(df, X)
    
    # Display summary
    log("\nModel Formula:")
    log("log(p/(1-p)) = β₀ + β_P*P_i + δ_U*U_pres + δ_D*D_pres + β_U*(U_pres*U_corr) + β_D*(D_pres*D_corr)")
    log("\nNote: Using cluster-robust standard errors (clustered by qid) to account for repeated measures")
    log("\nRegression Summary:")
    log(model.summary().as_text())
    
    # Display odds ratios
    odds = format_odds_ratios(model)
    log(f"\n{'─'*70}")
    log("Odds Ratios (exp(β))")
    log(f"{'─'*70}")
    log("\nParameter Interpretation:")
    log("- βP: Effect of parametric correctness")
    log("- δU/δD: Display effect of any cue (right or wrong)")
    log("- βU/βD: Additional boost when cue is also correct")
    log("- Composite effects: Total effect when prior is present AND correct (exp(δ + β))")
    log("- Selectivity: Benefit-to-harm ratio (exp(β)/exp(δ) = exp(β - δ))")
    log("\nOdds Ratios:")
    for param, value in odds.items():
        if isinstance(value, dict):  # Effects with CIs
            if 'odds_ratio' in value:  # Composite effects
                log(f"  {param:35s}: {value['odds_ratio']:6.3f} (95% CI: [{value['ci_lower']:.3f}, {value['ci_upper']:.3f}])")
            elif 'ratio' in value:  # Selectivity ratios
                log(f"  {param:35s}: {value['ratio']:6.3f} (95% CI: [{value['ci_lower']:.3f}, {value['ci_upper']:.3f}])")
        else:  # Simple odds ratios
            log(f"  {param:35s}: {value:6.3f}")
    
    # Prepare outputs
    regression_results = extract_regression_results(model, df)
    breakdown_json = format_breakdown_json(breakdown, unknown_breakdown, df)
    
    # Write files
    log(f"\n{'─'*70}")
    log("Saving Results")
    log(f"{'─'*70}")
    
    (out_dir / "logistic_regression_summary.txt").write_text("\n".join(log_buffer))
    (out_dir / "logistic_regression_results.json").write_text(json.dumps(regression_results, indent=2, ensure_ascii=False))
    (out_dir / "logistic_regression_breakdown.json").write_text(json.dumps(breakdown_json, indent=2, ensure_ascii=False))
    
    log(f"\n✓ Saved:")
    log(f"  - logistic_regression_summary.txt")
    log(f"  - logistic_regression_results.json") 
    log(f"  - logistic_regression_breakdown.json")
    log(f"  in {out_dir}")

if __name__ == "__main__":
    main()