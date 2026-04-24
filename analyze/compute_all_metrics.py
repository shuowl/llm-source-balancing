#!/usr/bin/env python3
"""
compute_all_metrics.py
======================
Orchestration script to compute all three types of metrics for experiments:
1. Logistic regression analysis 
2. Choice-level metrics analysis
3. Distribution metrics analysis


Usage
-----
# Process all experiments in results directories (can specify multiple):
python analyze/compute_all_metrics.py --results-dir results

# Parallel processing options:
python analyze/compute_all_metrics.py --results-dir results --num-workers 12

Parallel Processing Notes
-------------------------
- The script processes experiments in parallel by default (up to 16 workers)
- Default workers = min(CPU count, 16) to prevent resource oversubscription
- Use --num-workers to control parallelism (set to 1 for sequential processing)
- On shared systems with limited vCPUs, use fewer workers than your allocation
  Example: If you have 16 vCPUs, use --num-workers 8-12 for best performance
- Too many workers can cause failures due to resource limits or I/O bottlenecks
"""

import argparse
import pathlib
import subprocess
import sys
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Lock
from functools import partial
import threading

# ─── make utils import-able ───────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ---------------------------------------------------------------------

def get_results_dir(exp_name: str, custom_results_dir: pathlib.Path = None) -> pathlib.Path:
    """Get the results directory for an experiment.
    
    If custom_results_dir is provided, use it as the parent directory.
    Otherwise, use the default ROOT/results directory.
    """
    if custom_results_dir:
        return custom_results_dir / exp_name
    return ROOT / "results" / exp_name

def check_file_exists(filepath: pathlib.Path) -> bool:
    """Check if a file exists."""
    return filepath.exists()

def check_all_metrics_exist(exp_name: str, metrics_to_compute: list[str], custom_results_dir: pathlib.Path = None) -> bool:
    """Check if ALL requested metric files exist for an experiment.
    
    Returns True only if ALL metric files exist, False if ANY is missing.
    This implements the all-or-nothing logic for --results-dir mode.
    """
    results_dir = get_results_dir(exp_name, custom_results_dir)
    
    metric_files = {
        "logistic": "logistic_regression_results.json",
        "choice": "choice_metrics.json",
        "distribution": "distribution_metrics.json"
    }
    
    for metric in metrics_to_compute:
        if metric in metric_files:
            filepath = results_dir / metric_files[metric]
            if not filepath.exists():
                return False
    
    return True

def run_command_if_needed(cmd: list[str], output_file: pathlib.Path, skip_existing: bool = False, quiet: bool = False):
    """Run command unless output file exists and skip_existing is True."""
    if skip_existing and output_file.exists():
        if not quiet:
            print(f"  ✓ Skipping, {output_file.name} already exists")
        return True
    
    try:
        if not quiet:
            print(f"    Running: {' '.join(cmd)}")
        # Suppress output in quiet mode
        if quiet:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            # Check if there was an error in stderr
            if result.stderr and "error" in result.stderr.lower():
                return False
        else:
            subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"  ✗ Error running command: {' '.join(cmd)}")
            print(f"    Return code: {e.returncode}")
            if e.stderr:
                print(f"    Error: {e.stderr}")
        return False
    except Exception as e:
        if not quiet:
            print(f"  ✗ Unexpected error: {e}")
        return False

def compute_logistic_metrics(exp_name: str, skip_existing: bool = False, custom_results_dir: pathlib.Path = None, quiet: bool = False) -> bool:
    """Compute logistic regression analysis for an experiment."""
    results_dir = get_results_dir(exp_name, custom_results_dir)
    output_file = results_dir / "logistic_regression_results.json"
    
    if skip_existing and output_file.exists():
        if not quiet:
            print(f"  ✓ Logistic regression results already exist")
        return True
    
    cmd = [
        sys.executable, 
        str(ROOT / "analyze" / "logistic_regression_analysis.py"),
        "--experiment_name", exp_name
    ]
    
    # Add custom results directory if provided
    if custom_results_dir:
        cmd.extend(["--results-dir", str(custom_results_dir)])
    
    return run_command_if_needed(cmd, output_file, skip_existing, quiet)

def compute_choice_metrics(exp_name: str, skip_existing: bool = False, custom_results_dir: pathlib.Path = None, quiet: bool = False) -> bool:
    """Compute choice-level metrics for an experiment."""
    results_dir = get_results_dir(exp_name, custom_results_dir)
    output_file = results_dir / "choice_metrics.json"
    
    if skip_existing and output_file.exists():
        if not quiet:
            print(f"  ✓ Choice metrics already exist")
        return True
    
    cmd = [
        sys.executable,
        str(ROOT / "analyze" / "choice_metrics_analysis.py"),
        "--experiment_name", exp_name
    ]
    
    # Add custom results directory if provided
    if custom_results_dir:
        cmd.extend(["--results-dir", str(custom_results_dir)])
    
    return run_command_if_needed(cmd, output_file, skip_existing, quiet)

def process_single_experiment(exp_data: dict, metrics_to_compute: list[str], skip_existing: bool, args) -> tuple[str, int, bool, str]:
    """Process a single experiment. Returns (exp_name, success_count, overall_success, error_msg)."""
    exp_name = exp_data["name"]
    custom_results_dir = exp_data.get("results_dir", None)
    error_msg = ""
    
    try:
        # Check if results directory exists
        results_dir = get_results_dir(exp_name, custom_results_dir)
        if not results_dir.exists():
            return exp_name, 0, False, f"Results directory not found: {results_dir}"
        
        # Check if merged_results.jsonl exists (needed for all metrics)
        merged_results_file = results_dir / "merged_results.jsonl"
        has_merged_results = merged_results_file.exists()
        
        # For --results-dir mode without --skip-existing: always recompute all metrics
        if args.results_dir and not args.skip_existing:
            pass  # Will recompute all
        elif args.results_dir and args.skip_existing and check_all_metrics_exist(exp_name, metrics_to_compute, custom_results_dir):
            # Only skip if --skip-existing is provided AND all metrics exist
            return exp_name, len(metrics_to_compute), True, ""
        elif args.results_dir and args.skip_existing:
            # If using --results-dir with --skip-existing and ANY metric is missing, recompute ALL
            pass
        
        exp_success = True
        success_count = 0
        
        # Compute requested metrics
        # When using --results-dir without --skip-existing, we force recomputation
        skip_existing_local = args.skip_existing
        
        if "logistic" in metrics_to_compute:
            if has_merged_results:
                success = compute_logistic_metrics(exp_name, skip_existing_local, custom_results_dir, quiet=True)
                if success:
                    success_count += 1
                else:
                    exp_success = False
                    error_msg = f"Logistic regression analysis failed"
                    raise RuntimeError(error_msg)
        
        if "choice" in metrics_to_compute:
            if has_merged_results:
                success = compute_choice_metrics(exp_name, skip_existing_local, custom_results_dir, quiet=True)
                if success:
                    success_count += 1
                else:
                    exp_success = False
                    error_msg = f"Choice metrics analysis failed"
                    raise RuntimeError(error_msg)
        
        if "distribution" in metrics_to_compute:
            success = compute_distribution_metrics(exp_name, skip_existing_local, custom_results_dir, quiet=True)
            if success:
                success_count += 1
            else:
                exp_success = False
                error_msg = f"Distribution metrics analysis failed"
                raise RuntimeError(error_msg)
                
    except Exception as e:
        exp_success = False
        if not error_msg:
            error_msg = str(e)
    
    return exp_name, success_count, exp_success, error_msg


def compute_distribution_metrics(exp_name: str, skip_existing: bool = False, custom_results_dir: pathlib.Path = None, quiet: bool = False) -> bool:
    """Compute distribution metrics for an experiment."""
    results_dir = get_results_dir(exp_name, custom_results_dir)
    output_file = results_dir / "distribution_metrics.json"
    
    if skip_existing and output_file.exists():
        if not quiet:
            print(f"  ✓ Distribution metrics already exist")
        return True
    
    # Check if merged_results.jsonl exists
    merged_results_file = results_dir / "merged_results.jsonl"
    if not merged_results_file.exists():
        if not quiet:
            print(f"  ✗ No merged_results.jsonl found, skipping distribution metrics")
        return False
    
    cmd = [
        sys.executable,
        str(ROOT / "analyze" / "distribution_metrics_analysis.py"),
        "--experiment_name", exp_name
    ]
    
    # Add custom results directory if provided
    if custom_results_dir:
        cmd.extend(["--results-dir", str(custom_results_dir)])
    
    return run_command_if_needed(cmd, output_file, skip_existing, quiet)

def main():
    parser = argparse.ArgumentParser(description="Compute all metrics for experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to config YAML file")
    group.add_argument("--experiment-names", nargs="+", help="List of experiment names (results directories)")
    group.add_argument("--results-dir", nargs="+", help="Process all experiment subdirectories in the given results directory/directories")
    parser.add_argument("--metrics-only", 
                       help="Comma-separated list of metrics to compute (logistic,choice,distribution)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip computation if output files already exist")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of parallel workers (default: number of CPU cores)")
    args = parser.parse_args()
    
    # Get list of experiments
    if args.config:
        # Load from config file
        config_path = pathlib.Path(args.config)
        if not config_path.exists():
            sys.exit(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        experiments = config.get("experiments", [])
        if not experiments:
            sys.exit("No experiments found in config file")
        experiment_names = [exp["name"] for exp in experiments]
    elif args.experiment_names:
        # Use provided experiment names
        experiment_names = args.experiment_names
        experiments = [{"name": name} for name in experiment_names]
    else:
        # Process all subdirectories in results directories
        experiments = []
        experiment_to_dir = {}  # Map experiment name to its parent directory
        
        for results_dir_str in args.results_dir:
            results_dir = pathlib.Path(results_dir_str)
            if not results_dir.exists():
                print(f"Warning: Results directory not found: {results_dir}")
                continue
            
            # Get all subdirectories (each is an experiment)
            for subdir in sorted(results_dir.iterdir()):
                if subdir.is_dir() and not subdir.name.startswith('.') and subdir.name != '.cache':
                    # Check if it has merged_results.jsonl to be a valid experiment dir
                    if (subdir / "merged_results.jsonl").exists():
                        exp_name = subdir.name
                        # Store the experiment with its parent directory
                        experiments.append({"name": exp_name, "results_dir": results_dir})
                        experiment_to_dir[exp_name] = results_dir
        
        if not experiments:
            sys.exit(f"No valid experiment directories found in {args.results_dir}")
        
        experiment_names = [exp["name"] for exp in experiments]
        print(f"Found {len(experiments)} experiment directories across {len(args.results_dir)} results directories")
    
    # Parse metrics filter
    all_metrics = ["logistic", "choice", "distribution"]
    if args.metrics_only:
        requested_metrics = [m.strip() for m in args.metrics_only.split(",")]
        invalid_metrics = set(requested_metrics) - set(all_metrics)
        if invalid_metrics:
            sys.exit(f"Invalid metrics: {invalid_metrics}. Valid: {all_metrics}")
        metrics_to_compute = requested_metrics
    else:
        metrics_to_compute = all_metrics
    
    print(f"Computing metrics for {len(experiment_names)} experiments")
    print(f"Metrics to compute: {', '.join(metrics_to_compute)}")
    if args.skip_existing:
        print("Skipping existing output files")
    
    # Determine number of workers (limit to reasonable number)
    if args.num_workers:
        num_workers = args.num_workers
    else:
        # Use CPU count but cap at 16 to avoid overwhelming the system
        num_workers = min(cpu_count(), 16)
    print(f"Using {num_workers} parallel workers")
    print()
    
    success_count = 0
    total_metrics = len(experiment_names) * len(metrics_to_compute)
    
    # Create a partial function with fixed arguments
    process_func = partial(process_single_experiment, 
                          metrics_to_compute=metrics_to_compute,
                          skip_existing=args.skip_existing,
                          args=args)
    
    # Process experiments in parallel
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = []
        with tqdm(total=len(experiments), desc="Experiments") as pbar:
            for result in pool.imap_unordered(process_func, experiments):
                exp_name, exp_success_count, exp_success, error_msg = result
                results.append((exp_name, exp_success_count, exp_success, error_msg))
                success_count += exp_success_count
                pbar.update(1)
                
                # Print summary for each completed experiment
                if exp_success:
                    pbar.write(f"✓ {exp_name}: {exp_success_count}/{len(metrics_to_compute)} metrics computed")
                else:
                    pbar.write(f"✗ {exp_name}: Failed - {error_msg}")
    
    # Count failures
    failed_experiments = [(name, error_msg) for name, _, success, error_msg in results if not success]
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{total_metrics} metric computations completed successfully")
    
    if success_count == total_metrics:
        print("✓ All metrics computed successfully!")
    else:
        print(f"⚠ {total_metrics - success_count} metric computations failed or were skipped")
        
    if failed_experiments:
        print(f"\nFailed experiments ({len(failed_experiments)}):")
        for name, error_msg in failed_experiments:
            print(f"  - {name}: {error_msg}")

if __name__ == "__main__":
    main()
