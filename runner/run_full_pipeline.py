#!/usr/bin/env python3
"""
Run the full pipeline for each experiment in a config file sequentially.

For each experiment:
1. Run bare probe to generate canonical wrong answers
2. Generate tier sentences
3. Run all probe variants

Usage:
    # Run all experiments in config (defaults to all GPUs with data parallelism)
    python runner/run_full_pipeline.py --config experiments/generated/exp1_config.yaml

    # Run with custom tier generation batch size and concurrency
    python runner/run_full_pipeline.py --config experiments/generated/exp1_config.yaml \
        --tier-batch-size 200 --max-concurrent 50

    # Run with custom probe batch size and 4 HF GPUs
    python runner/run_full_pipeline.py --config experiments/generated/exp1_config.yaml \
        --probe-batch-size 4 --hf-gpus 4

    # Run with vLLM TP-sharded across 2 GPUs (for large reasoning models)
    python runner/run_full_pipeline.py --config experiments/generated/exp1_config.yaml \
        --vllm-tp-size 2 --hf-gpus 4

    # Run tier1-only mode (no API calls for tier2 generation)
    python runner/run_full_pipeline.py --config experiments/generated/exp1_config.yaml \
        --tier1-only

    # Start from a specific experiment (skip earlier ones)
    python runner/run_full_pipeline.py --config experiments/generated/exp1_config.yaml \
        --start-from csqa__llama3_8b__d1nu1nin__nocot

    # Dry run to see what commands would be executed
    python runner/run_full_pipeline.py --config experiments/generated/exp1_config.yaml --dry-run

Arguments:
    --config: Path to experiment config YAML file (required)
    --tier-batch-size: Batch size for tier generation (default: 100)
    --probe-batch-size: Batch size for probe execution (default: 8)
    --batch-size: DEPRECATED umbrella for both (sets both unless specific flags are provided)
    --max-concurrent: Max concurrent API calls for tier generation (default: 100)
    --tier1-only: Only generate tier1 data (no API calls for tier2)
    --vllm-tp-size: Tensor parallel size for vLLM (single replica sharded across N GPUs)
    --hf-gpus: Number of GPUs for HF accelerate data parallelism
    --start-from: Start from this experiment name (skip earlier ones)
    --dry-run: Print commands without executing them
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}\n")
    
    try:
        # Run command with output going directly to terminal (not captured)
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            logger.error(f"\nCommand failed with return code {result.returncode}")
            return False
            
        logger.info(f"\n✓ {description} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline for all experiments in config")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML file")
    # Deprecated umbrella flag: if provided, sets both tier/probe batch sizes unless specific flags are given
    parser.add_argument("--batch-size", type=int, default=None,
                       help="DEPRECATED: Use --tier-batch-size and --probe-batch-size. If provided, sets both unless overridden.")
    parser.add_argument("--tier-batch-size", type=int, default=100,
                       help="Batch size for tier generation (default: 100)")
    parser.add_argument("--probe-batch-size", type=int, default=8,
                       help="Batch size for probe execution (default: 8)")
    parser.add_argument("--max-concurrent", type=int, default=100,
                       help="Max concurrent API calls for tier generation (default: 100)")
    parser.add_argument("--tier1-only", action="store_true",
                       help="Only generate tier1 data (no API calls for tier2)")
    parser.add_argument("--vllm-tp-size", type=int, default=None,
                       help="Tensor parallel size for vLLM (single replica sharded across N GPUs)")
    parser.add_argument("--hf-gpus", type=str, default=None,
                       help="Number of GPUs for HF accelerate data parallelism (1-N or 'all')")
    parser.add_argument("--start-from", type=str, default=None,
                       help="Start from this experiment name (skip earlier ones)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing them")
    args = parser.parse_args()

    # Backward-compatibility: map deprecated --batch-size to both, unless specific flags provided
    if args.batch_size is not None:
        if "--tier-batch-size" not in sys.argv:
            args.tier_batch_size = args.batch_size
        if "--probe-batch-size" not in sys.argv:
            args.probe_batch_size = args.batch_size
    
    # Load config file
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    experiments = config.get("experiments", [])
    if not experiments:
        logger.error("No experiments found in config file")
        sys.exit(1)
        
    logger.info(f"Found {len(experiments)} experiments in config")
    
    # Log tier1-only mode if active
    if args.tier1_only:
        logger.info("Running in TIER1-ONLY mode (no API calls for tier2 generation)")
    
    # Process start_from flag
    start_index = 0
    if args.start_from:
        for i, exp in enumerate(experiments):
            if exp["name"] == args.start_from:
                start_index = i
                logger.info(f"Starting from experiment: {args.start_from} (index {i})")
                break
        else:
            logger.error(f"Experiment '{args.start_from}' not found in config")
            sys.exit(1)
    
    # Process each experiment
    for i, exp in enumerate(experiments[start_index:], start=start_index):
        exp_name = exp["name"]
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"# Processing experiment {i+1}/{len(experiments)}: {exp_name}")
        logger.info(f"{'#'*80}\n")
        
        # Step 1: Run bare probe
        cmd1 = [
            "python", "experiments/run_experiments.py",
            "--config", str(config_path),
            "--experiment", exp_name,
            "--probe-variant", "bare",
            "--batch-size", str(args.probe_batch_size)
        ]
        
        # Add GPU configuration parameters if specified
        if args.vllm_tp_size is not None:
            cmd1.extend(["--vllm-tp-size", str(args.vllm_tp_size)])
        if args.hf_gpus is not None:
            cmd1.extend(["--hf-gpus", str(args.hf_gpus)])
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd1)}")
        else:
            if not run_command(cmd1, f"Step 1: Bare probe for {exp_name}"):
                logger.error(f"Failed to run bare probe for {exp_name}")
                logger.error("Stopping pipeline due to error")
                sys.exit(1)
        
        # Step 2: Generate tier sentences
        cmd2 = [
            "python", "core/dataset_tier_generator.py",
            "--config", str(config_path),
            "--experiment", exp_name,
            "--batch-size", str(args.tier_batch_size),
            "--max-concurrent", str(args.max_concurrent)
        ]
        
        # Add tier1-only flag if specified
        if args.tier1_only:
            cmd2.append("--tier1-only")
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd2)}")
        else:
            if not run_command(cmd2, f"Step 2: Tier generation for {exp_name}"):
                logger.error(f"Failed to generate tiers for {exp_name}")
                logger.error("Stopping pipeline due to error")
                sys.exit(1)
        
        # Step 3: Run all probe variants
        cmd3 = [
            "python", "experiments/run_experiments.py",
            "--config", str(config_path),
            "--experiment", exp_name,
            "--batch-size", str(args.probe_batch_size)
        ]
        
        # Add GPU configuration parameters if specified
        if args.vllm_tp_size is not None:
            cmd3.extend(["--vllm-tp-size", str(args.vllm_tp_size)])
        if args.hf_gpus is not None:
            cmd3.extend(["--hf-gpus", str(args.hf_gpus)])
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd3)}")
        else:
            if not run_command(cmd3, f"Step 3: All probe variants for {exp_name}"):
                logger.error(f"Failed to run all probes for {exp_name}")
                logger.error("Stopping pipeline due to error")
                sys.exit(1)
        
        logger.info(f"\n✓ Completed all steps for {exp_name}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Pipeline completed for all experiments")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()