#!/usr/bin/env python3
"""
Cloud-only test runner for the BLINK benchmark.

This script dispatches benchmark runs for supported cloud models. It no longer
manages or starts local servers.

Usage:
    # Single model (OpenAI)
    python auto_mode_runner.py --models gpt-4o --task Relative_Depth --dataset BLINK

    # Single model (OpenRouter)
    python auto_mode_runner.py --models qwen/qwen3-vl-8b-instruct \
        --task Semantic_Correspondence --dataset BLINK

    # Multiple models
    python auto_mode_runner.py --models gpt-4o qwen/qwen3-vl-8b-instruct \
        --task Relative_Depth --dataset BLINK --num_threads 16
"""

import argparse
import subprocess
import sys
from typing import List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    base_url: str
    provider_only: Optional[str] = None


# Model configurations
MODEL_CONFIGS = {
    # OpenAI model
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        base_url="https://api.openai.com/v1",
    ),
    # OpenRouter model
    "qwen/qwen3-vl-8b-instruct": ModelConfig(
        name="qwen/qwen3-vl-8b-instruct",
        base_url="https://openrouter.ai/api/v1",
        provider_only="parasail/bf16,alibaba",
    ),
}


def run_benchmark(
    model_config: ModelConfig,
    task: str,
    dataset: str,
    extra_args: List[str]
):
    """
    Run benchmark for a single model.

    Args:
        model_config: Model configuration
        task: Task name (Relative_Depth or Semantic_Correspondence)
        dataset: Dataset type (BLINK, DA-2K, SPair-71k)
        extra_args: Additional arguments to pass to manual_mode_runner.py
    """
    base_url = model_config.base_url
    logger.info(f"Using API at {base_url}")

    # Build command
    cmd = [
        "uv", "run", "python", "manual_mode_runner.py",
        "--model_names", model_config.name,
        "--task_name", task,
        "--dataset_type", dataset,
        "--openai_base_url", base_url,
    ]
    if model_config.provider_only:
        cmd.extend(["--provider-only", model_config.provider_only])

    cmd.extend(extra_args)

    logger.info(f"Running benchmark for {model_config.name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        # Run benchmark
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False  # Show output in real-time
        )
        logger.info(f"Benchmark completed for {model_config.name}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark failed for {model_config.name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run BLINK benchmark with cloud-hosted models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Core arguments
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to test (cloud models only)"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["Relative_Depth", "Semantic_Correspondence"],
        help="Task to run"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["BLINK", "DA-2K", "SPair-71k"],
        help="Dataset to use"
    )

    # Pass-through arguments for manual_mode_runner.py
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads for parallel execution"
    )
    parser.add_argument(
        "--run_time",
        type=int,
        default=1,
        help="Number of times to run each configuration"
    )
    parser.add_argument(
        "--show_scripts",
        action="store_true",
        help="Print all generated commands"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show commands without executing"
    )
    parser.add_argument(
        "--debug_run",
        action="store_true",
        help="Run only first 10 samples"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force fresh start without resuming"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=25,
        help="Save progress every N queries"
    )

    args = parser.parse_args()

    # Validate task-dataset combination
    valid_combinations = {
        "Relative_Depth": ["BLINK", "DA-2K"],
        "Semantic_Correspondence": ["BLINK", "SPair-71k"]
    }

    if args.dataset not in valid_combinations[args.task]:
        parser.error(
            f"Invalid combination: {args.task} does not support {args.dataset}. "
            f"Valid datasets for {args.task}: {valid_combinations[args.task]}"
        )

    # Build extra arguments to pass through
    extra_args = [
        "--num_threads", str(args.num_threads),
        "--run_time", str(args.run_time),
        "--checkpoint_interval", str(args.checkpoint_interval),
    ]

    if args.show_scripts:
        extra_args.append("--show_scripts")
    if args.dry_run:
        extra_args.append("--dry_run")
    if args.debug_run:
        extra_args.append("--debug_run")
    if args.overwrite:
        extra_args.append("--overwrite")

    try:
        # Process each model
        for model_name in args.models:
            # Get model config
            if model_name not in MODEL_CONFIGS:
                logger.error(
                    f"Unknown model: {model_name}. "
                    f"Available models: {list(MODEL_CONFIGS.keys())}"
                )
                continue

            model_config = MODEL_CONFIGS[model_name]

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing model: {model_name}")
            logger.info(f"Task: {args.task}, Dataset: {args.dataset}")
            logger.info(f"{'='*60}\n")

            try:
                run_benchmark(
                    model_config,
                    args.task,
                    args.dataset,
                    extra_args
                )
            except Exception as e:
                logger.error(f"Failed to run benchmark for {model_name}: {e}")
                if len(args.models) == 1:
                    # If only one model, re-raise the exception
                    raise
                # Otherwise continue with next model
                continue

        logger.info("\n" + "="*60)
        logger.info("All benchmarks completed!")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
