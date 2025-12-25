#!/usr/bin/env python3
"""
Evaluation Script for RL-Trained Models

This script evaluates your RL-trained model and compares it to the base model
to see if RL training improved performance.

Usage:
    python scripts/evaluate.py experiments/medqa-baseline
    python scripts/evaluate.py experiments/medqa-baseline --base-model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/evaluate.py --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import verifiers as vf


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL-trained models")

    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (e.g., experiments/medqa-baseline)"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to compare against (default: auto-detect from config)"
    )

    parser.add_argument(
        "--env",
        type=str,
        default="medqa",
        help="Environment to evaluate on (default: medqa)"
    )

    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100)"
    )

    parser.add_argument(
        "--use-think",
        action="store_true",
        help="Use chain-of-thought reasoning tags"
    )

    parser.add_argument(
        "--shuffle-answers",
        action="store_true",
        help="Shuffle answer options (tests formatting robustness)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final",
        help="Which checkpoint to evaluate (default: final)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: experiment_dir/eval_results.json)"
    )

    return parser.parse_args()


def load_config(experiment_dir: Path) -> Dict[str, Any]:
    """Load experiment config if available"""
    # Try to find config file
    config_path = experiment_dir / "config.toml"

    if not config_path.exists():
        print(f"⚠️  Config not found at {config_path}")
        return {}

    # Parse TOML (simple approach, you could use tomli library)
    print(f"📄 Loading config from {config_path}")
    # For now, return empty dict (would need tomli to parse)
    return {}


def evaluate_model(
    model_path: str,
    env_name: str,
    num_examples: int,
    use_think: bool = False,
    shuffle_answers: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a model on an environment

    Returns dict with metrics like accuracy, reward, etc.
    """
    print(f"\n📊 Evaluating {model_path}...")
    print(f"   Environment: {env_name}")
    print(f"   Examples: {num_examples}")
    print(f"   Use think: {use_think}")
    print(f"   Shuffle: {shuffle_answers}")

    # Load environment
    try:
        env = vf.load_environment(
            env_name,
            use_think=use_think,
            shuffle_answers=shuffle_answers
        )
    except TypeError:
        # If shuffle_answers not supported
        print("   ⚠️  shuffle_answers not supported, loading without it")
        env = vf.load_environment(env_name, use_think=use_think)

    # Limit to num_examples
    if num_examples < len(env.eval_dataset):
        eval_data = env.eval_dataset.select(range(num_examples))
    else:
        eval_data = env.eval_dataset

    print(f"   Loaded {len(eval_data)} examples")

    # For now, use vf-eval CLI (proper implementation would use vf.evaluate)
    # This is a simplified version - in practice you'd call the evaluation API
    print(f"   ⚠️  TODO: Implement actual evaluation")
    print(f"   For now, run manually:")
    print(f"   uv run vf-eval {env_name} --model {model_path} -n {num_examples}")

    return {
        "model": model_path,
        "env": env_name,
        "num_examples": len(eval_data),
        "accuracy": None,  # Would come from actual eval
        "reward": None,
    }


def print_comparison(base_results: Dict, trained_results: Dict):
    """Print comparison between base and trained models"""
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)

    print("\n🔵 Base Model:")
    print(f"   Model: {base_results['model']}")
    print(f"   Accuracy: {base_results.get('accuracy', 'N/A')}")
    print(f"   Reward: {base_results.get('reward', 'N/A')}")

    print("\n🟢 Trained Model:")
    print(f"   Model: {trained_results['model']}")
    print(f"   Accuracy: {trained_results.get('accuracy', 'N/A')}")
    print(f"   Reward: {trained_results.get('reward', 'N/A')}")

    # Calculate improvement if we have numbers
    if base_results.get('accuracy') and trained_results.get('accuracy'):
        improvement = trained_results['accuracy'] - base_results['accuracy']
        pct_change = (improvement / base_results['accuracy']) * 100

        print(f"\n{'📈' if improvement > 0 else '📉'} Improvement:")
        print(f"   Absolute: {improvement:+.2%}")
        print(f"   Relative: {pct_change:+.1f}%")

        if improvement > 0.05:
            print("\n   ✅ Great! RL training helped significantly!")
        elif improvement > 0.01:
            print("\n   ✅ Good! RL training helped.")
        elif improvement > 0:
            print("\n   ⚠️  Small improvement. Try tuning hyperparameters.")
        else:
            print("\n   ❌ No improvement. Check:")
            print("      - Reward function working correctly?")
            print("      - Learning rate too low/high?")
            print("      - Enough training data?")

    print("\n" + "="*60)


def main():
    args = parse_args()

    # Validate experiment directory
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"❌ Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)

    print("="*60)
    print("🧪 MedARC RL Model Evaluation")
    print("="*60)
    print(f"\nExperiment: {exp_dir}")

    # Load config to get base model if not specified
    config = load_config(exp_dir)

    # Determine base model
    if args.base_model:
        base_model = args.base_model
    elif config.get("model", {}).get("name_or_path"):
        base_model = config["model"]["name_or_path"]
    else:
        # Default fallback
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"⚠️  Base model not specified, using default: {base_model}")

    # Determine trained model path
    checkpoint_dir = exp_dir / args.checkpoint
    if not checkpoint_dir.exists():
        # Try with "checkpoint-" prefix
        checkpoint_dir = exp_dir / f"checkpoint-{args.checkpoint}"
        if not checkpoint_dir.exists():
            print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
            print(f"   Available checkpoints in {exp_dir}:")
            for item in exp_dir.iterdir():
                if item.is_dir() and (item.name.startswith("checkpoint-") or item.name == "final"):
                    print(f"   - {item.name}")
            sys.exit(1)

    trained_model = str(checkpoint_dir)

    print(f"\n🔵 Base model: {base_model}")
    print(f"🟢 Trained model: {trained_model}")

    # Evaluate base model
    print("\n" + "-"*60)
    print("Evaluating BASE model...")
    print("-"*60)
    base_results = evaluate_model(
        base_model,
        args.env,
        args.num_examples,
        args.use_think,
        args.shuffle_answers
    )

    # Evaluate trained model
    print("\n" + "-"*60)
    print("Evaluating TRAINED model...")
    print("-"*60)
    trained_results = evaluate_model(
        trained_model,
        args.env,
        args.num_examples,
        args.use_think,
        args.shuffle_answers
    )

    # Compare results
    print_comparison(base_results, trained_results)

    # Save results
    output_path = args.output or exp_dir / "eval_results.json"
    results = {
        "base_model": base_results,
        "trained_model": trained_results,
        "config": {
            "env": args.env,
            "num_examples": args.num_examples,
            "use_think": args.use_think,
            "shuffle_answers": args.shuffle_answers,
        }
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)

    print("\n💡 Next steps:")
    print("   1. Review results in wandb")
    print("   2. Try different hyperparameters if needed")
    print("   3. Test on full dataset if results look good")
    print("   4. Try other environments (MedCaseReasoning, etc.)")


if __name__ == "__main__":
    main()
