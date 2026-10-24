"""Master Reproducibility Script — runs the entire GovRe evaluation pipeline.

Usage:
    python run_all.py              # Full pipeline (requires OpenRouter API key)
    python run_all.py --local-only # Only runs components that don't need API

Execution order:
    1. Build/load FAISS indexes for all domains (original + edited)
    2. Run main evaluation (System A vs System B, 3 domains, 3 seeds)
    3. Run policy-change robustness experiment
    4. Run human validation study
    5. Generate publication figures and summary tables
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="GovRe: Full Evaluation Pipeline")
    parser.add_argument("--local-only", action="store_true",
                        help="Run only local components (no API calls)")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip FAISS index rebuild")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    print("=" * 70)
    print(" GovRe: Governance Grounding Research — Full Pipeline")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Mode: {'LOCAL ONLY' if args.local_only else 'FULL (API + Local)'}")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check API key
    from src.config import OPENROUTER_API_KEY, DEFAULT_OPENROUTER_MODEL
    if not args.local_only:
        if not OPENROUTER_API_KEY:
            print("ERROR: OPENROUTER_API_KEY not set. Set it in .env or run with --local-only")
            sys.exit(1)
        print(f"API Key: ...{OPENROUTER_API_KEY[-8:]}")
        print(f"Model: {DEFAULT_OPENROUTER_MODEL}")
    print()

    # Step 1: Build FAISS indexes
    if not args.skip_index:
        print("[1/5] Building FAISS indexes...")
        from src.index import build_all_indexes
        build_all_indexes(force_rebuild=True, build_edited=True)
        print("Done.\n")
    else:
        print("[1/5] Skipping FAISS index build.\n")

    # Step 2: Main evaluation
    if not args.local_only:
        print("[2/5] Running main evaluation (System A vs System B)...")
        from eval.run_eval import run_all_evaluations
        run_all_evaluations()
        print("Done.\n")
    else:
        print("[2/5] Skipping main evaluation (requires API).\n")

    # Step 3: Policy-change robustness experiment
    if not args.local_only:
        print("[3/5] Running policy-change robustness experiment...")
        from src.policy_edit_experiment import run_policy_edit_experiment
        run_policy_edit_experiment()
        print("Done.\n")
    else:
        print("[3/5] Skipping robustness experiment (requires API).\n")

    # Step 4: Human validation study
    print("[4/5] Running human validation study...")
    from src.human_validation import run_human_validation_study
    run_human_validation_study()
    print("Done.\n")

    # Step 5: Generate figures
    if not args.skip_figures:
        print("[5/5] Generating publication figures...")
        try:
            from src.generate_figures import generate_all_plots
            generate_all_plots()
            print("Done.\n")
        except Exception as e:
            print(f"Figure generation failed (may need results from step 2): {e}\n")
    else:
        print("[5/5] Skipping figure generation.\n")

    print("=" * 70)
    print(f" Pipeline complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Results directory: {PROJECT_ROOT / 'results'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
