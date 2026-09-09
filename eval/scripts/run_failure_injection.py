"""
Orchestrate the failure injection protocol (injection designer).

Three phases:
  1. Apply — read injection configs, apply mutations via apply_injection
  2. Pause — wait for the blinded evaluator to complete diagnosis
  3. Revert — restore all mutations from backups

Usage:
    cd backend && uv run python -m eval.scripts.run_failure_injection \
        --config-dir eval/results/failure_injection/experiment_run_001 \
        --db-url "$DATABASE_URL"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from eval.scripts.apply_injection import inject, revert


def load_injection_configs(config_dir: Path) -> list[dict]:
    """Load all per-session injection config files."""
    configs_dir = config_dir / "injection_configs"
    if not configs_dir.exists():
        print(f"ERROR: {configs_dir} does not exist")
        sys.exit(1)

    configs = []
    for config_path in sorted(configs_dir.glob("FI-*.json")):
        with open(config_path) as f:
            configs.append(json.load(f))
    return configs


def phase_apply(db_url: str, configs: list[dict], backups_path: Path) -> list[dict]:
    """Phase 1: Apply all injections and save backups."""
    print(f"\n{'='*60}")
    print("PHASE 1: APPLYING INJECTIONS")
    print(f"{'='*60}\n")

    backups = []
    for config in configs:
        sid = config["session_id"]
        ftype = config["failure_type"]
        target = config["target_config"]

        if target.get("placeholder"):
            print(f"  SKIP {sid} ({ftype}): placeholder target (dry-run artifact)")
            continue

        print(f"  Injecting {sid}: {ftype}...", end=" ", flush=True)
        try:
            backup = inject(db_url, ftype, target)
            backup["session_id"] = sid
            backup["failure_type"] = ftype
            backups.append(backup)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            print("\nAborting. Reverting already-applied injections...")
            phase_revert(db_url, backups)
            sys.exit(1)

    # Save backups
    backups_path.write_text(json.dumps(backups, indent=2, default=str))
    print(f"\n  Backups saved to {backups_path}")
    print(f"  {len(backups)} injections applied successfully.")
    return backups


def phase_pause() -> None:
    """Phase 2: Wait for the blinded evaluator to complete diagnosis."""
    print(f"\n{'='*60}")
    print("PHASE 2: AWAITING AUTHOR B DIAGNOSIS")
    print(f"{'='*60}\n")
    print("  All injections applied.")
    print("  The blinded evaluator may now run the audit metric suite on all 12 sessions.")
    print("  The blinded evaluator should fill in diagnostic_trace_log_template.json")
    print("  and save as diagnostic_trace_log.json in the experiment directory.\n")
    input("  Press Enter when the blinded evaluator has completed diagnosis... ")


def phase_revert(db_url: str, backups: list[dict]) -> None:
    """Phase 3: Revert all injections from backups."""
    print(f"\n{'='*60}")
    print("PHASE 3: REVERTING INJECTIONS")
    print(f"{'='*60}\n")

    errors = []
    for backup in backups:
        sid = backup["session_id"]
        ftype = backup["failure_type"]
        print(f"  Reverting {sid}: {ftype}...", end=" ", flush=True)
        try:
            revert(db_url, ftype, backup)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            errors.append({"session_id": sid, "failure_type": ftype, "error": str(e)})

    if errors:
        print(f"\n  WARNING: {len(errors)} revert(s) failed:")
        for err in errors:
            print(f"    {err['session_id']} ({err['failure_type']}): {err['error']}")
        print("  Manual intervention may be required.")
    else:
        print(f"\n  All {len(backups)} injections reverted successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate failure injection protocol (injection designer)"
    )
    parser.add_argument(
        "--config-dir", type=Path, required=True,
        help="Experiment directory containing injection_configs/",
    )
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="DATABASE_URL (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--revert-only", action="store_true",
        help="Skip apply/pause; just revert from existing backups.json",
    )
    parser.add_argument(
        "--apply-only", action="store_true",
        help="Apply injections and exit (no pause or revert). Use --revert-only later.",
    )
    args = parser.parse_args()

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: --db-url or DATABASE_URL required")
        sys.exit(1)

    backups_path = args.config_dir / "backups.json"

    if args.revert_only:
        if not backups_path.exists():
            print(f"ERROR: {backups_path} not found")
            sys.exit(1)
        with open(backups_path) as f:
            backups = json.load(f)
        phase_revert(db_url, backups)
        return

    configs = load_injection_configs(args.config_dir)
    if not configs:
        print("ERROR: No injection configs found")
        sys.exit(1)

    print(f"Loaded {len(configs)} injection configs from {args.config_dir}")

    backups = phase_apply(db_url, configs, backups_path)

    if args.apply_only:
        print(f"\n  --apply-only: Injections applied. Revert later with --revert-only.")
        return

    phase_pause()
    phase_revert(db_url, backups)

    print(f"\n{'='*60}")
    print("PROTOCOL COMPLETE")
    print(f"{'='*60}")
    print("  Next: run score_and_unblind.py to reconcile results.")


if __name__ == "__main__":
    main()
