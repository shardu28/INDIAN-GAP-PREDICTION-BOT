"""
main.py
=======
Entry point for the Indian Gap Prediction Bot pipeline.

Usage:
    python src/main.py                    # live mode
    python src/main.py --mode backtest    # backtest mode
    python src/main.py --send-signal      # live + send email signal
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_collector import run_data_collection

logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Indian Gap Prediction Bot"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "backtest"],
        default="live",
        help="Run mode: live (default) or backtest",
    )
    parser.add_argument(
        "--send-signal",
        action="store_true",
        help="Send email signal after analysis (Phase 4+)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting pipeline | mode={args.mode}")

    # --- Phase 1: Data Collection ---
    results = run_data_collection(mode=args.mode)

    # --- Phase 2+: Placeholder (will be uncommented in future phases) ---
    # from feature_engineering import run_feature_engineering
    # features = run_feature_engineering(results)

    # from regime_detector import run_regime_detection
    # regime = run_regime_detection(features)

    # from probability_model import run_probability_model
    # probs = run_probability_model(features, regime)

    # from option_selector import run_option_selection
    # signal = run_option_selection(probs)

    # if args.send_signal:
    #     from notifier import send_email_signal
    #     send_email_signal(signal)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
