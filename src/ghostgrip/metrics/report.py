"""Print the results table for a logged session, or build and print a synthetic one.

    python -m ghostgrip.metrics.report data/local/sessions/s1              # an existing session
    python -m ghostgrip.metrics.report data/local/sessions/s1 --csv out.csv --plot out.png
    python -m ghostgrip.metrics.report --synthetic-demo data/local/demo    # pipeline check

Real and synthetic data are reported separately and never pooled. The synthetic demo is a check
that the modules fit together, not an experiment.
"""
import argparse
from pathlib import Path

from .analysis import (assist_availability, format_summary, load_frame_rows, load_trial_records,
                       plot_success_rates, prediction_error, summarize_trials, write_summary_csv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Results for a logged GhostGrip session.")
    parser.add_argument("session", nargs="?", help="session folder holding frames.csv / trials.csv")
    parser.add_argument("--synthetic-demo", metavar="OUT_DIR",
                        help="run the synthetic traces through the pipeline into OUT_DIR, then report")
    parser.add_argument("--seeds", type=int, default=2, help="seeds per scenario for the demo (default 2)")
    parser.add_argument("--csv", help="also write the summary table to this CSV file")
    parser.add_argument("--plot", help="also write a success-rate bar chart (needs matplotlib)")
    args = parser.parse_args(argv)

    if args.synthetic_demo:
        from .synthetic_session import run_synthetic_session
        counts = run_synthetic_session(args.synthetic_demo, seeds=tuple(range(args.seeds)))
        print(f"synthetic session written to {args.synthetic_demo}: {counts}")
        folder = args.synthetic_demo
    elif args.session:
        folder = args.session
    else:
        parser.error("give a session folder, or --synthetic-demo OUT_DIR")
    if not Path(folder, "trials.csv").exists():
        print(f"no trials.csv in {folder}")
        return 1

    summaries = summarize_trials(load_trial_records(folder))
    if not summaries:
        print("the session has no trials yet")
        return 1
    frames = load_frame_rows(folder)
    print(format_summary(summaries, prediction_error(frames), assist_availability(frames)))
    if args.csv:
        write_summary_csv(args.csv, summaries)
        print(f"summary table written to {args.csv}")
    if args.plot:
        plot_success_rates(summaries, args.plot)
        print(f"plot written to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
