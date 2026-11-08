"""Print the results table for logged sessions, or build and print a synthetic one.

    python -m ghostgrip.metrics.report data/local/sessions/s1                 # one session
    python -m ghostgrip.metrics.report data/local/sessions/P01 data/local/sessions/P02 data/local/sessions/P03
    python -m ghostgrip.metrics.report data/local/sessions/s1 --csv out.csv --plot out.png
    python -m ghostgrip.metrics.report --synthetic-demo data/local/demo       # pipeline check

With several session folders you get one table per participant, then one descriptive table over all
of them (the plan asks for both; with so few people it says nothing about significance). Real and
synthetic data are reported separately and never pooled. The synthetic demo is a check that the
modules fit together, not an experiment.
"""
import argparse
from pathlib import Path

from .analysis import (assist_availability, format_summary, load_frame_rows, load_trial_records,
                       plot_success_rates, prediction_error, summarize_trials, write_summary_csv)


def _table(records, frames) -> str:
    return format_summary(summarize_trials(records), prediction_error(frames), assist_availability(frames))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Results for logged GhostGrip sessions.")
    parser.add_argument("sessions", nargs="*", help="session folder(s) holding frames.csv / trials.csv")
    parser.add_argument("--synthetic-demo", metavar="OUT_DIR",
                        help="run the synthetic traces through the pipeline into OUT_DIR, then report")
    parser.add_argument("--seeds", type=int, default=2, help="seeds per scenario for the demo (default 2)")
    parser.add_argument("--csv", help="also write the (combined) summary table to this CSV file")
    parser.add_argument("--plot", help="also write a success-rate bar chart (needs matplotlib)")
    args = parser.parse_args(argv)

    if args.synthetic_demo:
        from .synthetic_session import run_synthetic_session
        counts = run_synthetic_session(args.synthetic_demo, seeds=tuple(range(args.seeds)))
        print(f"synthetic session written to {args.synthetic_demo}: {counts}")
        folders = [args.synthetic_demo]
    elif args.sessions:
        folders = args.sessions
    else:
        parser.error("give a session folder, or --synthetic-demo OUT_DIR")

    loaded = []
    for folder in folders:
        if not Path(folder, "trials.csv").exists():
            print(f"no trials.csv in {folder}")
            return 1
        records = load_trial_records(folder)
        if not records:
            print(f"the session {folder} has no trials yet")
            return 1
        loaded.append((folder, records, load_frame_rows(folder)))

    ids = [{r.session_id for r in records} for _, records, _ in loaded]
    for i, first in enumerate(ids):
        for j in range(i + 1, len(ids)):
            clash = first & ids[j]
            if clash:
                print(f"error: session id {sorted(clash)[0]!r} appears in both {loaded[i][0]} and {loaded[j][0]}; "
                      "combining them would merge different participants' trials. Give each participant "
                      "their own session id.")
                return 1

    for folder, records, frames in loaded:
        if len(loaded) > 1:
            print(f"===== SESSION {Path(folder).name} =====")
        print(_table(records, frames))
    all_records = [r for _, records, _ in loaded for r in records]
    all_frames = [row for _, _, frames in loaded for row in frames]
    if len(loaded) > 1:
        print(f"===== ALL {len(loaded)} SESSIONS TOGETHER (descriptive only) =====")
        print(_table(all_records, all_frames))
    summaries = summarize_trials(all_records)
    if args.csv:
        write_summary_csv(args.csv, summaries)
        print(f"summary table written to {args.csv}")
    if args.plot:
        plot_success_rates(summaries, args.plot)
        print(f"plot written to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
