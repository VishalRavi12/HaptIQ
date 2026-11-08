"""Logging and results for GhostGrip experiments (technical doc 10, Phase 1 plan D5).

    logger.py          SessionLogger: three CSV tables (frames, device, trials), restart-safe
    analysis.py        success rate, task time, slip / over-grip rates, prediction error, intent F1
    synthetic_session.py  runs the synthetic traces through the whole headless pipeline
    report.py          command line:  python -m ghostgrip.metrics.report
"""
