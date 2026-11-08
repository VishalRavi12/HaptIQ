"""The volunteer study: who does which condition in which order, a run sheet, and an audit.

    PYTHONPATH=src python -m ghostgrip.metrics.protocol --participants 4
    PYTHONPATH=src python -m ghostgrip.metrics.protocol --audit data/local/sessions/P01 --participant 1

The four conditions (A baseline, B AI only, C haptic only, D both) are given to each participant in
an order from a balanced Latin square: every condition appears once in every position, and every
condition follows every other exactly once, so practice and tiredness effects are spread evenly
(technical doc 10.1). With four participants all four orders are used once; more repeat them.

Each participant is one session (P01, P02, ...). Within it the experimenter presses F1-F4 to switch
condition, 1 / 2 / 3 to choose glass, sponge or block, and N for each new trial. The measured session
runs with the app's --evaluation flag, which freezes the assist profile and refuses manual gain overrides,
so the conditions stay comparable. See
docs/experiment_protocol.md for the full procedure, consent wording and validity rules.

The audit checks that what was logged matches the plan (order, number of attempts) and that every
run's metadata says it was frozen (--evaluation), camera-driven and servo-off. It cannot tell whether
anyone followed the spoken instructions, and it never edits or drops a trial.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from contracts import CONDITION_CODES
from .analysis import load_trial_records

OBJECTS = ("glass", "sponge", "block")
OBJECT_KEYS = {"glass": "1", "sponge": "2", "block": "3"}
CONDITION_KEYS = {"A": "F1", "B": "F2", "C": "F3", "D": "F4"}
CONDITION_LABELS = {"A": "Baseline", "B": "AI only", "C": "Haptic only", "D": "AI + Haptic"}
DEFAULT_TRIALS_PER_OBJECT = 2
PRACTICE_TRIALS = 3


def balanced_latin_square(n: int = 4) -> List[List[int]]:
    """Williams design for an even n: each value once per position, each ordered neighbour pair once."""
    if n < 2 or n % 2:
        raise ValueError("a balanced Latin square this way needs an even size of at least 2")
    first = [0]
    low, high = 1, n - 1
    while len(first) < n:
        first.append(low)
        low += 1
        if len(first) < n:
            first.append(high)
            high -= 1
    return [[(value + shift) % n for value in first] for shift in range(n)]


def condition_orders() -> List[Tuple[str, ...]]:
    """The four condition orders, as letters."""
    return [tuple(CONDITION_CODES[i] for i in row) for row in balanced_latin_square(len(CONDITION_CODES))]


@dataclass(frozen=True)
class ParticipantPlan:
    number: int
    session_id: str
    order: Tuple[str, ...]
    trials_per_condition: int          # per object x objects

    @property
    def total_trials(self) -> int:
        return self.trials_per_condition * len(self.order)


def make_plans(participants: int, trials_per_object: int = DEFAULT_TRIALS_PER_OBJECT,
               start: int = 1) -> List[ParticipantPlan]:
    if participants < 1 or participants > 50:
        raise ValueError("participants must be between 1 and 50")
    if trials_per_object < 1 or trials_per_object > 10:
        raise ValueError("trials per object must be between 1 and 10")
    if start < 1:
        raise ValueError("the first participant number must be at least 1")
    orders = condition_orders()
    return [ParticipantPlan(n, f"P{n:02d}", orders[(n - 1) % len(orders)], trials_per_object * len(OBJECTS))
            for n in range(start, start + participants)]


def run_sheet(plan: ParticipantPlan, trials_per_object: int = DEFAULT_TRIALS_PER_OBJECT,
              hardware: str = "mock") -> str:
    """What the experimenter does for one participant, step by step."""
    first = plan.order[0]
    lines = [f"PARTICIPANT {plan.session_id}   order {' -> '.join(plan.order)}   "
             f"({plan.total_trials} measured trials, {plan.trials_per_condition} per condition)",
             f"  consent given [ ]   anonymous id {plan.session_id} written down [ ]   other camera apps closed [ ]",
             "  practice (not analysed): ",
             f"    python src/ghostgrip/ui/app.py --camera --hardware {hardware} --condition A "
             f"--session-id {plan.session_id}_practice",
             f"    do {PRACTICE_TRIALS} trials with the participant (key N after each), then press Esc",
             "  measured session:",
             f"    python src/ghostgrip/ui/app.py --camera --hardware {hardware} --condition {first} "
             f"--evaluation --session-id {plan.session_id}"]
    for i, code in enumerate(plan.order, 1):
        switch = "already selected by --condition" if i == 1 else f"press {CONDITION_KEYS[code]}"
        extra = ("   (cues on: press H once to arm; keep the servo at the table stand, not on the person)"
                 if code in ("C", "D") and hardware != "mock" else "")
        lines.append(f"    block {i}: condition {code} ({CONDITION_LABELS[code]}): {switch}{extra}")
        for obj in OBJECTS:
            lines.append(f"        press {OBJECT_KEYS[obj]} ({obj}), then {trials_per_object} trial(s); press N after each   [ ]")
    lines += ["    when all four blocks are done press Esc (an unfinished trial is logged as incomplete)",
              f"  afterwards: PYTHONPATH=src python -m ghostgrip.metrics.protocol --audit data/local/sessions/{plan.session_id} "
              f"--participant {plan.number}",
              "  write down anything unusual (camera lost, lighting changed, someone interrupted); never delete a trial"]
    return "\n".join(lines)


def audit_run_metadata(folder) -> List[str]:
    """Check the run_N_meta.json files the app writes: was every run a frozen, camera-driven,
    servo-off measured run? Missing information is a problem, never assumed to be fine."""
    folder = Path(folder)
    paths = sorted(folder.glob("run_*_meta.json"))
    if not paths:
        return ["no run_*_meta.json found, so it cannot be confirmed that the runs used --evaluation"]
    problems = []
    for path in paths:
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            problems.append(f"{path.name} cannot be read")
            continue
        run = path.stem.replace("_meta", "")
        if meta.get("evaluation") is not True:
            problems.append(f"{run}: not started with --evaluation, so the assist profile could learn")
        if meta.get("assist_profile_frozen") is not True:
            problems.append(f"{run}: the assist profile was not frozen")
        if meta.get("manual_assist_gain") is not None:
            problems.append(f"{run}: a manual assist gain was set ({meta.get('manual_assist_gain')})")
        if meta.get("servo_enabled") is not False:
            problems.append(f"{run}: servo_enabled was not false; the protocol keeps the servo off")
        if meta.get("input") != "camera" or meta.get("origin") != "real":
            problems.append(f"{run}: input was {meta.get('input')!r} labelled {meta.get('origin')!r}, not a live camera "
                            "labelled real")
    return problems


@dataclass(frozen=True)
class AuditResult:
    problems: List[str]
    attempts: Dict[str, int]
    order_seen: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def audit_session(folder, plan: ParticipantPlan) -> AuditResult:
    """Compare a logged session with its plan. Every attempt counts, valid or not."""
    records = [r for r in load_trial_records(folder) if r.session_id == plan.session_id]
    problems: List[str] = []
    if not records:
        return AuditResult([f"no trials were logged for {plan.session_id}"], {}, ())
    other = {r.session_id for r in load_trial_records(folder)} - {plan.session_id}
    if other:
        problems.append(f"the folder also holds other session ids: {sorted(other)}")
    attempts = {code: 0 for code in plan.order}
    first_start: Dict[str, int] = {}
    for r in sorted(records, key=lambda r: (r.run, r.start_ms if r.start_ms is not None else 0)):
        attempts[r.condition] = attempts.get(r.condition, 0) + 1
        first_start.setdefault(r.condition, len(first_start))
    seen = tuple(sorted(first_start, key=first_start.get))
    if seen != plan.order:
        problems.append(f"conditions ran in the order {' -> '.join(seen)}, planned {' -> '.join(plan.order)}")
    for code in plan.order:
        if attempts.get(code, 0) < plan.trials_per_condition:
            problems.append(f"condition {code}: {attempts.get(code, 0)} attempt(s), planned {plan.trials_per_condition}")
        elif attempts[code] > plan.trials_per_condition:
            problems.append(f"condition {code}: {attempts[code]} attempts, planned {plan.trials_per_condition} "
                            "(extra attempts must be reported, not hidden)")
    unknown = sorted(set(attempts) - set(plan.order))
    if unknown:
        problems.append(f"unexpected condition(s) logged: {unknown}")
    problems += audit_run_metadata(folder)
    return AuditResult(problems, attempts, seen)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Condition orders, run sheets and an audit for the volunteer study.")
    parser.add_argument("--participants", type=int, default=4, help="how many run sheets to print (default 4)")
    parser.add_argument("--start", type=int, default=1, help="number of the first participant")
    parser.add_argument("--trials-per-object", type=int, default=DEFAULT_TRIALS_PER_OBJECT)
    parser.add_argument("--hardware", default="mock", help="mock, auto or a port; used in the printed commands")
    parser.add_argument("--audit", metavar="SESSION_DIR", help="check a logged session against its plan")
    parser.add_argument("--participant", type=int, help="participant number, needed with --audit")
    args = parser.parse_args(argv)
    try:
        if args.audit:
            if not args.participant:
                parser.error("--audit needs --participant N")
            plan = make_plans(1, args.trials_per_object, start=args.participant)[0]
            result = audit_session(args.audit, plan)
            print(f"audit of {args.audit} against participant {plan.session_id}, order {' -> '.join(plan.order)}")
            print("  attempts logged per condition: " + (", ".join(f"{c} {n}" for c, n in sorted(result.attempts.items())) or "none"))
            for problem in result.problems:
                print(f"  PROBLEM: {problem}")
            print("  matches the plan" if result.ok else "  does NOT match the plan; report the differences with the results")
            return 0 if result.ok else 1
        plans = make_plans(args.participants, args.trials_per_object, args.start)
        print("Condition orders (balanced Latin square): " + "   ".join(" -> ".join(o) for o in condition_orders()))
        print()
        for plan in plans:
            print(run_sheet(plan, args.trials_per_object, args.hardware))
            print()
        return 0
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
