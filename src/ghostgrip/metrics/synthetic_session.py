"""Run the synthetic traces through the whole headless pipeline and log it (pipeline check).

    predictor -> intent -> assist policy -> simulation world -> trial runner -> session logger

for each of the four conditions (A baseline, B AI, C haptic, D both). This proves the modules
fit together and that the metrics can be produced; it is NOT an experiment:

  * the traces are synthetic and every row is labelled "synthetic";
  * there is no person and no hardware: condition C and D's physical cue is not exercised
    (the device table stays empty), and the assistance in B and D is a few pixels of grab zone;
  * so the four conditions are expected to come out alike, and a difference here would mean
    a bug, not an effect.

The assistance profile is frozen for the whole session (plan: frozen while a condition is measured).
"""
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence

from contracts import CONDITION_CODES
from ..ai.adapt import AssistDecision, AssistPolicy
from ..ai.intent import IntentMachine, zones_for_scene
from ..ai.predict import ErrorTracker, KalmanPredictor
from ..ai.synthetic import SCENARIOS, make_scenario
from ..simulation.trial import CONDITIONS, READY, RUNNING, TrialRunner
from ..simulation.world import World
from .logger import SessionLogger

DEFAULT_SCENARIOS = tuple(SCENARIOS)
GAP_BETWEEN_TRACES_MS = 2000


def run_synthetic_session(folder, session_id: str = "synthetic-demo", seeds: Sequence[int] = (0, 1),
                          scenarios: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Log every (condition, scenario, seed) as one trial attempt. Returns counts of what happened."""
    scenarios = tuple(scenarios) if scenarios is not None else DEFAULT_SCENARIOS
    counts = {"traces": 0, "finished": 0, "incomplete": 0, "never_started": 0, "frames": 0}
    policy = AssistPolicy()
    with SessionLogger(folder, session_id, origin="synthetic") as log, policy.evaluation():
        clock = 0                                       # one continuous host clock across traces
        world = World("glass")
        trial = TrialRunner("glass", CONDITIONS[CONDITION_CODES[0]], session_id)   # one runner: ids stay unique
        for code in CONDITION_CODES:
            condition = CONDITIONS[code]
            trial.condition = condition
            for name in scenarios:
                for seed in seeds:
                    states = make_scenario(name, seed)
                    world.reset()
                    trial.next_trial()
                    predictor, tracker = KalmanPredictor(), ErrorTracker()
                    machine = IntentMachine(zones_for_scene("glass"))
                    previous, last_t, outcome = None, clock, None
                    for raw in states:
                        hand = replace(raw, t_ms=raw.t_ms + clock)
                        last_t = hand.t_ms
                        prediction = predictor.update(hand)
                        tracker.record(hand, prediction)
                        intent = machine.update(hand)
                        decision = policy.decide(hand, prediction,
                                                 predictor_trusted=not tracker.is_worse_than_baseline())
                        applied = decision if condition.ai_assist else AssistDecision(0.0, "assist_off_in_condition")
                        world.set_assistance(condition.ai_assist, applied.gain)
                        dt = 0.0 if previous is None else (hand.t_ms - previous) / 1000.0
                        previous = hand.t_ms
                        starting = trial.state == READY
                        outcome = trial.update(world, world.advance(hand, dt), hand)
                        if starting and trial.state != READY:       # the trial began on this frame
                            log.begin_trial(trial.trial_id, code, hand.t_ms)
                        running_id = trial.trial_id if trial.state != READY else None
                        log.log_frame(hand, prediction, intent, applied, trial_id=running_id, condition=code)
                        counts["frames"] += 1
                        if outcome is not None:
                            log.end_trial(outcome)
                            break
                    counts["traces"] += 1
                    if outcome is not None:
                        counts["finished"] += 1
                    elif trial.state == RUNNING:
                        log.abandon_trial(trial.trial_id, last_t, "trace_ended")
                        counts["incomplete"] += 1
                    else:
                        counts["never_started"] += 1
                    clock = last_t + GAP_BETWEEN_TRACES_MS
    return counts
