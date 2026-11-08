"""Tests for src/ghostgrip/ai/intent.py. No camera or hardware needed."""
import math
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from contracts import HandStateV2  # noqa: E402
from ghostgrip.ai import intent as I  # noqa: E402
from ghostgrip.ai.intent import (APPROACHING, CLOSING, CONTACT_CANDIDATE, HOLDING, IDLE,  # noqa: E402
                                 RELEASING, STATES, GraspZone, IntentMachine, IntentParams,
                                 timeline, zones_for_scene)
from ghostgrip.ai.synthetic import SCENARIOS, make_scenario  # noqa: E402

STEP = 33
GLASS = zones_for_scene("glass")[0]
ON_GLASS = GLASS.center_xy
FAR = (0.9, 0.1)


def hs(t, xy=ON_GLASS, q=0.0, dq=0.0, vel=(0.0, 0.0), ok=True, quality=0.9):
    return HandStateV2(t_ms=t, pointer_xy=xy, closure_q=q, dq_dt=dq, velocity_xy=vel,
                       stable_score=1.0 if ok else 0.0, tracking_ok=ok,
                       tracking_quality=quality if ok else 0.0)


class Driver:
    """Feeds frames at a steady rate and remembers the clock."""

    def __init__(self, machine=None, start=0):
        self.m = machine or IntentMachine([GLASS])
        self.t = start
        self.last = None

    def run(self, frames, **kwargs):
        for _ in range(frames):
            self.last = self.m.update(hs(self.t, **kwargs))
            self.t += STEP
        return self.last


def to_holding(driver):
    """Drive a machine from IDLE to HOLDING on the glass."""
    driver.run(6)                                  # on the glass, open hand -> APPROACHING
    driver.run(6, q=0.2, dq=0.3)                   # closing
    driver.run(6, q=0.34)                          # contact
    driver.run(6, q=0.5)                           # firm grip
    assert driver.m.state == HOLDING, driver.m.state
    return driver


class TestFullGrasp(unittest.TestCase):
    def test_clean_trace_walks_the_whole_sequence_in_order(self):
        log = timeline(make_scenario("clean"), IntentMachine([GLASS]))
        self.assertEqual([new for _, _, new, _ in log],
                         [APPROACHING, CLOSING, CONTACT_CANDIDATE, HOLDING, RELEASING, IDLE])
        self.assertEqual([old for _, old, _, _ in log],
                         [IDLE, APPROACHING, CLOSING, CONTACT_CANDIDATE, HOLDING, RELEASING])
        self.assertEqual([t for t, *_ in log], sorted(t for t, *_ in log))

    def test_the_object_is_named_while_it_is_the_target(self):
        machine = IntentMachine([GLASS])
        seen = {}
        for hand in make_scenario("clean"):
            result = machine.update(hand)
            seen.setdefault(result.state, set()).add(result.object_id)
        self.assertEqual(seen[IDLE], {None})
        for state in (APPROACHING, CLOSING, CONTACT_CANDIDATE, HOLDING, RELEASING):
            self.assertEqual(seen[state], {"glass"})

    def test_carrying_far_from_the_start_keeps_holding(self):
        machine = IntentMachine([GLASS])
        at_3500 = [machine.update(h) for h in make_scenario("clean")]
        carry = next(r for h, r in zip(make_scenario("clean"), at_3500) if h.t_ms >= 3500)
        self.assertEqual(carry.state, HOLDING)

    def test_agrees_with_the_simulation_within_a_few_frames(self):
        # Partly by construction: intent uses the simulation's own thresholds. This checks the
        # rules are consistent, not that they match what a real person intends.
        from ghostgrip.simulation.world import World
        for name in ("clean", "noisy", "slow_close", "frame_drops"):
            with self.subTest(scenario=name):
                states = make_scenario(name)
                world, events, previous = World("glass"), {}, None
                for s in states:
                    dt = 0.0 if previous is None else (s.t_ms - previous) / 1000.0
                    for ev in world.advance(s, dt):
                        events.setdefault(ev.kind, s.t_ms)
                    previous = s.t_ms
                log = timeline(states, IntentMachine([GLASS]))
                first = {new: t for t, _, new, _ in log}
                self.assertLessEqual(abs(first[HOLDING] - events["HOLD"]), 4 * STEP)
                self.assertLessEqual(abs(first[RELEASING] - events["RELEASE"]), 4 * STEP)


class TestGuards(unittest.TestCase):
    def test_a_single_frame_spike_never_changes_state(self):
        d = Driver()
        d.run(6)
        self.assertEqual(d.m.state, APPROACHING)
        d.run(1, q=0.45)                          # one frame of contact-level closure
        d.run(6)
        self.assertEqual(d.m.state, APPROACHING)

    def test_two_frames_are_not_enough_but_three_are(self):
        d = Driver()
        d.run(6)
        d.run(2, q=0.35)
        d.run(1)
        self.assertEqual(d.m.state, APPROACHING)
        d.run(3, q=0.35)
        self.assertEqual(d.m.state, CONTACT_CANDIDATE)

    def test_pinching_far_from_any_object_does_nothing(self):
        d = Driver()
        d.run(30, xy=FAR, q=0.7, dq=0.5)
        self.assertEqual(d.m.state, IDLE)

    def test_a_hand_moving_away_is_not_approaching_but_one_heading_in_is(self):
        near = (GLASS.center_xy[0] + 90 / 960, GLASS.center_xy[1])          # 90 px right of the glass
        away = Driver()
        away.run(10, xy=near, vel=(0.3, 0.0))                                # moving right, away
        self.assertEqual(away.m.state, IDLE)
        toward = Driver()
        toward.run(10, xy=near, vel=(-0.3, 0.0))                             # moving left, toward it
        self.assertEqual(toward.m.state, APPROACHING)

    def test_a_firm_pinch_beside_the_object_is_not_contact(self):
        # Near the glass (inside the approach area) but outside its 44 px grab radius.
        beside = (GLASS.center_xy[0] + 90 / 960, GLASS.center_xy[1])
        d = Driver()
        d.run(6, xy=beside, vel=(-0.3, 0.0))
        self.assertEqual(d.m.state, APPROACHING)
        d.run(12, xy=beside, q=0.5, dq=0.0)                 # pinching hard, but not on the glass
        self.assertNotIn(d.m.state, (CONTACT_CANDIDATE, HOLDING))
        d.run(3, xy=ON_GLASS, q=0.5)                        # now on it: contact after 3 frames
        self.assertEqual(d.m.state, CONTACT_CANDIDATE)
        d.run(3, xy=ON_GLASS, q=0.5)                        # firm grip after 3 more
        self.assertEqual(d.m.state, HOLDING)

    def test_sliding_off_the_object_loses_contact(self):
        d = Driver()
        d.run(6)
        d.run(6, q=0.34)
        self.assertEqual(d.m.state, CONTACT_CANDIDATE)
        d.run(6, xy=(GLASS.center_xy[0] + 90 / 960, GLASS.center_xy[1]), q=0.34)
        self.assertEqual(d.m.state, CLOSING)

    def test_the_nearest_object_is_chosen(self):
        zones = zones_for_scene("sandbox")
        sponge = next(z for z in zones if z.object_id == "sponge")
        d = Driver(IntentMachine(zones))
        d.run(6, xy=sponge.center_xy)
        self.assertEqual((d.m.state, d.last.object_id), (APPROACHING, "sponge"))

    def test_contact_is_kept_through_jitter_but_lost_below_the_hysteresis(self):
        d = Driver()
        d.run(6)
        d.run(6, q=0.34)
        self.assertEqual(d.m.state, CONTACT_CANDIDATE)
        for q in (0.28, 0.33, 0.27, 0.34, 0.26, 0.33, 0.28, 0.34):      # wobbles above contact_q - 0.05
            d.run(1, q=q)
        self.assertEqual(d.m.state, CONTACT_CANDIDATE)
        d.run(4, q=0.20, dq=-0.1)
        self.assertEqual(d.m.state, CLOSING)

    def test_holding_is_not_released_by_a_dip_that_does_not_stay(self):
        d = to_holding(Driver())
        for q in (0.2, 0.5, 0.2, 0.5, 0.2, 0.5):
            d.run(1, q=q)
        self.assertEqual(d.m.state, HOLDING)
        d.run(4, q=0.2)
        self.assertEqual(d.m.state, RELEASING)

    def test_holding_has_a_hysteresis_band_below_the_contact_threshold(self):
        # contact_q is 0.30 and the hysteresis 0.05, so the grip is only "relaxing" below 0.25.
        d = to_holding(Driver())
        d.run(20, q=0.27)                                   # under contact_q but inside the band
        self.assertEqual(d.m.state, HOLDING)
        d.run(4, q=0.23)                                    # below the band
        self.assertEqual(d.m.state, RELEASING)

    def test_a_slow_start_to_closing_is_not_mistaken_for_reopening(self):
        # A hand that has just begun to close is below open_q too; that must not cancel closing.
        d = Driver()
        d.run(6)
        d.run(4, q=0.05, dq=0.2)
        self.assertEqual(d.m.state, CLOSING)
        d.run(12, q=0.08, dq=0.2)                           # still barely closed, still closing
        self.assertEqual(d.m.state, CLOSING)
        d.run(4, q=0.10, dq=-0.3)                           # now actively opening
        self.assertEqual(d.m.state, APPROACHING)

    def test_slow_closure_is_one_clean_closing_step(self):
        log = timeline(make_scenario("slow_close"), IntentMachine([GLASS]))
        self.assertEqual([new for _, _, new, _ in log],
                         [APPROACHING, CLOSING, CONTACT_CANDIDATE, HOLDING, RELEASING, IDLE])

    def test_a_jittery_hand_on_the_contact_threshold_does_not_flap(self):
        states = make_scenario("q_jitter")
        crossings = sum(1 for a, b in zip(states, states[1:])
                        if (a.closure_q - 0.30) * (b.closure_q - 0.30) < 0)
        log = timeline(states, IntentMachine([GLASS]))
        self.assertGreater(crossings, 10)                  # a plain threshold would flip this often
        self.assertLessEqual(len(log), 3)

    def test_releasing_goes_back_to_idle_and_a_regrip_returns_to_holding(self):
        d = to_holding(Driver())
        d.run(4, q=0.2)
        self.assertEqual(d.m.state, RELEASING)
        d.run(4, q=0.5)
        self.assertEqual(d.m.state, HOLDING)
        d.run(4, q=0.2)
        d.run(4, q=0.05)
        self.assertEqual(d.m.state, IDLE)

    def test_a_release_that_goes_nowhere_times_out(self):
        d = to_holding(Driver())
        d.run(4, q=0.2)
        self.assertEqual(d.m.state, RELEASING)
        results = [d.run(1, q=0.2) for _ in range(40)]      # hovering between open and hold for > 1 s
        timed_out = [r for r in results if r.changed and r.reason == "release_timeout"]
        self.assertEqual(len(timed_out), 1)
        self.assertEqual(timed_out[0].state, IDLE)
        self.assertEqual(results[results.index(timed_out[0]) - 1].state, RELEASING)
        # The hand is still parked on the glass, so after the cooldown it starts approaching again.
        self.assertEqual(d.m.state, APPROACHING)

    def test_cooldown_blocks_an_immediate_new_approach(self):
        d = to_holding(Driver())
        d.run(4, q=0.2)
        d.run(4, q=0.05)
        self.assertEqual(d.m.state, IDLE)
        blocked = d.run(3, q=0.0)
        self.assertEqual((blocked.state, blocked.reason), (IDLE, "cooldown"))
        d.run(15, q=0.0)                                    # well past 250 ms
        self.assertEqual(d.m.state, APPROACHING)


class TestUnreliableInput(unittest.TestCase):
    def test_a_short_loss_freezes_the_state(self):
        d = to_holding(Driver())
        d.run(5, ok=False)                                  # about 165 ms
        self.assertEqual((d.m.state, d.last.reason), (HOLDING, "frozen"))
        d.run(3, q=0.5)
        self.assertEqual(d.m.state, HOLDING)

    def test_a_long_loss_returns_to_idle(self):
        d = to_holding(Driver())
        d.run(12, ok=False)                                 # about 400 ms
        self.assertEqual(d.m.state, IDLE)
        self.assertIsNone(d.last.object_id)

    def test_the_transition_reason_is_tracking_lost(self):
        d = to_holding(Driver())
        reasons = []
        for _ in range(12):
            reasons.append(d.run(1, ok=False))
        self.assertIn("tracking_lost", [r.reason for r in reasons if r.changed])

    def test_loss_discards_half_finished_evidence(self):
        d = Driver()
        d.run(6)
        d.run(2, q=0.35)
        d.run(2, ok=False)
        d.run(2, q=0.35)                                    # 2 + 2 frames must not add up to 3
        self.assertEqual(d.m.state, APPROACHING)

    def test_low_quality_cannot_start_a_grasp(self):
        d = Driver()
        result = d.run(20, quality=0.3)
        self.assertEqual((d.m.state, result.reason), (IDLE, "unreliable"))
        d.run(6)
        self.assertEqual(d.m.state, APPROACHING)

    def test_low_quality_cannot_strengthen_a_grasp(self):
        d = Driver()
        d.run(6)
        d.run(6, q=0.34)
        self.assertEqual(d.m.state, CONTACT_CANDIDATE)
        d.run(10, q=0.5, quality=0.3)                       # firm grip, but unreliable frames
        self.assertEqual(d.m.state, CONTACT_CANDIDATE)
        d.run(4, q=0.5)
        self.assertEqual(d.m.state, HOLDING)

    def test_low_quality_does_not_drop_a_grasp_but_a_release_still_counts(self):
        d = to_holding(Driver())
        d.run(30, q=0.5, quality=0.3)                       # a long unreliable stretch
        self.assertEqual(d.m.state, HOLDING)
        d.run(4, q=0.1, quality=0.3)                        # opening is accepted even when unreliable
        self.assertEqual(d.m.state, RELEASING)

    def test_repeated_and_out_of_order_frames_are_ignored(self):
        d = to_holding(Driver())
        before = d.last
        again = d.m.update(hs(d.t - STEP, q=0.0))
        older = d.m.update(hs(d.t - 5 * STEP, q=0.0))
        self.assertEqual((again.reason, older.reason), ("ignored", "ignored"))
        self.assertEqual(d.m.state, HOLDING)
        self.assertEqual(d.m.ignored_samples, 2)
        self.assertEqual(before.state, HOLDING)

    def test_a_long_gap_discards_partial_evidence(self):
        d = Driver()
        d.run(6)
        d.run(2, q=0.35)
        d.t += 800
        d.run(1, q=0.35)
        self.assertEqual(d.m.state, APPROACHING)

    def test_non_finite_numbers_are_treated_as_unusable_not_a_crash(self):
        d = to_holding(Driver())
        for bad in (dict(q=float("nan")), dict(xy=(float("inf"), 0.5)), dict(dq=float("nan")),
                    dict(vel=(float("nan"), 0.0))):
            result = d.run(1, **bad)
            self.assertEqual(result.reason, "frozen")
        self.assertEqual(d.m.state, HOLDING)


class TestSetup(unittest.TestCase):
    def test_invalid_zones_are_rejected(self):
        ok = dict(object_id="x", center_xy=(0.5, 0.5), radius_px=40.0, contact_q=0.3, hold_q=0.4)
        GraspZone(**ok)
        for change in (dict(center_xy=(1.5, 0.5)), dict(center_xy=(0.5,)), dict(radius_px=0.0),
                       dict(contact_q=0.5, hold_q=0.4), dict(contact_q=0.0), dict(hold_q=1.5),
                       dict(radius_px=float("nan"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                GraspZone(**{**ok, **change})

    def test_invalid_params_are_rejected(self):
        for change in (dict(confirm_frames=0), dict(min_quality=1.5), dict(close_rate=0.0),
                       dict(leave_factor=0.5), dict(open_q=1.0), dict(cooldown_ms=-1),
                       dict(approach_speed_px=float("nan"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                IntentParams(**change)

    def test_invalid_machine_setup_is_rejected(self):
        with self.assertRaises(ValueError):
            IntentMachine([])
        with self.assertRaises(ValueError):
            IntentMachine([GLASS, GLASS])
        with self.assertRaises(ValueError):
            IntentMachine([GLASS], world_size=(0, 540))

    def test_zones_follow_the_simulation_numbers(self):
        from ghostgrip.simulation.objects import PROFILES
        from ghostgrip.simulation.world import GRAB_MARGIN
        glass = PROFILES["glass"]
        self.assertEqual(GLASS.radius_px, glass.radius + GRAB_MARGIN)
        self.assertEqual(GLASS.contact_q, glass.contact_threshold)
        self.assertAlmostEqual(GLASS.hold_q, 0.40)
        self.assertEqual([z.object_id for z in zones_for_scene("sandbox")], ["glass", "sponge", "block"])
        with self.assertRaises(ValueError):
            zones_for_scene("kitchen")

    def test_default_rules_match_the_simulation_and_config(self):
        from ghostgrip.config import load_config
        from ghostgrip.simulation import world
        p = IntentParams()
        self.assertEqual(p.loss_timeout_ms, world.TRACKING_TIMEOUT_MS)
        self.assertEqual(p.min_quality, world.QUALITY_MIN)
        self.assertEqual(p.open_q, world.OPEN_Q)
        self.assertEqual(p.hysteresis, world.Q_HYST)
        self.assertEqual(p.confirm_frames, load_config().ai.grip_confirm_frames)

    def test_from_config_uses_the_configured_confirm_frames(self):
        import copy
        import yaml
        from ghostgrip import config as C
        data = copy.deepcopy(yaml.safe_load(C.DEFAULT_PATH.read_text()))
        data["ai"]["grip_confirm_frames"] = 5
        machine = IntentMachine.from_config([GLASS], C.parse_config(data))
        self.assertEqual(machine.params.confirm_frames, 5)
        d = Driver(machine)
        d.run(4)
        self.assertEqual(d.m.state, IDLE)                   # 4 frames are not enough any more
        d.run(2)
        self.assertEqual(d.m.state, APPROACHING)


class TestOnSyntheticTraces(unittest.TestCase):
    ALLOWED = {
        IDLE: {APPROACHING},
        APPROACHING: {CLOSING, CONTACT_CANDIDATE, IDLE},
        CLOSING: {CONTACT_CANDIDATE, APPROACHING, IDLE},
        CONTACT_CANDIDATE: {HOLDING, CLOSING, APPROACHING, IDLE},
        HOLDING: {RELEASING, IDLE},
        RELEASING: {IDLE, HOLDING},
    }

    def test_every_transition_on_every_scenario_is_a_legal_one(self):
        for name in SCENARIOS:
            for seed in (0, 1, 2):
                with self.subTest(scenario=name, seed=seed):
                    machine = IntentMachine([GLASS])
                    previous = machine.state
                    for hand in make_scenario(name, seed):
                        result = machine.update(hand)
                        self.assertIn(result.state, STATES)
                        if result.changed:
                            legal = self.ALLOWED[previous] | ({IDLE} if result.reason == "tracking_lost" else set())
                            self.assertIn(result.state, legal, f"{previous} -> {result.state}")
                            previous = result.state
                        else:
                            self.assertEqual(result.state, previous)

    def test_noise_does_not_change_the_story(self):
        clean = [new for _, _, new, _ in timeline(make_scenario("clean"), IntentMachine([GLASS]))]
        for seed in range(5):
            noisy = [new for _, _, new, _ in timeline(make_scenario("noisy", seed), IntentMachine([GLASS]))]
            self.assertEqual(noisy, clean)

    def test_short_dropout_and_glitch_do_not_break_the_grasp(self):
        for name in ("dropout_short", "glitch", "frame_drops", "fast_reach", "low_quality"):
            with self.subTest(scenario=name):
                states = [new for _, _, new, _ in timeline(make_scenario(name), IntentMachine([GLASS]))]
                self.assertEqual(states[-3:], [HOLDING, RELEASING, IDLE])

    def test_a_long_dropout_ends_the_grasp_like_the_simulation_does(self):
        log = timeline(make_scenario("dropout_long"), IntentMachine([GLASS]))
        self.assertEqual((log[-1][2], log[-1][3]), (IDLE, "tracking_lost"))

    def test_the_command_prints_a_timeline(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            I.main(["clean"])
        self.assertIn("HOLDING", out.getvalue())
        self.assertIn("6 transitions", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            I.main(["all"])


if __name__ == "__main__":
    unittest.main()
