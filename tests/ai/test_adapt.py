"""Tests for src/ghostgrip/ai/adapt.py. No camera or hardware needed."""
import math
from pathlib import Path
import random
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from contracts import HandStateV2, Prediction, TrialOutcome  # noqa: E402
from ghostgrip.ai.adapt import (AdaptationConfig, AdaptationProfile, AssistPolicy,  # noqa: E402
                                neutral_profile, profile_from_dict, trial_jitter)
from ghostgrip.ai.predict import ErrorTracker, KalmanPredictor, LOST_SIGMA  # noqa: E402
from ghostgrip.ai.synthetic import SCENARIOS, make_scenario  # noqa: E402


def hand(t=1000, x=0.5, y=0.5, ok=True, quality=0.9):
    return HandStateV2(t_ms=t, pointer_xy=(x, y), closure_q=0.0, dq_dt=0.0, velocity_xy=(0.0, 0.0),
                       stable_score=1.0 if ok else 0.0, tracking_ok=ok,
                       tracking_quality=quality if ok else 0.0)


def pred(sigma=0.0, t=1000, sigma_y=None):
    return Prediction(t_query_ms=t, horizon_ms=120, future_pointer_xy=(0.5, 0.5),
                      sigma_xy=(sigma, sigma if sigma_y is None else sigma_y))


def outcome(trial=1, success=True, reason="", session="s"):
    return TrialOutcome(session_id=session, trial_id=trial, condition="D", success=success,
                        fail_reason=reason, start_ms=0, end_ms=1000)


class TestGain(unittest.TestCase):
    def setUp(self):
        self.policy = AssistPolicy()                 # neutral profile
        self.tol = self.policy.profile.grasp_tolerance

    def test_a_certain_prediction_gives_the_users_full_share(self):
        d = self.policy.decide(hand(), pred(0.0))
        self.assertEqual(d.reason, "ok")
        self.assertAlmostEqual(d.gain, self.tol)

    def test_gain_falls_linearly_with_uncertainty_and_reaches_zero_at_the_limit(self):
        limit = self.policy.config.uncertainty_limit
        half = self.policy.decide(hand(), pred(limit / 2))
        self.assertAlmostEqual(half.gain, 0.5 * self.tol)
        for sigma in (limit, limit * 1.5, 0.5):
            d = self.policy.decide(hand(), pred(sigma))
            self.assertEqual((d.gain, d.reason), (0.0, "uncertain"))

    def test_gain_never_rises_as_uncertainty_grows(self):
        gains = [self.policy.gain(hand(), pred(s / 1000)) for s in range(0, 200, 5)]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_the_worse_axis_decides(self):
        both = self.policy.gain(hand(), pred(0.08, sigma_y=0.08))
        one_bad = self.policy.gain(hand(), pred(0.01, sigma_y=0.08))
        self.assertAlmostEqual(both, one_bad)

    def test_every_unsafe_situation_gives_exactly_zero(self):
        cases = {
            "tracking_lost": (hand(ok=False), pred(0.0), True),
            "low_quality": (hand(quality=0.49), pred(0.0), True),
            "no_prediction": (hand(), pred(LOST_SIGMA), True),
            "invalid_prediction": (hand(), pred(float("nan")), True),
            "stale_prediction": (hand(t=2000), pred(0.0, t=1000), True),
            "predictor_worse_than_baseline": (hand(), pred(0.0), False),
        }
        for reason, (h, p, trusted) in cases.items():
            with self.subTest(reason=reason):
                d = self.policy.decide(h, p, predictor_trusted=trusted)
                self.assertEqual((d.gain, d.reason), (0.0, reason))

    def test_a_prediction_from_the_future_is_rejected_too(self):
        self.assertEqual(self.policy.decide(hand(t=1000), pred(0.0, t=1500)).reason, "stale_prediction")

    def test_gain_is_always_between_zero_and_one(self):
        rng = random.Random(1)
        c = AdaptationConfig()
        for _ in range(500):
            jitter = rng.uniform(0, c.jitter_max)
            policy = AssistPolicy(profile=AdaptationProfile(jitter, c.tolerance_for(jitter)))
            g = policy.gain(hand(quality=rng.uniform(0, 1)), pred(rng.uniform(-1, 2)))
            self.assertTrue(math.isfinite(g) and 0.0 <= g <= 1.0)

    def test_a_profile_outside_the_limits_is_refused_at_the_start(self):
        c = AdaptationConfig()
        for bad in (AdaptationProfile(-0.1, 0.5), AdaptationProfile(c.jitter_max * 2, 0.5),
                    AdaptationProfile(float("nan"), 0.5), AdaptationProfile(0.001, 1.5),
                    AdaptationProfile(0.001, 0.0), AdaptationProfile(0.001, float("nan")),
                    AdaptationProfile(0.001, 0.5, trials=-1)):
            with self.subTest(profile=bad), self.assertRaises(ValueError):
                AssistPolicy(profile=bad)


class TestWithTheRealPredictor(unittest.TestCase):
    def test_assistance_follows_the_predictors_confidence_through_a_dropout(self):
        policy, predictor = AssistPolicy(), KalmanPredictor()
        gains = {}
        for state in make_scenario("dropout_long"):
            p = predictor.update(state)
            gains[state.t_ms] = policy.decide(state, p)
        warmup = gains[0]
        self.assertEqual(warmup.gain, 0.0)                              # nothing is known yet
        settled = gains[2000]
        self.assertEqual(settled.reason, "ok")
        self.assertGreater(settled.gain, 0.3)
        lost = [d for t, d in gains.items() if 3100 <= t <= 3500]
        self.assertTrue(all(d.gain == 0.0 for d in lost))
        recovered = gains[4500]
        self.assertGreater(recovered.gain, 0.3)

    def test_the_predictor_settles_well_inside_the_uncertainty_limit(self):
        # Drift guard between predict.py and adapt.py: if the predictor's sigma ever grew past the
        # limit, assistance would silently vanish everywhere.
        predictor = KalmanPredictor()
        sigmas = []
        for s in make_scenario("clean"):
            p = predictor.update(s)
            if s.t_ms > 3000:
                sigmas.append(max(p.sigma_xy))
        self.assertLess(max(sigmas), AdaptationConfig().uncertainty_limit / 2)

    def test_a_losing_predictor_switches_assistance_off(self):
        policy, predictor, tracker = AssistPolicy(), KalmanPredictor(), ErrorTracker()
        trusted_at_end = None
        for s in make_scenario("q_jitter"):                 # the Kalman filter loses here
            p = predictor.update(s)
            tracker.record(s, p)
            trusted_at_end = not tracker.is_worse_than_baseline()
        self.assertFalse(trusted_at_end)
        d = policy.decide(s, p, predictor_trusted=trusted_at_end)
        self.assertEqual((d.gain, d.reason), (0.0, "predictor_worse_than_baseline"))

    def test_the_gain_drives_the_simulation_within_its_bounded_effect(self):
        from ghostgrip.simulation.world import MAX_ASSIST_FRAC, World
        policy, predictor = AssistPolicy(), KalmanPredictor()
        world, peak = World("glass"), 0.0
        for s in make_scenario("clean"):
            g = policy.gain(s, predictor.update(s))
            world.set_assistance(True, g)
            peak = max(peak, world.snapshot().grab_radius_extra_frac)
        self.assertGreater(peak, 0.0)
        self.assertLessEqual(peak, MAX_ASSIST_FRAC)


class TestAdaptation(unittest.TestCase):
    def setUp(self):
        self.p = AssistPolicy()
        self.c = self.p.config

    def test_the_update_is_the_documented_moving_average(self):
        before = self.p.profile.jitter
        after = self.p.end_trial(outcome(1), 0.001)
        self.assertAlmostEqual(after.jitter, 0.85 * before + 0.15 * 0.001)
        self.assertEqual(after.trials, 1)
        self.assertAlmostEqual(after.grasp_tolerance, self.c.tolerance_for(after.jitter))

    def test_a_steady_user_gets_less_and_a_shaky_user_more(self):
        steady, shaky = AssistPolicy(), AssistPolicy()
        for i in range(60):
            steady.end_trial(outcome(i), 0.0)
            shaky.end_trial(outcome(i), self.c.jitter_max)
        self.assertAlmostEqual(steady.profile.grasp_tolerance, self.c.tolerance_min, places=2)
        self.assertAlmostEqual(shaky.profile.grasp_tolerance, self.c.tolerance_max, places=2)

    def test_one_outlier_cannot_swing_the_profile(self):
        before = self.p.profile.jitter
        for outlier in (1e9, 1e308):
            p = AssistPolicy()
            p.end_trial(outcome(1), outlier)
            self.assertLessEqual(p.profile.jitter - before, self.c.alpha * self.c.jitter_max + 1e-12)
            self.assertLessEqual(p.profile.jitter, self.c.jitter_max)

    def test_non_finite_or_missing_jitter_changes_nothing(self):
        for bad in (float("nan"), float("inf"), -1.0, None):
            with self.subTest(jitter=bad):
                p = AssistPolicy()
                self.assertEqual(p.end_trial(outcome(1), bad), p.profile)
                self.assertEqual((p.updates, p.profile.trials), (0, 0))

    def test_the_profile_stays_inside_its_limits_whatever_is_fed_in(self):
        rng = random.Random(7)
        p = AssistPolicy()
        for i in range(2000):
            p.end_trial(outcome(i), rng.choice([0.0, 1e-9, 0.002, 0.004, 1.0, 1e12, rng.random()]))
            self.assertTrue(0.0 <= p.profile.jitter <= self.c.jitter_max)
            self.assertTrue(self.c.tolerance_min <= p.profile.grasp_tolerance <= self.c.tolerance_max)

    def test_frozen_means_no_update(self):
        before = self.p.profile
        self.p.freeze()
        self.assertEqual(self.p.end_trial(outcome(1), self.c.jitter_max), before)
        self.assertEqual(self.p.updates, 0)
        self.p.unfreeze()
        self.assertNotEqual(self.p.end_trial(outcome(1), self.c.jitter_max), before)

    def test_evaluation_freezes_and_always_restores(self):
        with self.p.evaluation():
            self.assertTrue(self.p.frozen)
            self.p.end_trial(outcome(1), 0.004)
        self.assertFalse(self.p.frozen)
        self.assertEqual(self.p.updates, 0)
        self.p.freeze()
        with self.p.evaluation():
            pass
        self.assertTrue(self.p.frozen)                       # it was frozen before, and stays frozen
        self.p.unfreeze()
        with self.assertRaises(RuntimeError):
            with self.p.evaluation():
                raise RuntimeError("trial crashed")
        self.assertFalse(self.p.frozen)

    def test_the_profile_does_not_change_during_a_trial(self):
        # Per-frame decisions never touch the profile; only end_trial does.
        before, predictor = self.p.profile, KalmanPredictor()
        for s in make_scenario("noisy"):
            self.p.decide(s, predictor.update(s))
        self.assertEqual(self.p.profile, before)

    def test_failed_and_invalid_trials_do_not_count(self):
        for case in (outcome(1, success=False, reason="timeout"),
                     outcome(2, success=False, reason="damaged_overgrip"),
                     outcome(3, success=False, reason="tracking_lost"),
                     outcome(4, success=True, reason="odd")):
            self.assertEqual(self.p.end_trial(case, 0.002), self.p.profile)
        self.assertEqual((self.p.updates, self.p.profile.trials, self.p.skipped), (0, 0, 4))

    def test_each_trial_counts_once(self):
        self.p.end_trial(outcome(5), 0.004)
        after_first = self.p.profile
        self.p.end_trial(outcome(5), 0.004)
        self.assertEqual(self.p.profile, after_first)
        self.p.end_trial(outcome(5, session="other"), 0.004)   # a different session is a different trial
        self.assertEqual(self.p.profile.trials, 2)


class TestSetup(unittest.TestCase):
    def test_invalid_configs_are_rejected(self):
        for change in (dict(alpha=0.0), dict(alpha=1.5), dict(uncertainty_limit=0.0),
                       dict(uncertainty_limit=2.0), dict(min_quality=-0.1), dict(jitter_max=0.0),
                       dict(tolerance_min=0.9, tolerance_max=0.5), dict(tolerance_max=1.5),
                       dict(max_prediction_age_ms=-1), dict(alpha=float("nan")),
                       dict(jitter_max=float("inf"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                AdaptationConfig(**change)

    def test_a_new_user_starts_in_the_middle(self):
        c = AdaptationConfig()
        p = neutral_profile(c)
        self.assertAlmostEqual(p.jitter, c.jitter_max / 2)
        self.assertAlmostEqual(p.grasp_tolerance, (c.tolerance_min + c.tolerance_max) / 2)

    def test_defaults_match_config_yaml_and_the_simulation(self):
        from ghostgrip.config import load_config
        from ghostgrip.simulation import world
        cfg = load_config()
        self.assertEqual(AdaptationConfig().alpha, cfg.ai.adapt_alpha)
        self.assertEqual(AdaptationConfig().min_quality, world.QUALITY_MIN)
        self.assertEqual(cfg.ai.low_confidence_behavior, "disable_assist")   # the only supported mode
        self.assertEqual(AssistPolicy.from_config(cfg).config.alpha, cfg.ai.adapt_alpha)

    def test_the_profile_round_trips_and_bad_data_is_rejected(self):
        c = AdaptationConfig()
        p = AssistPolicy()
        p.end_trial(outcome(1), 0.003)
        rebuilt = profile_from_dict(p.profile.to_dict(), c)
        self.assertEqual(rebuilt, p.profile)
        for bad in ({}, {"jitter": "x", "trials": 1}, {"jitter": -1, "trials": 0},
                    {"jitter": 1.0, "trials": 0}, {"jitter": float("nan"), "trials": 0},
                    {"jitter": 0.001, "trials": -2}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                profile_from_dict(bad, c)


class TestFrozenByConfig(unittest.TestCase):
    def config(self, freeze):
        import copy
        import yaml
        from ghostgrip import config as C
        data = copy.deepcopy(yaml.safe_load(C.DEFAULT_PATH.read_text()))
        data["ai"]["freeze_adaptation"] = freeze
        return C.parse_config(data)

    def test_the_committed_settings_start_the_policy_frozen(self):
        from ghostgrip.config import load_config
        policy = AssistPolicy.from_config(load_config())
        self.assertTrue(policy.frozen)
        before = policy.profile
        policy.end_trial(outcome(1), 0.004)
        self.assertEqual((policy.profile, policy.updates), (before, 0))

    def test_practice_mode_lets_it_learn(self):
        policy = AssistPolicy.from_config(self.config(False))
        self.assertFalse(policy.frozen)
        before = policy.profile
        policy.end_trial(outcome(1), 0.004)
        self.assertNotEqual(policy.profile, before)

    def test_the_apps_own_pipeline_comes_out_frozen_and_does_not_learn(self):
        # The app builds its policy through from_config, so a measured run cannot drift by accident.
        from ghostgrip.config import load_config
        from ghostgrip.ui.integration import AIPipeline
        ai = AIPipeline(load_config(), "glass")
        before = ai.policy.profile
        ai.end_trial(outcome(1))
        self.assertTrue(ai.policy.frozen)
        self.assertEqual(ai.policy.profile, before)

    def test_the_apps_pipeline_learns_only_when_the_setting_says_practice(self):
        from ghostgrip.ui.integration import AIPipeline
        ai = AIPipeline(self.config(False), "glass")
        ai.trial_states = [hand(t=i * 33, x=0.3 + 0.003 * i + (0.004 if i % 2 else 0.0)) for i in range(20)]
        before = ai.policy.profile
        ai.end_trial(outcome(1))
        self.assertNotEqual(ai.policy.profile, before)


class TestTrialJitter(unittest.TestCase):
    def line(self, n=20, step=33, noise=None):
        rng = random.Random(3)
        return [hand(t=i * step, x=0.2 + 0.01 * i + (rng.gauss(0, noise) if noise else 0.0),
                     y=0.5 + (rng.gauss(0, noise) if noise else 0.0)) for i in range(n)]

    def test_smooth_straight_motion_has_no_jitter(self):
        self.assertAlmostEqual(trial_jitter(self.line()), 0.0, places=9)

    def test_wobble_is_measured_and_grows_with_noise(self):
        small, big = trial_jitter(self.line(noise=0.001)), trial_jitter(self.line(noise=0.01))
        self.assertGreater(small, 0.0)
        self.assertGreater(big, small * 3)

    def test_uneven_time_steps_do_not_look_like_jitter(self):
        times, t = [], 0
        for step in [20, 60, 33, 45, 25, 70, 28, 33, 40, 55] * 2:
            t += step
            times.append(t)
        states = [hand(t=t, x=0.1 + 0.0004 * t) for t in times]          # perfectly straight, uneven timing
        self.assertAlmostEqual(trial_jitter(states), 0.0, places=9)

    def test_lost_frames_are_skipped_and_nothing_to_measure_is_none(self):
        states = self.line()
        states[5] = hand(t=states[5].t_ms, ok=False)
        self.assertAlmostEqual(trial_jitter(states), 0.0, places=9)
        self.assertIsNone(trial_jitter([]))
        self.assertIsNone(trial_jitter(states[:2]))
        self.assertIsNone(trial_jitter([hand(t=i * 33, ok=False) for i in range(10)]))

    def test_repeated_timestamps_are_not_divided_by(self):
        states = [hand(t=0), hand(t=0), hand(t=0), hand(t=33)]
        self.assertIsNone(trial_jitter(states))

    def test_scenarios_order_sensibly(self):
        self.assertGreater(trial_jitter(make_scenario("noisy")), trial_jitter(make_scenario("clean")))
        for name in SCENARIOS:
            value = trial_jitter(make_scenario(name))
            self.assertTrue(value is not None and math.isfinite(value) and value >= 0.0)
            self.assertLess(value, AdaptationConfig().jitter_max)   # synthetic data fits the ceiling


if __name__ == "__main__":
    unittest.main()
