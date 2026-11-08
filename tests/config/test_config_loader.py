"""Tests for src/ghostgrip/config.py and the committed config.yaml. No hardware needed."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from ghostgrip import config as C  # noqa: E402
from ghostgrip.config import ConfigError, load_config, parse_config, to_haptic_config  # noqa: E402


def good():
    """A fresh copy of the committed settings as a plain dictionary."""
    return copy.deepcopy(yaml.safe_load(C.DEFAULT_PATH.read_text(encoding="utf-8")))


def with_servo(**changes):
    data = good()
    data["hardware"].update(servo_enabled=True, servo_neutral_deg=90, servo_min_deg=80, servo_max_deg=100)
    data["hardware"].update(changes)
    return data


class TestCommittedFile(unittest.TestCase):
    def test_default_file_exists_and_loads(self):
        self.assertTrue(C.DEFAULT_PATH.is_file())
        cfg = load_config()
        self.assertEqual(cfg.ai.prediction_horizon_ms, 120)
        self.assertEqual(cfg.hardware.baud, 115200)

    def test_adaptation_is_not_frozen_by_the_committed_file(self):
        # A measured run is frozen by the app's --evaluation flag, not by the settings file.
        self.assertFalse(load_config().ai.freeze_adaptation)

    def test_freeze_adaptation_must_be_a_real_boolean(self):
        for bad in ("yes", 1, 0, None, "false"):
            data = good()
            data["ai"]["freeze_adaptation"] = bad
            with self.subTest(value=bad), self.assertRaises(ConfigError):
                parse_config(data)
        data = good()
        data["ai"]["freeze_adaptation"] = False
        self.assertFalse(parse_config(data).ai.freeze_adaptation)

    def test_servo_is_off_and_unverified_in_the_committed_file(self):
        hw = load_config().hardware
        self.assertFalse(hw.servo_enabled)
        self.assertEqual((hw.servo_neutral_deg, hw.servo_min_deg, hw.servo_max_deg), ("VERIFY",) * 3)

    def test_raw_video_is_off_and_participant_is_anonymous(self):
        exp = load_config().experiment
        self.assertFalse(exp.save_raw_video)
        self.assertEqual(exp.participant_id, "anonymous")

    def test_matches_the_simulation_constants(self):
        # Drift guard: if someone changes world.py or config.yaml alone, this fails.
        from ghostgrip.simulation import world
        sim = load_config().simulation
        self.assertEqual((sim.width, sim.height, sim.physics_hz),
                         (world.WORLD_W, world.WORLD_H, world.PHYSICS_HZ))

    def test_matches_the_hardware_defaults_and_firmware_limits(self):
        from ghostgrip.hardware.haptics import DEFAULT_CONFIG
        hw = load_config().hardware
        self.assertEqual(hw.cue_cooldown_ms, DEFAULT_CONFIG["cue_cooldown_ms"])
        self.assertEqual(hw.cue_duration_ms, DEFAULT_CONFIG["cue_duration_ms"])
        self.assertEqual(hw.min_tracking_quality, DEFAULT_CONFIG["min_tracking_quality"])
        self.assertEqual(hw.contact_offset_deg, DEFAULT_CONFIG["contact_offset_deg"])
        # The firmware pins.h window and cue cap are copied into config.py; keep them in sync.
        pins = (REPO_ROOT / "firmware" / "ghostgrip_uno" / "pins.h").read_text()
        low, high = C.FIRMWARE_ANGLE_WINDOW
        self.assertIn(f"SERVO_MIN_DEG     = {low};", pins)
        self.assertIn(f"SERVO_MAX_DEG     = {high};", pins)
        firmware = (REPO_ROOT / "firmware" / "ghostgrip_uno" / "ghostgrip_uno.ino").read_text()
        self.assertIn(f"MAX_CUE_MS      = {C.FIRMWARE_MAX_CUE_MS};", firmware)
        self.assertIn(f"WATCHDOG_MS {hw.watchdog_ms}UL", firmware)


class TestStructure(unittest.TestCase):
    def test_unknown_section_is_rejected(self):
        data = good()
        data["cammera"] = {}
        with self.assertRaisesRegex(ConfigError, "unknown section"):
            parse_config(data)

    def test_unknown_key_is_rejected(self):
        data = good()
        data["ai"]["adapt_alfa"] = 0.1
        with self.assertRaisesRegex(ConfigError, "unknown setting"):
            parse_config(data)

    def test_missing_key_is_rejected(self):
        data = good()
        del data["ai"]["adapt_alpha"]
        with self.assertRaisesRegex(ConfigError, "missing setting"):
            parse_config(data)

    def test_missing_section_is_rejected(self):
        data = good()
        del data["experiment"]
        with self.assertRaisesRegex(ConfigError, "experiment"):
            parse_config(data)

    def test_not_a_mapping_is_rejected(self):
        for bad in (None, [], "text", 3):
            with self.assertRaises(ConfigError):
                parse_config(bad)

    def test_unreadable_file_and_bad_yaml(self):
        with self.assertRaisesRegex(ConfigError, "cannot read"):
            load_config(REPO_ROOT / "does_not_exist.yaml")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.yaml"
            path.write_text("camera: [unclosed\n")
            with self.assertRaisesRegex(ConfigError, "not valid YAML"):
                load_config(path)

    def test_a_valid_file_on_disk_round_trips(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.yaml"
            data = good()
            data["camera"]["index"] = 1
            path.write_text(yaml.safe_dump(data))
            self.assertEqual(load_config(path).camera.index, 1)


class TestValues(unittest.TestCase):
    def check_rejected(self, section, key, value):
        data = good()
        data[section][key] = value
        with self.assertRaises(ConfigError, msg=f"{section}.{key}={value!r}"):
            parse_config(data)

    def test_wrong_types_are_rejected(self):
        self.check_rejected("camera", "index", "0")
        self.check_rejected("camera", "mirror", "yes")
        self.check_rejected("camera", "target_fps", 30.5)
        self.check_rejected("ai", "adapt_alpha", "0.15")
        self.check_rejected("hardware", "enabled", 1)
        self.check_rejected("hardware", "port", "")

    def test_a_bool_is_not_accepted_as_a_number(self):
        self.check_rejected("camera", "target_fps", True)
        self.check_rejected("ai", "adapt_alpha", True)

    def test_out_of_range_values_are_rejected(self):
        self.check_rejected("camera", "target_fps", 0)
        self.check_rejected("simulation", "physics_hz", 5000)
        self.check_rejected("ai", "prediction_horizon_ms", 0)
        self.check_rejected("ai", "prediction_horizon_ms", 5000)
        self.check_rejected("ai", "grip_confirm_frames", 0)
        self.check_rejected("hardware", "min_tracking_quality", 1.5)
        self.check_rejected("hardware", "cue_cooldown_ms", -1)

    def test_nan_and_infinity_are_rejected(self):
        self.check_rejected("ai", "adapt_alpha", float("nan"))
        self.check_rejected("hardware", "min_tracking_quality", float("inf"))

    def test_adapt_alpha_must_be_above_zero(self):
        self.check_rejected("ai", "adapt_alpha", 0.0)
        self.check_rejected("ai", "adapt_alpha", 1.01)
        data = good()
        data["ai"]["adapt_alpha"] = 1
        self.assertEqual(parse_config(data).ai.adapt_alpha, 1.0)

    def test_unknown_low_confidence_behavior_is_rejected(self):
        self.check_rejected("ai", "low_confidence_behavior", "keep_assisting")

    def test_cue_cannot_be_longer_than_the_firmware_allows(self):
        self.check_rejected("hardware", "cue_duration_ms", C.FIRMWARE_MAX_CUE_MS + 1)
        data = good()
        data["hardware"]["cue_duration_ms"] = C.FIRMWARE_MAX_CUE_MS
        self.assertEqual(parse_config(data).hardware.cue_duration_ms, C.FIRMWARE_MAX_CUE_MS)

    def test_keepalive_must_leave_room_inside_the_watchdog(self):
        self.check_rejected("hardware", "keepalive_ms", 400)   # 2 x 400 > 500
        data = good()
        data["hardware"]["keepalive_ms"] = 250                 # exactly half is allowed
        self.assertEqual(parse_config(data).hardware.keepalive_ms, 250)

    def test_participant_id_must_be_anonymous_looking(self):
        for bad in ("jane@example.com", "Jane Doe", "a" * 33, "", "  ", "x/y"):
            self.check_rejected("experiment", "participant_id", bad)
        for ok in ("anonymous", "P01", "session_3-a"):
            data = good()
            data["experiment"]["participant_id"] = ok
            self.assertEqual(parse_config(data).experiment.participant_id, ok)


class TestServoSafety(unittest.TestCase):
    def test_servo_off_may_keep_the_placeholder(self):
        self.assertFalse(parse_config(good()).hardware.servo_enabled)

    def test_servo_on_with_placeholder_is_refused(self):
        for key in ("servo_neutral_deg", "servo_min_deg", "servo_max_deg"):
            data = with_servo()
            data["hardware"][key] = "VERIFY"
            with self.assertRaisesRegex(ConfigError, "VERIFY"):
                parse_config(data)

    def test_servo_on_with_verified_angles_in_the_window_is_accepted(self):
        hw = parse_config(with_servo()).hardware
        self.assertTrue(hw.servo_enabled)
        self.assertEqual((hw.servo_min_deg, hw.servo_neutral_deg, hw.servo_max_deg), (80, 90, 100))

    def test_servo_angles_outside_the_firmware_window_are_refused(self):
        with self.assertRaisesRegex(ConfigError, "window"):
            parse_config(with_servo(servo_min_deg=70))
        with self.assertRaisesRegex(ConfigError, "window"):
            parse_config(with_servo(servo_max_deg=110))

    def test_servo_angles_must_be_ordered(self):
        with self.assertRaisesRegex(ConfigError, "window"):
            parse_config(with_servo(servo_neutral_deg=79))
        with self.assertRaisesRegex(ConfigError, "window"):
            parse_config(with_servo(servo_min_deg=95, servo_neutral_deg=90))

    def test_servo_angles_must_be_whole_degrees(self):
        for bad in (90.5, "90", True, -5, 400):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                parse_config(with_servo(servo_neutral_deg=bad))

    def test_servo_cannot_be_on_while_hardware_is_off(self):
        with self.assertRaisesRegex(ConfigError, "hardware.enabled"):
            parse_config(with_servo(enabled=False))

    def test_a_missing_servo_setting_is_refused(self):
        data = good()
        del data["hardware"]["servo_enabled"]
        with self.assertRaisesRegex(ConfigError, "missing"):
            parse_config(data)


class TestHapticPolicyBridge(unittest.TestCase):
    def test_to_haptic_config_feeds_the_policy(self):
        from ghostgrip.hardware.haptics import HapticPolicy
        policy = HapticPolicy(to_haptic_config(load_config()))
        self.assertFalse(policy.config["servo_enabled"])
        decision = policy.filter(event_id=1, kind="CONTACT", tracking_quality=1.0, link_armed=True)
        self.assertEqual(decision.action, "cue")
        self.assertTrue(decision.cue.no_contact_mode)

    def test_verified_servo_values_reach_the_policy(self):
        from ghostgrip.hardware.haptics import HapticPolicy
        cfg = parse_config(with_servo())
        policy = HapticPolicy(to_haptic_config(cfg))
        decision = policy.filter(event_id=1, kind="CONTACT", tracking_quality=1.0, link_armed=True)
        self.assertEqual(decision.cue.safe_angle_deg, 95)    # neutral 90 + offset 5, inside 80-100
        self.assertFalse(decision.cue.no_contact_mode)

    def test_low_quality_is_suppressed_using_the_configured_threshold(self):
        from ghostgrip.hardware.haptics import HapticPolicy
        policy = HapticPolicy(to_haptic_config(load_config()))
        decision = policy.filter(event_id=1, kind="CONTACT", tracking_quality=0.59, link_armed=True)
        self.assertEqual(decision.action, "suppressed")


if __name__ == "__main__":
    unittest.main()
