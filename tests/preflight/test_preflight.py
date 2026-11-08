"""Tests for src/ghostgrip/preflight.py. The camera and serial port are faked: no hardware needed."""
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from ghostgrip import preflight as P  # noqa: E402
from ghostgrip.preflight import FAIL, OK, WARN, Probes, available_modes, run_checks  # noqa: E402


class FakeProbes(Probes):
    def __init__(self, missing=(), python=(3, 11, 9), model=True, calibration=True, camera=(True, "a frame was read"),
                 port="/dev/cu.usbserial-1", camera_raises=False, port_raises=False):
        self.missing, self._python, self.model, self.calibration = set(missing), python, model, calibration
        self._camera, self._port = camera, port
        self.camera_raises, self.port_raises = camera_raises, port_raises
        self.camera_calls = 0

    def python_version(self):
        return self._python

    def import_module(self, name):
        return (False, "No module named " + name) if name in self.missing else (True, "9.9")

    def file_size(self, path):
        if path.name == "hand_landmarker.task":
            return 4_000_000 if self.model else None
        if path.name == "calibration.json":
            return 300 if self.calibration else None
        return None

    def camera(self, index):
        self.camera_calls += 1
        if self.camera_raises:
            raise RuntimeError("driver exploded")
        return self._camera

    def serial_port(self):
        if self.port_raises:
            raise OSError("permission denied")
        return self._port


def statuses(checks):
    return {c.name: c.status for c in checks}


def modes(checks):
    return {name: works for name, works, _ in available_modes(checks)}


class TestChecks(unittest.TestCase):
    def test_a_fully_equipped_laptop_is_all_ok(self):
        checks = run_checks(FakeProbes())
        self.assertEqual(set(statuses(checks).values()), {OK})
        self.assertTrue(all(modes(checks).values()))

    def test_missing_vision_packages_only_cost_the_live_camera_mode(self):
        checks = run_checks(FakeProbes(missing={"cv2", "mediapipe"}))
        s = statuses(checks)
        self.assertEqual((s["package cv2"], s["package mediapipe"]), (WARN, WARN))
        m = modes(checks)
        self.assertFalse(m["live camera hand tracking"])
        self.assertTrue(m["mouse simulation and synthetic replay"])

    def test_missing_pygame_or_yaml_is_a_fail(self):
        for module in ("pygame", "yaml"):
            with self.subTest(module=module):
                checks = run_checks(FakeProbes(missing={module}))
                self.assertEqual(statuses(checks)[f"package {module}"], FAIL)
                self.assertFalse(modes(checks)["mouse simulation and synthetic replay"])

    def test_a_missing_model_or_calibration_or_camera_or_board_is_only_a_warning(self):
        s = statuses(run_checks(FakeProbes(model=False, calibration=False, camera=(False, "no permission"), port=None)))
        for name in ("hand model", "calibration", "camera", "arduino port"):
            self.assertEqual(s[name], WARN, name)

    def test_the_hints_say_what_to_run(self):
        details = {c.name: c.detail for c in run_checks(FakeProbes(model=False, calibration=False, port=None))}
        self.assertIn("download_hand_model.py", details["hand model"])
        self.assertIn("--calibrate", details["calibration"])
        self.assertIn("--mock", details["arduino port"])

    def test_the_camera_can_be_skipped_and_is_then_not_opened(self):
        probes = FakeProbes()
        checks = run_checks(probes, camera=False)
        self.assertEqual(probes.camera_calls, 0)
        self.assertEqual(statuses(checks)["camera"], WARN)
        self.assertFalse(modes(checks)["live camera hand tracking"])

    def test_a_probe_that_crashes_is_reported_not_raised(self):
        checks = run_checks(FakeProbes(camera_raises=True, port_raises=True))
        s = {c.name: c for c in checks}
        for name in ("camera", "arduino port"):
            self.assertEqual(s[name].status, WARN)
            self.assertIn("the check itself failed", s[name].detail)

    def test_an_unusual_python_is_a_warning_not_a_failure(self):
        self.assertEqual(statuses(run_checks(FakeProbes(python=(3, 13, 1))))["python"], WARN)
        self.assertEqual(statuses(run_checks(FakeProbes(python=(3, 12, 0))))["python"], OK)

    def test_an_invalid_config_is_a_fail_and_a_missing_one_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(statuses(run_checks(FakeProbes(), root=root))["config.yaml"], FAIL)   # no file
            shutil.copy(REPO_ROOT / "config.yaml", root / "config.yaml")
            self.assertEqual(statuses(run_checks(FakeProbes(), root=root))["config.yaml"], OK)
            text = (root / "config.yaml").read_text().replace("servo_enabled: false", "servo_enabled: true")
            (root / "config.yaml").write_text(text)                     # servo on while angles are VERIFY
            checks = run_checks(FakeProbes(), root=root)
            self.assertEqual(statuses(checks)["config.yaml"], FAIL)
            self.assertIn("VERIFY", {c.name: c.detail for c in checks}["config.yaml"])

    def test_the_committed_config_reports_the_servo_off(self):
        checks = {c.name: c for c in run_checks(FakeProbes(), root=REPO_ROOT)}
        self.assertIn("servo OFF", checks["config.yaml"].detail)

    def test_real_probes_never_raise_on_this_machine(self):
        checks = run_checks(camera=False)                    # no camera: no permission prompt in tests
        self.assertGreaterEqual(len(checks), 9)
        self.assertTrue(all(c.status in (OK, WARN, FAIL) for c in checks))


class TestMain(unittest.TestCase):
    def run_main(self, probes, argv=()):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = P.main(list(argv), probes)
        return code, out.getvalue()

    def test_exit_code_is_zero_unless_something_fails(self):
        code, text = self.run_main(FakeProbes())
        self.assertEqual(code, 0)
        self.assertIn("Available now:", text)
        self.assertIn("[ OK ]", text)
        self.assertEqual(self.run_main(FakeProbes(model=False, port=None))[0], 0)    # warnings are not failures
        self.assertEqual(self.run_main(FakeProbes(missing={"pygame"}))[0], 1)

    def test_no_camera_flag_is_honoured(self):
        probes = FakeProbes()
        code, text = self.run_main(probes, ["--no-camera"])
        self.assertEqual((code, probes.camera_calls), (0, 0))
        self.assertIn("skipped", text)

    def test_the_launcher_script_runs_and_reports(self):
        import subprocess
        result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "preflight.py"), "--no-camera"],
                                capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT))
        self.assertIn("Available now:", result.stdout)
        self.assertIn(result.returncode, (0, 1))


if __name__ == "__main__":
    unittest.main()
