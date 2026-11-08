"""Preflight: what can this laptop do right now? (Phase 1 plan D6, technical doc 14.1)

    python scripts/preflight.py              # full check, including opening the camera
    python scripts/preflight.py --no-camera  # skip the camera (it may trigger a permission prompt)

It reports; it never crashes and never moves hardware. Each line is one of:
    OK    works
    WARN  a capability is missing or untested, and the project has a fallback
    FAIL  something every mode needs is broken (pygame, pyyaml, config.yaml)

and it ends with the modes that are available. The exit code is 1 only when something FAILs.
"""
import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
OK, WARN, FAIL = "OK", "WARN", "FAIL"
SUPPORTED_PYTHON = ((3, 11), (3, 12))
MODEL_RELATIVE = Path("assets") / "hand_landmarker.task"
CALIBRATION_RELATIVE = Path("data") / "local" / "calibration.json"
MIN_MODEL_BYTES = 1000


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


class Probes:
    """Everything that touches the machine. Tests replace this with a fake."""

    def python_version(self) -> Tuple[int, int, int]:
        return tuple(sys.version_info[:3])

    def import_module(self, name: str) -> Tuple[bool, str]:
        try:
            module = importlib.import_module(name)
        except Exception as error:                      # a broken install can raise anything
            return False, f"{type(error).__name__}: {error}"
        return True, str(getattr(module, "__version__", "installed"))

    def file_size(self, path: Path) -> Optional[int]:
        try:
            return path.stat().st_size if path.is_file() else None
        except OSError:
            return None

    def camera(self, index: int) -> Tuple[bool, str]:
        import cv2
        capture = cv2.VideoCapture(index)
        try:
            if not capture.isOpened():
                return False, f"could not open camera {index} (in use, or no permission)"
            ok, _ = capture.read()
            return ok, "a frame was read" if ok else "opened, but no frame arrived"
        finally:
            capture.release()

    def serial_port(self) -> Optional[str]:
        from ghostgrip.hardware.serial_link import find_port
        return find_port()


def _safe(name: str, probe, on_error: str = WARN) -> Check:
    """Run one check; whatever goes wrong becomes a line in the report, not a traceback."""
    try:
        return probe()
    except Exception as error:
        return Check(name, on_error, f"the check itself failed: {type(error).__name__}: {error}")


def run_checks(probes: Optional[Probes] = None, root: Path = REPO_ROOT, camera: bool = True,
               camera_index: int = 0) -> List[Check]:
    probes = probes or Probes()
    checks: List[Check] = []

    def python():
        v = probes.python_version()
        if (v[0], v[1]) in SUPPORTED_PYTHON:
            return Check("python", OK, f"{v[0]}.{v[1]}.{v[2]}")
        return Check("python", WARN, f"{v[0]}.{v[1]}.{v[2]}; the project is tested on 3.11 and 3.12 "
                                     "(MediaPipe may have no wheel for other versions)")
    checks.append(_safe("python", python))

    for module, needed_for, level in (("pygame", "simulation and UI", FAIL), ("yaml", "config.yaml", FAIL),
                                      ("cv2", "live camera", WARN), ("mediapipe", "hand tracking", WARN),
                                      ("serial", "Arduino link", WARN)):
        def package(module=module, needed_for=needed_for, level=level):
            ok, detail = probes.import_module(module)
            return Check(f"package {module}", OK if ok else level,
                         f"{detail}" if ok else f"missing or broken ({detail}); needed for {needed_for}; "
                                                f"run: pip install -r requirements.txt")
        checks.append(_safe(f"package {module}", package, on_error=level))

    def config():
        from ghostgrip.config import ConfigError, load_config
        try:
            cfg = load_config(root / "config.yaml")
        except ConfigError as error:
            return Check("config.yaml", FAIL, str(error))
        servo = "ON" if cfg.hardware.servo_enabled else "OFF"
        learning = "frozen" if cfg.ai.freeze_adaptation else "LEARNING (practice mode)"
        return Check("config.yaml", OK, f"valid; hardware {'enabled' if cfg.hardware.enabled else 'disabled'}; "
                                        f"servo {servo}; assist profile {learning}")
    checks.append(_safe("config.yaml", config, on_error=FAIL))

    def model():
        size = probes.file_size(root / MODEL_RELATIVE)
        if size is None or size < MIN_MODEL_BYTES:
            return Check("hand model", WARN, f"{MODEL_RELATIVE} missing; run: python download_hand_model.py")
        return Check("hand model", OK, f"{MODEL_RELATIVE} ({size // 1024} KB)")
    checks.append(_safe("hand model", model))

    def calibration():
        if probes.file_size(root / CALIBRATION_RELATIVE) is None:
            return Check("calibration", WARN, "no saved profile; run: python vision_preview.py --calibrate")
        return Check("calibration", OK, f"{CALIBRATION_RELATIVE} found (stays on this laptop, not in git)")
    checks.append(_safe("calibration", calibration))

    def camera_check():
        if not camera:
            return Check("camera", WARN, "skipped (--no-camera)")
        ok, detail = probes.camera(camera_index)
        return Check("camera", OK if ok else WARN, detail)
    checks.append(_safe("camera", camera_check))

    def serial():
        port = probes.serial_port()
        if port is None:
            return Check("arduino port", WARN, "no board found; use the mock: "
                                               "python -m ghostgrip.hardware.cli --mock cue CONTACT")
        return Check("arduino port", OK, f"{port} (found; nothing was sent to it)")
    checks.append(_safe("arduino port", serial))
    return checks


def available_modes(checks: List[Check]) -> List[Tuple[str, bool, str]]:
    """Which ways of running the project work, from the checks."""
    ok = {c.name: c.status == OK for c in checks}
    core = ok.get("package pygame", False) and ok.get("package yaml", False) and ok.get("config.yaml", False)
    live = (core and ok.get("package cv2", False) and ok.get("package mediapipe", False)
            and ok.get("hand model", False) and ok.get("camera", False))
    board = ok.get("package serial", False) and ok.get("arduino port", False)
    return [
        ("mouse simulation and synthetic replay", core, "needs pygame, pyyaml and a valid config.yaml"),
        ("live camera hand tracking", live, "needs OpenCV, MediaPipe, the hand model and a camera"),
        ("real Arduino (LED cues)", board, "needs pyserial and a board; otherwise use --mock"),
        ("mock Arduino and tests", ok.get("config.yaml", False), "no hardware needed"),
    ]


def format_report(checks: List[Check]) -> str:
    lines = [f"[{c.status:^4}] {c.name:<18} {c.detail}" for c in checks]
    lines += ["", "Available now:"]
    lines += [f"  {'yes' if works else 'NO ':<3} {name}" + ("" if works else f"   ({why})")
              for name, works, why in available_modes(checks)]
    return "\n".join(lines)


def main(argv=None, probes: Optional[Probes] = None) -> int:
    parser = argparse.ArgumentParser(description="Report what this laptop can do for GhostGrip.")
    parser.add_argument("--no-camera", action="store_true", help="do not try to open the camera")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args(argv)
    checks = run_checks(probes, camera=not args.no_camera, camera_index=args.camera_index)
    print(format_report(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
