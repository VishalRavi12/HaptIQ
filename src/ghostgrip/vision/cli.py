"""Print shared hand states without a window, with optional local CSV recording.

From the root: PYTHONPATH=src python -m ghostgrip.vision.cli --frames 150
"""
import argparse
from contextlib import ExitStack
from datetime import datetime

from ..config import load_config
from .camera import DEFAULT_PROFILE, ROOT, LiveHandSource
from .trace import TraceRecorder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Settings YAML; defaults to repository config.yaml')
    parser.add_argument('--camera', type=int, help='Override configured camera index')
    parser.add_argument('--profile', default=str(DEFAULT_PROFILE))
    parser.add_argument('--frames', type=int, default=0, help='Stop after N states; 0 means until Ctrl+C')
    parser.add_argument('--record', action='store_true', help='Save states, not images, under data/local')
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error('--frames must be nonnegative')
    try:
        config = load_config(args.config)
        with ExitStack() as stack:
            source = stack.enter_context(LiveHandSource(profile_path=args.profile,
                                         camera_index=args.camera, config=config.camera))
            recorder = None
            if args.record:
                path = ROOT / 'data/local' / ('trace_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
                recorder = stack.enter_context(TraceRecorder(path))
                print(f'Recording hand states (no video) to {path}', flush=True)
            for count, state in enumerate(source, 1):
                if recorder is not None:
                    recorder.write(state)
                print(state, flush=True)
                if source.tracker.last_error:
                    print(source.reason, flush=True)
                    return 1
                if args.frames and count >= args.frames:
                    break
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'Perception source failed: {error}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
