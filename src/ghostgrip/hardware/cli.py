"""Member C command line tool.

  python -m ghostgrip.hardware.cli cue CONTACT        # real board, auto-detected port
  python -m ghostgrip.hardware.cli --mock cue CONTACT # fake board, no hardware needed
  python -m ghostgrip.hardware.cli stop
  python -m ghostgrip.hardware.cli monitor 10         # print sensor packets for 10 s

Run from the repo root with PYTHONPATH=src (the root holds the shared contracts.py).
"""
import argparse
import time

from .haptics import HapticPolicy
from .mock_arduino import MockArduino
from .protocol import Ack
from .serial_link import SerialLink, open_serial


def connect(args):
    transport = MockArduino() if args.mock else open_serial(args.port)
    return SerialLink(transport)


def run_cue(link, kind):
    reply = link.arm()
    print('ARM ->', reply)
    if not isinstance(reply, Ack):
        print('Board did not arm; no cue sent.')
        return 1
    decision = HapticPolicy().filter(event_id=1, kind=kind, tracking_quality=1.0, link_armed=link.armed)
    if decision.action != 'cue':
        print('Policy said', decision.action, decision.reason)
        link.stop()
        return 1
    reply = link.cue(decision.cue)
    print('CUE ->', reply)
    # Keep the watchdog fed while the cue plays, then disarm.
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        link.keepalive()
        time.sleep(0.1)
    print('STOP ->', link.stop())
    return 0 if isinstance(reply, Ack) else 1


def run_monitor(link, seconds):
    end = time.monotonic() + seconds
    last = None
    while time.monotonic() < end:
        if isinstance(link.transport, MockArduino):
            link.transport.sample()
        link.poll()
        if link.device is not None and link.device != last:
            d = link.device
            print(f'uptime={d.uptime_ms} armed={d.armed} touch={d.touch} piezo={d.piezo} '
                  f'rotary={d.rotary_raw} accel={"absent" if d.ax_g is None else (d.ax_g, d.ay_g, d.az_g)}')
            last = d
        time.sleep(0.05)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='GhostGrip hardware tool (Member C)')
    parser.add_argument('--port', default=None, help='serial port; auto-detected if omitted')
    parser.add_argument('--mock', action='store_true', help='use the fake board instead of hardware')
    sub = parser.add_subparsers(dest='command', required=True)
    cue = sub.add_parser('cue', help='arm, send one cue, then disarm')
    cue.add_argument('kind', choices=['CONTACT', 'HOLD', 'SLIP', 'DAMAGE'])
    sub.add_parser('stop', help='disarm immediately')
    monitor = sub.add_parser('monitor', help='print sensor packets')
    monitor.add_argument('seconds', type=float, nargs='?', default=10)
    args = parser.parse_args(argv)

    link = connect(args)
    try:
        if args.command == 'cue':
            return run_cue(link, args.kind)
        if args.command == 'stop':
            print('STOP ->', link.stop())
            return 0
        return run_monitor(link, args.seconds)
    finally:
        link.close()


if __name__ == '__main__':
    raise SystemExit(main())
