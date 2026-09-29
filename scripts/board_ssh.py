"""SSH/SFTP board runner; password from PLD_BOARD_PASS or interactive prompt.
Run: uv run --with paramiko python scripts/board_ssh.py --help
--sudo supplies the password on stdin and loads PYNQ/XRT via a login shell.
"""
import argparse
import getpass
import os
import shlex
import sys
import time

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--sudo", action="store_true")
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("command")
    for mode in ("put", "get"):
        p = sub.add_parser(mode)
        p.add_argument("source")
        p.add_argument("destination")
    args = parser.parse_args()
    password = os.environ.get("PLD_BOARD_PASS") or getpass.getpass("Board password: ")
    with paramiko.SSHClient() as client:
        client.load_system_host_keys()
        # Unknown keys are accepted for this LAN session, never persisted.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(os.environ.get("PLD_BOARD_HOST", "192.168.2.99"),
                       username=os.environ.get("PLD_BOARD_USER", "xilinx"),
                       password=password, timeout=10, auth_timeout=10,
                       banner_timeout=10, allow_agent=False, look_for_keys=False)
        if args.mode != "run":
            with client.open_sftp() as sftp:
                getattr(sftp, args.mode)(args.source, args.destination)
            print(f"{args.mode} {args.source} -> {args.destination}")
            return 0
        command = args.command
        if args.sudo:
            command = "sudo -S -p '' bash -lc " + shlex.quote(command)
        stdin, stdout, _ = client.exec_command(command, timeout=args.timeout)
        if args.sudo:
            stdin.write(password + "\n")
            stdin.flush()
        stdin.channel.shutdown_write()
        channel = stdout.channel
        deadline = time.monotonic() + args.timeout
        while True:
            for ready, receive, stream in (
                (channel.recv_ready, channel.recv, sys.stdout),
                (channel.recv_stderr_ready, channel.recv_stderr, sys.stderr),
            ):
                if ready():
                    stream.write(receive(65536).decode(errors="replace"))
                    stream.flush()
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                return channel.recv_exit_status()
            if time.monotonic() >= deadline:
                channel.close()
                raise TimeoutError("SSH deadline exceeded; remote hardware state is NOT confirmed stopped")
            time.sleep(0.01)


if __name__ == "__main__":
    raise SystemExit(main())
