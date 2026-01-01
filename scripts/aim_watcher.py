#!/usr/bin/env python3
"""
Aim Watcher - Streaming metrics uploader for burn-ppo

Watches the runs directory and streams metrics from all runs to Aim in real-time.
Automatically detects new runs as they are created.
Tracks file offset per run in .aim_offset to resume without duplicating logs.

Usage:
    uv run aim_watcher.py ../runs

Setup:
    cd scripts
    uv sync           # Install deps
    uv run aim init   # Initialize Aim repo (once)
    uv run aim up     # Start Aim UI in separate terminal
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from aim import Run


class RunManager:
    """Manages the global pool of open Aim runs to enforce limits"""

    def __init__(self, trackers: dict, max_open: int):
        self.trackers = trackers
        self.max_open = max_open
        self.open_count = 0

    def can_open(self) -> bool:
        """Check if we can open another run"""
        return self.open_count < self.max_open

    def on_open(self):
        """Called when a run is opened"""
        self.open_count += 1

    def on_close(self):
        """Called when a run is closed"""
        self.open_count = max(0, self.open_count - 1)

    def close_oldest_idle(self) -> bool:
        """Close the oldest idle run to make room. Returns True if one was closed."""
        oldest_tracker = None
        oldest_time = float("inf")

        for tracker in self.trackers.values():
            if tracker.aim_run is not None:
                if tracker.last_activity_time < oldest_time:
                    oldest_time = tracker.last_activity_time
                    oldest_tracker = tracker

        if oldest_tracker:
            print(f">>> Closing oldest idle run to make room: {oldest_tracker.run_name}")
            oldest_tracker.close()
            return True
        return False

    def ensure_capacity(self):
        """Ensure there's capacity for a new run, closing old ones if needed"""
        while not self.can_open():
            if not self.close_oldest_idle():
                break  # No more runs to close


class RunTracker:
    """Manages Aim Run and metrics reading for a single training run"""

    def __init__(
        self, run_dir: Path, aim_repo: Path, print_interval: int, manager: RunManager
    ):
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.metrics_path = run_dir / "metrics.jsonl"
        self.offset_path = run_dir / ".aim_offset"
        self.hash_path = run_dir / ".aim_run_hash"
        self.offset = self._load_offset()

        # Store aim_repo for lazy initialization
        self.aim_repo = aim_repo
        self.manager = manager
        self.aim_run = None  # Lazy init - only open when new data arrives
        self.last_activity_time = time.time()

        # Track last printed step per metric to avoid spam
        self.print_interval = print_interval
        self.last_printed_step: dict[str, int] = {}

        print(f">>> Tracking run: {self.run_name}")

    def _load_offset(self) -> int:
        """Load last processed offset from file"""
        if self.offset_path.exists():
            try:
                return int(self.offset_path.read_text().strip())
            except (ValueError, OSError):
                pass
        return 0

    def _save_offset(self):
        """Save current offset to file"""
        try:
            self.offset_path.write_text(str(self.offset))
        except OSError as e:
            print(f"Warning: Could not save offset: {e}", file=sys.stderr)

    def _init_aim_run(self):
        """Initialize the Aim run (lazy, called on first data)"""
        run_hash = None
        if self.hash_path.exists():
            run_hash = self.hash_path.read_text().strip()

        self.aim_run = Run(
            run_hash=run_hash,
            repo=str(self.aim_repo),
            experiment=self.run_name,
        )

        # Save hash for future resumption (if new run)
        if run_hash is None:
            self.hash_path.write_text(self.aim_run.hash)

        self.last_activity_time = time.time()
        print(f">>> Opened Aim run: {self.run_name}")

    def _ensure_open(self):
        """Ensure the Aim run is open, initializing if needed"""
        if self.aim_run is None:
            self.manager.ensure_capacity()
            self._init_aim_run()
            self.manager.on_open()

    def poll(self) -> bool:
        """Read and process any new lines in the metrics file. Returns True if new data found."""
        if not self.metrics_path.exists():
            return False

        try:
            with open(self.metrics_path, "r") as f:
                f.seek(self.offset)
                lines = f.readlines()
                new_offset = f.tell()

            if not lines:
                return False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    metric = json.loads(line)
                    self._log_metric(metric)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON: {e}", file=sys.stderr)

            self.offset = new_offset
            self._save_offset()
            self.last_activity_time = time.time()
            return True

        except OSError as e:
            # Handle "Too many open files" by closing this run and retrying later
            if "Too many open files" in str(e) or e.errno == 24:
                print(
                    f"Warning: File descriptor limit hit for {self.run_name}, "
                    "closing run to recover",
                    file=sys.stderr,
                )
                self.close()
            else:
                print(f"Warning: Could not read metrics: {e}", file=sys.stderr)
            return False

    def _log_metric(self, metric: dict):
        """Log a single metric to Aim"""
        try:
            self._ensure_open()

            metric_type = metric.get("type")
            step = metric.get("step", 0)

            if metric_type == "hparams":
                # Log hyperparameters
                data = metric.get("data", {})
                for key, value in data.items():
                    self.aim_run[key] = value
                print(f"[{self.run_name}] Logged hyperparameters: {list(data.keys())}")

            elif metric_type == "scalar":
                name = metric.get("name", "unknown")
                value = metric.get("value", 0.0)
                self.aim_run.track(value, name=name, step=step)

                # Print if enabled and: not a _single metric AND N+ steps since last print
                if self.print_interval > 0 and "_single" not in name:
                    last_step = self.last_printed_step.get(name, -self.print_interval)
                    if step - last_step >= self.print_interval:
                        print(f"[{self.run_name}] Step {step}: {name} = {value:.4f}")
                        self.last_printed_step[name] = step

        except OSError as e:
            if "Too many open files" in str(e) or e.errno == 24:
                print(
                    f"Warning: File descriptor limit hit logging to {self.run_name}, "
                    "closing run to recover",
                    file=sys.stderr,
                )
                self.close()
            else:
                raise

    def check_idle_timeout(self, timeout: float) -> bool:
        """Close run if idle too long. Returns True if closed."""
        if self.aim_run is None:
            return False

        elapsed = time.time() - self.last_activity_time
        if elapsed >= timeout:
            self.close()
            print(f">>> Closed idle run: {self.run_name} (inactive for {elapsed:.0f}s)")
            return True
        return False

    def close(self):
        """Close the Aim run"""
        if self.aim_run is not None:
            try:
                self.aim_run.close()
            except Exception as e:
                print(f"Warning: Error closing run {self.run_name}: {e}", file=sys.stderr)
            self.aim_run = None
            self.manager.on_close()


def validate_run_hashes(runs_dir: Path, aim_repo: Path):
    """Reset runs whose Aim run hash no longer exists (repo was recreated).

    When the .aim directory is deleted and recreated, the stored .aim_run_hash
    files point to non-existent runs. This function detects this and resets
    those runs so they re-upload all metrics from the beginning.
    """
    chunks_dir = aim_repo / ".aim" / "meta" / "chunks"

    if not chunks_dir.exists():
        # Aim repo doesn't exist or is empty - reset all runs
        print(">>> Aim repo appears empty, will reset any existing run tracking")

    reset_count = 0
    try:
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue

            hash_file = run_dir / ".aim_run_hash"
            offset_file = run_dir / ".aim_offset"

            if hash_file.exists():
                try:
                    run_hash = hash_file.read_text().strip()
                    hash_dir = chunks_dir / run_hash

                    if not hash_dir.exists():
                        print(f">>> Reset run {run_dir.name}: Aim repo was recreated")
                        hash_file.unlink()
                        if offset_file.exists():
                            offset_file.unlink()
                        reset_count += 1
                except OSError as e:
                    print(
                        f"Warning: Could not validate run {run_dir.name}: {e}",
                        file=sys.stderr,
                    )
    except OSError as e:
        print(f"Warning: Could not scan runs directory: {e}", file=sys.stderr)

    if reset_count > 0:
        print(f">>> Reset {reset_count} run(s) due to Aim repo recreation")


def main():
    parser = argparse.ArgumentParser(
        description="Watch all runs and stream metrics to Aim"
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Path to runs directory (parent of individual run folders)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=0,
        help="Print metrics every N steps (default: 0 = no printing)",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=60.0,
        help="Close idle runs after N seconds (default: 60.0, 0 = never)",
    )
    parser.add_argument(
        "--max-open-runs",
        type=int,
        default=3,
        help="Maximum concurrent open Aim runs (default: 3)",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()

    if not runs_dir.exists():
        print(f"Error: Runs directory does not exist: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Aim watcher for runs directory: {runs_dir}")

    # Use scripts directory as Aim repo location
    aim_repo = Path(__file__).parent

    # Validate run hashes on startup (detect if Aim repo was recreated)
    validate_run_hashes(runs_dir, aim_repo)

    # Dictionary of active run trackers
    trackers: dict[str, RunTracker] = {}

    # Manager to enforce max open runs limit
    manager = RunManager(trackers, args.max_open_runs)

    def register_run(run_dir: Path):
        """Register a new run for tracking"""
        run_name = run_dir.name
        if run_name in trackers:
            return  # Already tracking
        if run_name.startswith("."):
            return  # Skip hidden directories
        try:
            trackers[run_name] = RunTracker(
                run_dir, aim_repo, args.print_interval, manager
            )
        except Exception as e:
            print(f"Warning: Could not track run {run_name}: {e}", file=sys.stderr)

    # Scan for existing runs
    for subdir in sorted(runs_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            register_run(subdir)

    if not trackers:
        print("No existing runs found. Waiting for new runs...")

    print(f"Watching for new runs... (Ctrl+C to stop)")

    # Handle graceful shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main polling loop
    while not shutdown_requested:
        time.sleep(args.poll_interval)

        # Scan for new run directories
        try:
            for subdir in runs_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    register_run(subdir)
        except OSError as e:
            print(f"Warning: Could not scan runs directory: {e}", file=sys.stderr)

        # Poll all trackers for new metrics
        for tracker in trackers.values():
            tracker.poll()

        # Check for idle timeouts
        if args.idle_timeout > 0:
            for tracker in trackers.values():
                tracker.check_idle_timeout(args.idle_timeout)

    # Graceful shutdown
    print("\nShutting down...")
    for tracker in trackers.values():
        tracker.close()
    print("Done.")


if __name__ == "__main__":
    main()
