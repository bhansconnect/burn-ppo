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
from typing import Callable

from aim import Run
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class MetricsHandler(FileSystemEventHandler):
    """Handler for metrics.jsonl file changes"""

    def __init__(self, run: Run, metrics_path: Path, offset_path: Path, run_name: str):
        self.run = run
        self.metrics_path = metrics_path
        self.offset_path = offset_path
        self.run_name = run_name
        self.offset = self._load_offset()

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

    def process_new_lines(self):
        """Read and process any new lines in the metrics file"""
        if not self.metrics_path.exists():
            return

        try:
            with open(self.metrics_path, "r") as f:
                f.seek(self.offset)
                lines = f.readlines()
                new_offset = f.tell()

            if not lines:
                return

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

        except OSError as e:
            print(f"Warning: Could not read metrics: {e}", file=sys.stderr)

    def _log_metric(self, metric: dict):
        """Log a single metric to Aim"""
        metric_type = metric.get("type")
        step = metric.get("step", 0)

        if metric_type == "hparams":
            # Log hyperparameters
            data = metric.get("data", {})
            for key, value in data.items():
                self.run[key] = value
            print(f"[{self.run_name}] Logged hyperparameters: {list(data.keys())}")

        elif metric_type == "scalar":
            name = metric.get("name", "unknown")
            value = metric.get("value", 0.0)
            self.run.track(value, name=name, step=step)
            # Only print occasionally to avoid spam
            if step % 1000 == 0:
                print(f"[{self.run_name}] Step {step}: {name} = {value:.4f}")

    def on_modified(self, event):
        """Called when the metrics file is modified"""
        if event.src_path == str(self.metrics_path):
            self.process_new_lines()


class RunTracker:
    """Manages Aim Run and MetricsHandler for a single training run"""

    def __init__(self, run_dir: Path, aim_repo: Path, observer: Observer):
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.metrics_path = run_dir / "metrics.jsonl"
        self.offset_path = run_dir / ".aim_offset"
        self.hash_path = run_dir / ".aim_run_hash"

        # Try to resume existing Aim run, or create new one
        run_hash = None
        if self.hash_path.exists():
            run_hash = self.hash_path.read_text().strip()

        self.aim_run = Run(
            run_hash=run_hash,
            repo=str(aim_repo),
            experiment=self.run_name,
        )

        # Save hash for future resumption (if new run)
        if run_hash is None:
            self.hash_path.write_text(self.aim_run.hash)

        # Create metrics handler
        self.handler = MetricsHandler(
            self.aim_run, self.metrics_path, self.offset_path, self.run_name
        )

        # Schedule with observer
        observer.schedule(self.handler, str(run_dir), recursive=False)

        # Process any existing metrics
        self.handler.process_new_lines()

        print(f">>> Tracking run: {self.run_name}")

    def poll(self):
        """Poll for new metrics (backup for missed fsevents)"""
        self.handler.process_new_lines()

    def close(self):
        """Close the Aim run"""
        self.aim_run.close()


class RunsDirectoryHandler(FileSystemEventHandler):
    """Handler for detecting new run directories"""

    def __init__(self, on_new_run: Callable[[Path], None]):
        self.on_new_run = on_new_run

    def on_created(self, event):
        """Called when a new file/directory is created"""
        if event.is_directory:
            run_dir = Path(event.src_path)
            # Small delay to let the directory be fully created
            time.sleep(0.1)
            if run_dir.is_dir():
                self.on_new_run(run_dir)


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
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()

    if not runs_dir.exists():
        print(f"Error: Runs directory does not exist: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Aim watcher for runs directory: {runs_dir}")

    # Use scripts directory as Aim repo location
    aim_repo = Path(__file__).parent

    # Dictionary of active run trackers
    trackers: dict[str, RunTracker] = {}

    # Set up observer
    observer = Observer()

    def register_run(run_dir: Path):
        """Register a new run for tracking"""
        run_name = run_dir.name
        if run_name in trackers:
            return  # Already tracking
        if run_name.startswith("."):
            return  # Skip hidden directories
        try:
            trackers[run_name] = RunTracker(run_dir, aim_repo, observer)
        except Exception as e:
            print(f"Warning: Could not track run {run_name}: {e}", file=sys.stderr)

    # Scan for existing runs
    for subdir in sorted(runs_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            register_run(subdir)

    if not trackers:
        print("No existing runs found. Waiting for new runs...")

    # Watch for new run directories
    runs_handler = RunsDirectoryHandler(register_run)
    observer.schedule(runs_handler, str(runs_dir), recursive=False)
    observer.start()

    print(f"Watching for new runs... (Ctrl+C to stop)")

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        observer.stop()
        observer.join()
        for tracker in trackers.values():
            tracker.close()
        print("Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep running and poll periodically
    try:
        while True:
            time.sleep(args.poll_interval)
            for tracker in list(trackers.values()):
                tracker.poll()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
