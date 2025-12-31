#!/usr/bin/env python3
"""
Aim Watcher - Streaming metrics uploader for burn-ppo

Watches metrics.jsonl and streams new metrics to Aim in real-time.
Tracks file offset in .aim_offset to resume without duplicating logs.

Usage:
    uv run aim_watcher.py ../runs/<run_name>

Setup:
    cd scripts
    uv sync           # Install deps
    uv run aim init   # Initialize Aim repo (once)
    uv run aim up     # Start Aim UI in separate terminal
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from aim import Run
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class MetricsHandler(FileSystemEventHandler):
    """Handler for metrics.jsonl file changes"""

    def __init__(self, run: Run, metrics_path: Path, offset_path: Path):
        self.run = run
        self.metrics_path = metrics_path
        self.offset_path = offset_path
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
            print(f"Logged hyperparameters: {list(data.keys())}")

        elif metric_type == "scalar":
            name = metric.get("name", "unknown")
            value = metric.get("value", 0.0)
            self.run.track(value, name=name, step=step)
            # Only print occasionally to avoid spam
            if step % 1000 == 0:
                print(f"Step {step}: {name} = {value:.4f}")

    def on_modified(self, event):
        """Called when the metrics file is modified"""
        if event.src_path == str(self.metrics_path):
            self.process_new_lines()


def main():
    parser = argparse.ArgumentParser(
        description="Watch metrics.jsonl and stream to Aim"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to run directory containing metrics.jsonl",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.jsonl"
    offset_path = run_dir / ".aim_offset"

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Get run name from directory
    run_name = run_dir.name

    print(f"Starting Aim watcher for run: {run_name}")
    print(f"Metrics file: {metrics_path}")
    print(f"Offset file: {offset_path}")

    # Initialize Aim run
    # Use scripts directory as Aim repo location
    aim_repo = Path(__file__).parent
    run = Run(
        run_hash=run_name,
        repo=str(aim_repo),
        experiment=run_name,
    )

    # Set up file watcher
    handler = MetricsHandler(run, metrics_path, offset_path)

    # Process any existing lines first
    handler.process_new_lines()

    # Set up observer
    observer = Observer()
    observer.schedule(handler, str(run_dir), recursive=False)
    observer.start()

    print("Watching for new metrics... (Ctrl+C to stop)")

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        observer.stop()
        observer.join()
        run.close()
        print("Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep running and poll periodically (in case fsevents misses something)
    try:
        while True:
            time.sleep(args.poll_interval)
            handler.process_new_lines()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
