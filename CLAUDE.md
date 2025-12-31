# CLAUDE.md

## Project Overview

PPO (Proximal Policy Optimization) implementation in Rust using the Burn ML library. Designed for discrete action games (board games like Connect Four, eventually Wingspan).

## Design Document

See [docs/DESIGN.md](docs/DESIGN.md) for:
- Philosophy and core principles
- Architecture overview
- Module responsibilities
- Extension points
- Testing philosophy

## Quick Reference

### Build & Run
```bash
cargo build --release
cargo run --release -- --config configs/cartpole.toml
cargo run --release -- --config configs/cartpole.toml --num-envs 64 --seed 123

# Peak performance build (slower compile, for benchmarks/distribution)
cargo build --profile release-lto
./target/release-lto/burn-ppo --config configs/cartpole.toml
```

| Profile | Build Time | Use Case |
|---------|------------|----------|
| `release` | ~2m15s clean, ~3-5s incremental | Daily development |
| `release-lto` | ~6min | Benchmarks, distribution |

### Resume Training
```bash
# Continue from last checkpoint (same config)
cargo run --release -- --resume runs/<run_name>

# Extend training duration
cargo run --release -- --resume runs/<run_name> --total-timesteps 2000000

# Fork with different config
cargo run --release -- --fork runs/<run_name>/checkpoints/best --learning-rate 0.0001
```

### Run Tests
```bash
cargo test
```

### Aim Metrics Viewer
```bash
cd scripts
uv sync
uv run aim init    # Once
uv run aim up      # Start server at localhost:43800
uv run aim_watcher.py ../runs  # Stream metrics from all runs
```

### Profiling with Tracy
```bash
# Build with Tracy instrumentation
cargo build --release --features tracy

# Run - Tracy GUI will auto-connect
cargo run --release --features tracy -- --config configs/cartpole.toml
```

Tracy profiler download: https://github.com/wolfpld/tracy/releases

## Key Files

| File | Purpose |
|------|---------|
| `src/main.rs` | Entry point, CLI, training loop |
| `src/config.rs` | TOML config + CLI overrides |
| `src/ppo.rs` | PPO algorithm (rollouts, GAE, update) |
| `src/network.rs` | ActorCritic neural network |
| `src/env.rs` | Environment trait, VecEnv |
| `src/envs/` | Environment implementations |
| `src/checkpoint.rs` | Save/restore training state |
| `src/metrics.rs` | JSON-lines logging |

## Configuration

All hyperparameters in TOML. CLI overrides any field:
- `--learning-rate 0.0003`
- `--num-envs 128`
- `--seed 42`

See `configs/cartpole.toml` for all options.

## Checkpoints

```
runs/<run_name>/checkpoints/
  step_10000/
  step_20000/
  latest -> step_20000/   # Most recent
  best -> step_10000/     # Highest avg return
```

Use `best` for inference/playing against the AI.
