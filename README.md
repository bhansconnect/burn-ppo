# burn-ppo

A robust, fast PPO (Proximal Policy Optimization) implementation in Rust using the [Burn ML](https://burn.dev) library. Designed for discrete action spaces and board games.

## Features

- **Fast GPU training** via Burn's WGPU backend (Metal/Vulkan/CUDA auto-detection)
- **TOML configuration** with CLI overrides for experiments
- **Checkpointing** with `best` and `latest` symlinks for easy access
- **JSON-lines metrics** with optional Aim streaming for visualization
- **Vectorized environments** for parallel rollout collection
- **Two test environments**: CartPole and Connect Four

## Quick Start

```bash
# Build in release mode (much faster)
cargo build --release

# Train on CartPole (default)
cargo run --release

# Train with custom config
cargo run --release -- --config configs/default.toml --seed 123

# Override specific parameters
cargo run --release -- --learning-rate 0.0003 --num-envs 64
```

## Configuration

Default configuration in `configs/default.toml`:

```toml
env = "cartpole"
num_envs = "auto"        # Scales to 2x CPU cores
num_steps = 128
learning_rate = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
total_timesteps = 1_000_000
```

CLI overrides use kebab-case: `--learning-rate`, `--num-envs`, `--total-timesteps`.

## Monitoring with Aim

Training logs to `runs/<run_name>/metrics.jsonl`. To visualize with Aim:

```bash
cd scripts
uv sync                     # Install Python dependencies
uv run aim init             # Initialize Aim repo (once)
uv run aim up               # Start Aim UI (http://localhost:43800)

# In another terminal
uv run aim_watcher.py ../runs/<run_name>  # Stream metrics
```

The watcher tracks file offsets, so you can restart it without duplicate logs.

## Project Structure

```
src/
  main.rs           # Training loop
  config.rs         # TOML + CLI configuration
  network.rs        # ActorCritic neural network
  ppo.rs            # PPO algorithm (GAE, clipped surrogate)
  env.rs            # Environment trait + VecEnv
  envs/
    cartpole.rs     # CartPole test environment
    connect_four.rs # Connect Four with self-play
  checkpoint.rs     # Model save/load
  metrics.rs        # JSON-lines logger

configs/            # TOML configuration files
scripts/            # Python Aim watcher
runs/               # Training outputs (per-run dirs)
docs/               # Design documentation
```

## Checkpointing

Checkpoints are saved to `runs/<run_name>/checkpoints/`:

- `step_00010000/` - Checkpoint at step 10000
- `latest -> step_00020000/` - Symlink to most recent
- `best -> step_00015000/` - Symlink to highest average return

Each checkpoint includes model weights, optimizer state, and training metadata (step count, returns history, etc.).

## Resuming Training

### Resume after crash (same config)

Continue training from the last checkpoint in an existing run:

```bash
cargo run --release -- --resume runs/<run_name>
```

This loads the config from the run directory and continues where training left off. The global step, optimizer state, and metrics all continue from the checkpoint.

### Extend training duration

To train beyond the original `total_timesteps`:

```bash
cargo run --release -- --resume runs/<run_name> --total-timesteps 2000000
```

Note: Only `--total-timesteps` can be overridden when resuming. Other config changes are ignored to preserve training consistency.

### Fork with different config

Create a new run starting from an existing checkpoint with different hyperparameters:

```bash
# Fork from best checkpoint with new learning rate
cargo run --release -- --fork runs/<run_name>/checkpoints/best \
    --learning-rate 0.0001 --total-timesteps 500000

# Fork from a specific step
cargo run --release -- --fork runs/<run_name>/checkpoints/step_00050000 \
    --learning-rate 0.0001
```

Forking:
- Creates a new run directory
- Preserves the global step from the checkpoint (graphs continue from that point)
- Allows any config changes (learning rate, hyperparameters, etc.)
- Starts fresh metrics but step numbers continue from the checkpoint

## Run Directory Structure

Each training run creates:

```
runs/<run_name>/
  config.toml       # Frozen config snapshot
  metrics.jsonl     # Streaming metrics
  checkpoints/      # Model checkpoints
```

## PPO Implementation Details

Implements all core details from the [ICLR blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/):

- Orthogonal weight initialization
- Adam epsilon = 1e-5
- Learning rate linear annealing
- GAE (lambda=0.95)
- Advantage normalization at minibatch level
- Clipped surrogate objective + value clipping
- Global gradient clipping (max norm 0.5)

## Extending

See `docs/DESIGN.md` for architecture decisions and extension points:

- Add new environments by implementing the `Environment` trait
- Modify reward shaping in the rollout collection loop
- Add auxiliary heads to the network

## Development

```bash
# Run tests
cargo test

# Check compilation
cargo check

# Build docs
cargo doc --open
```

## License

MIT
