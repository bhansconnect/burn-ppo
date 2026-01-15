# DESIGN.md - PPO Implementation in Rust with Burn ML

## Reference Implementation

This implementation follows the **13 core implementation details** from the ICLR Blog Track post:
**[The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)**

These details are critical for matching reference PPO performance:
1. Vectorized architecture (N envs × M steps)
2. **Orthogonal weight initialization** (gain √2 for hidden, 0.01 for policy, 1.0 for value)
3. Adam epsilon = 1e-5 (not default 1e-8)
4. Learning rate annealing (linear decay)
5. Generalized Advantage Estimation (GAE)
6. Mini-batch updates (shuffle and split)
7. Advantage normalization (per-minibatch, not full batch)
8. Clipped surrogate objective
9. Value function loss clipping (optional, may hurt performance)
10. Overall loss = policy_loss - entropy × coef + value_loss × coef
11. Global gradient clipping (L2 norm ≤ 0.5)
12. Debug variables (policy_loss, value_loss, entropy, clipfrac, approx_kl)
13. Shared network with separate heads (2×64 hidden layers)

---

## Philosophy

This implementation follows the "software you can love" / Handmade community philosophy:

### Core Principles

1. **Simplicity over abstraction**: Few traits, mostly concrete implementations. Code should be readable from top to bottom.

2. **Locality of behavior**: Most logic lives in few files. When reading the training loop, you see the training loop - not a maze of trait implementations.

3. **Explicit over implicit**: No hidden magic. Configuration, data flow, and control flow are visible.

4. **Designed for extension, not extensibility**: We design with future extensions in mind but do not build plugin systems or over-generalize interfaces.

5. **Performance through simplicity**: Fast code comes from understanding what the machine is doing, not from clever abstractions.

### Key Tradeoffs

| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Network architecture | Shared backbone | Separate networks | Simpler, standard for discrete games, easier to validate |
| Configuration | TOML + CLI overrides | Pure CLI / Pure config | Reproducible runs + quick experiments |
| Environment interface | Minimal trait | Full gym-like abstraction | Only what PPO needs, nothing more |
| Parallelism | Vectorized envs, batched inference | Async actors | Simpler mental model, sufficient for board games |
| Logging | JSON-lines + Aim | TensorBoard | Proper hyperparameter tracking, great UI |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                            main.rs                                   │
│  CLI parsing, config loading, device init, training loop, shutdown  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ config.rs │   │ network.rs│   │ metrics.rs│
            │           │   │           │   │           │
            │ Config    │   │ActorCritic│   │ JSONLogger│
            │ CliArgs   │   │ forward() │   │ log_scalar│
            │ merge()   │   │ init()    │   │ flush()   │
            └───────────┘   └───────────┘   └───────────┘
                                    │
                                    ▼
            ┌─────────────────────────────────────────────┐
            │                   ppo.rs                     │
            │                                              │
            │  RolloutBuffer    - stores trajectories      │
            │  collect_rollouts - batched env stepping     │
            │  compute_gae      - advantage estimation     │
            │  ppo_update       - clipped surrogate loss   │
            └─────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌─────────────────────────────────────────────┐
            │                   env.rs                     │
            │                                              │
            │  Environment trait  - minimal interface      │
            │  VecEnv            - parallel execution      │
            │  Auto-reset on done                          │
            └─────────────────────────────────────────────┘
                    │                       │
                    ▼                       ▼
            ┌───────────────┐       ┌────────────────┐
            │ cartpole.rs   │       │ connect_four.rs│
            │               │       │                │
            │ Classic       │       │ Self-play      │
            │ control task  │       │ board game     │
            └───────────────┘       └────────────────┘
```

---

## Core Abstractions

### Environment Trait

The only trait in the system. Deliberately minimal - just what PPO needs to interact with a game.

```rust
pub trait Environment: Send + Sync + 'static {
    fn reset(&mut self) -> Vec<f32>;
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool);
    fn observation_dim(&self) -> usize;
    fn action_count(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

**Why this interface:**
- `Vec<f32>` observations keep things simple - no generic observation types
- Single `usize` action for discrete action spaces only
- `(obs, reward, done)` tuple is the standard RL interface
- `Send + Sync` enables parallel execution in VecEnv

### RolloutBuffer

Stores trajectory data from environment rollouts. Shaped as `[num_steps, num_envs, ...]` to match the vectorized collection pattern.

**Key fields:** observations, actions, rewards, dones, values, log_probs, advantages, returns

**Design rationale:** Keep everything as tensors on device. Avoid CPU-GPU transfers during training. Compute advantages after rollout collection, not during.

### ActorCritic Network

Shared backbone MLP with separate policy and value heads. Uses Burn's `#[derive(Module)]` for automatic parameter handling.

**Architecture decisions:**
- Shared backbone (2 hidden layers, tanh activation) - standard for discrete games
- Orthogonal initialization with specific gains (hidden: √2, policy: 0.01, value: 1.0)
- Returns `(logits, value)` - logits go through categorical distribution for action sampling

---

## Module Responsibilities

### main.rs
Orchestration only. Parses CLI, loads config, initializes components, runs training loop, handles Ctrl+C. Should be readable as a high-level overview of the entire training process.

### config.rs
Configuration loading and merging. TOML file provides defaults, CLI args override specific fields. The `NumEnvs` enum supports `"auto"` (scales to CPU cores) or explicit values.

**Key design:** Every hyperparameter lives in Config. No magic constants scattered through code.

### network.rs
Neural network definition using Burn. Handles initialization (orthogonal with correct gains) and forward pass. The network is generic over Burn backend, enabling automatic GPU selection.

### ppo.rs
The algorithm itself. Three main functions:
- `collect_rollouts`: Run N parallel envs for M steps, batching inference
- `compute_gae`: Generalized Advantage Estimation (backward pass through time)
- `ppo_update`: Minibatch updates with clipped surrogate loss

**Critical implementation details (from ICLR blog):**
- Adam epsilon = 1e-5 (not default 1e-8)
- Advantage normalization at minibatch level, not full batch
- Gradient clipping by global L2 norm
- Value function clipping (optional but included)

### env.rs
Environment trait definition and VecEnv wrapper. VecEnv holds N environments and steps them in parallel, automatically resetting on episode termination.

**Design choice:** VecEnv owns the environments and tracks episode statistics. This keeps the training loop clean.

### metrics.rs
JSON-lines logging. Appends to `metrics.jsonl` in the run directory. Logs hyperparameters once at start, then scalars during training.

**Format choice:** JSON-lines enables streaming reads, easy parsing, and clean Aim integration.

### checkpoint.rs
Save and restore training state. Uses Burn's record system for model weights, plus JSON for metadata (step count, RNG state).

**Atomicity:** Write to temp file, then rename. Prevents corruption if killed mid-save.

**Best checkpoint tracking:** Maintains a `best` symlink pointing to the checkpoint with highest average episode return. Useful for playing against the AI - always points to peak performance even if training later degrades.

---

## Environments

### CartPole
Classic control task. Simple physics (cart + pole), 4D observation, 2 discrete actions. Used for validating the PPO implementation against known baselines.

**Validation target:** Average return > 195 for 100 consecutive episodes in < 200k steps.

### Connect Four
7x6 board game with self-play. Agent plays against itself (or a fixed policy for initial testing). 42D observation (flattened board), 7 discrete actions (column choice).

**Self-play approach:** During each environment step, after the agent moves, the opponent (same or different policy) also moves. This keeps the Environment interface simple.

---

## Metrics and Logging

### JSON-lines Format

```json
{"step": 0, "type": "hparams", "data": {"lr": 0.00025, ...}}
{"step": 1000, "type": "scalar", "name": "train/policy_loss", "value": 0.5}
```

**Why JSON-lines:**
- Append-only (crash-safe)
- Streaming reads (Aim watcher can tail the file)
- Human-readable for debugging
- Hyperparameters are structured data, not encoded in filenames

### Aim Integration

Separate Python script watches `metrics.jsonl` and streams to Aim. Tracks file offset to resume without duplication after restart.

**Separation of concerns:** Rust training writes logs. Python handles visualization. No Python dependency in the training binary.

### Key Metrics

| Metric | Purpose |
|--------|---------|
| `train/policy_loss` | PPO clipped surrogate objective |
| `train/value_loss` | Value function MSE |
| `train/entropy` | Policy randomness (should stay >0.1) |
| `train/approx_kl` | Policy change magnitude (should stay <0.02) |
| `train/clip_fraction` | How often clipping activates |
| `episode/return` | Training progress indicator |

---

## Extension Points

The implementation is designed so these extensions require minimal changes:

### Reward Shaping
Add an optional `RewardShaper` that transforms rewards in the rollout collection loop. Core PPO logic unchanged.

### Auxiliary Heads
Extend network output to include additional predictions. Add corresponding loss terms. The shared backbone architecture makes this natural.

### Split Actor/Critic
Config flag to use separate networks instead of shared backbone. Same training loop, different network initialization.

### Async Rollout/Training Overlap
Move rollout collection to a separate thread, communicate via channel. Allows CPU-bound env stepping to overlap with GPU-bound training. Requires periodic model weight sync.

### Teacher-Student Networks

For complex games, train smaller networks first, then use as teachers:

**Approach:**
1. Train small network (e.g., 2×64 hidden) to competence
2. Train larger student network (e.g., 3×256 hidden)
3. Student loss = PPO_loss + α × KL_divergence(student_policy || teacher_policy)
4. Anneal α from 1.0 → 0.0 over training

**Training Flow:**

```
Phase 1: Train Teacher (small network)
    └─ Train to convergence on task
    └─ Save best checkpoint

Phase 2: Train Student (large network)
    └─ Initialize fresh
    └─ For each batch:
        ├─ Get teacher logits (frozen)
        ├─ Get student logits
        ├─ Compute PPO loss (normal)
        ├─ Compute distillation loss: α × KL(student || teacher)
        ├─ Total loss = PPO_loss + distillation_loss
        └─ Update student only
    └─ Anneal α: 1.0 → 0.0 over N steps

Phase 3: Pure Student (optional)
    └─ Continue training without teacher
    └─ Student may surpass teacher
```

**Benefits:**
- Faster initial learning (student benefits from teacher's exploration)
- More stable training for complex action spaces
- Knowledge distillation improves sample efficiency
- Student can eventually exceed teacher (not constrained by teacher's optimum)

**When to Use:**
- Large action spaces where exploration is expensive
- Complex games where random initial policy struggles
- When smaller network achieves reasonable performance but larger network fails to train

**Implementation:**

```rust
pub struct TeacherStudentConfig {
    pub teacher_checkpoint: PathBuf,    // Path to trained teacher
    pub distillation_coef: f64,         // Initial α (default 1.0)
    pub anneal_steps: usize,            // Steps to anneal α to 0
    pub temperature: f64,               // Softmax temperature (default 1.0)
}
```

- Teacher network frozen, loaded from checkpoint
- Student receives teacher logits as soft targets
- Temperature parameter softens distributions (higher = softer)
- Config flag to enable/disable
- Optional: Use teacher for action sampling during early training (imitation warmup)

---

## Testing Philosophy

### Unit Tests
Test mathematical correctness of core algorithms in isolation:
- GAE computation against hand-calculated examples
- Advantage normalization produces zero mean, unit variance
- Clipped loss behaves correctly at boundary conditions

### Environment Tests
Verify environment implementations match expected behavior:
- Reset produces valid initial states
- Physics/game logic is correct
- Terminal conditions trigger appropriately

### Integration Tests
Ensure components work together:
- Checkpoint save/load preserves exact training state
- Full training loop runs without crashing
- Resume continues from correct step

### Validation
CartPole serves as the integration test with a known baseline. If PPO can't solve CartPole, something is broken.

---

## Configuration Reference

All hyperparameters live in TOML config files. CLI can override any field.

**Environment:** `num_envs` (auto-scales to CPUs), `num_steps` (rollout length)

**PPO:** `learning_rate`, `gamma`, `gae_lambda`, `clip_epsilon`, `entropy_coef`, `value_coef`, `max_grad_norm`

**Training:** `total_steps`, `num_epochs`, `num_minibatches`, `adam_epsilon`

**Network:** `hidden_size`, `num_hidden`

**Checkpointing:** `checkpoint_freq`

**Defaults follow ICLR blog recommendations** for discrete control tasks.

---

## Run Directory Structure

```
runs/<run_name>/
  config.toml       # Frozen config snapshot
  metrics.jsonl     # Streaming metrics
  checkpoints/
    step_10000/     # Checkpoint at step 10000
    step_20000/     # Checkpoint at step 20000
    ...
    latest -> step_20000/   # Symlink to most recent
    best -> step_10000/     # Symlink to highest avg return
```

Each run is self-contained and reproducible. The `best` symlink provides stable access to peak performance for inference/play.
