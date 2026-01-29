# Configuration Files

This directory contains example PPO configurations for different environments and network architectures.

## Basic Configurations

### Single-Agent Tasks
- **`cartpole.toml`** - Classic control task for testing and debugging
  - Single agent, simple physics
  - Good for validating new features

### Multi-Agent Games
- **`connect_four.toml`** - Two-player zero-sum game
  - Perfect information
  - Traditional MLP architecture

- **`skull.toml`** - Multi-player bluffing game (2-6 players)
  - Hidden information, strategic deception
  - Traditional MLP architecture

- **`liars_dice.toml`** - Four-player dice game
  - Hidden information, probability-based bluffing
  - Traditional MLP architecture

## CTDE (Centralized Training, Decentralized Execution)

CTDE is designed for multi-agent games with hidden information, where the critic benefits from seeing privileged global state during training, while the actor uses only local observations for deployment.

### When to Use CTDE

**Use CTDE for:**
- Multi-agent games with hidden information (poker, bluffing games)
- Games where global state significantly helps value estimation
- Competitive games requiring decentralized execution at deployment

**Don't use CTDE for:**
- Perfect information games (Chess, Go, Connect Four)
- Single-agent tasks
- Games where the observation already contains all relevant information

### CTDE Configurations

- **`skull_ctde.toml`** - Skull with CTDE architecture
  - Actor: 128x2 MLP (local observations only)
  - Critic: 256x3 MLP (global game state with all hidden info)
  - Demonstrates asymmetric network sizes

- **`liars_dice_ctde.toml`** - Liar's Dice with CTDE architecture
  - Actor: 256x2 MLP (public information only)
  - Critic: 512x3 MLP (all players' dice revealed)
  - Shows benefit of critic seeing hidden dice

## Configuration Parameters

### CTDE-Specific Parameters

```toml
network_type = "ctde"           # Enable CTDE architecture
global_state_dim = 200          # Must match environment's GLOBAL_STATE_DIM

# Actor network (deployment)
hidden_size = 128
num_hidden = 2

# Critic network (training only)
critic_hidden_size = 256
critic_num_hidden = 3
```

### Standard PPO Parameters

All configs support standard PPO hyperparameters:
- `learning_rate` - Learning rate (can be schedule: `[[rate, step], ...]`)
- `gamma` - Discount factor
- `gae_lambda` - GAE lambda parameter
- `clip_epsilon` - PPO clipping threshold
- `entropy_coef` - Entropy bonus coefficient
- `value_coef` - Value loss coefficient
- `max_grad_norm` - Gradient clipping threshold
- `target_kl` - KL divergence early stopping threshold

### Self-Play Parameters

- `opponent_pool_fraction` - Fraction of games against historical checkpoints
- `opponent_select_exponent` - Weighting for opponent selection (higher = prefer recent)

## Usage Examples

### Training with Standard Config
```bash
cargo run --release -- train --config configs/skull.toml
```

### Training with CTDE
```bash
cargo run --release -- train --config configs/skull_ctde.toml
```

### Evaluating a Model
```bash
cargo run --release -- eval --checkpoint runs/skull_001/checkpoints/best --num-games 100
```

## Documentation

See [docs/CTDE.md](../docs/CTDE.md) for detailed CTDE implementation guide.
See [docs/DESIGN.md](../docs/DESIGN.md) for overall architecture and design philosophy.
