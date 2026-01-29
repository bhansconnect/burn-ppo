# CTDE (Centralized Training, Decentralized Execution)

## Overview

CTDE is a multi-agent reinforcement learning architecture that separates training from execution:

- **Centralized Training**: The critic (value function) network has access to privileged global state information during training
- **Decentralized Execution**: The actor (policy) network only uses local observations, enabling deployment without access to global state

This architecture is particularly effective for:
- Multi-agent games with hidden information (poker, bluffing games, Skull, Liar's Dice)
- Scenarios where value estimation benefits from privileged information
- Competitive games requiring decentralized execution

## Architecture

### Traditional Actor-Critic (Single Network)
```
Observation → Shared Network → Policy Head (logits)
                             → Value Head (values)
```

### CTDE (Separate Networks)
```
Local Observation → Actor Network → Policy Head (logits)
Global State      → Critic Network → Value Head (values)
```

### Key Differences

| Component | Input | Size | Used During |
|-----------|-------|------|-------------|
| **Actor** | Local observations (player perspective) | Small, fast | Training + Deployment |
| **Critic** | Global state (privileged information) | Large, slow | Training only |

## When to Use CTDE

### ✅ Good Use Cases

1. **Hidden Information Games**
   - Poker, Liar's Dice, Skull
   - The critic can see all players' private information
   - The actor only sees public info + own private cards

2. **Partial Observability**
   - Games where players have limited views
   - Value estimation benefits from global state
   - Policy must work with local observations

3. **Multi-Agent Competitive Scenarios**
   - Self-play training
   - Opponent modeling
   - Nash equilibrium approximation

### ❌ Not Recommended For

1. **Single-Player Games**
   - CartPole, Atari, etc.
   - No benefit from separate networks

2. **Perfect Information Games**
   - Connect Four, Chess, Go
   - Local observation already contains all information
   - Extra complexity without benefit

3. **Simple Environments**
   - Small observation/state spaces
   - Computational overhead not worth it

## Configuration

### Required Config Fields

```toml
# Network architecture
network_type = "ctde"
global_state_dim = 200  # Must match Environment::GLOBAL_STATE_DIM

# Actor network (small, fast for deployment)
hidden_size = 128
num_hidden = 2
activation = "relu"

# Critic network (larger, training only)
critic_hidden_size = 256
critic_num_hidden = 3
```

### Global State Dimension

**CRITICAL**: `global_state_dim` in config **must** match `Environment::GLOBAL_STATE_DIM` in your environment implementation.

Example:
```rust
// In src/envs/skull.rs
impl Environment for Skull {
    const OBSERVATION_DIM: usize = 135;  // Local observation
    const GLOBAL_STATE_DIM: Option<usize> = Some(200);  // Global state
    // ...
}
```

```toml
# In configs/skull_ctde.toml
global_state_dim = 200  # Must match!
```

### Network Size Guidelines

**Actor Network** (deployed with agents):
- Keep small for fast inference
- Typical: 128-256 hidden units, 2-3 layers
- Must fit in memory for real-time play

**Critic Network** (training only):
- Can be larger since not deployed
- Typical: 256-512 hidden units, 3-4 layers
- Benefits from extra capacity for value estimation

## Environment Implementation

To support CTDE, your environment must implement:

### 1. Global State Dimension

```rust
impl Environment for MyGame {
    const GLOBAL_STATE_DIM: Option<usize> = Some(GLOBAL_DIM);
    // ...
}
```

### 2. Global State Method

```rust
fn global_state(&self) -> Vec<f32> {
    let mut state = Vec::with_capacity(GLOBAL_DIM);

    // Shared game state (one copy)
    state.extend_from_slice(&self.board_state);
    state.extend_from_slice(&self.public_info);

    // Per-player private information (ALL players, absolute indexing)
    for player in &self.players {
        state.extend_from_slice(&player.hand);
        state.extend_from_slice(&player.private_state);
    }

    // Pad to exact dimension if needed
    state.resize(GLOBAL_DIM, 0.0);

    state
}
```

### Key Differences: Local vs Global

**Local Observations** (`observation()`):
- **Player-relative**: Current player is always at index 0
- **Limited information**: Only what that player can see
- **Used by**: Actor network during training and deployment

**Global State** (`global_state()`):
- **Absolute indexing**: Player 0 is always at index 0
- **Privileged information**: All hidden information included
- **Used by**: Critic network during training only

Example for a 4-player game:
```rust
// Local observation for Player 2
fn observation(&self, player: usize) -> Vec<f32> {
    let mut obs = vec![];
    // Rotate so current player is first
    obs.extend(&self.players[player].hand);      // Me (index 0)
    obs.extend(&self.players[(player+1)%4].public);  // Next (index 1)
    obs.extend(&self.players[(player+2)%4].public);  // Across (index 2)
    obs.extend(&self.players[(player+3)%4].public);  // Previous (index 3)
    obs
}

// Global state (absolute)
fn global_state(&self) -> Vec<f32> {
    let mut state = vec![];
    // No rotation - absolute positions
    for player in &self.players {
        state.extend(&player.hand);    // ALL hands (hidden info)
        state.extend(&player.public);
    }
    state
}
```

## Training vs Evaluation

### Training
During training, both networks are used:
```rust
// Rollout collection
let logits = model.forward_actor(local_obs);     // Actor for actions
let values = model.forward_critic(global_state); // Critic for value estimates

// Loss computation
let (new_logits, new_values) = if model.is_ctde() {
    let logits = model.forward_actor(mb_obs);
    let values = model.forward_critic(mb_global_states);
    (logits, values)
} else {
    model.forward(mb_obs)
};
```

### Evaluation
During evaluation, only the actor is needed:
```rust
// Eval mode - no global state needed
let logits = if model.is_ctde() {
    model.forward_actor(obs_tensor)  // Only actor for action selection
} else {
    model.forward(obs_tensor).0
};
```

## Checkpointing

CTDE checkpoints include both networks:
- Actor network (deployed with agent)
- Critic network (only needed for resume training)

Metadata includes:
```json
{
  "network_type": "ctde",
  "global_state_dim": 200,
  "hidden_size": 128,
  "critic_hidden_size": 256,
  ...
}
```

## Common Issues

### Issue: Training crashes with dimension mismatch

**Cause**: `global_state_dim` in config doesn't match `Environment::GLOBAL_STATE_DIM`

**Solution**: Verify dimensions match exactly:
```bash
# Check environment
grep "GLOBAL_STATE_DIM" src/envs/skull.rs
# Check config
grep "global_state_dim" configs/skull_ctde.toml
```

### Issue: Panic "forward() called on CTDE network"

**Cause**: Code calling `.forward(obs)` on CTDE network

**Solution**: Check if CTDE and use separate methods:
```rust
// ❌ Wrong
let (logits, values) = model.forward(obs);

// ✅ Correct
let (logits, values) = if model.is_ctde() {
    let logits = model.forward_actor(local_obs);
    let values = model.forward_critic(global_state);
    (logits, values)
} else {
    model.forward(obs)
};
```

### Issue: Performance is worse than standard network

**Possible causes**:
1. Global state doesn't provide useful additional information
2. Critic network too small (increase `critic_hidden_size`)
3. Actor network too large (decrease `hidden_size` for faster learning)
4. Environment doesn't have enough hidden information to benefit from CTDE

## Examples

### Skull (4-player bluffing game)

```toml
# configs/skull_ctde.toml
env = "skull"
network_type = "ctde"
global_state_dim = 200  # Matches Skull::GLOBAL_STATE_DIM

# Actor: 135 obs → 128x128x2 → 20 actions
hidden_size = 128
num_hidden = 2

# Critic: 200 global → 256x256x256x3 → 4 values
critic_hidden_size = 256
critic_num_hidden = 3

[player_count]
type = "Fixed"
count = 4
```

**Local observation (135 dims)**:
- Own cards + position
- Public bids/challenges
- Game state from player perspective

**Global state (200 dims)**:
- ALL players' hidden cards
- Full bid history
- All private information

### Liar's Dice (2-player)

```toml
# configs/liars_dice_ctde.toml
env = "liars_dice"
network_type = "ctde"
global_state_dim = 120  # Matches LiarsDice::GLOBAL_STATE_DIM

hidden_size = 128
num_hidden = 2
critic_hidden_size = 256
critic_num_hidden = 2
```

**Local observation**:
- Own dice
- Bid history
- Public information

**Global state**:
- Both players' dice (hidden)
- Full game state

## Performance Considerations

### Training Speed
- **Slower** than standard networks (2x forward passes per step)
- Offset by better sample efficiency in hidden information games

### Inference Speed
- **Same** as standard network (only actor used)
- Actor can be smaller than combined network

### Memory Usage
- **Higher** during training (two networks)
- **Same** during deployment (only actor deployed)

## References

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) (MADDPG paper)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926) (COMA paper)
- Related configs: `configs/skull_ctde.toml`, `configs/liars_dice_ctde.toml`
