# PPO Troubleshooting Guide

This guide helps diagnose and fix common training issues based on metric observations.

## Quick Reference: Healthy Metric Ranges

| Metric | Healthy Range | Warning Signs |
|--------|---------------|---------------|
| approx_kl | < 0.02 | > 0.05 = policy changing too fast |
| clip_fraction | 0.05-0.20 | > 0.30 = clipping too much |
| entropy | > 0.1 | < 0.01 = policy collapsed |
| explained_variance | > 0.5 | < 0 = value function broken |
| value_loss | Decreasing | Exploding = LR too high |

## Understanding Key Metrics

### explained_variance

Measures how well the value function predicts returns:
- **1.0** = perfect predictions
- **0.0** = no better than predicting zero
- **< 0** = worse than predicting zero (broken)

This is the most important metric for debugging the value function.

### approx_kl

Approximate KL divergence between old and new policy:
- Measures how much the policy changed during an update
- Should stay below 0.02 for stable learning
- High values indicate policy updates are too aggressive

### clip_fraction

Fraction of samples where the PPO clipping was triggered:
- Low values (< 0.05) = updates are small, might be too conservative
- High values (> 0.30) = updates hit the clip boundary frequently
- Should stabilize around 0.05-0.20 during good training

### entropy

Policy entropy (randomness):
- Higher = more exploration
- Decreasing over time is normal as policy converges
- Collapsing to near-zero too early indicates premature convergence

---

## Common Problems and Fixes

### Returns not improving / stuck at low values

**Symptoms:**
- episode/return stays flat
- No upward trend after many updates

**Likely causes:**
- entropy_coef too low (premature convergence)
- learning_rate too high (instability)
- num_steps too short (not enough temporal context)

**Try:**
- Increase entropy_coef: `0.01 → 0.05`
- Decrease learning_rate: `2.5e-4 → 1e-4`
- Increase num_steps: `128 → 256`

---

### High approx_kl (> 0.02 consistently)

**Symptoms:**
- approx_kl regularly exceeds 0.02
- Training is unstable, returns oscillate

**Likely causes:**
- Policy updating too aggressively

**Try:**
- Decrease learning_rate
- Decrease num_epochs: `4 → 2`
- Decrease clip_epsilon: `0.2 → 0.1`

---

### Entropy collapsing to 0

**Symptoms:**
- entropy drops to near-zero early in training
- Policy becomes deterministic
- Returns plateau at suboptimal level

**Likely causes:**
- entropy_coef too low
- Local optimum trap
- Environment may have dead ends

**Try:**
- Increase entropy_coef: `0.01 → 0.1`
- Reset training with different seed
- Check if environment has dead ends or sparse rewards

---

### explained_variance negative

**Symptoms:**
- explained_variance < 0
- Value function predictions are harmful

**Likely causes:**
- Network too small for task complexity
- value_coef too low
- Returns scale varies wildly

**Try:**
- Increase hidden_size: `64 → 128`
- Increase value_coef: `0.5 → 1.0`
- Consider return normalization (requires code change)

---

### clip_fraction very high (> 0.4)

**Symptoms:**
- clip_fraction > 0.4
- Many updates hitting the clip boundary

**Likely causes:**
- Updates are too large relative to clip boundary

**Try:**
- Increase clip_epsilon: `0.2 → 0.3`
- Decrease learning_rate
- Decrease num_epochs

---

### Training looks good then suddenly collapses

**Symptoms:**
- Returns improving, then suddenly drop
- Value loss or policy loss spike
- "Catastrophic forgetting"

**Likely causes:**
- Gradient explosion
- Value function instability

**Try:**
- Ensure gradient clipping is enabled: `max_grad_norm: 0.5`
- Lower learning_rate
- Save checkpoints more frequently to recover from collapse
- Consider reducing value_coef if value_loss is spiking

---

## Environment-Specific Tips

### CartPole

- Should reach ~195 average return within 50k-100k steps
- If not converging, check that observation normalization isn't applied (CartPole observations are already well-scaled)

### Connect Four

- Self-play training may take longer to show improvement
- Monitor win rate against random policy separately
- Consider curriculum: train against random opponent first, then self-play

---

## Debugging Checklist

1. **Check config is valid:** Run with `--help` to see resolved values
2. **Monitor all metrics:** Use Aim to visualize trends, not just final values
3. **Compare to baselines:** CartPole should converge quickly; if not, something is wrong
4. **Check seeds:** Different seeds can have very different outcomes
5. **Verify environment:** Run a few manual steps to ensure it behaves correctly

---

## Metric Logging Reference

All metrics logged during training:

| Metric | Description |
|--------|-------------|
| train/policy_loss | PPO clipped surrogate objective |
| train/value_loss | Value function MSE |
| train/entropy | Policy entropy (exploration) |
| train/approx_kl | KL divergence between policy updates |
| train/clip_fraction | Fraction of samples clipped |
| train/learning_rate | Current learning rate (with annealing) |
| train/explained_variance | Value function prediction quality |
| train/total_loss | Combined loss (policy + value + entropy) |
| train/value_mean | Average value predictions |
| train/returns_mean | Average target returns |
| charts/SPS | Steps per second (throughput) |
| episode/return | Average return over last 100 episodes |
| episode/return_single | Per-episode return |
| episode/length | Episode length |

## Entropy

Quick notes on entropy:
1. Entropy annealing is generally not recommended. While it can work, it often hurts learning.
2. Adaptive entropy can work well and can help to find good entropy coefficients by analyzing metric logs.
   That said, it also makes it so that extending training is often broken.
   It also can lead to spending too much time in exploration or never lowering entropy enough to exploit.
   While ok to use, push for exploring static entropy coefficients that just work.
