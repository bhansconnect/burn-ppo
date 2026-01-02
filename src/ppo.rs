/// PPO Algorithm Implementation
///
/// Implements Proximal Policy Optimization with all 13 details from the ICLR blog:
/// - Vectorized architecture, orthogonal init, Adam epsilon 1e-5
/// - Learning rate annealing, GAE, minibatch shuffled updates
/// - Advantage normalization, clipped surrogate, value clipping
/// - Combined loss, gradient clipping

use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::Int;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::config::Config;
use crate::env::{Environment, EpisodeStats, VecEnv};
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;
use crate::profile::{gpu_sync, profile_function, profile_scope};
use crate::utils::{entropy_categorical, log_prob_categorical, normalize_advantages, sample_categorical};

/// Stores trajectory data from environment rollouts
///
/// Shape: [num_steps, num_envs] for most fields
/// For multi-player games, also stores per-player values and rewards.
#[derive(Debug)]
pub struct RolloutBuffer<B: Backend> {
    /// Observations [num_steps, num_envs, obs_dim]
    pub observations: Tensor<B, 3>,
    /// Actions taken [num_steps, num_envs]
    pub actions: Tensor<B, 2, Int>,
    /// Rewards for acting player [num_steps, num_envs]
    pub rewards: Tensor<B, 2>,
    /// Episode done flags [num_steps, num_envs]
    pub dones: Tensor<B, 2>,
    /// Value estimates for acting player [num_steps, num_envs]
    pub values: Tensor<B, 2>,
    /// Log probabilities of actions [num_steps, num_envs]
    pub log_probs: Tensor<B, 2>,

    // Multi-player support fields
    /// Values for ALL players [num_steps, num_envs, num_players]
    pub all_values: Tensor<B, 3>,
    /// Rewards for ALL players [num_steps, num_envs, num_players]
    pub all_rewards: Tensor<B, 3>,
    /// Which player acted each step [num_steps, num_envs]
    pub acting_players: Tensor<B, 2, Int>,
    /// Number of players (max 255)
    num_players: u8,

    /// GAE advantages (computed after rollout) [num_steps, num_envs]
    pub advantages: Option<Tensor<B, 2>>,
    /// Returns (values + advantages) [num_steps, num_envs]
    pub returns: Option<Tensor<B, 2>>,
}

impl<B: Backend> RolloutBuffer<B> {
    /// Create empty buffer with given dimensions
    ///
    /// `num_players` must be <= 255 (stored as u8)
    pub fn new(
        num_steps: usize,
        num_envs: usize,
        obs_dim: usize,
        num_players: u8,
        device: &B::Device,
    ) -> Self {
        let np = num_players as usize;
        Self {
            observations: Tensor::zeros([num_steps, num_envs, obs_dim], device),
            actions: Tensor::zeros([num_steps, num_envs], device),
            rewards: Tensor::zeros([num_steps, num_envs], device),
            dones: Tensor::zeros([num_steps, num_envs], device),
            values: Tensor::zeros([num_steps, num_envs], device),
            log_probs: Tensor::zeros([num_steps, num_envs], device),
            // Multi-player fields
            all_values: Tensor::zeros([num_steps, num_envs, np], device),
            all_rewards: Tensor::zeros([num_steps, num_envs, np], device),
            acting_players: Tensor::zeros([num_steps, num_envs], device),
            num_players,
            advantages: None,
            returns: None,
        }
    }

    /// Get number of players (max 255)
    pub fn num_players(&self) -> u8 {
        self.num_players
    }

    /// Flatten buffer for minibatch processing
    /// Returns (obs, actions, old_log_probs, advantages, returns, acting_players)
    /// Each with shape [num_steps * num_envs, ...]
    pub fn flatten(&self) -> (Tensor<B, 2>, Tensor<B, 1, Int>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1, Int>) {
        let [num_steps, num_envs, obs_dim] = self.observations.dims();
        let batch_size = num_steps * num_envs;

        let obs = self.observations.clone().reshape([batch_size, obs_dim]);
        let actions = self.actions.clone().flatten(0, 1);
        let log_probs = self.log_probs.clone().flatten(0, 1);
        let advantages = self.advantages.as_ref().expect("Advantages not computed").clone().flatten(0, 1);
        let returns = self.returns.as_ref().expect("Returns not computed").clone().flatten(0, 1);
        let acting_players = self.acting_players.clone().flatten(0, 1);

        (obs, actions, log_probs, advantages, returns, acting_players)
    }
}

/// Collect rollouts from vectorized environment
///
/// Runs num_steps in each of num_envs environments, storing trajectories.
/// Uses CPU collection with batch GPU transfer for performance.
/// If normalizer is provided, observations are normalized before model inference
/// using existing (lagged) statistics, then stats are updated at end of rollout.
pub fn collect_rollouts<B: Backend, E: Environment>(
    model: &ActorCritic<B>,
    vec_env: &mut VecEnv<E>,
    buffer: &mut RolloutBuffer<B>,
    num_steps: usize,
    device: &B::Device,
    rng: &mut impl Rng,
    mut normalizer: Option<&mut ObsNormalizer>,
) -> Vec<EpisodeStats> {
    profile_function!();
    let num_envs = vec_env.num_envs();
    let obs_dim = E::OBSERVATION_DIM;
    let num_players = E::NUM_PLAYERS;
    let mut all_completed = Vec::new();

    // Pre-allocate CPU vectors for batch collection
    let mut all_obs: Vec<f32> = Vec::with_capacity(num_steps * num_envs * obs_dim);
    let mut all_actions: Vec<i64> = Vec::with_capacity(num_steps * num_envs);
    let mut all_acting_rewards: Vec<f32> = Vec::with_capacity(num_steps * num_envs);
    let mut all_dones: Vec<f32> = Vec::with_capacity(num_steps * num_envs);
    let mut all_acting_values: Vec<f32> = Vec::with_capacity(num_steps * num_envs);
    let mut all_log_probs: Vec<f32> = Vec::with_capacity(num_steps * num_envs);

    // Multi-player data
    let mut all_values_flat: Vec<f32> = Vec::with_capacity(num_steps * num_envs * num_players);
    let mut all_rewards_flat: Vec<f32> = Vec::with_capacity(num_steps * num_envs * num_players);
    let mut all_acting_players: Vec<i64> = Vec::with_capacity(num_steps * num_envs);

    // Collect raw observations for normalizer stats update at end of rollout
    // This implements "lagged" normalization: normalize with existing stats,
    // then update stats for the NEXT rollout
    let collect_raw_obs = normalizer.is_some();
    let mut raw_obs_for_stats: Vec<f32> = if collect_raw_obs {
        Vec::with_capacity(num_steps * num_envs * obs_dim)
    } else {
        Vec::new()
    };

    for _step in 0..num_steps {
        profile_scope!("rollout_step");

        // Get current players BEFORE the step (who is about to act)
        let current_players = vec_env.get_current_players();

        // Get current observations
        let mut obs_flat = vec_env.get_observations();

        // Store raw observations BEFORE normalization for stats update
        if collect_raw_obs {
            raw_obs_for_stats.extend_from_slice(&obs_flat);
        }

        // Normalize using EXISTING (lagged) stats - don't update stats yet
        if let Some(ref norm) = normalizer {
            norm.normalize_batch(&mut obs_flat, obs_dim);
        }

        // Model inference: forward pass, sample actions, compute log probs, sync to CPU
        // All GPU ops batched together - timing includes actual compute (sync at end)
        let (actions_data, acting_values_data, log_probs_data, values_all_data) = {
            profile_scope!("model_inference");

            let obs_tensor: Tensor<B, 2> =
                Tensor::<B, 1>::from_floats(obs_flat.as_slice(), device).reshape([num_envs, obs_dim]);
            let (logits, values) = model.forward(obs_tensor);
            // values is [num_envs, num_players]

            let actions = sample_categorical(logits.clone(), rng, device);
            let log_probs = log_prob_categorical(logits, actions.clone());

            // Sync to CPU - this is where actual GPU compute happens
            let actions_data: Vec<i64> = actions
                .float()
                .into_data()
                .to_vec::<f32>()
                .expect("actions to vec")
                .into_iter()
                .map(|x| x as i64)
                .collect();

            // Get all player values [num_envs * num_players]
            let values_all_data: Vec<f32> = values.into_data().to_vec().expect("values to vec");

            // Extract acting player's value for each env
            let acting_values_data: Vec<f32> = current_players
                .iter()
                .enumerate()
                .map(|(e, &p)| values_all_data[e * num_players + p])
                .collect();

            let log_probs_data: Vec<f32> = log_probs.into_data().to_vec().expect("log_probs to vec");
            (actions_data, acting_values_data, log_probs_data, values_all_data)
        };

        // Convert actions to Vec<usize> for environment
        let actions_usize: Vec<usize> = actions_data.iter().map(|&a| a as usize).collect();

        // Step environment (CPU-bound)
        let (player_rewards, dones, completed) = {
            profile_scope!("env_step_cpu");
            let (_next_obs, player_rewards, dones, completed) = vec_env.step(&actions_usize);
            (player_rewards, dones, completed)
        };
        all_completed.extend(completed);

        // Extract acting player's reward for backward-compat single-player path
        let acting_rewards: Vec<f32> = player_rewards
            .iter()
            .zip(current_players.iter())
            .map(|(r, &p)| r.get(p).copied().unwrap_or(0.0))
            .collect();

        // Flatten all player rewards [num_envs, num_players] -> [num_envs * num_players]
        let rewards_flat: Vec<f32> = player_rewards.iter().flat_map(|r| {
            // Pad with zeros if rewards vec is shorter than num_players
            (0..num_players).map(|p| r.get(p).copied().unwrap_or(0.0))
        }).collect();

        // Append to CPU buffers
        all_obs.extend_from_slice(&obs_flat);
        all_actions.extend_from_slice(&actions_data);
        all_acting_rewards.extend_from_slice(&acting_rewards);
        all_dones.extend(dones.iter().map(|&d| if d { 1.0 } else { 0.0 }));
        all_acting_values.extend_from_slice(&acting_values_data);
        all_log_probs.extend_from_slice(&log_probs_data);

        // Multi-player data
        all_values_flat.extend_from_slice(&values_all_data);
        all_rewards_flat.extend_from_slice(&rewards_flat);
        all_acting_players.extend(current_players.iter().map(|&p| p as i64));
    }

    // Batch transfer to GPU
    {
        profile_scope!("batch_gpu_transfer");
        buffer.observations =
            Tensor::<B, 1>::from_floats(all_obs.as_slice(), device).reshape([num_steps, num_envs, obs_dim]);
        buffer.actions =
            Tensor::<B, 1, Int>::from_ints(all_actions.as_slice(), device).reshape([num_steps, num_envs]);
        buffer.rewards =
            Tensor::<B, 1>::from_floats(all_acting_rewards.as_slice(), device).reshape([num_steps, num_envs]);
        buffer.dones =
            Tensor::<B, 1>::from_floats(all_dones.as_slice(), device).reshape([num_steps, num_envs]);
        buffer.values =
            Tensor::<B, 1>::from_floats(all_acting_values.as_slice(), device).reshape([num_steps, num_envs]);
        buffer.log_probs =
            Tensor::<B, 1>::from_floats(all_log_probs.as_slice(), device).reshape([num_steps, num_envs]);

        // Multi-player data
        buffer.all_values =
            Tensor::<B, 1>::from_floats(all_values_flat.as_slice(), device).reshape([num_steps, num_envs, num_players]);
        buffer.all_rewards =
            Tensor::<B, 1>::from_floats(all_rewards_flat.as_slice(), device).reshape([num_steps, num_envs, num_players]);
        buffer.acting_players =
            Tensor::<B, 1, Int>::from_ints(all_acting_players.as_slice(), device).reshape([num_steps, num_envs]);
    }

    // Update normalizer stats at end of rollout with all RAW observations
    // This ensures stats are updated for the NEXT rollout, not the current one
    if let Some(norm) = normalizer.as_mut() {
        norm.update_batch(&raw_obs_for_stats, obs_dim);
    }

    all_completed
}

/// Compute Generalized Advantage Estimation
///
/// GAE(gamma, lambda) = sum_{t'>=t} (gamma*lambda)^{t'-t} * delta_{t'}
/// where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
pub fn compute_gae<B: Backend>(
    buffer: &mut RolloutBuffer<B>,
    last_values: Tensor<B, 1>,
    gamma: f32,
    gae_lambda: f32,
    device: &B::Device,
) {
    profile_function!();
    let [num_steps, num_envs] = buffer.rewards.dims();

    // Get tensor data for computation
    let rewards_data: Vec<f32> = buffer.rewards.clone().into_data().to_vec().expect("rewards");
    let dones_data: Vec<f32> = buffer.dones.clone().into_data().to_vec().expect("dones");
    let values_data: Vec<f32> = buffer.values.clone().into_data().to_vec().expect("values");
    let last_values_data: Vec<f32> = last_values.into_data().to_vec().expect("last values");

    let mut advantages = vec![0.0_f32; num_steps * num_envs];
    let mut last_gae = vec![0.0_f32; num_envs];

    // Backward pass through time
    for t in (0..num_steps).rev() {
        for e in 0..num_envs {
            let idx = t * num_envs + e;
            let reward = rewards_data[idx];
            let done = dones_data[idx];
            let value = values_data[idx];

            // Next value: either from next step or last_values (bootstrap)
            let next_value = if t == num_steps - 1 {
                last_values_data[e]
            } else {
                values_data[(t + 1) * num_envs + e]
            };

            // TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
            let delta = reward + gamma * next_value * (1.0 - done) - value;

            // GAE: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
            last_gae[e] = delta + gamma * gae_lambda * (1.0 - done) * last_gae[e];
            advantages[idx] = last_gae[e];
        }
    }

    // Store advantages and compute returns
    let advantages_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(advantages.as_slice(), device).reshape([num_steps, num_envs]);
    let returns_tensor = advantages_tensor.clone() + buffer.values.clone();

    buffer.advantages = Some(advantages_tensor);
    buffer.returns = Some(returns_tensor);
}

/// Compute Generalized Advantage Estimation for multi-player games
///
/// Two-pass algorithm:
/// 1. Attribute cumulative rewards: rewards between turns are credited to last action
/// 2. Compute GAE using per-player value chains
///
/// This handles games where players take turns and rewards from other players'
/// actions should be attributed to the acting player's previous action.
pub fn compute_gae_multiplayer<B: Backend>(
    buffer: &mut RolloutBuffer<B>,
    last_values: Tensor<B, 2>, // [num_envs, num_players]
    gamma: f32,
    gae_lambda: f32,
    num_players: u8,
    device: &B::Device,
) {
    let num_players = num_players as usize;
    profile_function!();
    let [num_steps, num_envs, _] = buffer.all_rewards.dims();

    // Extract data from tensors
    let all_rewards_data: Vec<f32> = buffer.all_rewards.clone().into_data().to_vec().expect("all_rewards");
    let all_values_data: Vec<f32> = buffer.all_values.clone().into_data().to_vec().expect("all_values");
    let dones_data: Vec<f32> = buffer.dones.clone().into_data().to_vec().expect("dones");
    // Convert through float to avoid IntElem type mismatches between backends
    let acting_players_data: Vec<usize> = buffer
        .acting_players
        .clone()
        .float()
        .into_data()
        .to_vec::<f32>()
        .expect("acting_players")
        .into_iter()
        .map(|x| x as usize)
        .collect();
    let last_values_data: Vec<f32> = last_values.into_data().to_vec().expect("last_values");

    // Pass 1: Attribute cumulative rewards to acting player
    // Walk backwards, accumulating rewards for each player until they act
    let mut attributed_rewards = vec![0.0_f32; num_steps * num_envs];
    let mut reward_carry = vec![vec![0.0_f32; num_players]; num_envs];

    for t in (0..num_steps).rev() {
        for e in 0..num_envs {
            let idx = t * num_envs + e;
            let acting_player = acting_players_data[idx];
            let done = dones_data[idx];

            // This player's immediate reward + any accumulated from other turns
            let r_idx = (t * num_envs + e) * num_players + acting_player;
            attributed_rewards[idx] = all_rewards_data[r_idx] + reward_carry[e][acting_player];
            reward_carry[e][acting_player] = 0.0;

            // Accumulate other players' rewards (they'll get credited when they next act)
            for p in 0..num_players {
                if p != acting_player {
                    reward_carry[e][p] += all_rewards_data[(t * num_envs + e) * num_players + p];
                }
            }

            // Reset on episode boundary
            if done > 0.5 {
                for p in 0..num_players {
                    reward_carry[e][p] = 0.0;
                }
            }
        }
    }

    // Pass 2: Compute GAE using per-player value chains
    let mut advantages = vec![0.0_f32; num_steps * num_envs];
    let mut gae_carry = vec![vec![0.0_f32; num_players]; num_envs];

    // Initialize next_value from bootstrap values
    let mut next_value: Vec<Vec<f32>> = (0..num_envs)
        .map(|e| {
            (0..num_players)
                .map(|p| last_values_data[e * num_players + p])
                .collect()
        })
        .collect();

    for t in (0..num_steps).rev() {
        for e in 0..num_envs {
            let idx = t * num_envs + e;
            let player = acting_players_data[idx];
            let reward = attributed_rewards[idx];
            let value = all_values_data[(t * num_envs + e) * num_players + player];
            let done = dones_data[idx];

            // TD error using this player's next value
            let delta = reward + gamma * next_value[e][player] * (1.0 - done) - value;

            // GAE for this player
            let advantage = delta + gamma * gae_lambda * (1.0 - done) * gae_carry[e][player];
            advantages[idx] = advantage;

            // Update carry for this player
            gae_carry[e][player] = advantage;
            next_value[e][player] = value;

            // Reset all players' GAE carry on episode boundary
            if done > 0.5 {
                for p in 0..num_players {
                    gae_carry[e][p] = 0.0;
                    next_value[e][p] = 0.0;
                }
            }
        }
    }

    // Store advantages
    let advantages_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(advantages.as_slice(), device).reshape([num_steps, num_envs]);

    // Compute returns = advantages + acting player's value
    let values_flat: Vec<f32> = (0..num_steps * num_envs)
        .map(|i| {
            let t = i / num_envs;
            let e = i % num_envs;
            let p = acting_players_data[t * num_envs + e];
            all_values_data[(t * num_envs + e) * num_players + p]
        })
        .collect();
    let values_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(values_flat.as_slice(), device).reshape([num_steps, num_envs]);

    buffer.advantages = Some(advantages_tensor.clone());
    buffer.returns = Some(advantages_tensor + values_tensor);
}

/// Compute explained variance: 1 - Var(returns - values) / Var(returns)
/// Values close to 1 indicate good value function predictions
pub fn compute_explained_variance(values: &[f32], returns: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 2.0 {
        return 0.0;
    }

    let mean_returns = returns.iter().sum::<f32>() / n;
    let var_returns = returns.iter().map(|r| (r - mean_returns).powi(2)).sum::<f32>() / n;

    if var_returns < 1e-8 {
        return 0.0;
    }

    let residuals: Vec<f32> = values.iter().zip(returns).map(|(v, r)| r - v).collect();
    let mean_residuals = residuals.iter().sum::<f32>() / n;
    let var_residuals = residuals
        .iter()
        .map(|r| (r - mean_residuals).powi(2))
        .sum::<f32>()
        / n;

    1.0 - var_residuals / var_returns
}

/// Training metrics from a single update
#[derive(Debug, Clone)]
pub struct UpdateMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    pub clip_fraction: f32,
    // Additional metrics for debugging
    pub explained_variance: f32,
    pub total_loss: f32,
    pub value_mean: f32,
    pub returns_mean: f32,
}

/// Perform PPO update on collected rollouts
///
/// Implements clipped surrogate objective with value clipping
pub fn ppo_update<B: burn::tensor::backend::AutodiffBackend>(
    model: ActorCritic<B>,
    buffer: &RolloutBuffer<B>,
    optimizer: &mut impl burn::optim::Optimizer<ActorCritic<B>, B>,
    config: &Config,
    learning_rate: f64,
    entropy_coef: f64,
    rng: &mut impl Rng,
) -> (ActorCritic<B>, UpdateMetrics) {
    profile_function!();
    let device = model.devices()[0].clone();
    let num_players = buffer.num_players() as usize;
    let (obs, actions, old_log_probs, advantages, returns, acting_players) = buffer.flatten();
    let batch_size = obs.dims()[0];
    let minibatch_size = batch_size / config.num_minibatches;

    // Accumulators for metrics
    let mut total_policy_loss = 0.0;
    let mut total_value_loss = 0.0;
    let mut total_entropy = 0.0;
    let mut total_approx_kl = 0.0;
    let mut total_clip_fraction = 0.0;
    let mut total_loss_sum = 0.0;
    let mut total_value_mean = 0.0;
    let mut total_returns_mean = 0.0;
    let mut num_updates = 0;

    let mut model = model;

    // Epoch loop (labeled for KL early stopping)
    'epoch_loop: for _epoch in 0..config.num_epochs {
        profile_scope!("ppo_epoch");

        // Shuffle indices for minibatches
        let mut indices: Vec<usize> = (0..batch_size).collect();
        indices.shuffle(rng);

        // Minibatch loop
        for mb_start in (0..batch_size).step_by(minibatch_size) {
            profile_scope!("ppo_minibatch");
            let mb_end = (mb_start + minibatch_size).min(batch_size);
            let mb_indices: Vec<i64> = indices[mb_start..mb_end].iter().map(|&i| i as i64).collect();
            let mb_indices_tensor: Tensor<B, 1, Int> = Tensor::from_ints(mb_indices.as_slice(), &device);

            // Gather minibatch data
            let mb_obs = obs.clone().select(0, mb_indices_tensor.clone());
            let mb_actions: Tensor<B, 1, Int> = actions.clone().select(0, mb_indices_tensor.clone());
            let mb_old_log_probs = old_log_probs.clone().select(0, mb_indices_tensor.clone());
            let mb_advantages_raw = advantages.clone().select(0, mb_indices_tensor.clone());
            let mb_returns = returns.clone().select(0, mb_indices_tensor.clone());
            let mb_acting_players: Tensor<B, 1, Int> = acting_players.clone().select(0, mb_indices_tensor);

            // Normalize advantages at minibatch level (critical for stability)
            let mb_advantages = normalize_advantages(mb_advantages_raw);

            // Forward pass with GPU sync for accurate timing
            let (logits, all_values) = {
                profile_scope!("minibatch_forward");
                let result = model.forward(mb_obs);
                gpu_sync!(result.0); // Force sync to get accurate forward timing
                result
            };

            // Extract acting player's value for each sample in minibatch
            let mb_size = logits.dims()[0];
            let values: Tensor<B, 1> = if num_players == 1 {
                // Single-player: just extract player 0
                all_values.slice([0..mb_size, 0..1]).flatten(0, 1)
            } else {
                // Multi-player: extract acting player's value for each sample
                // This is done on CPU since gather operations on GPU can be slow for small tensors
                let all_values_data: Vec<f32> = all_values.clone().into_data().to_vec().expect("values");
                // Convert through float to avoid IntElem type mismatches between backends
                let acting_data: Vec<usize> = mb_acting_players
                    .clone()
                    .float()
                    .into_data()
                    .to_vec::<f32>()
                    .expect("acting")
                    .into_iter()
                    .map(|x| x as usize)
                    .collect();
                let values_vec: Vec<f32> = (0..mb_size)
                    .map(|i| {
                        let player = acting_data[i];
                        all_values_data[i * num_players + player]
                    })
                    .collect();
                Tensor::<B, 1>::from_floats(values_vec.as_slice(), &device)
            };

            // Compute losses and backward pass (combined for accurate timing)
            let (grads, loss, policy_loss_mean, value_loss, entropy, approx_kl, clip_fraction, values_mean, returns_mean) = {
                profile_scope!("loss_and_backward");

                let new_log_probs = log_prob_categorical(logits.clone(), mb_actions);
                let entropy = entropy_categorical(logits);

                // Policy loss (clipped surrogate objective)
                let log_ratio = new_log_probs.clone() - mb_old_log_probs;
                let ratio = log_ratio.clone().exp();

                let policy_loss_1 = -mb_advantages.clone() * ratio.clone();
                let policy_loss_2 = -mb_advantages.clone()
                    * ratio.clone().clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon);
                let policy_loss: Tensor<B, 1> = policy_loss_1.clone().max_pair(policy_loss_2);
                let policy_loss_mean = policy_loss.mean();

                // Value loss (optionally clipped)
                let value_loss = if config.clip_value {
                    let mb_old_values = buffer.values.clone().flatten(0, 1).select(
                        0,
                        Tensor::from_ints(
                            indices[mb_start..mb_end].iter().map(|&i| i as i64).collect::<Vec<_>>().as_slice(),
                            &device,
                        ),
                    );
                    let values_clipped = mb_old_values.clone()
                        + (values.clone() - mb_old_values.clone())
                            .clamp(-config.clip_epsilon, config.clip_epsilon);
                    let value_loss_1 = (values.clone() - mb_returns.clone()).powf_scalar(2.0);
                    let value_loss_2 = (values_clipped - mb_returns.clone()).powf_scalar(2.0);
                    value_loss_1.max_pair(value_loss_2).mean() * 0.5
                } else {
                    (values.clone() - mb_returns.clone()).powf_scalar(2.0).mean() * 0.5
                };

                // Entropy bonus
                let entropy_loss = -entropy.clone().mean() * entropy_coef;

                // Combined loss
                let loss = policy_loss_mean.clone() + value_loss.clone() * config.value_coef + entropy_loss;

                // Compute approximate KL divergence for logging
                // KL ≈ (ratio - 1) - log(ratio) is the unbiased low-variance estimator
                // from "Approximating KL Divergence" (Schulman)
                let approx_kl = ((ratio.clone() - 1.0) - log_ratio.clone()).mean();

                // Compute clip fraction for logging
                let clip_fraction = (ratio.clone() - 1.0)
                    .abs()
                    .greater_elem(config.clip_epsilon)
                    .float()
                    .mean();

                let values_mean = values.mean();
                let returns_mean = mb_returns.mean();

                // Backward pass
                let grads = loss.backward();

                (grads, loss, policy_loss_mean, value_loss, entropy, approx_kl, clip_fraction, values_mean, returns_mean)
            };

            // Optimizer step with GPU sync for accurate timing
            model = {
                profile_scope!("optimizer_step");
                let grads = GradientsParams::from_grads(grads, &model);
                let updated = optimizer.step(learning_rate.into(), model, grads);
                gpu_sync!(updated.layers[0].weight.val()); // Force sync after weight update
                updated
            };

            // Batched metric extraction - single GPU sync instead of 8
            {
                profile_scope!("extract_metrics");

                // Concatenate all scalar metrics into one tensor for single GPU->CPU transfer
                // Use reshape to convert 0-dim scalars to 1-dim, then cat along dim 0
                let metrics_tensor: Tensor<B, 1> = Tensor::cat(
                    vec![
                        (-policy_loss_mean).reshape([1]),  // Negate for display
                        value_loss.reshape([1]),
                        entropy.mean().reshape([1]),
                        approx_kl.reshape([1]),
                        clip_fraction.reshape([1]),
                        loss.reshape([1]),
                        values_mean.reshape([1]),
                        returns_mean.reshape([1]),
                    ],
                    0,
                );

                let metrics_data: Vec<f32> = metrics_tensor.into_data().to_vec().expect("metrics");

                total_policy_loss += metrics_data[0];
                total_value_loss += metrics_data[1];
                total_entropy += metrics_data[2];
                total_approx_kl += metrics_data[3];
                total_clip_fraction += metrics_data[4];
                total_loss_sum += metrics_data[5];
                total_value_mean += metrics_data[6];
                total_returns_mean += metrics_data[7];
                num_updates += 1;

                // KL early stopping: stop epoch if KL divergence exceeds threshold
                if let Some(target) = config.target_kl {
                    if metrics_data[3] > target as f32 {
                        break 'epoch_loop;
                    }
                }
            }
        }
    }

    // Compute explained variance from full buffer
    let values_data: Vec<f32> = buffer
        .values
        .clone()
        .into_data()
        .to_vec()
        .expect("values to vec");
    let returns_data: Vec<f32> = buffer
        .returns
        .as_ref()
        .expect("returns computed")
        .clone()
        .into_data()
        .to_vec()
        .expect("returns to vec");
    let explained_variance = compute_explained_variance(&values_data, &returns_data);

    // Average metrics
    let metrics = UpdateMetrics {
        policy_loss: total_policy_loss / num_updates as f32,
        value_loss: total_value_loss / num_updates as f32,
        entropy: total_entropy / num_updates as f32,
        approx_kl: total_approx_kl / num_updates as f32,
        clip_fraction: total_clip_fraction / num_updates as f32,
        explained_variance,
        total_loss: total_loss_sum / num_updates as f32,
        value_mean: total_value_mean / num_updates as f32,
        returns_mean: total_returns_mean / num_updates as f32,
    };

    (model, metrics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rollout_buffer_creation() {
        let device = Default::default();
        let buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(128, 4, 4, 1u8, &device);

        assert_eq!(buffer.observations.dims(), [128, 4, 4]);
        assert_eq!(buffer.actions.dims(), [128, 4]);
        assert_eq!(buffer.rewards.dims(), [128, 4]);
        assert_eq!(buffer.all_values.dims(), [128, 4, 1]);
        assert_eq!(buffer.all_rewards.dims(), [128, 4, 1]);
        assert_eq!(buffer.acting_players.dims(), [128, 4]);
    }

    #[test]
    fn test_rollout_buffer_multiplayer() {
        let device = Default::default();
        let buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(64, 8, 86, 2u8, &device);

        assert_eq!(buffer.observations.dims(), [64, 8, 86]);
        assert_eq!(buffer.all_values.dims(), [64, 8, 2]);
        assert_eq!(buffer.all_rewards.dims(), [64, 8, 2]);
        assert_eq!(buffer.num_players(), 2u8);
    }

    #[test]
    fn test_gae_computation() {
        let device = Default::default();
        let num_steps = 4;
        let num_envs = 2;
        let num_players = 1u8;

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(num_steps, num_envs, 1, num_players, &device);

        // Set up simple rewards and values for testing
        buffer.rewards = Tensor::from_floats(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            &device,
        );
        buffer.values = Tensor::from_floats(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            &device,
        );
        buffer.dones = Tensor::zeros([num_steps, num_envs], &device);

        let last_values: Tensor<TestBackend, 1> = Tensor::from_floats([0.5, 0.5], &device);

        compute_gae(&mut buffer, last_values, 0.99, 0.95, &device);

        assert!(buffer.advantages.is_some());
        assert!(buffer.returns.is_some());

        // Verify advantages are non-zero
        let adv_data: Vec<f32> = buffer.advantages.unwrap().into_data().to_vec().expect("advantages");
        assert!(adv_data.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn test_gae_multiplayer() {
        let device = Default::default();
        let num_steps = 4;
        let num_envs = 2;
        let num_players = 2u8;

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(num_steps, num_envs, 86, num_players, &device);

        // Set up multi-player data
        // All rewards and values for both players
        buffer.all_rewards = Tensor::from_floats(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]], // Final rewards: env 0 P0 wins, env 1 P1 wins
            ],
            &device,
        );
        buffer.all_values = Tensor::from_floats(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
            ],
            &device,
        );
        // Alternating players: P0, P1, P0, P1
        buffer.acting_players = Tensor::from_ints([[0, 0], [1, 1], [0, 0], [1, 1]], &device);
        buffer.dones = Tensor::from_floats(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
            &device,
        );

        let last_values: Tensor<TestBackend, 2> = Tensor::from_floats(
            [[0.0, 0.0], [0.0, 0.0]], // Terminal state, values don't matter
            &device,
        );

        compute_gae_multiplayer(&mut buffer, last_values, 0.99, 0.95, num_players, &device);

        assert!(buffer.advantages.is_some());
        assert!(buffer.returns.is_some());
    }

    #[test]
    fn test_buffer_flatten() {
        let device = Default::default();
        let buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(4, 2, 3, 1u8, &device);

        // Set advantages and returns
        let mut buffer = buffer;
        buffer.advantages = Some(Tensor::ones([4, 2], &device));
        buffer.returns = Some(Tensor::ones([4, 2], &device));

        let (obs, actions, log_probs, advantages, returns, acting_players) = buffer.flatten();

        assert_eq!(obs.dims(), [8, 3]);
        assert_eq!(actions.dims(), [8]);
        assert_eq!(log_probs.dims(), [8]);
        assert_eq!(advantages.dims(), [8]);
        assert_eq!(returns.dims(), [8]);
        assert_eq!(acting_players.dims(), [8]);
    }
}
