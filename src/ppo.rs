//! PPO Algorithm Implementation
//!
//! Implements Proximal Policy Optimization with all 13 details from the ICLR blog:
//! - Vectorized architecture, orthogonal init, Adam epsilon 1e-5
//! - Learning rate annealing, GAE, minibatch shuffled updates
//! - Advantage normalization, clipped surrogate, value clipping
//! - Combined loss, gradient clipping

// Tensor operations use expect/unwrap for internal invariants that cannot fail
// when tensor shapes are correct (which is guaranteed by construction)

use std::collections::HashMap;

use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::Int;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::config::Config;
use crate::env::{Environment, EpisodeStats, VecEnv};
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;
use crate::opponent_pool::{EnvState, OpponentPool};
use crate::profile::{gpu_sync, profile_function, profile_scope};
use crate::utils::{
    entropy_categorical, log_prob_categorical, normalize_advantages, sample_categorical,
};

/// Flattened buffer data for minibatch processing:
/// `(obs, actions, old_log_probs, advantages, returns, acting_players, old_values, valid_indices)`
/// - `valid_indices` is `None` for self-play, `Some(indices)` for opponent pool training
/// - When `valid_indices` is present, caller should use it to select only learner turn data
type FlattenedBuffer<B> = (
    Tensor<B, 2>,              // obs
    Tensor<B, 1, Int>,         // actions
    Tensor<B, 1>,              // old_log_probs
    Tensor<B, 1>,              // advantages
    Tensor<B, 1>,              // returns
    Tensor<B, 1, Int>,         // acting_players
    Tensor<B, 1>,              // old_values (flattened)
    Option<Tensor<B, 1, Int>>, // valid_indices for opponent pool training
);

/// Stores trajectory data from environment rollouts
///
/// Shape: [`num_steps`, `num_envs`] for most fields
/// For multi-player games, also stores per-player values and rewards.
#[derive(Debug)]
pub struct RolloutBuffer<B: Backend> {
    /// Observations [`num_steps`, `num_envs`, `obs_dim`]
    pub observations: Tensor<B, 3>,
    /// Actions taken [`num_steps`, `num_envs`]
    pub actions: Tensor<B, 2, Int>,
    /// Rewards for acting player [`num_steps`, `num_envs`]
    pub rewards: Tensor<B, 2>,
    /// Episode done flags [`num_steps`, `num_envs`]
    pub dones: Tensor<B, 2>,
    /// Value estimates for acting player [`num_steps`, `num_envs`]
    pub values: Tensor<B, 2>,
    /// Log probabilities of actions [`num_steps`, `num_envs`]
    pub log_probs: Tensor<B, 2>,

    // Multi-player support fields
    /// Values for ALL players [`num_steps`, `num_envs`, `num_players`]
    pub all_values: Tensor<B, 3>,
    /// Rewards for ALL players [`num_steps`, `num_envs`, `num_players`]
    pub all_rewards: Tensor<B, 3>,
    /// Which player acted each step [`num_steps`, `num_envs`]
    pub acting_players: Tensor<B, 2, Int>,
    /// Number of players (max 255)
    num_players: u8,

    /// GAE advantages (computed after rollout) [`num_steps`, `num_envs`]
    pub advantages: Option<Tensor<B, 2>>,
    /// Returns (values + advantages) [`num_steps`, `num_envs`]
    pub returns: Option<Tensor<B, 2>>,
    /// Valid mask for opponent pool training [`num_steps`, `num_envs`]
    /// 1.0 for learner turns (on-policy data), 0.0 for opponent turns (off-policy, masked out)
    pub valid_mask: Option<Tensor<B, 2>>,
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
            valid_mask: None,
        }
    }

    /// Get number of players (max 255)
    pub const fn num_players(&self) -> u8 {
        self.num_players
    }

    /// Flatten buffer for minibatch processing
    /// Returns (obs, actions, `old_log_probs`, advantages, returns, `acting_players`, `old_values`, `valid_indices`)
    /// Each with shape [`num_steps` * `num_envs`, ...]
    ///
    /// If `valid_mask` is set (opponent pool training), returns `valid_indices` containing
    /// indices of learner turns only. Caller should use these to select valid data before
    /// any computation to ensure opponent turn data is completely excluded.
    ///
    /// # Panics
    /// Panics if `compute_gae` was not called first (advantages/returns not set)
    #[expect(clippy::cast_possible_wrap, reason = "batch indices fit in i64")]
    pub fn flatten(&self) -> FlattenedBuffer<B> {
        let [num_steps, num_envs, obs_dim] = self.observations.dims();
        let batch_size = num_steps * num_envs;

        let obs = self.observations.clone().reshape([batch_size, obs_dim]);
        let actions = self.actions.clone().flatten(0, 1);
        let log_probs = self.log_probs.clone().flatten(0, 1);
        let advantages = self
            .advantages
            .as_ref()
            .expect("Advantages not computed")
            .clone()
            .flatten(0, 1);
        let returns = self
            .returns
            .as_ref()
            .expect("Returns not computed")
            .clone()
            .flatten(0, 1);
        let acting_players = self.acting_players.clone().flatten(0, 1);
        let old_values = self.values.clone().flatten(0, 1);

        // Compute valid indices if mask present (opponent pool training)
        // These indices identify learner turns - caller uses them to select only valid data
        let valid_indices = if let Some(ref mask) = self.valid_mask {
            let mask_flat: Tensor<B, 1> = mask.clone().flatten(0, 1);
            let mask_data: Vec<f32> = mask_flat.into_data().to_vec().expect("mask data");
            let indices: Vec<i64> = mask_data
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 0.5)
                .map(|(i, _)| i as i64)
                .collect();
            let device = self.observations.device();
            Some(Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device))
        } else {
            None
        };

        (
            obs,
            actions,
            log_probs,
            advantages,
            returns,
            acting_players,
            old_values,
            valid_indices,
        )
    }
}

/// Collect rollouts from vectorized environment
///
/// Runs `num_steps` in each of `num_envs` environments, storing trajectories.
/// Uses CPU collection with batch GPU transfer for performance.
/// If normalizer is provided, observations are normalized before model inference
/// using existing (lagged) statistics, then stats are updated at end of rollout.
#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "action indices and player counts are small positive values"
)]
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

            let obs_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(obs_flat.as_slice(), device)
                .reshape([num_envs, obs_dim]);
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

            let log_probs_data: Vec<f32> =
                log_probs.into_data().to_vec().expect("log_probs to vec");
            (
                actions_data,
                acting_values_data,
                log_probs_data,
                values_all_data,
            )
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
        let rewards_flat: Vec<f32> = player_rewards
            .iter()
            .flat_map(|r| {
                // Pad with zeros if rewards vec is shorter than num_players
                (0..num_players).map(|p| r.get(p).copied().unwrap_or(0.0))
            })
            .collect();

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
        buffer.observations = Tensor::<B, 1>::from_floats(all_obs.as_slice(), device)
            .reshape([num_steps, num_envs, obs_dim]);
        buffer.actions = Tensor::<B, 1, Int>::from_ints(all_actions.as_slice(), device)
            .reshape([num_steps, num_envs]);
        buffer.rewards = Tensor::<B, 1>::from_floats(all_acting_rewards.as_slice(), device)
            .reshape([num_steps, num_envs]);
        buffer.dones = Tensor::<B, 1>::from_floats(all_dones.as_slice(), device)
            .reshape([num_steps, num_envs]);
        buffer.values = Tensor::<B, 1>::from_floats(all_acting_values.as_slice(), device)
            .reshape([num_steps, num_envs]);
        buffer.log_probs = Tensor::<B, 1>::from_floats(all_log_probs.as_slice(), device)
            .reshape([num_steps, num_envs]);

        // Multi-player data
        buffer.all_values = Tensor::<B, 1>::from_floats(all_values_flat.as_slice(), device)
            .reshape([num_steps, num_envs, num_players]);
        buffer.all_rewards = Tensor::<B, 1>::from_floats(all_rewards_flat.as_slice(), device)
            .reshape([num_steps, num_envs, num_players]);
        buffer.acting_players =
            Tensor::<B, 1, Int>::from_ints(all_acting_players.as_slice(), device)
                .reshape([num_steps, num_envs]);
    }

    // Update normalizer stats at end of rollout with all RAW observations
    // This ensures stats are updated for the NEXT rollout, not the current one
    if let Some(norm) = normalizer.as_mut() {
        norm.update_batch(&raw_obs_for_stats, obs_dim);
    }

    all_completed
}

/// Information about a completed episode in opponent pool training
#[derive(Debug)]
#[expect(dead_code, reason = "fields reserved for pool evaluation metrics")]
pub struct OpponentEpisodeCompletion {
    /// Environment index
    pub env_idx: usize,
    /// Placements for all players (1 = first, 2 = second, etc.)
    pub placements: Vec<usize>,
    /// Pool indices of opponents in this game
    pub opponent_pool_indices: Vec<usize>,
    /// Whether this env was playing against opponents (vs self-play)
    pub is_opponent_game: bool,
}

/// Collect rollouts with opponent pool training
///
/// A fraction of envs play against historical opponents, the rest do self-play.
/// For opponent games, only the learner's experiences are stored (on-policy requirement).
///
/// Returns `(episode_stats, opponent_completions)` where `opponent_completions`
/// contains info for rating updates.
#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_arguments,
    clippy::explicit_iter_loop,
    reason = "action indices and player counts are small positive values"
)]
pub fn collect_rollouts_with_opponents<B: Backend, E: Environment>(
    model: &ActorCritic<B>,
    opponent_pool: &mut OpponentPool<B>,
    vec_env: &mut VecEnv<E>,
    buffer: &mut RolloutBuffer<B>,
    env_states: &mut [EnvState],
    num_opponent_envs: usize,
    num_steps: usize,
    device: &B::Device,
    rng: &mut impl Rng,
    mut normalizer: Option<&mut ObsNormalizer>,
    _current_step: usize,
) -> (Vec<EpisodeStats>, Vec<OpponentEpisodeCompletion>) {
    profile_function!();
    let num_envs = vec_env.num_envs();
    let obs_dim = E::OBSERVATION_DIM;
    let num_players = E::NUM_PLAYERS;
    let mut all_completed = Vec::new();
    let mut opponent_completions = Vec::new();

    // Track which indices in each buffer position contain learner data
    // For self-play envs: all turns are learner data
    // For opponent envs: only learner's turns are stored
    let mut valid_indices: Vec<bool> = Vec::with_capacity(num_steps * num_envs);

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

    // Collect raw observations for normalizer stats update
    let collect_raw_obs = normalizer.is_some();
    let mut raw_obs_for_stats: Vec<f32> = if collect_raw_obs {
        Vec::with_capacity(num_steps * num_envs * obs_dim)
    } else {
        Vec::new()
    };

    for _step in 0..num_steps {
        profile_scope!("rollout_step_opponents");

        // Get current players BEFORE the step
        let current_players = vec_env.get_current_players();

        // Get current observations (keep raw for opponent normalization)
        let obs_flat_raw = vec_env.get_observations();

        // Store raw observations for normalizer stats update
        if collect_raw_obs {
            raw_obs_for_stats.extend_from_slice(&obs_flat_raw);
        }

        // Partition envs by which model should act
        let mut learner_env_indices: Vec<usize> = Vec::new();
        let mut opponent_batches: HashMap<usize, Vec<usize>> = HashMap::new();

        for (env_idx, &current_player) in current_players.iter().enumerate() {
            if env_idx >= num_opponent_envs {
                // Self-play env: learner always acts
                learner_env_indices.push(env_idx);
            } else {
                // Opponent env: check if it's learner's turn
                let env_state = &env_states[env_idx];
                if current_player == env_state.learner_position {
                    learner_env_indices.push(env_idx);
                } else {
                    // Opponent's turn - look up which opponent model
                    let pool_idx = env_state.position_to_opponent[current_player]
                        .expect("position_to_opponent should have opponent index");
                    opponent_batches.entry(pool_idx).or_default().push(env_idx);
                }
            }
        }

        // Prepare actions for all envs (will be filled in by model forward passes)
        let mut all_env_actions: Vec<usize> = vec![0; num_envs];
        let mut all_env_values: Vec<Vec<f32>> = vec![vec![0.0; num_players]; num_envs];
        let mut all_env_log_probs: Vec<f32> = vec![0.0; num_envs];

        // Forward pass for learner model
        if !learner_env_indices.is_empty() {
            profile_scope!("learner_forward");

            // Gather observations for learner envs
            let mut learner_obs: Vec<f32> = learner_env_indices
                .iter()
                .flat_map(|&env_idx| {
                    let start = env_idx * obs_dim;
                    obs_flat_raw[start..start + obs_dim].iter().copied()
                })
                .collect();

            // Apply learner's normalizer
            if let Some(ref norm) = normalizer {
                norm.normalize_batch(&mut learner_obs, obs_dim);
            }

            let batch_size = learner_env_indices.len();
            let obs_tensor: Tensor<B, 2> =
                Tensor::<B, 1>::from_floats(learner_obs.as_slice(), device)
                    .reshape([batch_size, obs_dim]);
            let (logits, values) = model.forward(obs_tensor);

            let actions = sample_categorical(logits.clone(), rng, device);
            let log_probs = log_prob_categorical(logits, actions.clone());

            // Sync to CPU
            let actions_data: Vec<i64> = actions
                .float()
                .into_data()
                .to_vec::<f32>()
                .expect("actions to vec")
                .into_iter()
                .map(|x| x as i64)
                .collect();
            let values_data: Vec<f32> = values.into_data().to_vec().expect("values to vec");
            let log_probs_data: Vec<f32> =
                log_probs.into_data().to_vec().expect("log_probs to vec");

            // Scatter results back to env arrays
            for (batch_idx, &env_idx) in learner_env_indices.iter().enumerate() {
                all_env_actions[env_idx] = actions_data[batch_idx] as usize;
                all_env_log_probs[env_idx] = log_probs_data[batch_idx];

                // Extract all player values for this env
                let value_start = batch_idx * num_players;
                all_env_values[env_idx] =
                    values_data[value_start..value_start + num_players].to_vec();
            }
        }

        // Forward pass for each opponent model
        for (pool_idx, env_indices) in opponent_batches.iter() {
            if env_indices.is_empty() {
                continue;
            }

            profile_scope!("opponent_forward");

            // Gather observations for opponent envs (from raw)
            let mut opp_obs: Vec<f32> = env_indices
                .iter()
                .flat_map(|&env_idx| {
                    let start = env_idx * obs_dim;
                    obs_flat_raw[start..start + obs_dim].iter().copied()
                })
                .collect();

            // Load opponent model and normalizer together (single borrow)
            let (opp_model, opp_norm) = opponent_pool
                .get_model_and_normalizer(*pool_idx)
                .expect("opponent model should load");

            // Apply opponent's normalizer if present
            if let Some(norm) = opp_norm {
                norm.normalize_batch(&mut opp_obs, obs_dim);
            }

            let batch_size = env_indices.len();
            let obs_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(opp_obs.as_slice(), device)
                .reshape([batch_size, obs_dim]);
            let (logits, _values) = opp_model.forward(obs_tensor);

            // Sample actions (opponents don't need log probs or values for training)
            let actions = sample_categorical(logits, rng, device);

            let actions_data: Vec<i64> = actions
                .float()
                .into_data()
                .to_vec::<f32>()
                .expect("actions to vec")
                .into_iter()
                .map(|x| x as i64)
                .collect();

            // Scatter actions to env arrays
            for (batch_idx, &env_idx) in env_indices.iter().enumerate() {
                all_env_actions[env_idx] = actions_data[batch_idx] as usize;
            }
        }

        // Step environment
        let (player_rewards, dones, completed) = {
            profile_scope!("env_step_cpu");
            let (_next_obs, player_rewards, dones, completed) = vec_env.step(&all_env_actions);
            (player_rewards, dones, completed)
        };

        // Process completed episodes
        for stats in &completed {
            let env_idx = stats.env_index;
            let is_opponent_game = env_idx < num_opponent_envs;

            // Get placements from GameOutcome if available
            let placements = stats.outcome.as_ref().map(|o| o.0.clone());

            if is_opponent_game {
                if let Some(ref places) = placements {
                    // Track opponent completion for rating update
                    let env_state = &env_states[env_idx];
                    let opponent_indices: Vec<usize> = env_state
                        .position_to_opponent
                        .iter()
                        .filter_map(|&opt| opt)
                        .collect();

                    opponent_completions.push(OpponentEpisodeCompletion {
                        env_idx,
                        placements: places.clone(),
                        opponent_pool_indices: opponent_indices,
                        is_opponent_game: true,
                    });
                }

                // Shuffle positions for next episode (same opponents until rotation)
                env_states[env_idx].shuffle_positions(num_players, rng);

                // Apply pending rotation if any
                env_states[env_idx].apply_pending_rotation();
            }
        }
        all_completed.extend(completed);

        // Store data - for opponent envs, only store if it's learner's turn
        for (env_idx, &current_player) in current_players.iter().enumerate() {
            let is_opponent_env = env_idx < num_opponent_envs;
            let is_learner_turn = if is_opponent_env {
                current_player == env_states[env_idx].learner_position
            } else {
                true // Self-play: all turns are learner
            };

            valid_indices.push(is_learner_turn);

            // Always store data in buffer slots (will be masked later)
            // Store normalized observations for learner envs
            let obs_start = env_idx * obs_dim;
            let mut env_obs = obs_flat_raw[obs_start..obs_start + obs_dim].to_vec();
            // Normalize with learner's normalizer for buffer storage
            if let Some(ref norm) = normalizer {
                norm.normalize_batch(&mut env_obs, obs_dim);
            }
            all_obs.extend_from_slice(&env_obs);

            all_actions.push(all_env_actions[env_idx] as i64);
            all_log_probs.push(all_env_log_probs[env_idx]);

            // Get acting player's reward and value
            let acting_reward = player_rewards
                .get(env_idx)
                .and_then(|r| r.get(current_player))
                .copied()
                .unwrap_or(0.0);
            all_acting_rewards.push(acting_reward);

            let acting_value = all_env_values
                .get(env_idx)
                .and_then(|v| v.get(current_player))
                .copied()
                .unwrap_or(0.0);
            all_acting_values.push(acting_value);

            all_dones.push(if dones.get(env_idx).copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            });

            // Multi-player data
            all_values_flat.extend_from_slice(&all_env_values[env_idx]);
            let rewards_flat: Vec<f32> = (0..num_players)
                .map(|p| {
                    player_rewards
                        .get(env_idx)
                        .and_then(|r| r.get(p))
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect();
            all_rewards_flat.extend_from_slice(&rewards_flat);
            all_acting_players.push(current_player as i64);
        }
    }

    // Batch transfer to GPU
    {
        profile_scope!("batch_gpu_transfer");
        buffer.observations = Tensor::<B, 1>::from_floats(all_obs.as_slice(), device)
            .reshape([num_steps, num_envs, obs_dim]);
        buffer.actions = Tensor::<B, 1, Int>::from_ints(all_actions.as_slice(), device)
            .reshape([num_steps, num_envs]);
        buffer.rewards = Tensor::<B, 1>::from_floats(all_acting_rewards.as_slice(), device)
            .reshape([num_steps, num_envs]);
        buffer.dones = Tensor::<B, 1>::from_floats(all_dones.as_slice(), device)
            .reshape([num_steps, num_envs]);
        buffer.values = Tensor::<B, 1>::from_floats(all_acting_values.as_slice(), device)
            .reshape([num_steps, num_envs]);
        buffer.log_probs = Tensor::<B, 1>::from_floats(all_log_probs.as_slice(), device)
            .reshape([num_steps, num_envs]);

        buffer.all_values = Tensor::<B, 1>::from_floats(all_values_flat.as_slice(), device)
            .reshape([num_steps, num_envs, num_players]);
        buffer.all_rewards = Tensor::<B, 1>::from_floats(all_rewards_flat.as_slice(), device)
            .reshape([num_steps, num_envs, num_players]);
        buffer.acting_players =
            Tensor::<B, 1, Int>::from_ints(all_acting_players.as_slice(), device)
                .reshape([num_steps, num_envs]);

        // Create valid mask from valid_indices (1.0 for learner turns, 0.0 for opponent turns)
        let valid_mask_data: Vec<f32> = valid_indices
            .iter()
            .map(|&valid| if valid { 1.0 } else { 0.0 })
            .collect();
        buffer.valid_mask = Some(
            Tensor::<B, 1>::from_floats(valid_mask_data.as_slice(), device)
                .reshape([num_steps, num_envs]),
        );
    }

    // Update normalizer stats
    if let Some(norm) = normalizer.as_mut() {
        norm.update_batch(&raw_obs_for_stats, obs_dim);
    }

    (all_completed, opponent_completions)
}

/// Compute Generalized Advantage Estimation
///
/// GAE(gamma, lambda) = sum_{t'>=t} (gamma*lambda)^{t'-t} * delta_{t'}
/// where `delta_t` = `r_t` + gamma * V(s_{t+1}) - `V(s_t)`
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
    let rewards_data: Vec<f32> = buffer
        .rewards
        .clone()
        .into_data()
        .to_vec()
        .expect("rewards");
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
            let delta = (gamma * next_value).mul_add(1.0 - done, reward) - value;

            // GAE: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
            last_gae[e] = (gamma * gae_lambda * (1.0 - done)).mul_add(last_gae[e], delta);
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
#[expect(clippy::cast_sign_loss, reason = "tensor indices are non-negative")]
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
    let all_rewards_data: Vec<f32> = buffer
        .all_rewards
        .clone()
        .into_data()
        .to_vec()
        .expect("all_rewards");
    let all_values_data: Vec<f32> = buffer
        .all_values
        .clone()
        .into_data()
        .to_vec()
        .expect("all_values");
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
            for (p, carry) in reward_carry[e].iter_mut().enumerate() {
                if p != acting_player {
                    *carry += all_rewards_data[(t * num_envs + e) * num_players + p];
                }
            }

            // Reset on episode boundary
            if done > 0.5 {
                reward_carry[e].fill(0.0);
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
            let delta = (gamma * next_value[e][player]).mul_add(1.0 - done, reward) - value;

            // GAE for this player
            let advantage =
                (gamma * gae_lambda * (1.0 - done)).mul_add(gae_carry[e][player], delta);
            advantages[idx] = advantage;

            // Update carry for this player
            gae_carry[e][player] = advantage;
            next_value[e][player] = value;

            // Reset all players' GAE carry on episode boundary
            if done > 0.5 {
                gae_carry[e].fill(0.0);
                next_value[e].fill(0.0);
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
    let var_returns = returns
        .iter()
        .map(|r| (r - mean_returns).powi(2))
        .sum::<f32>()
        / n;

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

/// Extract a scalar f32 from a 1D tensor with a single element.
/// Used for metric extraction after backward pass.
#[inline]
fn scalar<B: burn::prelude::Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data().as_slice::<f32>().expect("scalar")[0]
}

/// Scalar metrics extracted from a single minibatch update
///
/// These are extracted immediately after `backward()` to free the autodiff graph.
/// Using `.inner().into_scalar()` converts autodiff tensors to plain f32 values,
/// allowing the computation graph to be garbage collected.
#[derive(Debug, Clone, Copy, Default)]
struct MinibatchMetrics {
    policy_loss: f32,
    value_loss: f32,
    entropy: f32,
    approx_kl: f32,
    clip_fraction: f32,
    total_loss: f32,
    value_mean: f32,
    returns_mean: f32,
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
/// Implements clipped surrogate objective with value clipping.
/// Takes buffer with inner (non-autodiff) backend tensors and converts them
/// to autodiff tensors for gradient computation.
#[expect(clippy::cast_possible_wrap, reason = "tensor indices are non-negative")]
pub fn ppo_update<B: burn::tensor::backend::AutodiffBackend>(
    model: ActorCritic<B>,
    buffer: &RolloutBuffer<B::InnerBackend>,
    optimizer: &mut impl burn::optim::Optimizer<ActorCritic<B>, B>,
    config: &Config,
    learning_rate: f64,
    entropy_coef: f64,
    rng: &mut impl Rng,
) -> (ActorCritic<B>, UpdateMetrics) {
    profile_function!();
    let device = model.devices()[0].clone();
    let num_players = buffer.num_players() as usize;

    // Get flattened data from inner backend buffer
    // Keep as InnerBackend tensors - convert to autodiff per-minibatch to ensure
    // each minibatch has an independent computation graph that can be fully freed
    let (
        obs_inner,
        actions_inner,
        old_log_probs_inner,
        advantages_inner,
        returns_inner,
        acting_players_inner,
        old_values_inner,
        valid_indices,
    ) = buffer.flatten();

    // If opponent pool training, filter to only learner turn data
    // This completely excludes opponent turn data from all computation
    let (
        obs_inner,
        actions_inner,
        old_log_probs_inner,
        advantages_inner,
        returns_inner,
        acting_players_inner,
        old_values_inner,
    ) = if let Some(indices) = valid_indices {
        (
            obs_inner.select(0, indices.clone()),
            actions_inner.select(0, indices.clone()),
            old_log_probs_inner.select(0, indices.clone()),
            advantages_inner.select(0, indices.clone()),
            returns_inner.select(0, indices.clone()),
            acting_players_inner.select(0, indices.clone()),
            old_values_inner.select(0, indices),
        )
    } else {
        (
            obs_inner,
            actions_inner,
            old_log_probs_inner,
            advantages_inner,
            returns_inner,
            acting_players_inner,
            old_values_inner,
        )
    };

    let batch_size = obs_inner.dims()[0];
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
            let mb_size = mb_end - mb_start;

            // Skip minibatches that are too small for stable normalization
            // variance requires at least 2 samples to avoid NaN (sample var divides by n-1)
            if mb_size < 2 {
                continue;
            }

            let mb_indices: Vec<i64> = indices[mb_start..mb_end]
                .iter()
                .map(|&i| i as i64)
                .collect();

            // Create indices tensor on InnerBackend
            let mb_indices_inner: Tensor<B::InnerBackend, 1, Int> =
                Tensor::from_ints(mb_indices.as_slice(), &device);

            // Select minibatch data on InnerBackend (no autodiff graph created)
            // Data is already filtered to learner turns only (if opponent pool training)
            let mb_obs_inner = obs_inner.clone().select(0, mb_indices_inner.clone());
            let mb_actions_inner = actions_inner.clone().select(0, mb_indices_inner.clone());
            let mb_old_log_probs_inner = old_log_probs_inner
                .clone()
                .select(0, mb_indices_inner.clone());
            let mb_advantages_inner = advantages_inner.clone().select(0, mb_indices_inner.clone());
            let mb_returns_inner = returns_inner.clone().select(0, mb_indices_inner.clone());
            let mb_acting_players_inner = acting_players_inner
                .clone()
                .select(0, mb_indices_inner.clone());
            let mb_old_values_inner = old_values_inner.clone().select(0, mb_indices_inner);

            // Convert to autodiff for THIS minibatch only (fresh independent graph)
            // Each minibatch's graph is independent and can be fully freed at loop end
            let mb_obs: Tensor<B, 2> = Tensor::from_inner(mb_obs_inner);
            let mb_actions: Tensor<B, 1, Int> = Tensor::from_inner(mb_actions_inner);
            let mb_old_log_probs: Tensor<B, 1> = Tensor::from_inner(mb_old_log_probs_inner);
            let mb_advantages_raw: Tensor<B, 1> = Tensor::from_inner(mb_advantages_inner);
            let mb_returns: Tensor<B, 1> = Tensor::from_inner(mb_returns_inner);
            let mb_acting_players: Tensor<B, 1, Int> = Tensor::from_inner(mb_acting_players_inner);
            let mb_old_values: Tensor<B, 1> = Tensor::from_inner(mb_old_values_inner);

            // Normalize advantages at minibatch level (critical for stability)
            let mb_advantages = normalize_advantages(mb_advantages_raw);

            // Forward pass with GPU sync for accurate timing
            #[expect(clippy::let_and_return, reason = "scoped for profiling")]
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
                // Multi-player: extract acting player's value using gather (maintains autodiff)
                // gather(dim, indices) selects elements along dim using per-element indices
                let indices_2d = mb_acting_players.unsqueeze_dim(1); // [mb_size] -> [mb_size, 1]
                all_values.gather(1, indices_2d).squeeze_dims(&[1]) // [mb_size, num_players] -> [mb_size]
            };

            // Compute losses, backward pass, and extract metrics immediately
            //
            // CRITICAL: Extract metrics as scalars immediately after backward() to free
            // the autodiff graph. Returning autodiff tensors would retain the entire
            // computation graph until they're consumed, causing memory accumulation.
            let (grads, metrics) = {
                profile_scope!("loss_and_backward");

                let new_log_probs = log_prob_categorical(logits.clone(), mb_actions);
                let entropy = entropy_categorical(logits);

                // Policy loss (clipped surrogate objective)
                let log_ratio = new_log_probs.clone() - mb_old_log_probs;
                let ratio = log_ratio.clone().exp();

                // Capture inner versions for metrics BEFORE further loss computation
                // These don't participate in autodiff, so they won't retain the graph
                let ratio_inner = ratio.clone().inner();
                let log_ratio_inner = log_ratio.inner();

                let policy_loss_1 = -mb_advantages.clone() * ratio.clone();
                let policy_loss_2 = -mb_advantages
                    * ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon);
                let policy_loss: Tensor<B, 1> = policy_loss_1.max_pair(policy_loss_2);
                let policy_loss_mean = policy_loss.mean();

                // Value loss (optionally clipped)
                // Data is already filtered to learner turns only
                let value_loss = if config.clip_value {
                    let values_clipped = mb_old_values.clone()
                        + (values.clone() - mb_old_values)
                            .clamp(-config.clip_epsilon, config.clip_epsilon);
                    let value_loss_1 = (values.clone() - mb_returns.clone()).powf_scalar(2.0);
                    let value_loss_2 = (values_clipped - mb_returns.clone()).powf_scalar(2.0);
                    value_loss_1.max_pair(value_loss_2).mean() * 0.5
                } else {
                    (values.clone() - mb_returns.clone())
                        .powf_scalar(2.0)
                        .mean()
                        * 0.5
                };

                // Entropy bonus
                let entropy_loss = -entropy.clone().mean() * entropy_coef;

                // Combined loss
                let loss = policy_loss_mean.clone()
                    + value_loss.clone() * config.value_coef
                    + entropy_loss;

                // Capture values for metrics before backward
                let values_mean = values.mean();
                let returns_mean = mb_returns.mean();

                // Backward pass
                let grads = loss.backward();

                // Extract ALL metrics as scalars immediately (frees autodiff graph)
                // Using inner tensors for approx_kl and clip_fraction avoids retaining
                // the ratio/log_ratio autodiff graphs
                let metrics = MinibatchMetrics {
                    policy_loss: scalar((-policy_loss_mean).inner().reshape([1])),
                    value_loss: scalar(value_loss.inner().reshape([1])),
                    entropy: scalar(entropy.inner().mean().reshape([1])),
                    approx_kl: scalar(
                        ((ratio_inner.clone() - 1.0) - log_ratio_inner)
                            .mean()
                            .reshape([1]),
                    ),
                    clip_fraction: scalar(
                        (ratio_inner - 1.0)
                            .abs()
                            .greater_elem(config.clip_epsilon)
                            .float()
                            .mean()
                            .reshape([1]),
                    ),
                    total_loss: scalar(loss.inner().reshape([1])),
                    value_mean: scalar(values_mean.inner().reshape([1])),
                    returns_mean: scalar(returns_mean.inner().reshape([1])),
                };

                (grads, metrics)
            };

            // Optimizer step with GPU sync for accurate timing
            #[expect(clippy::let_and_return, reason = "scoped for profiling")]
            {
                model = {
                    profile_scope!("optimizer_step");
                    let grads = GradientsParams::from_grads(grads, &model);
                    let updated = optimizer.step(learning_rate, model, grads);
                    gpu_sync!(updated.layers[0].weight.val()); // Force sync after weight update
                    updated
                };
            }

            // Accumulate metrics from struct (scalars already extracted in loss_and_backward)
            {
                profile_scope!("extract_metrics");

                total_policy_loss += metrics.policy_loss;
                total_value_loss += metrics.value_loss;
                total_entropy += metrics.entropy;
                total_approx_kl += metrics.approx_kl;
                total_clip_fraction += metrics.clip_fraction;
                total_loss_sum += metrics.total_loss;
                total_value_mean += metrics.value_mean;
                total_returns_mean += metrics.returns_mean;
                num_updates += 1;

                // KL early stopping: stop epoch if KL divergence exceeds threshold
                if let Some(target) = config.target_kl {
                    if metrics.approx_kl > target as f32 {
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

        let mut buffer: RolloutBuffer<TestBackend> =
            RolloutBuffer::new(num_steps, num_envs, 1, num_players, &device);

        // Set up simple rewards and values for testing
        buffer.rewards =
            Tensor::from_floats([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], &device);
        buffer.values =
            Tensor::from_floats([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], &device);
        buffer.dones = Tensor::zeros([num_steps, num_envs], &device);

        let last_values: Tensor<TestBackend, 1> = Tensor::from_floats([0.5, 0.5], &device);

        compute_gae(&mut buffer, last_values, 0.99, 0.95, &device);

        assert!(buffer.advantages.is_some());
        assert!(buffer.returns.is_some());

        // Verify advantages are non-zero
        let adv_data: Vec<f32> = buffer
            .advantages
            .unwrap()
            .into_data()
            .to_vec()
            .expect("advantages");
        assert!(adv_data.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn test_gae_multiplayer() {
        let device = Default::default();
        let num_steps = 4;
        let num_envs = 2;
        let num_players = 2u8;

        let mut buffer: RolloutBuffer<TestBackend> =
            RolloutBuffer::new(num_steps, num_envs, 86, num_players, &device);

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
        buffer.dones =
            Tensor::from_floats([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], &device);

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

        let (
            obs,
            actions,
            log_probs,
            advantages,
            returns,
            acting_players,
            old_values,
            valid_indices,
        ) = buffer.flatten();

        assert_eq!(obs.dims(), [8, 3]);
        assert_eq!(actions.dims(), [8]);
        assert_eq!(log_probs.dims(), [8]);
        assert_eq!(advantages.dims(), [8]);
        assert_eq!(returns.dims(), [8]);
        assert_eq!(acting_players.dims(), [8]);
        assert_eq!(old_values.dims(), [8]);
        assert!(
            valid_indices.is_none(),
            "valid_indices should be None without valid_mask"
        );
    }

    #[test]
    fn test_explained_variance_perfect_prediction() {
        // Values perfectly predict returns
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let returns = vec![1.0, 2.0, 3.0, 4.0];
        let ev = compute_explained_variance(&values, &returns);
        assert!((ev - 1.0).abs() < 1e-5, "Expected 1.0, got {ev}");
    }

    #[test]
    fn test_explained_variance_zero_correlation() {
        // Values have no correlation with returns
        let values = vec![0.0, 0.0, 0.0, 0.0];
        let returns = vec![1.0, 2.0, 3.0, 4.0];
        let ev = compute_explained_variance(&values, &returns);
        // With zero mean values, explained variance should be negative
        // since residuals will have higher variance than returns
        assert!(ev < 1.0, "Expected < 1.0, got {ev}");
    }

    #[test]
    fn test_explained_variance_constant_returns() {
        // Returns have no variance
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let returns = vec![2.5, 2.5, 2.5, 2.5];
        let ev = compute_explained_variance(&values, &returns);
        // Should return 0 since var(returns) is near zero
        assert!((ev - 0.0).abs() < 1e-5, "Expected 0.0, got {ev}");
    }

    #[test]
    fn test_explained_variance_single_element() {
        // Too few elements
        let values = vec![1.0];
        let returns = vec![1.0];
        let ev = compute_explained_variance(&values, &returns);
        assert!((ev - 0.0).abs() < 1e-5, "Expected 0.0 for single element");
    }

    #[test]
    fn test_explained_variance_empty() {
        let values: Vec<f32> = vec![];
        let returns: Vec<f32> = vec![];
        let ev = compute_explained_variance(&values, &returns);
        assert!((ev - 0.0).abs() < 1e-5, "Expected 0.0 for empty input");
    }

    #[test]
    fn test_explained_variance_partial_prediction() {
        // Values partially predict returns
        let values = vec![1.0, 2.5, 3.0, 4.5];
        let returns = vec![1.0, 2.0, 3.0, 4.0];
        let ev = compute_explained_variance(&values, &returns);
        // Should be positive but less than 1
        assert!(ev > 0.0 && ev < 1.0, "Expected 0 < ev < 1, got {ev}");
    }

    #[test]
    fn test_valid_mask_produces_correct_indices() {
        // Test that valid_mask correctly filters to only learner turn indices
        let device = Default::default();
        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(4, 2, 3, 1u8, &device);

        // Set up required fields
        buffer.advantages = Some(Tensor::ones([4, 2], &device));
        buffer.returns = Some(Tensor::ones([4, 2], &device));

        // Set valid_mask: alternating learner (1.0) and opponent (0.0) turns
        // Shape [4, 2] flattens to [8] indices: 0,1,2,3,4,5,6,7
        // Mask: [[1,0], [0,1], [1,0], [0,1]] -> flattened: [1,0,0,1,1,0,0,1]
        // Valid indices should be: 0, 3, 4, 7
        buffer.valid_mask = Some(Tensor::from_floats(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            &device,
        ));

        let (_, _, _, _, _, _, _, valid_indices) = buffer.flatten();

        assert!(
            valid_indices.is_some(),
            "valid_indices should be Some with valid_mask"
        );
        let indices = valid_indices.unwrap();
        assert_eq!(indices.dims(), [4], "Should have 4 valid indices");

        let indices_data: Vec<i64> = indices.into_data().to_vec().expect("indices data");
        assert_eq!(
            indices_data,
            vec![0, 3, 4, 7],
            "Valid indices should match mask pattern"
        );
    }

    #[test]
    fn test_valid_mask_all_learner_turns() {
        // When all turns are learner turns, valid_indices should include all
        let device = Default::default();
        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(4, 2, 3, 1u8, &device);

        buffer.advantages = Some(Tensor::ones([4, 2], &device));
        buffer.returns = Some(Tensor::ones([4, 2], &device));

        // All 1.0 = all learner turns
        buffer.valid_mask = Some(Tensor::ones([4, 2], &device));

        let (_, _, _, _, _, _, _, valid_indices) = buffer.flatten();

        let indices = valid_indices.expect("should have valid_indices");
        assert_eq!(indices.dims(), [8], "All 8 positions should be valid");

        let indices_data: Vec<i64> = indices.into_data().to_vec().expect("indices data");
        assert_eq!(indices_data, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_valid_mask_no_learner_turns() {
        // Edge case: no learner turns (shouldn't happen in practice, but test robustness)
        let device = Default::default();
        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(4, 2, 3, 1u8, &device);

        buffer.advantages = Some(Tensor::ones([4, 2], &device));
        buffer.returns = Some(Tensor::ones([4, 2], &device));

        // All 0.0 = all opponent turns
        buffer.valid_mask = Some(Tensor::zeros([4, 2], &device));

        let (_, _, _, _, _, _, _, valid_indices) = buffer.flatten();

        let indices = valid_indices.expect("should have valid_indices");
        assert_eq!(indices.dims(), [0], "No positions should be valid");
    }

    #[test]
    fn test_select_filters_data_correctly() {
        // Test that selecting with valid_indices correctly filters tensor data
        let device = Default::default();

        // Create observation data where we can verify filtering
        // Obs shape: [8, 3] - 8 timesteps, 3 features
        let obs: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
                [18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0],
            ],
            &device,
        );

        // Select indices 0, 3, 4, 7 (simulating learner turns)
        let indices: Tensor<TestBackend, 1, Int> = Tensor::from_ints([0, 3, 4, 7], &device);
        let filtered = obs.select(0, indices);

        assert_eq!(filtered.dims(), [4, 3], "Should have 4 selected rows");

        let filtered_data: Vec<f32> = filtered.into_data().to_vec().expect("filtered data");
        // Row 0: [0, 1, 2], Row 3: [9, 10, 11], Row 4: [12, 13, 14], Row 7: [21, 22, 23]
        assert_eq!(
            filtered_data,
            vec![0.0, 1.0, 2.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0]
        );
    }
}
