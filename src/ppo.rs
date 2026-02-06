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
use crate::normalization::{ObsNormalizer, PopArtNormalizer, ReturnNormalizer};
use crate::opponent_pool::{EnvState, OpponentPool};
use crate::profile::{profile_function, profile_scope};
use crate::utils::{
    apply_action_mask, entropy_categorical, log_prob_categorical, normalize_advantages,
    sample_categorical,
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
    Option<Tensor<B, 2>>,      // privileged_obs for CTDE (batch_size, privileged_obs_dim)
);

/// Stores trajectory data from environment rollouts
///
/// Shape: [`num_steps`, `num_envs`] for most fields
/// For multi-player games, also stores per-player rewards for attribution.
#[derive(Debug)]
pub struct RolloutBuffer<B: Backend> {
    /// Observations [`num_steps`, `num_envs`, `obs_dim`]
    pub observations: Tensor<B, 3>,
    /// Privileged observations for CTDE critic [`num_steps`, `num_envs`, `privileged_obs_dim`]
    /// Only populated when using CTDE networks
    pub privileged_obs: Option<Tensor<B, 3>>,
    /// Actions taken [`num_steps`, `num_envs`]
    pub actions: Tensor<B, 2, Int>,
    /// Rewards for acting player [`num_steps`, `num_envs`]
    pub rewards: Tensor<B, 2>,
    /// Episode done flags [`num_steps`, `num_envs`]
    pub dones: Tensor<B, 2>,
    /// Value estimates for acting player [`num_steps`, `num_envs`] (single scalar)
    pub values: Tensor<B, 2>,
    /// Log probabilities of actions [`num_steps`, `num_envs`]
    pub log_probs: Tensor<B, 2>,

    // Multi-player support fields
    /// Rewards for ALL players [`num_steps`, `num_envs`, `num_players`]
    /// (needed for reward attribution in GAE)
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
    /// Action masks for each step [`num_steps`, `num_envs`, `num_actions`]
    /// true = valid action, false = invalid action
    /// Used during PPO update to ensure `log_probs` and entropy are computed
    /// on the same masked distribution as during rollout collection.
    pub action_masks: Option<Tensor<B, 3>>,
}

impl<B: Backend> RolloutBuffer<B> {
    /// Create empty buffer with given dimensions
    ///
    /// `num_players` must be <= 255 (stored as u8)
    /// `privileged_obs_dim`: Optional privileged observation dimension for CTDE networks
    pub fn new(
        num_steps: usize,
        num_envs: usize,
        obs_dim: usize,
        privileged_obs_dim: Option<usize>,
        num_players: u8,
        device: &B::Device,
    ) -> Self {
        let np = num_players as usize;
        let privileged_obs =
            privileged_obs_dim.map(|dim| Tensor::zeros([num_steps, num_envs, dim], device));
        Self {
            observations: Tensor::zeros([num_steps, num_envs, obs_dim], device),
            privileged_obs,
            actions: Tensor::zeros([num_steps, num_envs], device),
            rewards: Tensor::zeros([num_steps, num_envs], device),
            dones: Tensor::zeros([num_steps, num_envs], device),
            values: Tensor::zeros([num_steps, num_envs], device),
            log_probs: Tensor::zeros([num_steps, num_envs], device),
            // Multi-player fields
            all_rewards: Tensor::zeros([num_steps, num_envs, np], device),
            acting_players: Tensor::zeros([num_steps, num_envs], device),
            num_players,
            advantages: None,
            returns: None,
            valid_mask: None,
            action_masks: None,
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

        // Reshape privileged observations if present (for CTDE)
        let privileged_obs = self.privileged_obs.as_ref().map(|po| {
            let [_num_steps, _num_envs, priv_dim] = po.dims();
            po.clone().reshape([batch_size, priv_dim])
        });

        (
            obs,
            actions,
            log_probs,
            advantages,
            returns,
            acting_players,
            old_values,
            valid_indices,
            privileged_obs,
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
    mut return_normalizer: Option<&mut ReturnNormalizer>,
    popart: Option<&PopArtNormalizer>,
) -> (Vec<EpisodeStats>, Vec<Vec<f32>>) {
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

    // Multi-player data (only rewards need per-player tracking for attribution)
    let mut all_rewards_flat: Vec<f32> = Vec::with_capacity(num_steps * num_envs * num_players);
    let mut all_acting_players: Vec<i64> = Vec::with_capacity(num_steps * num_envs);

    // Track each player's last value prediction for GAE bootstrap
    // From each player's perspective, the rollout "ends" at their last action
    let mut last_value_per_player: Vec<Vec<f32>> = vec![vec![0.0; num_players]; num_envs];

    // Privileged observations for CTDE (collected if buffer has privileged_obs field)
    let collect_privileged_obs = buffer.privileged_obs.is_some();
    let mut all_privileged_obs: Vec<f32> = if collect_privileged_obs {
        // Get global state dim from first env
        let first_global = vec_env.get_privileged_obs();
        let privileged_obs_dim = first_global.len() / num_envs;
        Vec::with_capacity(num_steps * num_envs * privileged_obs_dim)
    } else {
        Vec::new()
    };

    // Action masks (collected if environment provides them)
    let mut all_action_masks: Option<Vec<f32>> = None;
    let mut num_actions: usize = 0;

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

        // Get global states for CTDE (if enabled)
        if collect_privileged_obs {
            let privileged_obs_flat = vec_env.get_privileged_obs();
            all_privileged_obs.extend_from_slice(&privileged_obs_flat);
        }

        // Store raw observations BEFORE normalization for stats update
        if collect_raw_obs {
            raw_obs_for_stats.extend_from_slice(&obs_flat);
        }

        // Normalize using EXISTING (lagged) stats - don't update stats yet
        if let Some(ref norm) = normalizer {
            norm.normalize_batch(&mut obs_flat, obs_dim);
        }

        // Get action masks for all environments
        let action_masks = vec_env.get_action_masks();

        // Collect action masks if environment provides them
        if let Some(ref masks) = action_masks {
            if all_action_masks.is_none() {
                // First step: initialize collection with capacity
                num_actions = masks.len() / num_envs;
                all_action_masks = Some(Vec::with_capacity(num_steps * num_envs * num_actions));
            }
            // Convert bool to f32 and extend
            all_action_masks
                .as_mut()
                .expect("all_action_masks initialized above")
                .extend(masks.iter().map(|&v| if v { 1.0 } else { 0.0 }));
        }

        // Model inference: forward pass, sample actions, compute log probs, sync to CPU
        // All GPU ops batched together - timing includes actual compute (sync at end)
        let (actions_data, acting_values_data, log_probs_data) = {
            profile_scope!("model_inference");

            let obs_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(obs_flat.as_slice(), device)
                .reshape([num_envs, obs_dim]);

            // Handle CTDE networks: separate forward passes for actor and critic
            let (logits, values) = if model.is_ctde() {
                let privileged_obs_flat = vec_env.get_privileged_obs();
                let privileged_obs_dim = privileged_obs_flat.len() / num_envs;
                let privileged_obs_tensor =
                    Tensor::<B, 1>::from_floats(privileged_obs_flat.as_slice(), device)
                        .reshape([num_envs, privileged_obs_dim]);
                let logits = model.forward_actor(obs_tensor.clone());
                let values = model.forward_critic(privileged_obs_tensor, obs_tensor);
                (logits, values)
            } else {
                model.forward(obs_tensor)
            };
            // values is [num_envs, 1] - single scalar per environment (acting player's value)

            // Apply action mask to prevent sampling invalid actions
            let masked_logits = apply_action_mask(logits, action_masks);
            let actions = sample_categorical(masked_logits.clone(), rng, device);
            let log_probs = log_prob_categorical(masked_logits, actions.clone());

            // Sync to CPU - this is where actual GPU compute happens
            let actions_data: Vec<i64> = actions
                .float()
                .into_data()
                .to_vec::<f32>()
                .expect("actions to vec")
                .into_iter()
                .map(|x| x as i64)
                .collect();

            // Get scalar values [num_envs] - already the acting player's value
            let mut acting_values_data: Vec<f32> =
                values.into_data().to_vec().expect("values to vec");

            // Denormalize values if PopArt is active
            // After rescaling, model outputs normalized values - convert back to raw for GAE
            if let Some(popart_norm) = popart {
                popart_norm.denormalize(&mut acting_values_data);
            }

            let log_probs_data: Vec<f32> =
                log_probs.into_data().to_vec().expect("log_probs to vec");
            (actions_data, acting_values_data, log_probs_data)
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
        let mut acting_rewards: Vec<f32> = player_rewards
            .iter()
            .zip(current_players.iter())
            .map(|(r, &p)| r.get(p).copied().unwrap_or(0.0))
            .collect();

        // Apply return normalization to rewards
        // In self-play/single-player, all turns are learner turns
        if let Some(ref mut return_norm) = return_normalizer {
            for (env_idx, ((reward, &done), &player)) in acting_rewards
                .iter_mut()
                .zip(dones.iter())
                .zip(current_players.iter())
                .enumerate()
            {
                // Update rolling return for this player
                return_norm.update_return(env_idx, player, *reward);
                // Update variance stats (all turns are learner in self-play)
                return_norm.update_variance_stats(env_idx, player);
                // Normalize the reward
                *reward = return_norm.normalize(*reward);
                // Reset rolling return on episode end (after stats captured)
                if done {
                    return_norm.reset_player(env_idx, player);
                }
            }
        }

        // Flatten all player rewards [num_envs, num_players] -> [num_envs * num_players]
        // Use normalized acting player rewards to ensure consistency with single-player path
        let rewards_flat: Vec<f32> = {
            let mut flat = Vec::with_capacity(num_envs * num_players);
            for (env_idx, r) in player_rewards.iter().enumerate() {
                let acting_player = current_players[env_idx];
                for p in 0..num_players {
                    let reward = if p == acting_player {
                        // Use the (possibly normalized) acting reward
                        acting_rewards[env_idx]
                    } else {
                        // Non-acting players get their raw reward (typically 0)
                        r.get(p).copied().unwrap_or(0.0)
                    };
                    flat.push(reward);
                }
            }
            flat
        };

        // Append to CPU buffers
        all_obs.extend_from_slice(&obs_flat);
        all_actions.extend_from_slice(&actions_data);
        all_acting_rewards.extend_from_slice(&acting_rewards);
        all_dones.extend(dones.iter().map(|&d| if d { 1.0 } else { 0.0 }));
        all_acting_values.extend_from_slice(&acting_values_data);
        all_log_probs.extend_from_slice(&log_probs_data);

        // Multi-player data (per-player rewards for attribution)
        all_rewards_flat.extend_from_slice(&rewards_flat);
        all_acting_players.extend(current_players.iter().map(|&p| p as i64));

        // Update each player's last value when they act
        for (e, &player) in current_players.iter().enumerate() {
            last_value_per_player[e][player] = acting_values_data[e];
        }
    }

    // Batch transfer to GPU
    {
        profile_scope!("batch_gpu_transfer");
        buffer.observations = Tensor::<B, 1>::from_floats(all_obs.as_slice(), device)
            .reshape([num_steps, num_envs, obs_dim]);

        // Populate global states if CTDE is enabled
        if let Some(ref mut privileged_obs_tensor) = buffer.privileged_obs {
            let privileged_obs_dim = all_privileged_obs.len() / (num_steps * num_envs);
            *privileged_obs_tensor =
                Tensor::<B, 1>::from_floats(all_privileged_obs.as_slice(), device).reshape([
                    num_steps,
                    num_envs,
                    privileged_obs_dim,
                ]);
        }

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

        // Multi-player data (per-player rewards for attribution)
        buffer.all_rewards = Tensor::<B, 1>::from_floats(all_rewards_flat.as_slice(), device)
            .reshape([num_steps, num_envs, num_players]);
        buffer.acting_players =
            Tensor::<B, 1, Int>::from_ints(all_acting_players.as_slice(), device)
                .reshape([num_steps, num_envs]);

        // Store action masks if collected
        buffer.action_masks = all_action_masks.map(|masks| {
            Tensor::<B, 1>::from_floats(masks.as_slice(), device).reshape([
                num_steps,
                num_envs,
                num_actions,
            ])
        });
    }

    // Update normalizer stats at end of rollout with all RAW observations
    // This ensures stats are updated for the NEXT rollout, not the current one
    if let Some(norm) = normalizer.as_mut() {
        norm.update_batch(&raw_obs_for_stats, obs_dim);
    }

    (all_completed, last_value_per_player)
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
    /// Pre-computed placements for rating history (learner first, then opponents in order).
    /// Built before shuffle to ensure correct state access.
    pub rating_placements: Vec<usize>,
    /// Whether this env was playing against opponents (vs self-play)
    pub is_opponent_game: bool,
    /// Learner's seat position (captured before shuffle for win rate tracking)
    pub learner_position: usize,
    /// Map from seat position to opponent pool index (captured before shuffle)
    pub position_to_opponent: Vec<Option<usize>>,
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
    mut return_normalizer: Option<&mut ReturnNormalizer>,
    _current_step: usize,
    popart: Option<&PopArtNormalizer>,
    actual_player_count: Option<usize>,
) -> (
    Vec<EpisodeStats>,
    Vec<OpponentEpisodeCompletion>,
    Vec<Vec<f32>>,
) {
    profile_function!();
    let num_envs = vec_env.num_envs();
    let obs_dim = E::OBSERVATION_DIM;
    let num_players = E::NUM_PLAYERS;
    // Use actual_player_count for game logic (e.g., opponent positions) when available
    let game_num_players = actual_player_count.unwrap_or(num_players);
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

    // Multi-player data (only rewards need per-player tracking for attribution)
    let mut all_rewards_flat: Vec<f32> = Vec::with_capacity(num_steps * num_envs * num_players);
    let mut all_acting_players: Vec<i64> = Vec::with_capacity(num_steps * num_envs);

    // Track each player's last value prediction for GAE bootstrap
    // From each player's perspective, the rollout "ends" at their last action
    let mut last_value_per_player: Vec<Vec<f32>> = vec![vec![0.0; num_players]; num_envs];

    // Action masks (collected if environment provides them)
    let mut collected_action_masks: Option<Vec<f32>> = None;
    let mut stored_action_count: usize = 0;

    // Privileged observations for CTDE (collected if buffer has privileged_obs field)
    let collect_privileged_obs = buffer.privileged_obs.is_some();
    let mut all_privileged_obs: Vec<f32> = if collect_privileged_obs {
        // Get global state dim from first env
        let first_global = vec_env.get_privileged_obs();
        let privileged_obs_dim = first_global.len() / num_envs;
        Vec::with_capacity(num_steps * num_envs * privileged_obs_dim)
    } else {
        Vec::new()
    };

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

        // Get global states for CTDE (if enabled)
        if collect_privileged_obs {
            let privileged_obs_flat = vec_env.get_privileged_obs();
            all_privileged_obs.extend_from_slice(&privileged_obs_flat);
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

        // Get action masks for all environments
        let all_action_masks = vec_env.get_action_masks();
        let action_count = all_action_masks.as_ref().map_or(0, |m| m.len() / num_envs);

        // Collect action masks if environment provides them
        if let Some(ref masks) = all_action_masks {
            if collected_action_masks.is_none() {
                // First step: initialize collection with capacity
                stored_action_count = action_count;
                collected_action_masks =
                    Some(Vec::with_capacity(num_steps * num_envs * action_count));
            }
            // Convert bool to f32 and extend
            collected_action_masks
                .as_mut()
                .expect("collected_action_masks initialized above")
                .extend(masks.iter().map(|&v| if v { 1.0 } else { 0.0 }));
        }

        // Prepare actions for all envs (will be filled in by model forward passes)
        let mut all_env_actions: Vec<usize> = vec![0; num_envs];
        let mut all_env_values: Vec<f32> = vec![0.0; num_envs]; // Single scalar per env
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

            // Extract action masks for learner envs
            let learner_masks: Option<Vec<bool>> = all_action_masks.as_ref().map(|all_masks| {
                learner_env_indices
                    .iter()
                    .flat_map(|&env_idx| {
                        let start = env_idx * action_count;
                        all_masks[start..start + action_count].iter().copied()
                    })
                    .collect()
            });

            // Apply learner's normalizer
            if let Some(ref norm) = normalizer {
                norm.normalize_batch(&mut learner_obs, obs_dim);
            }

            let batch_size = learner_env_indices.len();
            let obs_tensor: Tensor<B, 2> =
                Tensor::<B, 1>::from_floats(learner_obs.as_slice(), device)
                    .reshape([batch_size, obs_dim]);

            // Handle CTDE networks: separate forward passes for actor and critic
            let (logits, values) = if model.is_ctde() {
                // Get full global states (all envs)
                let all_privileged_obs_flat = vec_env.get_privileged_obs();
                let privileged_obs_dim = all_privileged_obs_flat.len() / num_envs;

                // Extract global states for learner envs
                let learner_privileged_obs: Vec<f32> = learner_env_indices
                    .iter()
                    .flat_map(|&env_idx| {
                        let start = env_idx * privileged_obs_dim;
                        all_privileged_obs_flat[start..start + privileged_obs_dim]
                            .iter()
                            .copied()
                    })
                    .collect();

                let privileged_obs_tensor =
                    Tensor::<B, 1>::from_floats(learner_privileged_obs.as_slice(), device)
                        .reshape([batch_size, privileged_obs_dim]);
                let logits = model.forward_actor(obs_tensor.clone());
                let values = model.forward_critic(privileged_obs_tensor, obs_tensor);
                (logits, values)
            } else {
                model.forward(obs_tensor)
            };

            // Apply action mask to prevent sampling invalid actions
            let masked_logits = apply_action_mask(logits, learner_masks);
            let actions = sample_categorical(masked_logits.clone(), rng, device);
            let log_probs = log_prob_categorical(masked_logits, actions.clone());

            // Sync to CPU
            let actions_data: Vec<i64> = actions
                .float()
                .into_data()
                .to_vec::<f32>()
                .expect("actions to vec")
                .into_iter()
                .map(|x| x as i64)
                .collect();
            let mut values_data: Vec<f32> = values.into_data().to_vec().expect("values to vec");
            let log_probs_data: Vec<f32> =
                log_probs.into_data().to_vec().expect("log_probs to vec");

            // Denormalize values if PopArt is active
            // After rescaling, model outputs normalized values - convert back to raw for GAE
            if let Some(popart_norm) = popart {
                popart_norm.denormalize(&mut values_data);
            }

            // Scatter results back to env arrays (values are already scalars)
            for (batch_idx, &env_idx) in learner_env_indices.iter().enumerate() {
                all_env_actions[env_idx] = actions_data[batch_idx] as usize;
                all_env_log_probs[env_idx] = log_probs_data[batch_idx];
                all_env_values[env_idx] = values_data[batch_idx];
                // Update this player's last value for GAE bootstrap
                let player = current_players[env_idx];
                last_value_per_player[env_idx][player] = values_data[batch_idx];
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

            // Extract action masks for opponent envs
            let opp_masks: Option<Vec<bool>> = all_action_masks.as_ref().map(|all_masks| {
                env_indices
                    .iter()
                    .flat_map(|&env_idx| {
                        let start = env_idx * action_count;
                        all_masks[start..start + action_count].iter().copied()
                    })
                    .collect()
            });

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

            // Handle CTDE opponent models: separate forward passes for actor and critic
            let (logits, _values) = if opp_model.is_ctde() {
                // Get full global states (all envs)
                let all_privileged_obs_flat = vec_env.get_privileged_obs();
                let privileged_obs_dim = all_privileged_obs_flat.len() / num_envs;

                // Extract global states for opponent envs
                let opp_privileged_obs: Vec<f32> = env_indices
                    .iter()
                    .flat_map(|&env_idx| {
                        let start = env_idx * privileged_obs_dim;
                        all_privileged_obs_flat[start..start + privileged_obs_dim]
                            .iter()
                            .copied()
                    })
                    .collect();

                let privileged_obs_tensor =
                    Tensor::<B, 1>::from_floats(opp_privileged_obs.as_slice(), device)
                        .reshape([batch_size, privileged_obs_dim]);
                let logits = opp_model.forward_actor(obs_tensor.clone());
                let values = opp_model.forward_critic(privileged_obs_tensor, obs_tensor);
                (logits, values)
            } else {
                opp_model.forward(obs_tensor)
            };

            // Apply action mask and sample (opponents don't need log probs or values for training)
            let masked_logits = apply_action_mask(logits, opp_masks);
            let actions = sample_categorical(masked_logits, rng, device);

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
                    // IMPORTANT: Build rating_placements BEFORE shuffle modifies env_state
                    let env_state = &env_states[env_idx];
                    let opponent_indices: Vec<usize> = env_state
                        .position_to_opponent
                        .iter()
                        .filter_map(|&opt| opt)
                        .collect();

                    // Build rating placements while state is still valid
                    // Format: [learner_placement, opponent1_placement, opponent2_placement, ...]
                    let mut rating_placements = vec![places[env_state.learner_position]];
                    for (pos, slot) in env_state.position_to_opponent.iter().enumerate() {
                        if slot.is_some() {
                            rating_placements.push(places[pos]);
                        }
                    }

                    opponent_completions.push(OpponentEpisodeCompletion {
                        env_idx,
                        placements: places.clone(),
                        opponent_pool_indices: opponent_indices,
                        rating_placements,
                        is_opponent_game: true,
                        // Capture position state BEFORE shuffle for win rate tracking
                        learner_position: env_state.learner_position,
                        position_to_opponent: env_state.position_to_opponent.clone(),
                    });
                }

                // Resample opponents for next game (without replacement, qi-weighted)
                env_states[env_idx].assigned_opponents = opponent_pool.sample_all_slots();

                // Shuffle positions for next episode with new opponents
                env_states[env_idx].shuffle_positions(game_num_players, rng);
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
            let done = dones.get(env_idx).copied().unwrap_or(false);
            let mut acting_reward = player_rewards
                .get(env_idx)
                .and_then(|r| r.get(current_player))
                .copied()
                .unwrap_or(0.0);

            // Apply return normalization
            // Update rolling return for current player, but only update variance stats for learner turns
            if let Some(ref mut return_norm) = return_normalizer {
                // Update rolling return for this player in this env
                return_norm.update_return(env_idx, current_player, acting_reward);

                // Only update variance stats for learner turns
                // (opponent turn data should not influence normalization statistics)
                if is_learner_turn {
                    return_norm.update_variance_stats(env_idx, current_player);
                }

                // Normalize the reward
                acting_reward = return_norm.normalize(acting_reward);

                // Reset rolling return on episode end (after stats captured)
                if done {
                    return_norm.reset_player(env_idx, current_player);
                }
            }
            all_acting_rewards.push(acting_reward);

            // Value is already scalar (acting player's value)
            let acting_value = all_env_values.get(env_idx).copied().unwrap_or(0.0);
            all_acting_values.push(acting_value);

            all_dones.push(if done { 1.0 } else { 0.0 });

            // Multi-player data (per-player rewards for attribution)
            // Use normalized acting_reward for consistency
            let rewards_flat: Vec<f32> = (0..num_players)
                .map(|p| {
                    if p == current_player {
                        // Use the (possibly normalized) acting reward
                        acting_reward
                    } else {
                        // Non-acting players get raw reward (typically 0)
                        player_rewards
                            .get(env_idx)
                            .and_then(|r| r.get(p))
                            .copied()
                            .unwrap_or(0.0)
                    }
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

        // Populate global states if CTDE is enabled
        if let Some(ref mut privileged_obs_tensor) = buffer.privileged_obs {
            let privileged_obs_dim = all_privileged_obs.len() / (num_steps * num_envs);
            *privileged_obs_tensor =
                Tensor::<B, 1>::from_floats(all_privileged_obs.as_slice(), device).reshape([
                    num_steps,
                    num_envs,
                    privileged_obs_dim,
                ]);
        }

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

        // Multi-player data (per-player rewards for attribution)
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

        // Store action masks if collected
        buffer.action_masks = collected_action_masks.map(|masks| {
            Tensor::<B, 1>::from_floats(masks.as_slice(), device).reshape([
                num_steps,
                num_envs,
                stored_action_count,
            ])
        });
    }

    // Update normalizer stats
    if let Some(norm) = normalizer.as_mut() {
        norm.update_batch(&raw_obs_for_stats, obs_dim);
    }

    (all_completed, opponent_completions, last_value_per_player)
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
///
/// # Bootstrap values
/// `last_value_per_player` contains each player's last value prediction from the rollout.
/// From each player's perspective, the rollout "ends" at their last action.
/// Players who never acted in the rollout have 0 bootstrap (conservative: no future estimate).
#[expect(clippy::cast_sign_loss, reason = "tensor indices are non-negative")]
pub fn compute_gae_multiplayer<B: Backend>(
    buffer: &mut RolloutBuffer<B>,
    last_value_per_player: &[Vec<f32>], // [num_envs][num_players] - each player's last value
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
    // Values are already scalar per step (acting player's value)
    let values_data: Vec<f32> = buffer.values.clone().into_data().to_vec().expect("values");
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

    // Pass 1: Attribute cumulative rewards to acting player
    // Walk backwards, accumulating rewards for each player until they act
    let mut attributed_rewards = vec![0.0_f32; num_steps * num_envs];
    let mut reward_carry = vec![vec![0.0_f32; num_players]; num_envs];

    for t in (0..num_steps).rev() {
        for e in 0..num_envs {
            let idx = t * num_envs + e;
            let acting_player = acting_players_data[idx];
            let done = dones_data[idx];

            // Reset BEFORE processing: clears rewards from future episodes
            // (already processed in our backwards iteration)
            if done > 0.5 {
                reward_carry[e].fill(0.0);
            }

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
        }
    }

    // Pass 2: Compute GAE using per-player value chains
    // Note: Even with single value output, we still track per-player GAE carry
    // because different players may have different "continuation" values when
    // the acting player changes between steps.
    let mut advantages = vec![0.0_f32; num_steps * num_envs];
    let mut gae_carry = vec![vec![0.0_f32; num_players]; num_envs];

    // Initialize next_value from each player's last value prediction
    // From each player's perspective, the rollout "ends" at their last action.
    // Players who never acted have 0 (conservative: no future estimate beyond rollout).
    let mut next_value: Vec<Vec<f32>> = last_value_per_player.to_vec();

    for t in (0..num_steps).rev() {
        for e in 0..num_envs {
            let idx = t * num_envs + e;
            let player = acting_players_data[idx];
            let reward = attributed_rewards[idx];
            // Value is already scalar (acting player's value at this step)
            let value = values_data[idx];
            let done = dones_data[idx];

            // Reset BEFORE processing: clears carry from future episodes
            // (already processed in our backwards iteration)
            if done > 0.5 {
                gae_carry[e].fill(0.0);
                // Reset next_value for non-acting players only:
                // - Acting player: keep for bootstrapping earlier same-player steps
                // - Others: reset to prevent bleed from future episodes (their future
                //   rewards are already in attributed_rewards via reward carry)
                for (p, nv) in next_value[e].iter_mut().enumerate() {
                    if p != player {
                        *nv = 0.0;
                    }
                }
            }

            // TD error using this player's next value
            let delta = (gamma * next_value[e][player]).mul_add(1.0 - done, reward) - value;

            // GAE for this player
            let advantage =
                (gamma * gae_lambda * (1.0 - done)).mul_add(gae_carry[e][player], delta);
            advantages[idx] = advantage;

            // Update carry for this player
            gae_carry[e][player] = advantage;
            next_value[e][player] = value;
        }
    }

    // Store advantages
    let advantages_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(advantages.as_slice(), device).reshape([num_steps, num_envs]);

    // Compute returns = advantages + value (already scalar per step)
    let values_tensor = buffer.values.clone();

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

/// Raw tensor values extracted during loss computation for metrics
///
/// All tensors are `InnerBackend` (no autodiff overhead).
/// Used to compute metrics AFTER `backward()` when no autodiff tensors exist.
struct RawMinibatchValues<B: Backend> {
    // From policy computation (needed for approx_kl, clip_fraction)
    log_ratio: Tensor<B, 1>,
    ratio: Tensor<B, 1>,
    entropy: Tensor<B, 1>,

    // From forward pass (needed for value_error stats)
    values: Tensor<B, 1>,

    // Targets passed through (needed for value_error computation)
    returns: Tensor<B, 1>,

    // Per-sample count of valid actions (only when action masks present)
    valid_counts: Option<Tensor<B, 1>>,
}

/// Scalar metrics extracted from a single minibatch update
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
    // Diagnostic metrics
    adv_mean_raw: f32,
    adv_std_raw: f32,
    adv_min_raw: f32,
    adv_max_raw: f32,
    value_error_mean: f32,
    value_error_std: f32,
    value_error_max: f32,
    // Valid action metrics (only when action masks present)
    avg_valid_actions: Option<f32>,
    entropy_valid_pct: Option<f32>,
}

/// Training metrics from a single update
#[derive(Debug, Clone)]
pub struct UpdateMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub entropy_scaled: f32,
    pub approx_kl: f32,
    pub clip_fraction: f32,
    // Additional metrics for debugging
    pub explained_variance: f32,
    pub total_loss: f32,
    pub value_mean: f32,
    pub returns_mean: f32,
    // Diagnostic metrics
    pub adv_mean_raw: f32,
    pub adv_std_raw: f32,
    pub adv_min_raw: f32,
    pub adv_max_raw: f32,
    pub value_error_mean: f32,
    pub value_error_std: f32,
    pub value_error_max: f32,
    // Value normalization metrics (only set when normalize_values enabled)
    pub value_norm_target_mean: Option<f32>,
    pub value_norm_target_std: Option<f32>,
    pub value_norm_rescale_mag: Option<f32>,
    // Valid action metrics (only when action masks present)
    pub avg_valid_actions: Option<f32>,
    pub entropy_valid_pct: Option<f32>,
}

/// Compute PPO loss with minimal autodiff scope
///
/// All intermediate autodiff tensors are dropped when this function returns.
/// Only `loss` escapes as autodiff for backward pass.
///
/// Returns:
/// - `loss`: Autodiff tensor for backward pass (0-dim scalar)
/// - `policy_loss_scalar`: Policy loss value for metrics
/// - `value_loss_scalar`: Value loss value for metrics
/// - `raw`: Inner tensors for computing remaining metrics after backward
#[expect(
    clippy::too_many_arguments,
    reason = "minibatch data requires many inputs"
)]
fn compute_minibatch_loss<B: burn::tensor::backend::AutodiffBackend>(
    model: &ActorCritic<B>,
    mb_obs: Tensor<B::InnerBackend, 2>,
    mb_privileged_obs: Option<Tensor<B::InnerBackend, 2>>,
    mb_actions: Tensor<B::InnerBackend, 1, Int>,
    mb_old_log_probs: Tensor<B::InnerBackend, 1>,
    mb_advantages_normalized: Tensor<B::InnerBackend, 1>,
    mb_returns: Tensor<B::InnerBackend, 1>,
    mb_old_values: Tensor<B::InnerBackend, 1>,
    _mb_acting_players: &Tensor<B::InnerBackend, 1, Int>,
    mb_action_masks: Option<Tensor<B::InnerBackend, 2>>,
    config: &Config,
    entropy_coef: f64,
    num_players: usize,
) -> (
    Tensor<B, 1>,                        // loss (autodiff) - scalar as 1D tensor
    f32,                                 // policy_loss_scalar
    f32,                                 // value_loss_scalar
    RawMinibatchValues<B::InnerBackend>, // raw values for metrics
)
where
    B::FloatElem: Into<f32>,
{
    // Forward pass (creates autodiff tensors)
    // For CTDE: use separate actor/critic forward passes
    // For non-CTDE: use standard forward pass
    let (logits, values_2d) = if model.is_ctde() {
        let local_obs_autodiff = Tensor::from_inner(mb_obs);
        let privileged_obs_autodiff =
            Tensor::from_inner(mb_privileged_obs.expect("CTDE requires privileged_obs in buffer"));
        let logits = model.forward_actor(local_obs_autodiff.clone());
        let values = model.forward_critic(privileged_obs_autodiff, local_obs_autodiff);
        (logits, values)
    } else {
        model.forward(Tensor::from_inner(mb_obs))
    };

    // Values are already scalar per sample [batch, 1] -> flatten to [batch]
    let mb_size = logits.dims()[0];
    let _ = num_players; // unused with single value output
    let values: Tensor<B, 1> = values_2d.slice([0..mb_size, 0..1]).flatten(0, 1);

    // IMMEDIATELY extract inner for metrics (before loss uses it)
    let values_inner = values.clone().inner();

    // Compute per-sample valid action counts before consuming the mask
    let valid_counts = mb_action_masks
        .as_ref()
        .map(|mask| mask.clone().sum_dim(1).squeeze_dims(&[1]));

    // Apply action mask
    let masked_logits = if let Some(mask) = mb_action_masks {
        let mask_additive = (mask - 1.0) * 1e9;
        logits + Tensor::from_inner(mask_additive)
    } else {
        logits
    };

    // Policy computations - extract inner IMMEDIATELY after each
    let new_log_probs = log_prob_categorical(masked_logits.clone(), Tensor::from_inner(mb_actions));

    let entropy = entropy_categorical(masked_logits);
    let entropy_inner = entropy.clone().inner();

    let log_ratio = new_log_probs - Tensor::from_inner(mb_old_log_probs);
    let log_ratio_inner = log_ratio.clone().inner();

    let ratio = log_ratio.exp();
    let ratio_inner = ratio.clone().inner();

    // Policy loss (clipped surrogate)
    let neg_advantages: Tensor<B, 1> = Tensor::from_inner(-mb_advantages_normalized);
    let policy_loss_1 = neg_advantages.clone() * ratio.clone();
    let policy_loss_2 =
        neg_advantages * ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon);
    let policy_loss: Tensor<B, 1> = policy_loss_1.max_pair(policy_loss_2);
    let policy_loss_mean = policy_loss.mean();

    // Extract scalar BEFORE combining into total loss
    let policy_loss_scalar: f32 = policy_loss_mean.clone().inner().into_scalar().into();

    // Value loss
    let value_loss = if config.clip_value {
        let old_values: Tensor<B, 1> = Tensor::from_inner(mb_old_values);
        let values_clipped = old_values.clone()
            + (values.clone() - old_values).clamp(-config.clip_epsilon, config.clip_epsilon);
        let value_loss_1 = (values - Tensor::from_inner(mb_returns.clone())).powf_scalar(2.0);
        let value_loss_2 =
            (values_clipped - Tensor::from_inner(mb_returns.clone())).powf_scalar(2.0);
        value_loss_1.max_pair(value_loss_2).mean() * 0.5
    } else {
        (values - Tensor::from_inner(mb_returns.clone()))
            .powf_scalar(2.0)
            .mean()
            * 0.5
    };

    // Extract scalar BEFORE combining into total loss
    let value_loss_scalar: f32 = value_loss.clone().inner().into_scalar().into();

    // Entropy bonus and combined loss
    let entropy_loss = -entropy.mean() * entropy_coef;
    let loss = policy_loss_mean + value_loss * config.value_coef + entropy_loss;

    // Package raw values for metrics (all InnerBackend)
    let raw = RawMinibatchValues {
        log_ratio: log_ratio_inner,
        ratio: ratio_inner,
        entropy: entropy_inner,
        values: values_inner,
        returns: mb_returns,
        valid_counts,
    };

    // All intermediate autodiff tensors are DROPPED when this function returns.
    // Only `loss` escapes as autodiff.
    (loss, policy_loss_scalar, value_loss_scalar, raw)
}

/// Compute metrics from raw inner tensors (no autodiff)
///
/// Called AFTER `backward()` when no autodiff tensors exist.
fn compute_minibatch_metrics<B: Backend>(
    raw: RawMinibatchValues<B>,
    // Pre-extracted scalars from loss computation
    policy_loss: f32,
    value_loss: f32,
    total_loss: f32,
    // Advantage stats (computed before normalization)
    adv_mean_raw: f32,
    adv_std_raw: f32,
    adv_min_raw: f32,
    adv_max_raw: f32,
    clip_epsilon: f32,
) -> MinibatchMetrics
where
    B::FloatElem: Into<f32>,
{
    // Approx KL: E[(ratio - 1) - log_ratio]
    let approx_kl: f32 = ((raw.ratio.clone() - 1.0) - raw.log_ratio)
        .mean()
        .into_scalar()
        .into();

    // Clip fraction: fraction of samples where ratio was clipped
    let clip_fraction: f32 = (raw.ratio - 1.0)
        .abs()
        .greater_elem(clip_epsilon)
        .float()
        .mean()
        .into_scalar()
        .into();

    // Value error stats: |V(s) - returns|
    let value_errors = (raw.values.clone() - raw.returns.clone()).abs();
    let value_error_mean: f32 = value_errors.clone().mean().into_scalar().into();
    let value_error_std: f32 = value_errors.clone().var(0).sqrt().into_scalar().into();
    let value_error_max: f32 = value_errors.max().into_scalar().into();

    // Entropy mean
    let entropy_per_sample = raw.entropy;
    let entropy: f32 = entropy_per_sample.clone().mean().into_scalar().into();

    // Valid action metrics (only when action masks present)
    let (avg_valid_actions, entropy_valid_pct) = if let Some(valid_counts) = raw.valid_counts {
        let avg: f32 = valid_counts.clone().mean().into_scalar().into();

        // Per-sample entropy/ln(valid_count), excluding forced moves (valid_count <= 1)
        let max_ent = valid_counts.clone().log();
        let has_choice = valid_counts.greater_elem(1.0).float();
        let choice_count: f32 = has_choice.clone().sum().into_scalar().into();

        let pct = if choice_count > 0.0 {
            let ratio = entropy_per_sample * has_choice.clone() / max_ent.clamp_min(1e-8);
            let ratio_sum: f32 = ratio.sum().into_scalar().into();
            ratio_sum / choice_count
        } else {
            0.0
        };
        (Some(avg), Some(pct))
    } else {
        (None, None)
    };

    // Value and returns means
    let value_mean: f32 = raw.values.mean().into_scalar().into();
    let returns_mean: f32 = raw.returns.mean().into_scalar().into();

    MinibatchMetrics {
        policy_loss,
        value_loss,
        entropy,
        approx_kl,
        clip_fraction,
        total_loss,
        value_mean,
        returns_mean,
        adv_mean_raw,
        adv_std_raw,
        adv_min_raw,
        adv_max_raw,
        value_error_mean,
        value_error_std,
        value_error_max,
        avg_valid_actions,
        entropy_valid_pct,
    }
}

/// Rescale value head weights for value normalization
///
/// When normalization statistics change, we rescale the value head to preserve
/// output semantics. This ensures value predictions when denormalized remain
/// consistent, preventing catastrophic forgetting when normalization statistics shift.
///
/// With single value output, the value head outputs a scalar per sample, so we
/// use player 0's statistics (unified across all players).
pub fn rescale_value_head_for_popart<B: Backend>(
    model: ActorCritic<B>,
    old_means: &[f64],
    old_stds: &[f64],
    new_means: &[f64],
    new_stds: &[f64],
    _num_players: usize, // Kept for API compatibility but not used (single value output)
) -> ActorCritic<B> {
    let value_head = model.value_head();
    let device = value_head.weight.device();

    // Get weight shape and data
    // weight shape: [input_dim, 1] (single value output)
    let weight = value_head.weight.val();
    let weight_shape = weight.shape();
    let input_dim = weight_shape.dims[0];
    let output_dim = weight_shape.dims[1]; // Should be 1 for single value output
    let weight_data: Vec<f32> = weight.into_data().to_vec().expect("weight data");

    // Compute scale factor using player 0's stats (unified for single value output)
    let scale = old_stds[0] / new_stds[0];

    // Rescale weights: W_new = W_old * scale
    let new_weight_data: Vec<f32> = weight_data
        .iter()
        .map(|&w| (f64::from(w) * scale) as f32)
        .collect();

    // Compute new bias: b_new = (b_old * σ_old + μ_old - μ_new) / σ_new
    let bias_data: Vec<f32> = value_head.bias.as_ref().map_or_else(
        || vec![0.0; output_dim],
        |b| b.val().into_data().to_vec().expect("bias data"),
    );

    let new_bias_data: Vec<f32> = bias_data
        .iter()
        .map(|&b_old| {
            ((f64::from(b_old) * old_stds[0] + old_means[0] - new_means[0]) / new_stds[0]) as f32
        })
        .collect();

    // Create fresh tensors from the computed data (these are leaf tensors)
    let new_weight_1d: Tensor<B, 1> = Tensor::from_floats(new_weight_data.as_slice(), &device);
    let new_weight: Tensor<B, 2> = new_weight_1d.reshape([input_dim, output_dim]);
    let new_bias: Tensor<B, 1> = Tensor::from_floats(new_bias_data.as_slice(), &device);

    // Use map() to preserve ParamIds - this keeps optimizer momentum/velocity state
    // intact instead of creating orphaned entries that accumulate over training
    let new_weight_param = value_head.weight.clone().map(|_| new_weight);
    let new_bias_param = value_head.bias.clone().map(|p| p.map(|_| new_bias));

    let new_value_head = burn::nn::Linear {
        weight: new_weight_param,
        bias: new_bias_param,
    };

    model.with_value_head(new_value_head)
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
    num_actions: usize,
    rng: &mut impl Rng,
    popart: Option<&mut PopArtNormalizer>,
) -> (ActorCritic<B>, UpdateMetrics)
where
    B::FloatElem: Into<f32>,
{
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
        privileged_obs_inner,
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
        privileged_obs_inner,
    ) = if let Some(indices) = valid_indices {
        (
            obs_inner.select(0, indices.clone()),
            actions_inner.select(0, indices.clone()),
            old_log_probs_inner.select(0, indices.clone()),
            advantages_inner.select(0, indices.clone()),
            returns_inner.select(0, indices.clone()),
            acting_players_inner.select(0, indices.clone()),
            old_values_inner.select(0, indices.clone()),
            privileged_obs_inner.map(|gs| gs.select(0, indices)),
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
            privileged_obs_inner,
        )
    };

    // Flatten and filter action masks (if present)
    // Shape: [num_steps * num_envs, num_actions] -> after filtering -> [batch_size, num_actions]
    let action_masks_inner: Option<Tensor<B::InnerBackend, 2>> =
        buffer.action_masks.as_ref().map(|masks| {
            let [num_steps, num_envs, num_actions] = masks.dims();
            let flattened = masks.clone().reshape([num_steps * num_envs, num_actions]);

            // Filter with valid_indices if opponent pool training
            if let Some(ref mask) = buffer.valid_mask {
                let mask_flat: Tensor<B::InnerBackend, 1> = mask.clone().flatten(0, 1);
                let mask_data: Vec<f32> = mask_flat.into_data().to_vec().expect("mask data");
                let indices: Vec<i64> = mask_data
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v > 0.5)
                    .map(|(i, _)| i as i64)
                    .collect();
                let device = masks.device();
                let indices_tensor =
                    Tensor::<B::InnerBackend, 1, Int>::from_ints(indices.as_slice(), &device);
                flattened.select(0, indices_tensor)
            } else {
                flattened
            }
        });

    let batch_size = obs_inner.dims()[0];

    // Accumulators for metrics
    let mut total_policy_loss = 0.0;
    let mut total_value_loss = 0.0;
    let mut total_entropy = 0.0;
    let mut total_approx_kl = 0.0;
    let mut total_clip_fraction = 0.0;
    let mut total_loss_sum = 0.0;
    let mut total_value_mean = 0.0;
    let mut total_returns_mean = 0.0;
    let mut total_adv_mean_raw = 0.0;
    let mut total_adv_std_raw = 0.0;
    let mut total_adv_min_raw = f32::INFINITY;
    let mut total_adv_max_raw = f32::NEG_INFINITY;
    let mut total_value_error_mean = 0.0;
    let mut total_value_error_std = 0.0;
    let mut total_value_error_max = f32::NEG_INFINITY;
    let mut total_avg_valid_actions = 0.0f32;
    let mut total_entropy_valid_pct = 0.0f32;
    let mut has_mask_metrics = false;
    let mut num_updates = 0;

    let mut model = model;

    // PopArt: Update statistics and rescale value head before training
    let mut popart = popart;
    let mut value_norm_rescale_mag: Option<f32> = None;
    let mut value_norm_target_sum = 0.0f64;
    let mut value_norm_target_sq_sum = 0.0f64;
    let mut value_norm_target_count = 0usize;

    if let Some(popart_norm) = popart.as_mut() {
        // Extract returns and acting players from flattened buffer
        let returns_data: Vec<f32> = returns_inner
            .clone()
            .into_data()
            .to_vec()
            .expect("returns data");
        let acting_players_data: Vec<usize> = acting_players_inner
            .clone()
            .into_data()
            .to_vec::<i64>()
            .expect("acting_players data")
            .into_iter()
            .map(|x| usize::try_from(x).expect("non-negative player index"))
            .collect();

        // Update statistics (returns old values for rescaling)
        let (old_means, old_stds) = popart_norm.update(&returns_data, &acting_players_data);

        // Rescale value head if initialized
        if popart_norm.is_initialized() {
            let new_means: Vec<f64> = (0..num_players).map(|p| popart_norm.mean(p)).collect();
            let new_stds: Vec<f64> = (0..num_players).map(|p| popart_norm.std(p)).collect();

            // Compute rescale magnitude (max ratio of old_std/new_std)
            let max_rescale = old_stds
                .iter()
                .zip(new_stds.iter())
                .map(|(&old, &new)| (old / new).abs())
                .fold(0.0f64, f64::max);
            value_norm_rescale_mag = Some(max_rescale as f32);

            model = rescale_value_head_for_popart(
                model,
                &old_means,
                &old_stds,
                &new_means,
                &new_stds,
                num_players,
            );
        }
    }

    // Epoch loop (labeled for KL early stopping)
    'epoch_loop: for _epoch in 0..config.num_epochs {
        profile_scope!("ppo_epoch");

        // Shuffle indices for minibatches
        let mut indices: Vec<usize> = (0..batch_size).collect();
        indices.shuffle(rng);

        // Minibatch loop - distribute samples evenly across minibatches
        // Example: 893 samples / 4 minibatches = 224, 223, 223, 223
        let base_mb_size = batch_size / config.num_minibatches;
        let remainder = batch_size % config.num_minibatches;

        let mut mb_start = 0;
        for mb_idx in 0..config.num_minibatches {
            profile_scope!("ppo_minibatch");
            // First `remainder` batches get one extra sample
            let mb_size = base_mb_size + usize::from(mb_idx < remainder);
            if mb_size == 0 {
                continue; // Skip if no samples (only if batch_size < num_minibatches)
            }
            let mb_end = mb_start + mb_size;

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
            let mb_returns_inner_raw = returns_inner.clone().select(0, mb_indices_inner.clone());
            let mb_acting_players_inner = acting_players_inner
                .clone()
                .select(0, mb_indices_inner.clone());
            let mb_old_values_inner = old_values_inner.clone().select(0, mb_indices_inner.clone());
            let mb_privileged_obs_inner = privileged_obs_inner
                .as_ref()
                .map(|gs| gs.clone().select(0, mb_indices_inner.clone()));

            // PopArt: Normalize returns and old_values for value loss computation
            // Buffer stores RAW values (denormalized during collection), but value loss
            // should be computed in normalized space for stability
            let (mb_returns_inner, mb_old_values_inner) = if let Some(ref popart_norm) = popart {
                let returns_vec: Vec<f32> = mb_returns_inner_raw
                    .clone()
                    .into_data()
                    .to_vec()
                    .expect("returns data");
                let old_values_vec: Vec<f32> = mb_old_values_inner
                    .clone()
                    .into_data()
                    .to_vec()
                    .expect("old_values data");
                let players_vec: Vec<usize> = mb_acting_players_inner
                    .clone()
                    .into_data()
                    .to_vec::<i64>()
                    .expect("players data")
                    .into_iter()
                    .map(|x| usize::try_from(x).expect("non-negative player index"))
                    .collect();

                let normalized_returns = popart_norm.normalize(&returns_vec, &players_vec);
                let normalized_old_values = popart_norm.normalize(&old_values_vec, &players_vec);

                // Track normalized target statistics
                for &v in &normalized_returns {
                    value_norm_target_sum += f64::from(v);
                    value_norm_target_sq_sum += f64::from(v) * f64::from(v);
                    value_norm_target_count += 1;
                }

                let device = mb_returns_inner_raw.device();
                (
                    Tensor::<B::InnerBackend, 1>::from_floats(
                        normalized_returns.as_slice(),
                        &device,
                    ),
                    Tensor::<B::InnerBackend, 1>::from_floats(
                        normalized_old_values.as_slice(),
                        &device,
                    ),
                )
            } else {
                (mb_returns_inner_raw, mb_old_values_inner)
            };

            // Select minibatch action masks (if present)
            let mb_action_masks_inner: Option<Tensor<B::InnerBackend, 2>> = action_masks_inner
                .as_ref()
                .map(|masks| masks.clone().select(0, mb_indices_inner));

            // Capture raw advantage stats as f32 scalars (InnerBackend - no autodiff overhead)
            let adv_mean_raw: f32 = mb_advantages_inner.clone().mean().into_scalar().into();
            let adv_std_raw: f32 = mb_advantages_inner
                .clone()
                .var(0)
                .sqrt()
                .into_scalar()
                .into();
            let adv_min_raw: f32 = mb_advantages_inner.clone().min().into_scalar().into();
            let adv_max_raw: f32 = mb_advantages_inner.clone().max().into_scalar().into();

            // Normalize advantages at minibatch level (InnerBackend - no autodiff)
            let mb_advantages_normalized =
                normalize_advantages::<B::InnerBackend>(mb_advantages_inner);

            // ═══════════════════════════════════════════════════════════════════
            // PHASE 1: Loss computation (autodiff scope isolated in function)
            // All intermediate autodiff tensors are dropped when function returns.
            // ═══════════════════════════════════════════════════════════════════
            let (loss, policy_loss_scalar, value_loss_scalar, raw_values) = {
                profile_scope!("loss_computation");
                let out = compute_minibatch_loss(
                    &model,
                    mb_obs_inner,
                    mb_privileged_obs_inner,
                    mb_actions_inner,
                    mb_old_log_probs_inner,
                    mb_advantages_normalized,
                    mb_returns_inner,
                    mb_old_values_inner,
                    &mb_acting_players_inner,
                    mb_action_masks_inner,
                    config,
                    entropy_coef,
                    num_players,
                );
                #[cfg(feature = "tracy")]
                let _ = B::sync(&device);
                out
            };
            // All intermediate autodiff tensors dropped!
            // Only `loss` is autodiff now.

            // Extract total loss scalar before backward
            let total_loss_scalar: f32 = loss.clone().inner().into_scalar().into();

            // ═══════════════════════════════════════════════════════════════════
            // PHASE 2: Backward pass (consumes loss)
            // ═══════════════════════════════════════════════════════════════════
            let grads = {
                profile_scope!("backward");
                let grads = loss.backward();
                #[cfg(feature = "tracy")]
                let _ = B::sync(&device);
                grads
            };
            // Now ZERO autodiff tensors exist!

            // ═══════════════════════════════════════════════════════════════════
            // PHASE 3: Compute metrics from inner tensors
            // ═══════════════════════════════════════════════════════════════════
            let metrics = {
                profile_scope!("compute_metrics");
                compute_minibatch_metrics(
                    raw_values,
                    policy_loss_scalar,
                    value_loss_scalar,
                    total_loss_scalar,
                    adv_mean_raw,
                    adv_std_raw,
                    adv_min_raw,
                    adv_max_raw,
                    config.clip_epsilon as f32,
                )
            };

            // Optimizer step
            model = {
                profile_scope!("optimizer_step");
                let grads = GradientsParams::from_grads(grads, &model);
                let updated = optimizer.step(learning_rate, model, grads);
                #[cfg(feature = "tracy")]
                let _ = B::sync(&device);
                updated
            };

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
                total_adv_mean_raw += metrics.adv_mean_raw;
                total_adv_std_raw += metrics.adv_std_raw;
                total_adv_min_raw = total_adv_min_raw.min(metrics.adv_min_raw);
                total_adv_max_raw = total_adv_max_raw.max(metrics.adv_max_raw);
                total_value_error_mean += metrics.value_error_mean;
                total_value_error_std += metrics.value_error_std;
                total_value_error_max = total_value_error_max.max(metrics.value_error_max);
                if let Some(avg) = metrics.avg_valid_actions {
                    total_avg_valid_actions += avg;
                    has_mask_metrics = true;
                }
                if let Some(pct) = metrics.entropy_valid_pct {
                    total_entropy_valid_pct += pct;
                }
                num_updates += 1;

                // KL early stopping: stop epoch if KL divergence exceeds threshold
                if let Some(target) = config.target_kl {
                    if metrics.approx_kl > target as f32 {
                        break 'epoch_loop;
                    }
                }
            }

            mb_start = mb_end;
        }
    }

    // Compute explained variance from buffer, filtering by valid_mask if present
    // (opponent pool training includes opponent turns that shouldn't be in the metric)
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

    let explained_variance = if let Some(ref mask) = buffer.valid_mask {
        // Filter to only learner turns (valid_mask > 0.5)
        let mask_data: Vec<f32> = mask.clone().into_data().to_vec().expect("mask to vec");
        let (filtered_values, filtered_returns): (Vec<f32>, Vec<f32>) = mask_data
            .iter()
            .zip(values_data.iter().zip(returns_data.iter()))
            .filter(|(&m, _)| m > 0.5)
            .map(|(_, (&v, &r))| (v, r))
            .unzip();
        compute_explained_variance(&filtered_values, &filtered_returns)
    } else {
        compute_explained_variance(&values_data, &returns_data)
    };

    // Compute value normalization target stats
    let (value_norm_target_mean, value_norm_target_std) = if value_norm_target_count > 0 {
        let mean = value_norm_target_sum / value_norm_target_count as f64;
        let variance = value_norm_target_sq_sum / value_norm_target_count as f64 - mean * mean;
        let std = variance.max(0.0).sqrt();
        (Some(mean as f32), Some(std as f32))
    } else {
        (None, None)
    };

    // Average metrics
    let entropy = total_entropy / num_updates as f32;
    let max_entropy = (num_actions as f32).ln();
    let metrics = UpdateMetrics {
        policy_loss: total_policy_loss / num_updates as f32,
        value_loss: total_value_loss / num_updates as f32,
        entropy,
        entropy_scaled: entropy / max_entropy,
        approx_kl: total_approx_kl / num_updates as f32,
        clip_fraction: total_clip_fraction / num_updates as f32,
        explained_variance,
        total_loss: total_loss_sum / num_updates as f32,
        value_mean: total_value_mean / num_updates as f32,
        returns_mean: total_returns_mean / num_updates as f32,
        adv_mean_raw: total_adv_mean_raw / num_updates as f32,
        adv_std_raw: total_adv_std_raw / num_updates as f32,
        adv_min_raw: total_adv_min_raw,
        adv_max_raw: total_adv_max_raw,
        value_error_mean: total_value_error_mean / num_updates as f32,
        value_error_std: total_value_error_std / num_updates as f32,
        value_error_max: total_value_error_max,
        value_norm_target_mean,
        value_norm_target_std,
        value_norm_rescale_mag,
        avg_valid_actions: if has_mask_metrics {
            Some(total_avg_valid_actions / num_updates as f32)
        } else {
            None
        },
        entropy_valid_pct: if has_mask_metrics {
            Some(total_entropy_valid_pct / num_updates as f32)
        } else {
            None
        },
    };

    // Cleanup memory in hopes of avoiding any leaks.
    let _ = B::sync(&device);
    B::memory_cleanup(&device);

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
        let buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(128, 4, 4, None, 1u8, &device);

        assert_eq!(buffer.observations.dims(), [128, 4, 4]);
        assert_eq!(buffer.actions.dims(), [128, 4]);
        assert_eq!(buffer.rewards.dims(), [128, 4]);
        assert_eq!(buffer.values.dims(), [128, 4]); // Single value per step
        assert_eq!(buffer.all_rewards.dims(), [128, 4, 1]);
        assert_eq!(buffer.acting_players.dims(), [128, 4]);
    }

    #[test]
    fn test_rollout_buffer_multiplayer() {
        let device = Default::default();
        let buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(64, 8, 86, None, 2u8, &device);

        assert_eq!(buffer.observations.dims(), [64, 8, 86]);
        assert_eq!(buffer.values.dims(), [64, 8]); // Single value per step (not per-player)
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
            RolloutBuffer::new(num_steps, num_envs, 1, None, num_players, &device);

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
            RolloutBuffer::new(num_steps, num_envs, 86, None, num_players, &device);

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
        // Values: acting player's value estimate at each step (uniform 0.5 for all)
        buffer.values =
            Tensor::from_floats([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], &device);
        // Alternating players: P0, P1, P0, P1
        buffer.acting_players = Tensor::from_ints([[0, 0], [1, 1], [0, 0], [1, 1]], &device);
        buffer.dones =
            Tensor::from_floats([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], &device);

        // Terminal state - bootstrap values don't matter, use zeros
        // Each player's last value from the rollout (P0 at step 2, P1 at step 3)
        // Terminal so these don't affect the result
        let last_value_per_player = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        compute_gae_multiplayer(
            &mut buffer,
            &last_value_per_player,
            0.99,
            0.95,
            num_players,
            &device,
        );

        assert!(buffer.advantages.is_some());
        assert!(buffer.returns.is_some());
    }

    #[test]
    fn test_gae_multiplayer_same_player_consecutive() {
        // P0 acts twice in a row, ending the episode
        // Verifies bootstrapping works for same-player chains

        let device = Default::default();
        let gamma = 0.99_f32;
        let lambda = 0.95_f32;

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(2, 1, 4, None, 2, &device);

        // P0 acts both steps, step 1 is terminal
        buffer.dones = Tensor::from_floats([[0.0], [1.0]], &device);
        buffer.acting_players = Tensor::from_ints([[0], [0]], &device);

        // P0 gets +1 reward at terminal step
        buffer.all_rewards = Tensor::from_floats([[[0.0, 0.0]], [[1.0, 0.0]]], &device);

        // Values: acting player's value at each step (P0 acts both steps)
        buffer.values = Tensor::from_floats(
            [[0.5], [0.8]], // P0 expects 0.5 at step 0, 0.8 at terminal
            &device,
        );

        // Terminal state - P0's last value was 0.8 at step 1, P1 never acted (gets 0)
        let last_value_per_player = vec![vec![0.8, 0.0]]; // [env][player]
        compute_gae_multiplayer(
            &mut buffer,
            &last_value_per_player,
            gamma,
            lambda,
            2,
            &device,
        );

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();

        // Step 1 (terminal): delta = 1.0 - 0.8 = 0.2
        let expected_step1 = 1.0 - 0.8;
        assert!(
            (advs[1] - expected_step1).abs() < 1e-5,
            "Step 1: expected {}, got {}",
            expected_step1,
            advs[1]
        );

        // Step 0: delta = 0.0 + gamma * V1 - V0 = 0.99 * 0.8 - 0.5 = 0.292
        // advantage = delta + gamma * lambda * A1 = 0.292 + 0.99 * 0.95 * 0.2
        let delta0 = gamma * 0.8 - 0.5;
        let expected_step0 = delta0 + gamma * lambda * expected_step1;
        assert!(
            (advs[0] - expected_step0).abs() < 1e-5,
            "Step 0: expected {}, got {}",
            expected_step0,
            advs[0]
        );
    }

    #[test]
    fn test_gae_multiplayer_different_player_terminal_no_bleed() {
        // Episode 1: P0 acts, P1 acts (terminal)
        // Episode 2: P0 acts
        // P0's advantage at step 0 should NOT bootstrap from Episode 2

        let device = Default::default();
        let gamma = 0.99_f32;

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(3, 1, 4, None, 2, &device);

        buffer.dones = Tensor::from_floats([[0.0], [1.0], [1.0]], &device);
        buffer.acting_players = Tensor::from_ints([[0], [1], [0]], &device);

        // Episode 1: P0 loses (-1), P1 wins (+1)
        // Episode 2: P0 wins (+1)
        buffer.all_rewards = Tensor::from_floats(
            [
                [[0.0, 0.0]],
                [[-1.0, 1.0]], // Episode 1 end
                [[1.0, -1.0]], // Episode 2 end
            ],
            &device,
        );

        // Give Episode 2 P0 a high value to detect bleed
        // Values: acting player's value at each step (P0 at 0,2; P1 at 1)
        buffer.values = Tensor::from_floats(
            [[0.0], [0.0], [0.9]], // P0 value=0.0, P1 value=0.0, Episode 2 P0 expects 0.9
            &device,
        );

        // Terminal at step 2 - P0's last value was 0.9, P1's was 0.0
        let last_value_per_player = vec![vec![0.9, 0.0]]; // [env][player]
        compute_gae_multiplayer(&mut buffer, &last_value_per_player, gamma, 0.95, 2, &device);

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();

        // Step 0 (P0, Episode 1): attributed_reward = r0[P0] + r1[P0] = 0 + (-1) = -1
        // With correct code: delta = -1 + gamma * 0 - 0 = -1 (next_value reset for P0)
        // With buggy code: delta = -1 + gamma * 0.9 - 0 = -0.109 (wrong!)
        //
        // P0 loses in Episode 1, so advantage should be strongly negative
        assert!(
            advs[0] < -0.5,
            "P0 step 0 should have strongly negative advantage (lost game), got {}",
            advs[0]
        );

        // If buggy code uses V2=0.9 from Episode 2:
        // delta = -1 + 0.99 * 0.9 = -0.109 (almost neutral - WRONG!)
        assert!(
            advs[0] < -0.9,
            "P0 advantage should be close to -1 (attributed reward), got {} (possible Episode 2 bleed)",
            advs[0]
        );
    }

    #[test]
    fn test_gae_multiplayer_reward_attribution_boundary() {
        // Verify rewards don't bleed from Episode 2 into Episode 1
        // Episode 1: steps 0-1, P1 wins
        // Episode 2: steps 2-3, P0 wins big (+10)

        let device = Default::default();

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(4, 1, 4, None, 2, &device);

        buffer.dones = Tensor::from_floats([[0.0], [1.0], [0.0], [1.0]], &device);
        buffer.acting_players = Tensor::from_ints([[0], [1], [0], [1]], &device);

        buffer.all_rewards = Tensor::from_floats(
            [
                [[0.0, 0.0]],
                [[-1.0, 1.0]], // Episode 1: P0 loses, P1 wins
                [[0.0, 0.0]],
                [[10.0, -10.0]], // Episode 2: P0 wins big
            ],
            &device,
        );

        buffer.values = Tensor::zeros([4, 1], &device);

        // Terminal at step 3 - all players had 0 values
        let last_value_per_player = vec![vec![0.0, 0.0]]; // [env][player]
        compute_gae_multiplayer(&mut buffer, &last_value_per_player, 0.99, 0.95, 2, &device);

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();

        // P0's advantage in Episode 1 should reflect losing (-1), not Episode 2's win
        assert!(
            advs[0] < 0.0,
            "P0 step 0 (Episode 1) should be negative (lost), got {}",
            advs[0]
        );

        // P1's advantage in Episode 1 should reflect winning (+1)
        assert!(
            advs[1] > 0.0,
            "P1 step 1 (Episode 1) should be positive (won), got {}",
            advs[1]
        );

        // Episode 2 should be independent
        assert!(
            advs[2] > 5.0,
            "P0 step 2 (Episode 2) should reflect +10 win, got {}",
            advs[2]
        );
    }

    #[test]
    fn test_gae_multiplayer_three_players() {
        // 3-player game: P0, P1, P2 take turns
        // Only P2 wins at terminal step

        let device = Default::default();

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(3, 1, 4, None, 3, &device);

        buffer.dones = Tensor::from_floats([[0.0], [0.0], [1.0]], &device);
        buffer.acting_players = Tensor::from_ints([[0], [1], [2]], &device);

        // P2 wins, others lose
        buffer.all_rewards = Tensor::from_floats(
            [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[-1.0, -1.0, 2.0]]],
            &device,
        );

        buffer.values = Tensor::zeros([3, 1], &device);

        // Terminal at step 2 - all players had 0 values (3 players)
        let last_value_per_player = vec![vec![0.0, 0.0, 0.0]]; // [env][player]
        compute_gae_multiplayer(&mut buffer, &last_value_per_player, 0.99, 0.95, 3, &device);

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();

        // P0 and P1 should have negative advantages (they lose)
        assert!(
            advs[0] < 0.0,
            "P0 should have negative advantage, got {}",
            advs[0]
        );
        assert!(
            advs[1] < 0.0,
            "P1 should have negative advantage, got {}",
            advs[1]
        );

        // P2 should have positive advantage (wins)
        assert!(
            advs[2] > 0.0,
            "P2 should have positive advantage, got {}",
            advs[2]
        );
    }

    #[test]
    fn test_gae_multiplayer_long_alternating_episode() {
        // 6-step episode: P0, P1, P0, P1, P0, P1 (terminal)
        // P0 acts at 0, 2, 4; P1 acts at 1, 3, 5
        // Tests that value chains work correctly over multiple turns

        let device = Default::default();
        let gamma = 0.99_f32;
        let lambda = 0.95_f32;

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(6, 1, 4, None, 2, &device);

        buffer.dones = Tensor::from_floats([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]], &device);
        buffer.acting_players = Tensor::from_ints([[0], [1], [0], [1], [0], [1]], &device);

        // Reward only at end: P0 wins
        buffer.all_rewards = Tensor::from_floats(
            [
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[1.0, -1.0]],
            ],
            &device,
        );

        // Values: acting player's value at each step (alternating P0/P1)
        // P0 acts at 0,2,4; P1 acts at 1,3,5
        buffer.values = Tensor::from_floats(
            [[0.3], [0.6], [0.5], [0.4], [0.7], [0.2]], // P0:0.3, P1:0.6, P0:0.5, P1:0.4, P0:0.7, P1:0.2
            &device,
        );

        // Terminal at step 5 - P0's last value was 0.7 (step 4), P1's was 0.2 (step 5)
        let last_value_per_player = vec![vec![0.7, 0.2]]; // [env][player]
        compute_gae_multiplayer(
            &mut buffer,
            &last_value_per_player,
            gamma,
            lambda,
            2,
            &device,
        );

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();

        // All P0 steps should have positive advantages
        assert!(
            advs[0] > 0.0,
            "P0 step 0 should be positive, got {}",
            advs[0]
        );
        assert!(
            advs[2] > 0.0,
            "P0 step 2 should be positive, got {}",
            advs[2]
        );
        assert!(
            advs[4] > 0.0,
            "P0 step 4 should be positive, got {}",
            advs[4]
        );

        // All P1 steps should have negative advantages
        assert!(
            advs[1] < 0.0,
            "P1 step 1 should be negative, got {}",
            advs[1]
        );
        assert!(
            advs[3] < 0.0,
            "P1 step 3 should be negative, got {}",
            advs[3]
        );
        assert!(
            advs[5] < 0.0,
            "P1 step 5 should be negative, got {}",
            advs[5]
        );

        // Advantages should decrease in magnitude as we approach terminal
        // (less uncertainty as game progresses)
        assert!(
            advs[0].abs() > advs[2].abs(),
            "Earlier advantages should be larger in magnitude"
        );
    }

    #[test]
    fn test_gae_multiplayer_different_player_terminal_exact() {
        // Verify exact advantage values when different player acts at terminal
        // Episode: Step 0 (P0 acts), Step 1 (P1 acts, done)

        let device = Default::default();

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(2, 1, 4, None, 2, &device);

        buffer.dones = Tensor::from_floats([[0.0], [1.0]], &device);
        buffer.acting_players = Tensor::from_ints([[0], [1]], &device);

        // P0 loses (-1), P1 wins (+1) at terminal step
        buffer.all_rewards = Tensor::from_floats([[[0.0, 0.0]], [[-1.0, 1.0]]], &device);

        // All values = 0 for simple calculation
        buffer.values = Tensor::zeros([2, 1], &device);

        // Terminal at step 1 - both players had 0 values
        let last_value_per_player = vec![vec![0.0, 0.0]]; // [env][player]
        compute_gae_multiplayer(&mut buffer, &last_value_per_player, 0.99, 0.95, 2, &device);

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();

        // Step 1 (P1, terminal): delta = 1 + 0 - 0 = 1, advantage = 1
        assert!(
            (advs[1] - 1.0).abs() < 1e-5,
            "P1 terminal advantage should be 1.0, got {}",
            advs[1]
        );

        // Step 0 (P0): attributed_reward = 0 + (-1) = -1 (P0's reward from step 1 via carry)
        // next_value[P0] = 0 (reset at episode boundary)
        // delta = -1 + gamma * 0 - 0 = -1, advantage = -1
        assert!(
            (advs[0] - (-1.0)).abs() < 1e-5,
            "P0 advantage should be -1.0, got {}",
            advs[0]
        );
    }

    #[test]
    fn test_gae_multiplayer_same_player_across_boundary() {
        // P0 acts at end of Episode 1 AND start of Episode 2
        // Verify P0's values don't bleed across episodes

        let device = Default::default();

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(3, 1, 4, None, 2, &device);

        // Episode 1: step 0 (P0), step 1 (P0, done)
        // Episode 2: step 2 (P0, done)
        buffer.dones = Tensor::from_floats([[0.0], [1.0], [1.0]], &device);
        buffer.acting_players = Tensor::from_ints([[0], [0], [0]], &device);

        // Episode 1: P0 loses (-1 at step 1)
        // Episode 2: P0 wins (+10 at step 2)
        buffer.all_rewards =
            Tensor::from_floats([[[0.0, 0.0]], [[-1.0, 0.0]], [[10.0, 0.0]]], &device);

        // Different values to detect bleed (P0 acts all steps)
        buffer.values = Tensor::from_floats(
            [[0.0], [0.0], [5.0]], // Step 0: 0.0, Step 1: 0.0, Step 2 (Episode 2): high value 5.0
            &device,
        );

        // Terminal at step 2 - P0's last value was 5.0, P1 never acted (0)
        let last_value_per_player = vec![vec![5.0, 0.0]]; // [env][player]
        compute_gae_multiplayer(&mut buffer, &last_value_per_player, 0.99, 0.95, 2, &device);

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();

        // Step 2 (Episode 2): delta = 10 - 5 = 5, advantage = 5
        assert!(
            (advs[2] - 5.0).abs() < 1e-5,
            "Episode 2 advantage should be 5.0, got {}",
            advs[2]
        );

        // Step 1 (Episode 1 terminal): delta = -1 - 0 = -1
        // P0's next_value should NOT be 5.0 from Episode 2
        assert!(
            (advs[1] - (-1.0)).abs() < 1e-5,
            "Episode 1 terminal advantage should be -1.0, got {}",
            advs[1]
        );

        // Step 0 (Episode 1): Should bootstrap from step 1's value (0), not step 2's value (5)
        // delta = 0 + gamma * 0 - 0 = 0
        // advantage = delta + gamma * lambda * (-1) = -0.9405
        let expected = -(0.99 * 0.95);
        assert!(
            (advs[0] - expected).abs() < 1e-5,
            "Episode 1 step 0 advantage should be {}, got {}",
            expected,
            advs[0]
        );
    }

    #[test]
    fn test_gae_multiplayer_multiple_envs_isolated() {
        // Two environments with different episode boundaries
        // Verify they don't interfere with each other

        let device = Default::default();

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(2, 2, 4, None, 2, &device);

        // Env 0: done at step 1, Env 1: not done
        buffer.dones = Tensor::from_floats([[0.0, 0.0], [1.0, 0.0]], &device);
        buffer.acting_players = Tensor::from_ints([[0, 0], [1, 1]], &device);

        // Env 0: P1 wins at step 1
        // Env 1: ongoing game, no terminal rewards
        buffer.all_rewards = Tensor::from_floats(
            [[[0.0, 0.0], [0.0, 0.0]], [[-1.0, 1.0], [0.0, 0.0]]],
            &device,
        );

        // Values: acting player's value at each step
        // Step 0: P0 acts in both envs - env0=0.5 (P0's), env1=0.3 (P0's)
        // Step 1: P1 acts in both envs - env0=0.4 (P1's), env1=0.4 (P1's)
        buffer.values = Tensor::from_floats([[0.5, 0.3], [0.4, 0.4]], &device);

        // Per-player bootstrap values:
        // Env 0: done at step 1, P0's last value=0.5, P1's last value=0.4
        // Env 1: not done, P0's last value=0.3, P1's last value=0.5 (bootstrap for continuation)
        let last_value_per_player = vec![
            vec![0.5, 0.4], // Env 0
            vec![0.3, 0.5], // Env 1 - P1 continues, bootstrap 0.5
        ];
        compute_gae_multiplayer(&mut buffer, &last_value_per_player, 0.99, 0.95, 2, &device);

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();
        // Layout: [step0_env0, step0_env1, step1_env0, step1_env1]

        // Env 0, step 1 (P1, done): delta = 1 + 0 - 0.4 = 0.6
        assert!(
            (advs[2] - 0.6).abs() < 1e-5,
            "Env 0 step 1 advantage should be 0.6, got {}",
            advs[2]
        );

        // Env 1, step 1 (P1, not done): should bootstrap from last_values
        // delta = 0 + gamma * 0.5 - 0.4 = 0.495 - 0.4 = 0.095
        let env1_step1_expected = 0.99 * 0.5 - 0.4;
        assert!(
            (advs[3] - env1_step1_expected).abs() < 1e-4,
            "Env 1 step 1 advantage should be ~{}, got {}",
            env1_step1_expected,
            advs[3]
        );
    }

    #[test]
    fn test_gae_multiplayer_no_done_flags() {
        // No episodes end - verify normal GAE computation works

        let device = Default::default();

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(3, 1, 4, None, 2, &device);

        // No done flags - single ongoing episode
        buffer.dones = Tensor::zeros([3, 1], &device);
        buffer.acting_players = Tensor::from_ints([[0], [1], [0]], &device);

        buffer.all_rewards =
            Tensor::from_floats([[[0.1, 0.0]], [[0.0, 0.2]], [[0.3, 0.0]]], &device);

        // Values: acting player's value at each step (all 0.5)
        buffer.values = Tensor::from_floats([[0.5], [0.5], [0.5]], &device);

        // No done flags - rollout ends with P0 at step 2
        // P0's last value=0.5 (step 2), P1's last value=0.5 (step 1)
        // Bootstrap with fresh value for next actor (P1) = 0.6
        let last_value_per_player = vec![vec![0.5, 0.6]]; // [env][player]
        compute_gae_multiplayer(&mut buffer, &last_value_per_player, 0.99, 0.95, 2, &device);

        let advs = buffer.advantages.unwrap().to_data();
        let advs = advs.as_slice::<f32>().unwrap();

        // All advantages should be computed (not NaN or zero due to bad resets)
        assert!(advs[0].is_finite(), "Step 0 advantage should be finite");
        assert!(advs[1].is_finite(), "Step 1 advantage should be finite");
        assert!(advs[2].is_finite(), "Step 2 advantage should be finite");

        // Step 2 (P0): With per-player bootstrap, P0 uses their own last value (0.5)
        // delta = 0.3 + gamma * 0.5 - 0.5 = 0.3 + 0.495 - 0.5 = 0.295
        let step2_delta = 0.3 + 0.99 * 0.5 - 0.5;
        assert!(
            (advs[2] - step2_delta).abs() < 1e-4,
            "Step 2 advantage should be ~{}, got {}",
            step2_delta,
            advs[2]
        );
    }

    #[test]
    fn test_buffer_flatten() {
        let device = Default::default();
        let buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(4, 2, 3, None, 1u8, &device);

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
            privileged_obs,
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
        assert!(
            privileged_obs.is_none(),
            "privileged_obs should be None without CTDE"
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
        let mut buffer: RolloutBuffer<TestBackend> =
            RolloutBuffer::new(4, 2, 3, None, 1u8, &device);

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

        let (_, _, _, _, _, _, _, valid_indices, _) = buffer.flatten();

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
        let mut buffer: RolloutBuffer<TestBackend> =
            RolloutBuffer::new(4, 2, 3, None, 1u8, &device);

        buffer.advantages = Some(Tensor::ones([4, 2], &device));
        buffer.returns = Some(Tensor::ones([4, 2], &device));

        // All 1.0 = all learner turns
        buffer.valid_mask = Some(Tensor::ones([4, 2], &device));

        let (_, _, _, _, _, _, _, valid_indices, _) = buffer.flatten();

        let indices = valid_indices.expect("should have valid_indices");
        assert_eq!(indices.dims(), [8], "All 8 positions should be valid");

        let indices_data: Vec<i64> = indices.into_data().to_vec().expect("indices data");
        assert_eq!(indices_data, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_valid_mask_no_learner_turns() {
        // Edge case: no learner turns (shouldn't happen in practice, but test robustness)
        let device = Default::default();
        let mut buffer: RolloutBuffer<TestBackend> =
            RolloutBuffer::new(4, 2, 3, None, 1u8, &device);

        buffer.advantages = Some(Tensor::ones([4, 2], &device));
        buffer.returns = Some(Tensor::ones([4, 2], &device));

        // All 0.0 = all opponent turns
        buffer.valid_mask = Some(Tensor::zeros([4, 2], &device));

        let (_, _, _, _, _, _, _, valid_indices, _) = buffer.flatten();

        let indices = valid_indices.expect("should have valid_indices");
        assert_eq!(indices.dims(), [0], "No positions should be valid");
    }

    #[test]
    fn test_select_filters_data_correctly() {
        // Test that selecting with valid_indices correctly filters tensor data
        let device = Default::default();

        // Create observation data where we can verify filtering
        // Obs shape: [8, 3] - 8 steps, 3 features
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

    /// Helper to compute minibatch boundaries using the same logic as `ppo_update`
    fn compute_minibatch_sizes(batch_size: usize, num_minibatches: usize) -> Vec<usize> {
        let base_mb_size = batch_size / num_minibatches;
        let remainder = batch_size % num_minibatches;

        let mut sizes = Vec::new();
        for mb_idx in 0..num_minibatches {
            let mb_size = base_mb_size + usize::from(mb_idx < remainder);
            if mb_size > 0 {
                sizes.push(mb_size);
            }
        }
        sizes
    }

    #[test]
    fn test_minibatch_distribution_evenly_divisible() {
        // 100 samples / 4 minibatches = 25 each
        let sizes = compute_minibatch_sizes(100, 4);
        assert_eq!(sizes, vec![25, 25, 25, 25]);
        assert_eq!(sizes.iter().sum::<usize>(), 100);
    }

    #[test]
    fn test_minibatch_distribution_with_remainder() {
        // 893 samples / 4 minibatches = 223 base + 1 remainder
        // First 1 batch gets 224, remaining 3 get 223
        let sizes = compute_minibatch_sizes(893, 4);
        assert_eq!(sizes, vec![224, 223, 223, 223]);
        assert_eq!(sizes.iter().sum::<usize>(), 893);
    }

    #[test]
    fn test_minibatch_distribution_larger_remainder() {
        // 14 samples / 4 minibatches = 3 base + 2 remainder
        // First 2 batches get 4, remaining 2 get 3
        let sizes = compute_minibatch_sizes(14, 4);
        assert_eq!(sizes, vec![4, 4, 3, 3]);
        assert_eq!(sizes.iter().sum::<usize>(), 14);
    }

    #[test]
    fn test_minibatch_distribution_small_batch() {
        // 3 samples / 4 minibatches - only 3 batches have samples
        let sizes = compute_minibatch_sizes(3, 4);
        assert_eq!(sizes, vec![1, 1, 1]);
        assert_eq!(sizes.iter().sum::<usize>(), 3);
    }

    #[test]
    fn test_minibatch_distribution_single_sample() {
        // 1 sample / 4 minibatches - only 1 batch has the sample
        let sizes = compute_minibatch_sizes(1, 4);
        assert_eq!(sizes, vec![1]);
        assert_eq!(sizes.iter().sum::<usize>(), 1);
    }

    #[test]
    fn test_minibatch_distribution_empty() {
        // 0 samples - no batches
        let sizes = compute_minibatch_sizes(0, 4);
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_minibatch_boundaries_correct() {
        // Verify that mb_start/mb_end boundaries don't overlap or skip samples
        let batch_size = 893;
        let num_minibatches = 4;
        let base_mb_size = batch_size / num_minibatches;
        let remainder = batch_size % num_minibatches;

        let mut mb_start = 0;
        let mut covered = vec![false; batch_size];

        for mb_idx in 0..num_minibatches {
            let mb_size = base_mb_size + usize::from(mb_idx < remainder);
            if mb_size == 0 {
                continue;
            }
            let mb_end = mb_start + mb_size;

            // Mark these indices as covered
            for c in &mut covered[mb_start..mb_end] {
                assert!(!*c, "Index covered twice");
                *c = true;
            }

            mb_start = mb_end;
        }

        // All indices should be covered exactly once
        assert!(covered.iter().all(|&c| c), "Not all indices covered");
    }
}
