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
use crate::utils::{entropy_categorical, log_prob_categorical, normalize_advantages, sample_categorical};

/// Stores trajectory data from environment rollouts
///
/// Shape: [num_steps, num_envs] for most fields
#[derive(Debug)]
pub struct RolloutBuffer<B: Backend> {
    /// Observations [num_steps, num_envs, obs_dim]
    pub observations: Tensor<B, 3>,
    /// Actions taken [num_steps, num_envs]
    pub actions: Tensor<B, 2, Int>,
    /// Rewards received [num_steps, num_envs]
    pub rewards: Tensor<B, 2>,
    /// Episode done flags [num_steps, num_envs]
    pub dones: Tensor<B, 2>,
    /// Value estimates [num_steps, num_envs]
    pub values: Tensor<B, 2>,
    /// Log probabilities of actions [num_steps, num_envs]
    pub log_probs: Tensor<B, 2>,
    /// GAE advantages (computed after rollout) [num_steps, num_envs]
    pub advantages: Option<Tensor<B, 2>>,
    /// Returns (values + advantages) [num_steps, num_envs]
    pub returns: Option<Tensor<B, 2>>,
}

impl<B: Backend> RolloutBuffer<B> {
    /// Create empty buffer with given dimensions
    pub fn new(num_steps: usize, num_envs: usize, obs_dim: usize, device: &B::Device) -> Self {
        Self {
            observations: Tensor::zeros([num_steps, num_envs, obs_dim], device),
            actions: Tensor::zeros([num_steps, num_envs], device),
            rewards: Tensor::zeros([num_steps, num_envs], device),
            dones: Tensor::zeros([num_steps, num_envs], device),
            values: Tensor::zeros([num_steps, num_envs], device),
            log_probs: Tensor::zeros([num_steps, num_envs], device),
            advantages: None,
            returns: None,
        }
    }

    /// Flatten buffer for minibatch processing
    /// Returns (obs, actions, old_log_probs, advantages, returns)
    /// Each with shape [num_steps * num_envs, ...]
    pub fn flatten(&self) -> (Tensor<B, 2>, Tensor<B, 1, Int>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let [num_steps, num_envs, obs_dim] = self.observations.dims();
        let batch_size = num_steps * num_envs;

        let obs = self.observations.clone().reshape([batch_size, obs_dim]);
        let actions = self.actions.clone().flatten(0, 1);
        let log_probs = self.log_probs.clone().flatten(0, 1);
        let advantages = self.advantages.as_ref().expect("Advantages not computed").clone().flatten(0, 1);
        let returns = self.returns.as_ref().expect("Returns not computed").clone().flatten(0, 1);

        (obs, actions, log_probs, advantages, returns)
    }
}

/// Collect rollouts from vectorized environment
///
/// Runs num_steps in each of num_envs environments, storing trajectories.
/// Uses CPU collection with batch GPU transfer for performance.
pub fn collect_rollouts<B: Backend, E: Environment>(
    model: &ActorCritic<B>,
    vec_env: &mut VecEnv<E>,
    buffer: &mut RolloutBuffer<B>,
    num_steps: usize,
    device: &B::Device,
    rng: &mut impl Rng,
) -> Vec<EpisodeStats> {
    let num_envs = vec_env.num_envs();
    let obs_dim = vec_env.observation_dim();
    let mut all_completed = Vec::new();

    // Pre-allocate CPU vectors for batch collection
    let mut all_obs: Vec<f32> = Vec::with_capacity(num_steps * num_envs * obs_dim);
    let mut all_actions: Vec<i64> = Vec::with_capacity(num_steps * num_envs);
    let mut all_rewards: Vec<f32> = Vec::with_capacity(num_steps * num_envs);
    let mut all_dones: Vec<f32> = Vec::with_capacity(num_steps * num_envs);
    let mut all_values: Vec<f32> = Vec::with_capacity(num_steps * num_envs);
    let mut all_log_probs: Vec<f32> = Vec::with_capacity(num_steps * num_envs);

    for _step in 0..num_steps {
        // Get current observations and convert to tensor
        let obs_flat = vec_env.get_observations();
        let obs_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(obs_flat.as_slice(), device).reshape([num_envs, obs_dim]);

        // Forward pass to get logits and values
        let (logits, values) = model.forward(obs_tensor);

        // Sample actions from categorical distribution
        let actions = sample_categorical(logits.clone(), rng, device);

        // Compute log probs of sampled actions
        let log_probs = log_prob_categorical(logits, actions.clone());

        // Convert to CPU data
        // Note: Convert Int tensor through float to avoid WGPU backend type mismatch
        let actions_data: Vec<i64> = actions
            .float()
            .into_data()
            .to_vec::<f32>()
            .expect("actions to vec")
            .into_iter()
            .map(|x| x as i64)
            .collect();
        let values_data: Vec<f32> = values.into_data().to_vec().expect("values to vec");
        let log_probs_data: Vec<f32> = log_probs.into_data().to_vec().expect("log_probs to vec");

        // Convert actions to Vec<usize> for environment
        let actions_usize: Vec<usize> = actions_data.iter().map(|&a| a as usize).collect();

        // Step environment
        let (_next_obs, rewards, dones, completed) = vec_env.step(&actions_usize);
        all_completed.extend(completed);

        // Append to CPU buffers
        all_obs.extend_from_slice(&obs_flat);
        all_actions.extend_from_slice(&actions_data);
        all_rewards.extend_from_slice(&rewards);
        all_dones.extend(dones.iter().map(|&d| if d { 1.0 } else { 0.0 }));
        all_values.extend_from_slice(&values_data);
        all_log_probs.extend_from_slice(&log_probs_data);
    }

    // Batch transfer to GPU
    buffer.observations =
        Tensor::<B, 1>::from_floats(all_obs.as_slice(), device).reshape([num_steps, num_envs, obs_dim]);
    buffer.actions =
        Tensor::<B, 1, Int>::from_ints(all_actions.as_slice(), device).reshape([num_steps, num_envs]);
    buffer.rewards =
        Tensor::<B, 1>::from_floats(all_rewards.as_slice(), device).reshape([num_steps, num_envs]);
    buffer.dones =
        Tensor::<B, 1>::from_floats(all_dones.as_slice(), device).reshape([num_steps, num_envs]);
    buffer.values =
        Tensor::<B, 1>::from_floats(all_values.as_slice(), device).reshape([num_steps, num_envs]);
    buffer.log_probs =
        Tensor::<B, 1>::from_floats(all_log_probs.as_slice(), device).reshape([num_steps, num_envs]);

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
    rng: &mut impl Rng,
) -> (ActorCritic<B>, UpdateMetrics) {
    let device = model.devices()[0].clone();
    let (obs, actions, old_log_probs, advantages, returns) = buffer.flatten();
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

    // Epoch loop
    for _epoch in 0..config.num_epochs {
        // Shuffle indices for minibatches
        let mut indices: Vec<usize> = (0..batch_size).collect();
        indices.shuffle(rng);

        // Minibatch loop
        for mb_start in (0..batch_size).step_by(minibatch_size) {
            let mb_end = (mb_start + minibatch_size).min(batch_size);
            let mb_indices: Vec<i64> = indices[mb_start..mb_end].iter().map(|&i| i as i64).collect();
            let mb_indices_tensor: Tensor<B, 1, Int> = Tensor::from_ints(mb_indices.as_slice(), &device);

            // Gather minibatch data
            let mb_obs = obs.clone().select(0, mb_indices_tensor.clone());
            let mb_actions: Tensor<B, 1, Int> = actions.clone().select(0, mb_indices_tensor.clone());
            let mb_old_log_probs = old_log_probs.clone().select(0, mb_indices_tensor.clone());
            let mb_advantages_raw = advantages.clone().select(0, mb_indices_tensor.clone());
            let mb_returns = returns.clone().select(0, mb_indices_tensor);

            // Normalize advantages at minibatch level (critical for stability)
            let mb_advantages = normalize_advantages(mb_advantages_raw);

            // Forward pass
            let (logits, values) = model.forward(mb_obs);
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
            let entropy_loss = -entropy.clone().mean() * config.entropy_coef;

            // Combined loss
            let loss = policy_loss_mean.clone() + value_loss.clone() * config.value_coef + entropy_loss;

            // Compute approximate KL divergence for logging
            let approx_kl = ((ratio.clone() - 1.0) - log_ratio).mean();

            // Compute clip fraction for logging
            let clip_fraction = (ratio.clone() - 1.0)
                .abs()
                .greater_elem(config.clip_epsilon)
                .float()
                .mean();

            // Accumulate metrics
            total_policy_loss += policy_loss_mean.clone().into_scalar().elem::<f32>();
            total_value_loss += value_loss.clone().into_scalar().elem::<f32>();
            total_entropy += entropy.mean().into_scalar().elem::<f32>();
            total_approx_kl += approx_kl.clone().into_scalar().elem::<f32>();
            total_clip_fraction += clip_fraction.clone().into_scalar().elem::<f32>();
            total_loss_sum += loss.clone().into_scalar().elem::<f32>();
            total_value_mean += values.mean().into_scalar().elem::<f32>();
            total_returns_mean += mb_returns.mean().into_scalar().elem::<f32>();
            num_updates += 1;

            // Backward pass and optimization
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // Gradient clipping by global L2 norm
            // Note: Burn's optimizers handle this internally with proper config

            model = optimizer.step(learning_rate.into(), model, grads);
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
        let buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(128, 4, 4, &device);

        assert_eq!(buffer.observations.dims(), [128, 4, 4]);
        assert_eq!(buffer.actions.dims(), [128, 4]);
        assert_eq!(buffer.rewards.dims(), [128, 4]);
    }

    #[test]
    fn test_gae_computation() {
        let device = Default::default();
        let num_steps = 4;
        let num_envs = 2;

        let mut buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(num_steps, num_envs, 1, &device);

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
    fn test_buffer_flatten() {
        let device = Default::default();
        let buffer: RolloutBuffer<TestBackend> = RolloutBuffer::new(4, 2, 3, &device);

        // Set advantages and returns
        let mut buffer = buffer;
        buffer.advantages = Some(Tensor::ones([4, 2], &device));
        buffer.returns = Some(Tensor::ones([4, 2], &device));

        let (obs, actions, log_probs, advantages, returns) = buffer.flatten();

        assert_eq!(obs.dims(), [8, 3]);
        assert_eq!(actions.dims(), [8]);
        assert_eq!(log_probs.dims(), [8]);
        assert_eq!(advantages.dims(), [8]);
        assert_eq!(returns.dims(), [8]);
    }
}
