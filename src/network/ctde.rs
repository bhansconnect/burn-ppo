//! CTDE (Centralized Training, Decentralized Execution) Actor-Critic Network
//!
//! This module implements a MAPPO-style network architecture where:
//! - Actor sees only local observations (decentralized execution)
//! - Critic sees local obs + privileged observations (centralized training)
//! - Value head outputs single scalar (acting player's value)
//!
//! Key terminology:
//! - `obs`: What the acting player can see (player-relative)
//! - `privileged_obs`: Hidden info not visible to actor (player-relative)

use burn::prelude::*;

use super::mlp::create_linear_orthogonal;
use crate::config::Config;
use crate::profile::{profile_function, profile_scope};

/// CTDE Actor-Critic with separate actor and critic networks
///
/// Design philosophy:
/// - Actor network is small and fast (used during deployment)
/// - Critic network can be larger (only used during training)
/// - Critic sees obs + `privileged_obs` (no redundancy)
/// - Value head outputs single scalar (acting player's value)
#[derive(Module, Debug)]
pub struct CtdeActorCritic<B: Backend> {
    // Actor network: obs -> policy_logits
    actor_layers: Vec<nn::Linear<B>>,
    policy_head: nn::Linear<B>,

    // Critic network: (privileged_obs, obs) -> value
    critic_layers: Vec<nn::Linear<B>>,
    pub value_head: nn::Linear<B>,

    // Configuration
    #[module(skip)]
    use_relu: bool,
    #[module(skip)]
    obs_dim: usize,
    #[module(skip)]
    privileged_obs_dim: usize,
    #[module(skip)]
    action_count: usize,
}

impl<B: Backend> CtdeActorCritic<B> {
    /// Create a new CTDE Actor-Critic network
    ///
    /// # Arguments
    /// * `obs_dim` - Dimension of observations (for actor, player-relative)
    /// * `privileged_obs_dim` - Dimension of privileged observations (hidden info, player-relative)
    /// * `action_count` - Number of discrete actions
    /// * `config` - Network configuration (hidden sizes, activation, etc.)
    /// * `device` - Device to create network on
    ///
    /// # Architecture
    /// Actor: `obs` -> hidden -> ... -> hidden -> `policy_logits`
    /// Critic: `(privileged_obs, obs)` -> hidden -> ... -> hidden -> `value` (scalar)
    ///
    /// Orthogonal initialization:
    /// - Hidden layers: sqrt(2) for `ReLU`, 1.0 for tanh
    /// - Policy head: 0.01 (stable initial policy)
    /// - Value head: 1.0
    pub fn new(
        obs_dim: usize,
        privileged_obs_dim: usize,
        action_count: usize,
        config: &Config,
        device: &B::Device,
    ) -> Self {
        let actor_hidden_size = config.hidden_size;
        let critic_hidden_size = config.critic_hidden_size.unwrap_or(config.hidden_size);
        let num_hidden = config.num_hidden;
        let critic_num_hidden = config.critic_num_hidden.unwrap_or(num_hidden);
        let use_relu = config.activation == "relu";

        let hidden_gain = if use_relu { 2.0_f64.sqrt() } else { 1.0 };

        // Build actor network
        let mut actor_layers = Vec::with_capacity(num_hidden);
        let mut in_size = obs_dim;

        for _ in 0..num_hidden {
            actor_layers.push(create_linear_orthogonal(
                in_size,
                actor_hidden_size,
                hidden_gain,
                device,
            ));
            in_size = actor_hidden_size;
        }

        let policy_head = create_linear_orthogonal(in_size, action_count, 0.01, device);

        // Build critic network
        // Critic sees privileged_obs + obs (full information, player-relative)
        let mut critic_layers = Vec::with_capacity(critic_num_hidden);
        in_size = privileged_obs_dim + obs_dim;

        for _ in 0..critic_num_hidden {
            critic_layers.push(create_linear_orthogonal(
                in_size,
                critic_hidden_size,
                hidden_gain,
                device,
            ));
            in_size = critic_hidden_size;
        }

        // Single value output (acting player's value)
        let value_head = create_linear_orthogonal(in_size, 1, 1.0, device);

        Self {
            actor_layers,
            policy_head,
            critic_layers,
            value_head,
            use_relu,
            obs_dim,
            privileged_obs_dim,
            action_count,
        }
    }

    /// Forward pass through actor network only
    ///
    /// # Arguments
    /// * `obs` - Observations [`batch_size`, `obs_dim`] (player-relative)
    ///
    /// # Returns
    /// Policy logits [`batch_size`, `action_count`]
    pub fn forward_actor(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        profile_function!();
        profile_scope!("actor");

        let mut x = obs;

        // Actor hidden layers with activation
        for layer in &self.actor_layers {
            x = layer.forward(x);
            x = if self.use_relu {
                burn::tensor::activation::relu(x)
            } else {
                burn::tensor::activation::tanh(x)
            };
        }

        // Policy head (no activation - raw logits for categorical distribution)
        self.policy_head.forward(x)
    }

    /// Forward pass through critic network only
    ///
    /// # Arguments
    /// * `privileged_obs` - Privileged observations [`batch_size`, `privileged_obs_dim`] (player-relative)
    /// * `obs` - Observations [`batch_size`, `obs_dim`] (player-relative)
    ///
    /// The critic sees `privileged_obs` (hidden info) + obs (what agent sees).
    /// Both are player-relative (current player at index 0).
    /// `privileged_obs` contains ONLY info not in obs (no redundancy).
    ///
    /// # Returns
    /// Value [`batch_size`, 1] (acting player's value)
    pub fn forward_critic(&self, privileged_obs: Tensor<B, 2>, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        profile_function!();
        profile_scope!("critic");

        // Concatenate privileged observations with regular observations
        let mut x = Tensor::cat(vec![privileged_obs, obs], 1);

        // Critic hidden layers with activation
        for layer in &self.critic_layers {
            x = layer.forward(x);
            x = if self.use_relu {
                burn::tensor::activation::relu(x)
            } else {
                burn::tensor::activation::tanh(x)
            };
        }

        // Value head (no activation - raw value prediction)
        self.value_head.forward(x)
    }

    /// Get dimensions for debugging/validation
    #[cfg(test)]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.obs_dim, self.privileged_obs_dim, self.action_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_ctde_creation() {
        let device = Default::default();
        let config = Config {
            hidden_size: 64,
            num_hidden: 2,
            critic_hidden_size: Some(128),
            critic_num_hidden: Some(2),
            activation: "relu".to_string(),
            ..Default::default()
        };

        let network = CtdeActorCritic::<TestBackend>::new(
            135, // obs_dim (e.g., Skull)
            50,  // privileged_obs_dim (hidden info only)
            7,   // action_count
            &config, &device,
        );

        let (obs, priv_obs, actions) = network.dimensions();
        assert_eq!(obs, 135);
        assert_eq!(priv_obs, 50);
        assert_eq!(actions, 7);
    }

    #[test]
    fn test_actor_forward_shape() {
        let device = Default::default();
        let config = Config {
            hidden_size: 64,
            num_hidden: 2,
            activation: "relu".to_string(),
            ..Default::default()
        };

        let network = CtdeActorCritic::<TestBackend>::new(
            10, // obs_dim
            20, // privileged_obs_dim
            5,  // action_count
            &config, &device,
        );

        let batch_size = 8;
        let obs = Tensor::<TestBackend, 2>::zeros([batch_size, 10], &device);
        let logits = network.forward_actor(obs);

        assert_eq!(logits.dims(), [batch_size, 5]);
    }

    #[test]
    fn test_critic_forward_shape() {
        let device = Default::default();
        let config = Config {
            hidden_size: 64,
            num_hidden: 2,
            activation: "relu".to_string(),
            ..Default::default()
        };

        let network = CtdeActorCritic::<TestBackend>::new(
            10, // obs_dim
            20, // privileged_obs_dim
            5,  // action_count
            &config, &device,
        );

        let batch_size = 8;
        let privileged_obs = Tensor::<TestBackend, 2>::zeros([batch_size, 20], &device);
        let obs = Tensor::<TestBackend, 2>::zeros([batch_size, 10], &device);
        let values = network.forward_critic(privileged_obs, obs);

        assert_eq!(values.dims(), [batch_size, 1]); // Single scalar value
    }

    #[test]
    fn test_separate_actor_critic_parameters() {
        let device = Default::default();
        let config = Config {
            hidden_size: 32,
            num_hidden: 1,
            critic_hidden_size: Some(64),
            critic_num_hidden: Some(1),
            activation: "tanh".to_string(),
            ..Default::default()
        };

        let network = CtdeActorCritic::<TestBackend>::new(
            10, // obs_dim
            20, // privileged_obs_dim
            5,  // action_count
            &config, &device,
        );

        // Verify actor and critic have different architectures
        assert_eq!(network.actor_layers.len(), 1);
        assert_eq!(network.critic_layers.len(), 1);

        // Verify actor hidden size
        let actor_weight_shape = network.actor_layers[0].weight.dims();
        assert_eq!(actor_weight_shape[1], 32); // output dim = hidden_size

        // Verify critic hidden size
        let critic_weight_shape = network.critic_layers[0].weight.dims();
        assert_eq!(critic_weight_shape[1], 64); // output dim = critic_hidden_size
    }
}
