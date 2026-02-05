//! CTDE (Centralized Training, Decentralized Execution) Actor-Critic Network
//!
//! This module implements a MAPPO-style network architecture where:
//! - Actor sees only local observations (decentralized execution)
//! - Critic sees global game state (centralized training)
//! - Both output per-player predictions for competitive games

use burn::prelude::*;

use super::mlp::create_linear_orthogonal;
use crate::config::Config;
use crate::profile::{profile_function, profile_scope};

/// CTDE Actor-Critic with separate actor and critic networks
///
/// Design philosophy:
/// - Actor network is small and fast (used during deployment)
/// - Critic network can be larger (only used during training)
/// - Both networks trained end-to-end with PPO loss
#[derive(Module, Debug)]
pub struct CtdeActorCritic<B: Backend> {
    // Actor network: local_obs -> policy_logits
    actor_layers: Vec<nn::Linear<B>>,
    policy_head: nn::Linear<B>,

    // Critic network: global_state -> per_player_values
    critic_layers: Vec<nn::Linear<B>>,
    pub value_head: nn::Linear<B>,

    // Configuration
    #[module(skip)]
    use_relu: bool,
    #[module(skip)]
    num_players: usize,
    #[module(skip)]
    local_obs_dim: usize,
    #[module(skip)]
    global_state_dim: usize,
    #[module(skip)]
    action_count: usize,
}

impl<B: Backend> CtdeActorCritic<B> {
    /// Create a new CTDE Actor-Critic network
    ///
    /// # Arguments
    /// * `local_obs_dim` - Dimension of local observations (for actor)
    /// * `global_state_dim` - Dimension of global state (for critic)
    /// * `action_count` - Number of discrete actions
    /// * `num_players` - Number of players (for value head output)
    /// * `config` - Network configuration (hidden sizes, activation, etc.)
    /// * `device` - Device to create network on
    ///
    /// # Architecture
    /// Actor: `local_obs` -> hidden -> ... -> hidden -> `policy_logits`
    /// Critic: `global_state` -> hidden -> ... -> hidden -> `per_player_values`
    ///
    /// Orthogonal initialization:
    /// - Hidden layers: sqrt(2) for `ReLU`, 1.0 for tanh
    /// - Policy head: 0.01 (stable initial policy)
    /// - Value head: 1.0
    pub fn new(
        local_obs_dim: usize,
        global_state_dim: usize,
        action_count: usize,
        num_players: usize,
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
        let mut in_size = local_obs_dim;

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
        // Critic sees global_state + local_obs (MAPPO-style agent-specific features)
        let mut critic_layers = Vec::with_capacity(critic_num_hidden);
        in_size = global_state_dim + local_obs_dim;

        for _ in 0..critic_num_hidden {
            critic_layers.push(create_linear_orthogonal(
                in_size,
                critic_hidden_size,
                hidden_gain,
                device,
            ));
            in_size = critic_hidden_size;
        }

        let value_head = create_linear_orthogonal(in_size, num_players, 1.0, device);

        Self {
            actor_layers,
            policy_head,
            critic_layers,
            value_head,
            use_relu,
            num_players,
            local_obs_dim,
            global_state_dim,
            action_count,
        }
    }

    /// Forward pass through actor network only
    ///
    /// # Arguments
    /// * `local_obs` - Local observations [`batch_size`, `local_obs_dim`]
    ///
    /// # Returns
    /// Policy logits [`batch_size`, `action_count`]
    pub fn forward_actor(&self, local_obs: Tensor<B, 2>) -> Tensor<B, 2> {
        profile_function!();
        profile_scope!("actor");

        let mut x = local_obs;

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
    /// * `global_state` - Global game state [`batch_size`, `global_state_dim`]
    /// * `local_obs` - Local observations [`batch_size`, `local_obs_dim`]
    ///
    /// The critic sees both global state (privileged info) AND local observations
    /// (what the agent knows). This MAPPO-style approach reduces value function bias
    /// in partially observable games by conditioning on the agent's information set.
    ///
    /// # Returns
    /// Per-player values [`batch_size`, `num_players`]
    pub fn forward_critic(
        &self,
        global_state: Tensor<B, 2>,
        local_obs: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        profile_function!();
        profile_scope!("critic");

        // Concatenate global state with local observation (MAPPO agent-specific features)
        let mut x = Tensor::cat(vec![global_state, local_obs], 1);

        // Critic hidden layers with activation
        for layer in &self.critic_layers {
            x = layer.forward(x);
            x = if self.use_relu {
                burn::tensor::activation::relu(x)
            } else {
                burn::tensor::activation::tanh(x)
            };
        }

        // Value head (no activation - raw value predictions)
        self.value_head.forward(x)
    }

    /// Get dimensions for debugging/validation
    #[cfg(test)]
    pub fn dimensions(&self) -> (usize, usize, usize, usize) {
        (
            self.local_obs_dim,
            self.global_state_dim,
            self.action_count,
            self.num_players,
        )
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
            135, // local_obs_dim (e.g., Skull)
            540, // global_state_dim (e.g., Skull 4-player concat)
            7,   // action_count
            4,   // num_players
            &config, &device,
        );

        let (local, global, actions, players) = network.dimensions();
        assert_eq!(local, 135);
        assert_eq!(global, 540);
        assert_eq!(actions, 7);
        assert_eq!(players, 4);
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
            10, // local_obs_dim
            20, // global_state_dim
            5,  // action_count
            2,  // num_players
            &config, &device,
        );

        let batch_size = 8;
        let local_obs = Tensor::<TestBackend, 2>::zeros([batch_size, 10], &device);
        let logits = network.forward_actor(local_obs);

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
            10, // local_obs_dim
            20, // global_state_dim
            5,  // action_count
            2,  // num_players
            &config, &device,
        );

        let batch_size = 8;
        let global_state = Tensor::<TestBackend, 2>::zeros([batch_size, 20], &device);
        let local_obs = Tensor::<TestBackend, 2>::zeros([batch_size, 10], &device);
        let values = network.forward_critic(global_state, local_obs);

        assert_eq!(values.dims(), [batch_size, 2]);
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
            10, // local_obs_dim
            20, // global_state_dim
            5,  // action_count
            2,  // num_players
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
