//! Neural network architectures for PPO training
//!
//! This module provides different network architectures that can be selected via config:
//! - MLP (Multi-Layer Perceptron): Default, works for any observation
//! - CNN (Convolutional Neural Network): For spatial observations like board games

mod cnn;
mod ctde;
mod mlp;

pub use cnn::CnnActorCritic;
pub use ctde::CtdeActorCritic;
pub use mlp::MlpActorCritic;

use burn::module::Module;
use burn::prelude::*;

use crate::config::Config;

/// Unified actor-critic network supporting multiple architectures
///
/// This enum wraps the different network implementations and provides
/// a common interface for training and inference.
#[derive(Module, Debug)]
pub enum ActorCriticNetwork<B: Backend> {
    /// Multi-layer perceptron (default)
    Mlp(MlpActorCritic<B>),
    /// Convolutional neural network for spatial observations
    Cnn(CnnActorCritic<B>),
    /// CTDE (Centralized Training, Decentralized Execution)
    Ctde(CtdeActorCritic<B>),
}

impl<B: Backend> ActorCriticNetwork<B> {
    /// Create a new actor-critic network based on config
    ///
    /// # Arguments
    /// * `obs_dim` - Total observation dimension
    /// * `obs_shape` - Optional spatial shape (height, width, channels) for CNN
    /// * `action_count` - Number of discrete actions
    /// * `num_players` - Number of value outputs (1 for single-agent, 2+ for multi-player)
    /// * `global_state_dim` - Global state dimension for CTDE (required when `network_type` is "ctde")
    /// * `config` - Configuration specifying network type and parameters
    /// * `device` - Compute device
    ///
    /// # Panics
    /// Panics if `network_type = "cnn"` but `obs_shape` is None
    /// Panics if `network_type = "ctde"` but `global_state_dim` is None
    pub fn new(
        obs_dim: usize,
        obs_shape: Option<(usize, usize, usize)>,
        action_count: usize,
        num_players: usize,
        global_state_dim: Option<usize>,
        config: &Config,
        device: &B::Device,
    ) -> Self {
        match config.network_type.as_str() {
            "mlp" => Self::Mlp(MlpActorCritic::new(
                obs_dim,
                action_count,
                num_players,
                config,
                device,
            )),
            "cnn" => Self::Cnn(CnnActorCritic::new(
                obs_dim,
                obs_shape,
                action_count,
                num_players,
                config,
                device,
            )),
            "ctde" => {
                let dim =
                    global_state_dim.expect("CTDE network requires global_state_dim parameter");
                Self::Ctde(CtdeActorCritic::new(
                    obs_dim,
                    dim,
                    action_count,
                    num_players,
                    config,
                    device,
                ))
            }
            other => panic!("Unknown network_type: {other}. Use 'mlp', 'cnn', or 'ctde'"),
        }
    }

    /// Forward pass returning action logits and N player values
    ///
    /// Input: observations [batch, `obs_dim`]
    /// Output: (logits [batch, `action_count`], values [batch, `num_players`])
    ///
    /// # Panics
    /// Panics if called on a CTDE network. CTDE requires separate inputs for actor and critic.
    /// Use `forward_actor(local_obs)` and `forward_critic(global_state)` instead.
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self {
            Self::Mlp(mlp) => mlp.forward(obs),
            Self::Cnn(cnn) => cnn.forward(obs),
            Self::Ctde(_) => {
                panic!(
                    "forward() called on CTDE network. CTDE requires separate inputs for actor and critic.\n\
                     Use forward_actor(local_obs) and forward_critic(global_state) instead.\n\
                     \n\
                     Example:\n\
                     if network.is_ctde() {{\n\
                         let logits = network.forward_actor(local_obs);\n\
                         let values = network.forward_critic(global_state);\n\
                     }} else {{\n\
                         let (logits, values) = network.forward(obs);\n\
                     }}\n\
                     \n\
                     See docs/CTDE.md for more information."
                );
            }
        }
    }

    /// Forward pass through actor network only (for CTDE)
    ///
    /// For non-CTDE networks, this is equivalent to calling `forward()` and taking the logits.
    pub fn forward_actor(&self, local_obs: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Self::Mlp(mlp) => mlp.forward(local_obs).0,
            Self::Cnn(cnn) => cnn.forward(local_obs).0,
            Self::Ctde(ctde) => ctde.forward_actor(local_obs),
        }
    }

    /// Forward pass through critic network only (for CTDE)
    ///
    /// For CTDE networks, the critic takes both global state and local observations
    /// to reduce value function bias in partially observable games.
    /// For non-CTDE networks, `local_obs` is used as input (same as actor).
    pub fn forward_critic(
        &self,
        global_state: Tensor<B, 2>,
        local_obs: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        match self {
            Self::Mlp(mlp) => mlp.forward(local_obs).1,
            Self::Cnn(cnn) => cnn.forward(local_obs).1,
            Self::Ctde(ctde) => ctde.forward_critic(global_state, local_obs),
        }
    }

    /// Check if this is a CNN network
    #[cfg(test)]
    pub const fn is_cnn(&self) -> bool {
        matches!(self, Self::Cnn(_))
    }

    /// Check if this is an MLP network
    #[cfg(test)]
    pub const fn is_mlp(&self) -> bool {
        matches!(self, Self::Mlp(_))
    }

    /// Get a reference to the value head linear layer
    ///
    /// Used by value normalization for weight rescaling.
    pub fn value_head(&self) -> &burn::nn::Linear<B> {
        match self {
            Self::Mlp(mlp) => &mlp.value_head,
            Self::Cnn(cnn) => &cnn.value_head,
            Self::Ctde(ctde) => &ctde.value_head,
        }
    }

    /// Replace the value head with a new one
    ///
    /// Used by value normalization to rescale weights when statistics change.
    /// Returns a new network with the updated value head.
    pub fn with_value_head(self, new_value_head: burn::nn::Linear<B>) -> Self {
        match self {
            Self::Mlp(mut mlp) => {
                mlp.value_head = new_value_head;
                Self::Mlp(mlp)
            }
            Self::Cnn(mut cnn) => {
                cnn.value_head = new_value_head;
                Self::Cnn(cnn)
            }
            Self::Ctde(mut ctde) => {
                ctde.value_head = new_value_head;
                Self::Ctde(ctde)
            }
        }
    }

    /// Check if this is a CTDE network
    pub const fn is_ctde(&self) -> bool {
        matches!(self, Self::Ctde(_))
    }
}

// Re-export the old name for backward compatibility during transition
// TODO: Remove this alias once all code is updated
pub type ActorCritic<B> = ActorCriticNetwork<B>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_network_selection_mlp() {
        let device = Default::default();
        let config = Config {
            network_type: "mlp".to_string(),
            ..Config::default()
        };

        let model: ActorCriticNetwork<TestBackend> =
            ActorCriticNetwork::new(4, None, 2, 1, None, &config, &device);

        assert!(model.is_mlp());
        assert!(!model.is_cnn());

        let obs = Tensor::zeros([8, 4], &device);
        let (logits, values) = model.forward(obs);
        assert_eq!(logits.dims(), [8, 2]);
        assert_eq!(values.dims(), [8, 1]);
    }

    #[test]
    fn test_network_selection_cnn() {
        let device = Default::default();
        let config = Config {
            network_type: "cnn".to_string(),
            num_conv_layers: 2,
            conv_channels: vec![32, 64],
            kernel_size: 3,
            cnn_fc_hidden_size: 64,
            cnn_num_fc_layers: 1,
            ..Config::default()
        };

        let model: ActorCriticNetwork<TestBackend> =
            ActorCriticNetwork::new(86, Some((6, 7, 2)), 7, 2, None, &config, &device);

        assert!(model.is_cnn());
        assert!(!model.is_mlp());

        let obs = Tensor::zeros([8, 86], &device);
        let (logits, values) = model.forward(obs);
        assert_eq!(logits.dims(), [8, 7]);
        assert_eq!(values.dims(), [8, 2]);
    }

    #[test]
    fn test_backward_compat_alias() {
        let device = Default::default();
        let config = Config::default();

        // Use the old alias
        let model: ActorCritic<TestBackend> =
            ActorCritic::new(4, None, 2, 1, None, &config, &device);

        let obs = Tensor::zeros([1, 4], &device);
        let (logits, values) = model.forward(obs);
        assert_eq!(logits.dims(), [1, 2]);
        assert_eq!(values.dims(), [1, 1]);
    }

    #[test]
    #[should_panic(expected = "forward() called on CTDE network")]
    fn test_ctde_forward_panics() {
        let device = Default::default();
        let config = Config {
            network_type: "ctde".to_string(),
            ..Config::default()
        };
        let model: ActorCriticNetwork<TestBackend> =
            ActorCriticNetwork::new(10, None, 5, 2, Some(20), &config, &device);
        let obs = Tensor::zeros([8, 10], &device);
        let _ = model.forward(obs); // Should panic
    }

    #[test]
    fn test_ctde_separate_forward_works() {
        let device = Default::default();
        let config = Config {
            network_type: "ctde".to_string(),
            ..Config::default()
        };
        let model: ActorCriticNetwork<TestBackend> =
            ActorCriticNetwork::new(10, None, 5, 2, Some(20), &config, &device);
        let local_obs = Tensor::zeros([8, 10], &device);
        let global_state = Tensor::zeros([8, 20], &device);

        let logits = model.forward_actor(local_obs.clone());
        let values = model.forward_critic(global_state, local_obs);

        assert_eq!(logits.dims(), [8, 5]);
        assert_eq!(values.dims(), [8, 2]);
    }

    #[test]
    fn test_ctde_is_ctde_method() {
        let device = Default::default();
        let config = Config {
            network_type: "ctde".to_string(),
            ..Config::default()
        };
        let model: ActorCriticNetwork<TestBackend> =
            ActorCriticNetwork::new(10, None, 5, 2, Some(20), &config, &device);

        assert!(model.is_ctde());
        assert!(!model.is_mlp());
        assert!(!model.is_cnn());
    }
}
