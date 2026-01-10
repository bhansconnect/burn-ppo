//! Neural network architectures for PPO training
//!
//! This module provides different network architectures that can be selected via config:
//! - MLP (Multi-Layer Perceptron): Default, works for any observation
//! - CNN (Convolutional Neural Network): For spatial observations like board games

mod cnn;
mod mlp;

pub use cnn::CnnActorCritic;
pub use mlp::MlpActorCritic;

use burn::module::Module;
use burn::prelude::*;

use crate::config::Config;
#[cfg(feature = "tracy")]
use crate::profile::gpu_sync;

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
}

impl<B: Backend> ActorCriticNetwork<B> {
    /// Create a new actor-critic network based on config
    ///
    /// # Arguments
    /// * `obs_dim` - Total observation dimension
    /// * `obs_shape` - Optional spatial shape (height, width, channels) for CNN
    /// * `action_count` - Number of discrete actions
    /// * `num_players` - Number of value outputs (1 for single-agent, 2+ for multi-player)
    /// * `config` - Configuration specifying network type and parameters
    /// * `device` - Compute device
    ///
    /// # Panics
    /// Panics if `network_type = "cnn"` but `obs_shape` is None
    pub fn new(
        obs_dim: usize,
        obs_shape: Option<(usize, usize, usize)>,
        action_count: usize,
        num_players: usize,
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
            other => panic!("Unknown network_type: {other}. Use 'mlp' or 'cnn'"),
        }
    }

    /// Forward pass returning action logits and N player values
    ///
    /// Input: observations [batch, `obs_dim`]
    /// Output: (logits [batch, `action_count`], values [batch, `num_players`])
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self {
            Self::Mlp(mlp) => mlp.forward(obs),
            Self::Cnn(cnn) => cnn.forward(obs),
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

    /// Returns a tensor suitable for GPU synchronization (tracy profiling)
    ///
    /// This forces a GPU sync by reading weight data, ensuring accurate profiling timing.
    #[cfg(feature = "tracy")]
    pub fn sync_optimize(&self) {
        match self {
            Self::Mlp(mlp) => gpu_sync!(mlp.layers[0].weight.val()),
            Self::Cnn(cnn) => gpu_sync!(cnn.conv_layers[0].weight.val()),
        }
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
            ActorCriticNetwork::new(4, None, 2, 1, &config, &device);

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
            ActorCriticNetwork::new(86, Some((6, 7, 2)), 7, 2, &config, &device);

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
        let model: ActorCritic<TestBackend> = ActorCritic::new(4, None, 2, 1, &config, &device);

        let obs = Tensor::zeros([1, 4], &device);
        let (logits, values) = model.forward(obs);
        assert_eq!(logits.dims(), [1, 2]);
        assert_eq!(values.dims(), [1, 1]);
    }
}
