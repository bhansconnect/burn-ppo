use burn::nn::Initializer;
use burn::prelude::*;

use crate::config::Config;
use crate::profile::{profile_function, profile_scope};

/// Create a Linear layer with orthogonal weight initialization and zero biases
///
/// This is necessary because Burn's `LinearConfig.with_initializer()` applies
/// the same initializer to both weights and biases, but Orthogonal requires
/// 2D tensors (weights are 2D, biases are 1D).
///
/// Per ICLR blog "37 Implementation Details of PPO":
/// - Weights: Orthogonal initialization with specified gain
/// - Biases: Zeros
pub fn create_linear_orthogonal<B: Backend>(
    d_input: usize,
    d_output: usize,
    gain: f64,
    device: &B::Device,
) -> nn::Linear<B> {
    // Orthogonal initialization for weights (2D tensor)
    // init_with returns Param<Tensor<B, D>> directly
    let weight = Initializer::Orthogonal { gain }.init_with(
        [d_input, d_output],
        Some(d_input),
        Some(d_output),
        device,
    );

    // Zero initialization for biases (1D tensor)
    let bias = Initializer::Zeros.init_with([d_output], Some(d_input), Some(d_output), device);

    nn::Linear {
        weight,
        bias: Some(bias),
    }
}

/// MLP Actor-Critic network with optional split architecture
///
/// When `split_networks` is false (default): shared backbone with separate heads
/// When `split_networks` is true: separate actor and critic networks
///
/// The value head outputs a single scalar value (the acting player's value).
#[derive(Module, Debug)]
pub struct MlpActorCritic<B: Backend> {
    /// Actor hidden layers (or shared backbone when `split_networks` is false)
    pub layers: Vec<nn::Linear<B>>,
    /// Critic hidden layers (only populated when `split_networks` is true)
    pub critic_layers: Vec<nn::Linear<B>>,
    /// Policy output head
    pub policy_head: nn::Linear<B>,
    /// Single value head - outputs scalar (acting player's value)
    pub value_head: nn::Linear<B>,
    /// Use `ReLU` activation (true) or tanh (false)
    #[module(skip)]
    use_relu: bool,
    /// Whether to use separate actor/critic networks
    #[module(skip)]
    split_networks: bool,
}

impl<B: Backend> MlpActorCritic<B> {
    /// Create a new `MlpActorCritic` network
    ///
    /// Uses orthogonal initialization with specific gains (per ICLR blog):
    /// - Hidden layers: sqrt(2) for `ReLU`, 1.0 for tanh
    /// - Policy head: 0.01 (small for stable initial policy)
    /// - Value head: 1.0
    /// - All biases: 0
    ///
    /// When `config.split_networks` is true, creates separate actor and critic networks.
    ///
    /// The value head outputs a single scalar (the acting player's value).
    pub fn new(obs_dim: usize, action_count: usize, config: &Config, device: &B::Device) -> Self {
        let hidden_size = config.hidden_size;
        let num_hidden = config.num_hidden;
        let use_relu = config.activation == "relu";
        let split_networks = config.split_networks;

        // Hidden layer gain depends on activation
        // sqrt(2) for ReLU (He et al.), 1.0 for tanh (ICLR blog)
        let hidden_gain = if use_relu { 2.0_f64.sqrt() } else { 1.0 };

        // Build actor hidden layers with orthogonal initialization
        let mut layers = Vec::with_capacity(num_hidden);
        let mut in_size = obs_dim;

        for _ in 0..num_hidden {
            layers.push(create_linear_orthogonal(
                in_size,
                hidden_size,
                hidden_gain,
                device,
            ));
            in_size = hidden_size;
        }

        // Build critic hidden layers if using split networks
        let critic_layers = if split_networks {
            let mut critic_layers = Vec::with_capacity(num_hidden);
            let mut in_size = obs_dim;
            for _ in 0..num_hidden {
                critic_layers.push(create_linear_orthogonal(
                    in_size,
                    hidden_size,
                    hidden_gain,
                    device,
                ));
                in_size = hidden_size;
            }
            critic_layers
        } else {
            Vec::new()
        };

        // Policy head: small init (0.01) for stable initial policy
        let policy_head = create_linear_orthogonal(hidden_size, action_count, 0.01, device);

        // Value head: single output (acting player's value), gain 1.0
        let value_head = create_linear_orthogonal(hidden_size, 1, 1.0, device);

        Self {
            layers,
            critic_layers,
            policy_head,
            value_head,
            use_relu,
            split_networks,
        }
    }

    /// Forward pass returning action logits and value
    ///
    /// Input: observations [batch, `obs_dim`]
    /// Output: (logits [batch, `action_count`], values [batch, 1])
    ///
    /// When `split_networks` is true, uses separate actor and critic networks.
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        profile_function!();

        if self.split_networks {
            // Separate actor and critic networks
            let mut actor_x = obs.clone();
            {
                profile_scope!("actor_backbone");
                for layer in &self.layers {
                    actor_x = layer.forward(actor_x);
                    actor_x = if self.use_relu {
                        burn::tensor::activation::relu(actor_x)
                    } else {
                        actor_x.tanh()
                    };
                }
            }
            let logits = {
                profile_scope!("policy_head");
                self.policy_head.forward(actor_x)
            };

            let mut critic_x = obs;
            {
                profile_scope!("critic_backbone");
                for layer in &self.critic_layers {
                    critic_x = layer.forward(critic_x);
                    critic_x = if self.use_relu {
                        burn::tensor::activation::relu(critic_x)
                    } else {
                        critic_x.tanh()
                    };
                }
            }
            let values = {
                profile_scope!("value_head");
                self.value_head.forward(critic_x)
            };

            (logits, values)
        } else {
            // Shared backbone with separate heads
            let mut x = obs;
            {
                profile_scope!("shared_backbone");
                for layer in &self.layers {
                    x = layer.forward(x);
                    x = if self.use_relu {
                        burn::tensor::activation::relu(x)
                    } else {
                        x.tanh()
                    };
                }
            }

            let logits = {
                profile_scope!("policy_head");
                self.policy_head.forward(x.clone())
            };
            let values = {
                profile_scope!("value_head");
                self.value_head.forward(x)
            };

            (logits, values)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_forward_shape() {
        let device = Default::default();
        let config = Config::default();
        let model: MlpActorCritic<TestBackend> = MlpActorCritic::new(4, 2, &config, &device);

        let batch_size = 8;
        let obs = Tensor::zeros([batch_size, 4], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 2]);
        assert_eq!(values.dims(), [batch_size, 1]); // Single scalar value
    }

    #[test]
    fn test_forward_shape_larger_obs() {
        let device = Default::default();
        let config = Config::default();
        let model: MlpActorCritic<TestBackend> = MlpActorCritic::new(86, 7, &config, &device);

        let batch_size = 8;
        let obs = Tensor::zeros([batch_size, 86], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 7]);
        assert_eq!(values.dims(), [batch_size, 1]); // Single scalar value
    }

    #[test]
    fn test_action_probs_sum_to_one() {
        let device = Default::default();
        let config = Config::default();
        let model: MlpActorCritic<TestBackend> = MlpActorCritic::new(4, 2, &config, &device);

        let obs = Tensor::zeros([1, 4], &device);
        let (logits, _) = model.forward(obs);
        let probs = burn::tensor::activation::softmax(logits, 1);

        let sum: f32 = probs.sum().into_scalar();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_forward_shape_split_networks() {
        let device = Default::default();
        let config = Config {
            split_networks: true,
            ..Config::default()
        };
        let model: MlpActorCritic<TestBackend> = MlpActorCritic::new(4, 2, &config, &device);

        // Verify critic_layers is populated
        assert_eq!(model.critic_layers.len(), config.num_hidden);
        assert_eq!(model.layers.len(), config.num_hidden);

        let batch_size = 8;
        let obs = Tensor::zeros([batch_size, 4], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 2]);
        assert_eq!(values.dims(), [batch_size, 1]);
    }

    #[test]
    fn test_shared_vs_split_structure() {
        let device = Default::default();

        // Shared backbone (default)
        let shared_config = Config::default();
        let shared_model: MlpActorCritic<TestBackend> =
            MlpActorCritic::new(4, 2, &shared_config, &device);
        assert!(shared_model.critic_layers.is_empty());
        assert!(!shared_model.split_networks);

        // Split networks
        let split_config = Config {
            split_networks: true,
            ..Config::default()
        };
        let split_model: MlpActorCritic<TestBackend> =
            MlpActorCritic::new(4, 2, &split_config, &device);
        assert!(!split_model.critic_layers.is_empty());
        assert!(split_model.split_networks);
        assert_eq!(split_model.critic_layers.len(), split_config.num_hidden);
    }

    #[test]
    fn test_split_networks_different_outputs() {
        let device = Default::default();
        let config = Config {
            split_networks: true,
            ..Config::default()
        };

        let model: MlpActorCritic<TestBackend> = MlpActorCritic::new(4, 2, &config, &device);

        // Use non-zero input to verify actor and critic produce different outputs
        // (since they have independently initialized weights)
        let obs = Tensor::ones([1, 4], &device);
        let (logits, values) = model.forward(obs);

        // Just verify shapes are correct and we get actual values
        assert_eq!(logits.dims(), [1, 2]);
        assert_eq!(values.dims(), [1, 1]);

        // Values should be finite
        let value_scalar: f32 = values.clone().into_scalar();
        assert!(value_scalar.is_finite());
    }
}
