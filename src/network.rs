use burn::nn::Initializer;
use burn::prelude::*;

use crate::config::Config;
use crate::profile::{profile_function, profile_scope};

/// Create a Linear layer with orthogonal weight initialization and zero biases
///
/// This is necessary because Burn's LinearConfig.with_initializer() applies
/// the same initializer to both weights and biases, but Orthogonal requires
/// 2D tensors (weights are 2D, biases are 1D).
///
/// Per ICLR blog "37 Implementation Details of PPO":
/// - Weights: Orthogonal initialization with specified gain
/// - Biases: Zeros
fn create_linear_orthogonal<B: Backend>(
    d_input: usize,
    d_output: usize,
    gain: f64,
    device: &B::Device,
) -> nn::Linear<B> {
    // Orthogonal initialization for weights (2D tensor)
    // init_with returns Param<Tensor<B, D>> directly
    let weight = Initializer::Orthogonal { gain }
        .init_with([d_input, d_output], Some(d_input), Some(d_output), device);

    // Zero initialization for biases (1D tensor)
    let bias = Initializer::Zeros
        .init_with([d_output], Some(d_input), Some(d_output), device);

    nn::Linear {
        weight,
        bias: Some(bias),
    }
}

/// Actor-Critic network with shared backbone and separate heads
#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    /// Hidden layers (shared backbone)
    pub layers: Vec<nn::Linear<B>>,
    /// Policy output head
    pub policy_head: nn::Linear<B>,
    /// Single value head with N outputs (one per player)
    pub value_head: nn::Linear<B>,
    /// Use ReLU activation (true) or tanh (false)
    #[module(skip)]
    use_relu: bool,
    /// Number of players (for value head output dimension)
    #[module(skip)]
    num_players: usize,
}

impl<B: Backend> ActorCritic<B> {
    /// Create a new ActorCritic network
    ///
    /// Uses orthogonal initialization with specific gains (per ICLR blog):
    /// - Hidden layers: sqrt(2) for ReLU, 1.0 for tanh
    /// - Policy head: 0.01 (small for stable initial policy)
    /// - Value head: 1.0
    /// - All biases: 0
    ///
    /// `num_players`: Number of value outputs (1 for single-agent, 2+ for multi-player)
    pub fn new(
        obs_dim: usize,
        action_count: usize,
        num_players: usize,
        config: &Config,
        device: &B::Device,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let num_hidden = config.num_hidden;
        let use_relu = config.activation == "relu";

        // Hidden layer gain depends on activation
        // sqrt(2) for ReLU (He et al.), 1.0 for tanh (ICLR blog)
        let hidden_gain = if use_relu { 2.0_f64.sqrt() } else { 1.0 };

        // Build hidden layers with orthogonal initialization
        let mut layers = Vec::with_capacity(num_hidden);
        let mut in_size = obs_dim;

        for _ in 0..num_hidden {
            layers.push(create_linear_orthogonal(in_size, hidden_size, hidden_gain, device));
            in_size = hidden_size;
        }

        // Policy head: small init (0.01) for stable initial policy
        let policy_head = create_linear_orthogonal(hidden_size, action_count, 0.01, device);

        // Value head: N outputs (one per player), gain 1.0
        let value_head = create_linear_orthogonal(hidden_size, num_players, 1.0, device);

        Self {
            layers,
            policy_head,
            value_head,
            use_relu,
            num_players,
        }
    }

    /// Forward pass returning action logits and N player values
    ///
    /// Input: observations [batch, obs_dim]
    /// Output: (logits [batch, action_count], values [batch, num_players])
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        profile_function!();
        let mut x = obs;

        // Shared backbone with configured activation
        {
            profile_scope!("async_backbone");
            for layer in &self.layers {
                x = layer.forward(x);
                x = if self.use_relu {
                    burn::tensor::activation::relu(x)
                } else {
                    x.tanh()
                };
            }
        }

        // Separate heads
        let logits = {
            profile_scope!("async_policy_head");
            self.policy_head.forward(x.clone())
        };
        let values = {
            profile_scope!("async_value_head");
            self.value_head.forward(x) // [batch, num_players]
        };

        (logits, values)
    }

    /// Get number of players (value head output dimension)
    pub fn num_players(&self) -> usize {
        self.num_players
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_forward_shape_single_player() {
        let device = Default::default();
        let config = Config::default();
        // Single-player: num_players = 1
        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, 1, &config, &device);

        let batch_size = 8;
        let obs = Tensor::zeros([batch_size, 4], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 2]);
        assert_eq!(values.dims(), [batch_size, 1]); // [batch, num_players]
    }

    #[test]
    fn test_forward_shape_multi_player() {
        let device = Default::default();
        let config = Config::default();
        // Two-player game
        let model: ActorCritic<TestBackend> = ActorCritic::new(86, 7, 2, &config, &device);

        let batch_size = 8;
        let obs = Tensor::zeros([batch_size, 86], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 7]);
        assert_eq!(values.dims(), [batch_size, 2]); // [batch, num_players]
    }

    #[test]
    fn test_action_probs_sum_to_one() {
        let device = Default::default();
        let config = Config::default();
        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, 1, &config, &device);

        let obs = Tensor::zeros([1, 4], &device);
        let (logits, _) = model.forward(obs);
        let probs = burn::tensor::activation::softmax(logits, 1);

        let sum: f32 = probs.sum().into_scalar();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_num_players() {
        let device = Default::default();
        let config = Config::default();

        let model1: ActorCritic<TestBackend> = ActorCritic::new(4, 2, 1, &config, &device);
        assert_eq!(model1.num_players(), 1);

        let model2: ActorCritic<TestBackend> = ActorCritic::new(86, 7, 2, &config, &device);
        assert_eq!(model2.num_players(), 2);
    }
}
