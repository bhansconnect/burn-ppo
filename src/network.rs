use burn::nn::Initializer;
use burn::prelude::*;

use crate::config::Config;

/// Actor-Critic network with shared backbone and separate heads
#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    layers: Vec<nn::Linear<B>>,
    policy_head: nn::Linear<B>,
    value_head: nn::Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    /// Create a new ActorCritic network
    ///
    /// Uses orthogonal initialization with specific gains:
    /// - Hidden layers: sqrt(2)
    /// - Policy head: 0.01 (small for stable initial policy)
    /// - Value head: 1.0
    pub fn new(obs_dim: usize, action_count: usize, config: &Config, device: &B::Device) -> Self {
        let hidden_size = config.hidden_size;
        let num_hidden = config.num_hidden;

        // Hidden layer initializer: orthogonal with gain sqrt(2)
        let hidden_init = Initializer::KaimingNormal {
            gain: 2.0_f64.sqrt(),
            fan_out_only: false,
        };

        // Build hidden layers
        let mut layers = Vec::with_capacity(num_hidden);
        let mut in_size = obs_dim;

        for _ in 0..num_hidden {
            layers.push(
                nn::LinearConfig::new(in_size, hidden_size)
                    .with_initializer(hidden_init.clone())
                    .init(device),
            );
            in_size = hidden_size;
        }

        // Policy head: small init for stable initial policy
        let policy_init = Initializer::KaimingNormal {
            gain: 0.01,
            fan_out_only: false,
        };
        let policy_head = nn::LinearConfig::new(hidden_size, action_count)
            .with_initializer(policy_init)
            .init(device);

        // Value head: gain 1.0
        let value_init = Initializer::KaimingNormal {
            gain: 1.0,
            fan_out_only: false,
        };
        let value_head = nn::LinearConfig::new(hidden_size, 1)
            .with_initializer(value_init)
            .init(device);

        Self {
            layers,
            policy_head,
            value_head,
        }
    }

    /// Forward pass returning action logits and state value
    ///
    /// Input: observations [batch, obs_dim]
    /// Output: (logits [batch, action_count], values [batch])
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let mut x = obs;

        // Shared backbone with tanh activation
        for layer in &self.layers {
            x = layer.forward(x);
            x = x.tanh();
        }

        // Separate heads
        let logits = self.policy_head.forward(x.clone());
        // Value head output is [batch, 1], squeeze dim 1 to get [batch]
        let value: Tensor<B, 1> = self.value_head.forward(x).squeeze_dims(&[1]);

        (logits, value)
    }

    /// Get action probabilities from logits using softmax
    pub fn action_probs(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::softmax(logits, 1)
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
        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, &config, &device);

        let batch_size = 8;
        let obs = Tensor::zeros([batch_size, 4], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 2]);
        assert_eq!(values.dims(), [batch_size]);
    }

    #[test]
    fn test_action_probs_sum_to_one() {
        let device = Default::default();
        let config = Config::default();
        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, &config, &device);

        let obs = Tensor::zeros([1, 4], &device);
        let (logits, _) = model.forward(obs);
        let probs = model.action_probs(logits);

        let sum: f32 = probs.sum().into_scalar();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
