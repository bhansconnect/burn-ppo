use burn::prelude::*;
use rand::Rng;

/// Sample actions from categorical distribution defined by logits
///
/// Uses the Gumbel-max trick for efficient sampling:
/// argmax(logits + Gumbel noise) ~ Categorical(softmax(logits))
pub fn sample_categorical<B: Backend>(
    logits: Tensor<B, 2>,
    rng: &mut impl Rng,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let [batch, num_actions] = logits.dims();

    // Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
    let uniform: Vec<f32> = (0..batch * num_actions)
        .map(|_| rng.gen_range(1e-10_f32..1.0_f32))
        .collect();
    let uniform = Tensor::<B, 1>::from_floats(uniform.as_slice(), device)
        .reshape([batch, num_actions]);

    let gumbel = -(-uniform.log()).log();

    // Add noise to logits and take argmax
    // argmax(1) returns [batch, 1], squeeze dim 1 to get [batch]
    let noisy_logits = logits + gumbel;
    noisy_logits.argmax(1).squeeze_dims(&[1])
}

/// Compute log probabilities of taken actions under categorical distribution
///
/// logits: [batch, num_actions]
/// actions: [batch]
/// returns: [batch]
pub fn log_prob_categorical<B: Backend>(
    logits: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let log_softmax = burn::tensor::activation::log_softmax(logits, 1);
    gather_1d(log_softmax, actions)
}

/// Compute entropy of categorical distribution from logits
///
/// H = -sum(p * log(p))
pub fn entropy_categorical<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 1> {
    let log_probs = burn::tensor::activation::log_softmax(logits.clone(), 1);
    let probs = burn::tensor::activation::softmax(logits, 1);
    // sum_dim(1) returns [batch, 1], squeeze dim 1 to get [batch]
    -(probs * log_probs).sum_dim(1).squeeze_dims(&[1])
}

/// Gather values at indices along dimension 1
/// Equivalent to: output[i] = input[i, indices[i]]
///
/// Uses Burn's `gather` operation which properly supports gradient flow,
/// unlike the previous boolean-mask approach which blocked gradients.
fn gather_1d<B: Backend>(input: Tensor<B, 2>, indices: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    // Expand indices from [batch] to [batch, 1] for gather
    let indices_2d = indices.unsqueeze_dim(1); // [batch, 1]

    // Gather along dimension 1: selects input[i, indices[i]] for each batch element
    let gathered = input.gather(1, indices_2d); // [batch, 1]

    // Squeeze back to [batch]
    gathered.squeeze_dims(&[1])
}

/// Normalize advantages to zero mean and unit variance
pub fn normalize_advantages<B: Backend>(advantages: Tensor<B, 1>) -> Tensor<B, 1> {
    let mean = advantages.clone().mean();
    let std = advantages.clone().var(0).sqrt();
    (advantages - mean) / (std + 1e-8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use rand::SeedableRng;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_sample_categorical_shape() {
        let device = Default::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let logits: Tensor<TestBackend, 2> = Tensor::zeros([8, 4], &device);
        let actions = sample_categorical(logits, &mut rng, &device);

        assert_eq!(actions.dims(), [8]);
    }

    #[test]
    fn test_sample_categorical_range() {
        let device = Default::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Heavily biased towards action 2
        let logits: Tensor<TestBackend, 2> =
            Tensor::from_floats([[0.0, 0.0, 100.0, 0.0]], &device);
        let actions = sample_categorical(logits, &mut rng, &device);

        let action: i64 = actions.into_scalar();
        assert_eq!(action, 2);
    }

    #[test]
    fn test_log_prob_categorical() {
        let device = Default::default();

        // Uniform logits
        let logits: Tensor<TestBackend, 2> = Tensor::zeros([2, 4], &device);
        let actions: Tensor<TestBackend, 1, Int> = Tensor::from_ints([0, 2], &device);

        let log_probs = log_prob_categorical(logits, actions);

        // With 4 actions and uniform distribution, log_prob = log(0.25) ≈ -1.386
        let expected = 0.25_f32.ln();
        let actual: f32 = log_probs.slice([0..1]).into_scalar();
        assert!((actual - expected).abs() < 1e-4);
    }

    #[test]
    fn test_entropy_categorical() {
        let device = Default::default();

        // Uniform logits -> maximum entropy
        let logits: Tensor<TestBackend, 2> = Tensor::zeros([1, 4], &device);
        let entropy = entropy_categorical(logits);

        // H = log(4) for uniform distribution over 4 actions
        let expected = 4.0_f32.ln();
        let actual: f32 = entropy.into_scalar();
        assert!((actual - expected).abs() < 1e-4);
    }

    #[test]
    fn test_normalize_advantages() {
        let device = Default::default();
        let advantages: Tensor<TestBackend, 1> = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device);

        let normalized = normalize_advantages(advantages);

        // Check mean is approximately 0
        let mean: f32 = normalized.clone().mean().into_scalar();
        assert!(mean.abs() < 1e-5);

        // Check std is approximately 1
        let std: f32 = normalized.var(0).sqrt().into_scalar();
        assert!((std - 1.0).abs() < 1e-4);
    }
}
