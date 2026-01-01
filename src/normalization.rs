/// Running observation normalizer using Welford's online algorithm
///
/// Tracks mean and variance of observations and normalizes new observations
/// to approximately zero mean and unit variance.
use serde::{Deserialize, Serialize};

/// Running statistics normalizer for observations
///
/// Uses Welford's online algorithm for numerically stable mean/variance updates.
/// Must be serialized with checkpoints for inference use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsNormalizer {
    /// Running mean for each observation dimension
    mean: Vec<f64>,
    /// Running variance for each observation dimension (using M2 for Welford)
    var: Vec<f64>,
    /// Number of samples seen
    count: f64,
    /// Clipping range for normalized values (typically 10.0)
    clip: f32,
}

impl ObsNormalizer {
    /// Create a new normalizer for observations of given dimension
    pub fn new(obs_dim: usize, clip: f32) -> Self {
        Self {
            mean: vec![0.0; obs_dim],
            var: vec![0.0; obs_dim], // M2 accumulator for Welford's algorithm
            count: 0.0,
            clip,
        }
    }

    /// Update running statistics with a batch of observations
    ///
    /// obs_batch: flat array of [batch_size * obs_dim]
    pub fn update_batch(&mut self, obs_batch: &[f32], obs_dim: usize) {
        let batch_size = obs_batch.len() / obs_dim;

        for i in 0..batch_size {
            self.count += 1.0;
            let offset = i * obs_dim;
            for j in 0..obs_dim {
                let x = obs_batch[offset + j] as f64;

                // Welford's online update
                let delta = x - self.mean[j];
                self.mean[j] += delta / self.count;
                let delta2 = x - self.mean[j];
                self.var[j] += delta * delta2;
            }
        }
    }

    /// Normalize a batch of observations in-place
    ///
    /// obs_batch: flat array of [batch_size * obs_dim], modified in place
    pub fn normalize_batch(&self, obs_batch: &mut [f32], obs_dim: usize) {
        // Need at least 2 samples for meaningful variance
        if self.count < 2.0 {
            return;
        }

        let batch_size = obs_batch.len() / obs_dim;

        for i in 0..batch_size {
            let offset = i * obs_dim;
            for j in 0..obs_dim {
                let variance = self.var[j] / self.count;
                let std = variance.sqrt().max(1e-8);
                let normalized = ((obs_batch[offset + j] as f64 - self.mean[j]) / std) as f32;
                obs_batch[offset + j] = normalized.clamp(-self.clip, self.clip);
            }
        }
    }

    /// Normalize a single observation, returning a new Vec
    pub fn normalize(&self, obs: &[f32]) -> Vec<f32> {
        // Need at least 2 samples for meaningful variance
        if self.count < 2.0 {
            return obs.to_vec();
        }

        obs.iter()
            .enumerate()
            .map(|(j, &x)| {
                let variance = self.var[j] / self.count;
                let std = variance.sqrt().max(1e-8);
                let normalized = ((x as f64 - self.mean[j]) / std) as f32;
                normalized.clamp(-self.clip, self.clip)
            })
            .collect()
    }

    /// Get the observation dimension
    pub fn obs_dim(&self) -> usize {
        self.mean.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer_creation() {
        let norm = ObsNormalizer::new(4, 10.0);
        assert_eq!(norm.obs_dim(), 4);
    }

    #[test]
    fn test_normalizer_update_and_normalize() {
        let mut norm = ObsNormalizer::new(2, 10.0);

        // Update with some observations
        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 observations of dim 2
        norm.update_batch(&obs, 2);

        // Mean should be approximately [3, 4]
        assert!((norm.mean[0] - 3.0).abs() < 0.1);
        assert!((norm.mean[1] - 4.0).abs() < 0.1);

        // Normalize should center around 0
        let mut test_obs = vec![3.0, 4.0];
        norm.normalize_batch(&mut test_obs, 2);

        // Values at the mean should normalize to ~0
        assert!(test_obs[0].abs() < 0.5);
        assert!(test_obs[1].abs() < 0.5);
    }

    #[test]
    fn test_normalizer_clipping() {
        let mut norm = ObsNormalizer::new(1, 5.0);

        // Update with narrow range
        let obs = vec![0.0, 1.0, 0.0, 1.0];
        norm.update_batch(&obs, 1);

        // Extreme value should be clipped
        let mut extreme = vec![1000.0];
        norm.normalize_batch(&mut extreme, 1);

        assert!(extreme[0] <= 5.0);
        assert!(extreme[0] >= -5.0);
    }
}
