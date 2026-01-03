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
    /// `obs_batch`: flat array of [`batch_size` * `obs_dim`]
    pub fn update_batch(&mut self, obs_batch: &[f32], obs_dim: usize) {
        let batch_size = obs_batch.len() / obs_dim;

        for i in 0..batch_size {
            self.count += 1.0;
            let offset = i * obs_dim;
            for j in 0..obs_dim {
                let x = f64::from(obs_batch[offset + j]);

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
    /// `obs_batch`: flat array of [`batch_size` * `obs_dim`], modified in place
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
                let normalized = ((f64::from(obs_batch[offset + j]) - self.mean[j]) / std) as f32;
                obs_batch[offset + j] = normalized.clamp(-self.clip, self.clip);
            }
        }
    }

    /// Normalize a single observation, returning a new Vec
    ///
    /// Used for single-observation inference (eval mode). Training uses `normalize_batch()`.
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
                let normalized = ((f64::from(x) - self.mean[j]) / std) as f32;
                normalized.clamp(-self.clip, self.clip)
            })
            .collect()
    }

    /// Get the observation dimension
    #[cfg(test)]
    pub const fn obs_dim(&self) -> usize {
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

    #[test]
    fn test_normalize_does_not_modify_stats() {
        // Verify that normalize_batch doesn't change normalizer state
        let mut norm = ObsNormalizer::new(2, 10.0);

        // Pre-populate with some data
        norm.update_batch(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);

        let count_before = norm.count;
        let mean_before = norm.mean.clone();
        let var_before = norm.var.clone();

        // Normalize some data - should NOT modify state
        let mut obs = vec![100.0, 200.0];
        norm.normalize_batch(&mut obs, 2);

        // Stats should be unchanged
        assert_eq!(norm.count, count_before);
        assert_eq!(norm.mean, mean_before);
        assert_eq!(norm.var, var_before);
    }

    #[test]
    fn test_normalize_with_insufficient_samples() {
        // Verify count < 2 returns identity (unnormalized)
        let mut norm = ObsNormalizer::new(2, 10.0);

        // Only 1 sample - not enough for variance
        norm.update_batch(&[5.0, 10.0], 2);
        assert_eq!(norm.count, 1.0);

        let mut obs = vec![3.0, 7.0];
        let original = obs.clone();
        norm.normalize_batch(&mut obs, 2);

        // Should return unchanged (identity transform)
        assert_eq!(obs, original);
    }

    #[test]
    fn test_welford_correctness() {
        // Verify Welford's algorithm computes correct mean and variance
        let mut norm = ObsNormalizer::new(1, 10.0);

        // Known data: [1, 2, 3, 4, 5] -> mean=3, population_variance=2
        norm.update_batch(&[1.0, 2.0, 3.0, 4.0, 5.0], 1);

        // Mean should be 3.0
        assert!((norm.mean[0] - 3.0).abs() < 1e-6);

        // Variance = M2/count = 2.0 (population variance)
        let variance = norm.var[0] / norm.count;
        assert!((variance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_update_equals_sequential() {
        // Verify batch update gives same result as sequential updates
        let mut norm1 = ObsNormalizer::new(1, 10.0);
        let mut norm2 = ObsNormalizer::new(1, 10.0);

        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Update all at once
        norm1.update_batch(&obs, 1);

        // Update one by one
        for &o in &obs {
            norm2.update_batch(&[o], 1);
        }

        // Results should be identical (Welford's is order-independent)
        assert!((norm1.mean[0] - norm2.mean[0]).abs() < 1e-10);
        assert!((norm1.var[0] - norm2.var[0]).abs() < 1e-10);
        assert_eq!(norm1.count, norm2.count);
    }

    #[test]
    fn test_lagged_normalization_behavior() {
        // Test that stats can be used for normalization BEFORE being updated
        // This simulates the correct PPO behavior: normalize with old stats,
        // then update stats for next rollout
        let mut norm = ObsNormalizer::new(1, 10.0);

        // Initial data to establish baseline stats
        norm.update_batch(&[0.0, 10.0], 1);
        // mean = 5.0, var = 50.0, count = 2

        // Save state before new batch
        let mean_before = norm.mean[0];
        let var_before = norm.var[0];

        // New observation to normalize
        let new_obs_raw = 15.0;
        let mut new_obs = vec![new_obs_raw];

        // Step 1: Normalize using EXISTING stats (before update)
        norm.normalize_batch(&mut new_obs, 1);

        // Expected normalized value: (15 - 5) / sqrt(50/2) = 10/5 = 2.0
        let expected = (f64::from(new_obs_raw) - mean_before) / (var_before / 2.0).sqrt();
        assert!((f64::from(new_obs[0]) - expected).abs() < 0.01);

        // Step 2: Update stats with raw observation (for next rollout)
        norm.update_batch(&[new_obs_raw], 1);

        // Stats should now be updated
        assert!(norm.count > 2.0);
        assert!((norm.mean[0] - mean_before).abs() > 0.1); // Mean changed
    }
}
