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

/// Running return normalizer for reward normalization
///
/// Normalizes rewards by dividing by the standard deviation of rolling discounted returns.
/// This follows the Stable Baselines3 `VecNormalize` approach:
/// 1. Track rolling discounted returns per player per environment
/// 2. Update running variance statistics on these returns (Welford's algorithm)
/// 3. Normalize rewards by dividing by sqrt(variance)
///
/// Key design: Tracks returns per-player-per-env (not just per-env) to match GAE's
/// gamma application - gamma is only applied between a player's own actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnNormalizer {
    /// Rolling discounted returns per environment per player `[num_envs][num_players]`
    /// Gamma is only applied when that specific player acts
    returns: Vec<Vec<f64>>,
    /// Running variance using Welford's algorithm (M2 accumulator)
    var: f64,
    /// Running mean (needed for Welford's variance, not used in normalization)
    mean: f64,
    /// Number of samples seen
    count: f64,
    /// Discount factor for return computation
    gamma: f64,
    /// Clipping range for normalized rewards
    clip: f32,
    /// Number of players
    num_players: usize,
    /// Small epsilon for numerical stability
    epsilon: f64,
}

impl ReturnNormalizer {
    /// Create a new return normalizer
    ///
    /// # Arguments
    /// * `num_envs` - Number of parallel environments
    /// * `num_players` - Number of players (1 for single-player games)
    /// * `gamma` - Discount factor (typically 0.99)
    /// * `clip` - Clipping range for normalized rewards (typically 10.0)
    pub fn new(num_envs: usize, num_players: usize, gamma: f64, clip: f32) -> Self {
        Self {
            returns: vec![vec![0.0; num_players]; num_envs],
            var: 0.0,
            mean: 0.0,
            count: 0.0,
            gamma,
            clip,
            num_players,
            epsilon: 1e-8,
        }
    }

    /// Update rolling return for a specific player in a specific environment
    ///
    /// Gamma is applied per-player, not per-step. This matches GAE's approach
    /// where gamma represents discounting between a player's own decision points.
    ///
    /// Note: Does NOT automatically reset on done - caller should call `reset_player`
    /// AFTER `update_variance_stats` to ensure the terminal return is captured.
    pub fn update_return(&mut self, env_idx: usize, player: usize, reward: f32) {
        // Apply gamma and add reward
        self.returns[env_idx][player] =
            self.returns[env_idx][player] * self.gamma + f64::from(reward);
    }

    /// Reset rolling return for a specific player (call after episode ends)
    pub fn reset_player(&mut self, env_idx: usize, player: usize) {
        self.returns[env_idx][player] = 0.0;
    }

    /// Add current rolling return to Welford variance statistics
    ///
    /// Call this only for learner turns to ensure variance reflects learner experience.
    pub fn update_variance_stats(&mut self, env_idx: usize, player: usize) {
        let x = self.returns[env_idx][player];
        self.count += 1.0;

        // Welford's online update
        let delta = x - self.mean;
        self.mean += delta / self.count;
        let delta2 = x - self.mean;
        self.var += delta * delta2;
    }

    /// Normalize a reward using current variance statistics
    ///
    /// Uses variance-only normalization (no mean subtraction) to preserve
    /// reward sign and magnitude relationships.
    pub fn normalize(&self, reward: f32) -> f32 {
        // Need at least 2 samples for meaningful variance
        if self.count < 2.0 {
            return reward;
        }

        let variance = self.var / self.count;
        let std = (variance + self.epsilon).sqrt();
        let normalized = f64::from(reward) / std;
        (normalized as f32).clamp(-self.clip, self.clip)
    }

    /// Reset all player returns for an environment (on episode end)
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "API for future use or convenience")
    )]
    pub fn reset_env(&mut self, env_idx: usize) {
        for player in 0..self.num_players {
            self.returns[env_idx][player] = 0.0;
        }
    }

    /// Convenience method for single-player: update returns, stats, and normalize all rewards
    ///
    /// For single-player games, this handles the full update cycle:
    /// 1. Update rolling returns for each env
    /// 2. Update variance statistics
    /// 3. Normalize rewards in-place
    /// 4. Reset rolling returns on episode end
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "API for future single-player convenience")
    )]
    pub fn update_and_normalize_all(&mut self, rewards: &mut [f32], dones: &[bool]) {
        let num_envs = rewards.len();
        assert_eq!(num_envs, dones.len());
        assert_eq!(num_envs, self.returns.len());

        for env_idx in 0..num_envs {
            let reward = rewards[env_idx];
            let done = dones[env_idx];

            // Update rolling return (player 0 for single-player)
            self.update_return(env_idx, 0, reward);

            // Update variance stats
            self.update_variance_stats(env_idx, 0);

            // Normalize reward
            rewards[env_idx] = self.normalize(reward);

            // Reset rolling return on episode end (after stats captured)
            if done {
                self.reset_player(env_idx, 0);
            }
        }
    }

    /// Get variance for debugging/logging
    #[cfg(test)]
    pub fn variance(&self) -> f64 {
        if self.count < 2.0 {
            0.0
        } else {
            self.var / self.count
        }
    }
}

/// Value normalization with weight rescaling
///
/// Implements the algorithm from "Learning values across many orders of magnitude"
/// (van Hasselt et al., 2016). Maintains running mean and standard deviation of value targets
/// (returns), and rescales value head weights when statistics change to preserve output semantics.
///
/// Key difference from `ReturnNormalizer`:
/// - `ReturnNormalizer`: Normalizes rewards during rollout collection
/// - Value normalization: Normalizes value targets during loss and rescales critic weights
///
/// Uses unified (single scalar) statistics across all players, since the network outputs
/// a single value estimate from the acting player's perspective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopArtNormalizer {
    /// Running mean
    mean: f64,
    /// Running M2 for Welford's variance computation
    var: f64,
    /// Sample count
    count: f64,
    /// Small epsilon for numerical stability
    epsilon: f64,
}

impl PopArtNormalizer {
    /// Create a new value normalizer
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            var: 0.0,
            count: 0.0,
            epsilon: 1e-4,
        }
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current standard deviation
    ///
    /// Returns 1.0 before initialization to avoid rescaling with invalid statistics.
    pub fn std(&self) -> f64 {
        if self.count < 2.0 {
            1.0
        } else {
            (self.var / self.count + self.epsilon).sqrt()
        }
    }

    /// Check if normalizer is initialized (has enough samples)
    pub fn is_initialized(&self) -> bool {
        self.count >= 2.0
    }

    /// Update statistics with a batch of returns
    ///
    /// Uses Welford's online algorithm for numerically stable mean/variance computation.
    ///
    /// # Returns
    /// Tuple of (`old_mean`, `old_std`) for value head rescaling
    pub fn update(&mut self, returns: &[f32]) -> (f64, f64) {
        let old_mean = self.mean;
        let old_std = self.std();

        for &ret in returns {
            let x = f64::from(ret);
            self.count += 1.0;
            let delta = x - self.mean;
            self.mean += delta / self.count;
            let delta2 = x - self.mean;
            self.var += delta * delta2;
        }

        (old_mean, old_std)
    }

    /// Normalize returns for loss computation: (return - mean) / std
    ///
    /// Returns identity if not initialized.
    pub fn normalize(&self, returns: &[f32]) -> Vec<f32> {
        if self.count < 2.0 {
            return returns.to_vec();
        }
        let mean = self.mean;
        let std = self.std();
        returns
            .iter()
            .map(|&r| ((f64::from(r) - mean) / std) as f32)
            .collect()
    }

    /// Denormalize values in-place: value * std + mean
    pub fn denormalize(&self, values: &mut [f32]) {
        if self.count < 2.0 {
            return;
        }
        let std = self.std();
        let mean = self.mean;
        for v in values.iter_mut() {
            *v = (f64::from(*v) * std + mean) as f32;
        }
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

    // ======== ReturnNormalizer Tests ========

    #[test]
    fn test_return_normalizer_creation() {
        let norm = ReturnNormalizer::new(4, 2, 0.99, 10.0);
        assert_eq!(norm.returns.len(), 4);
        assert_eq!(norm.returns[0].len(), 2);
        assert_eq!(norm.num_players, 2);
    }

    #[test]
    fn test_return_normalizer_rolling_return() {
        let mut norm = ReturnNormalizer::new(2, 1, 0.99, 10.0);

        // Update with reward 1.0 for env 0, player 0
        norm.update_return(0, 0, 1.0);
        assert!((norm.returns[0][0] - 1.0).abs() < 1e-6);

        // Update again - should apply gamma
        norm.update_return(0, 0, 1.0);
        // Expected: 0.99 * 1.0 + 1.0 = 1.99
        assert!((norm.returns[0][0] - 1.99).abs() < 1e-6);
    }

    #[test]
    fn test_return_normalizer_reset_player() {
        let mut norm = ReturnNormalizer::new(2, 1, 0.99, 10.0);

        // Build up some return
        norm.update_return(0, 0, 10.0);
        assert!(norm.returns[0][0] > 0.0);

        // reset_player should reset
        norm.reset_player(0, 0);
        assert_eq!(norm.returns[0][0], 0.0);
    }

    #[test]
    fn test_return_normalizer_variance_stats() {
        let mut norm = ReturnNormalizer::new(1, 1, 0.99, 10.0);

        // Add several samples - each is a separate episode
        for reward in [1.0, 2.0, 3.0, 4.0, 5.0] {
            norm.update_return(0, 0, reward);
            norm.update_variance_stats(0, 0);
            norm.reset_player(0, 0); // Reset after stats captured
        }

        // After 5 samples, variance should be computed
        assert_eq!(norm.count, 5.0);
        let variance = norm.variance();
        // Variance of [1,2,3,4,5] = 2.0
        assert!((variance - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_return_normalizer_normalize() {
        let mut norm = ReturnNormalizer::new(1, 1, 0.99, 10.0);

        // Build up variance stats - each is a separate episode
        for reward in [1.0, 2.0, 3.0, 4.0, 5.0] {
            norm.update_return(0, 0, reward);
            norm.update_variance_stats(0, 0);
            norm.reset_player(0, 0); // Reset after stats captured
        }

        // Normalize a reward
        let normalized = norm.normalize(2.0);
        // With variance ~2.0, std ~1.41, normalized should be ~1.41
        // (no mean subtraction, just divide by std)
        assert!(normalized > 0.0); // Positive reward stays positive
        assert!(normalized < 10.0); // Within clip bounds
    }

    #[test]
    fn test_return_normalizer_clipping() {
        let mut norm = ReturnNormalizer::new(1, 1, 0.99, 5.0); // Clip at 5

        // Build up very small variance
        for _ in 0..10 {
            norm.update_return(0, 0, 1.0);
            norm.update_variance_stats(0, 0);
            norm.reset_player(0, 0);
        }

        // Large reward should be clipped
        let normalized = norm.normalize(100.0);
        assert!(normalized <= 5.0);
        assert!(normalized >= -5.0);
    }

    #[test]
    fn test_return_normalizer_no_normalize_insufficient_samples() {
        let mut norm = ReturnNormalizer::new(1, 1, 0.99, 10.0);

        // Only 1 sample - should return raw reward
        norm.update_return(0, 0, 5.0);
        norm.update_variance_stats(0, 0);
        norm.reset_player(0, 0);

        let normalized = norm.normalize(10.0);
        assert_eq!(normalized, 10.0); // Unchanged
    }

    #[test]
    fn test_return_normalizer_per_player_tracking() {
        let mut norm = ReturnNormalizer::new(1, 2, 0.99, 10.0);

        // Player 0 acts
        norm.update_return(0, 0, 1.0);
        assert!((norm.returns[0][0] - 1.0).abs() < 1e-6);
        assert_eq!(norm.returns[0][1], 0.0); // Player 1 unchanged

        // Player 1 acts - player 0's return should NOT change
        norm.update_return(0, 1, 2.0);
        assert!((norm.returns[0][0] - 1.0).abs() < 1e-6); // Still 1.0, no gamma
        assert!((norm.returns[0][1] - 2.0).abs() < 1e-6);

        // Player 0 acts again - NOW gamma is applied
        norm.update_return(0, 0, 1.0);
        // Expected: 0.99 * 1.0 + 1.0 = 1.99
        assert!((norm.returns[0][0] - 1.99).abs() < 1e-6);
    }

    #[test]
    fn test_return_normalizer_reset_env() {
        let mut norm = ReturnNormalizer::new(2, 2, 0.99, 10.0);

        // Build up returns
        norm.update_return(0, 0, 1.0);
        norm.update_return(0, 1, 2.0);
        norm.update_return(1, 0, 3.0);

        // Reset env 0
        norm.reset_env(0);

        assert_eq!(norm.returns[0][0], 0.0);
        assert_eq!(norm.returns[0][1], 0.0);
        assert!((norm.returns[1][0] - 3.0).abs() < 1e-6); // Env 1 unchanged
    }

    #[test]
    fn test_return_normalizer_update_and_normalize_all() {
        let mut norm = ReturnNormalizer::new(3, 1, 0.99, 10.0);

        // First pass to build up stats
        let mut rewards = vec![1.0, 2.0, 3.0];
        let dones = vec![true, true, true];
        norm.update_and_normalize_all(&mut rewards, &dones);

        // Second pass
        let mut rewards2 = vec![1.0, 2.0, 3.0];
        let dones2 = vec![true, true, true];
        norm.update_and_normalize_all(&mut rewards2, &dones2);

        // Third pass - normalization should have effect now
        let mut rewards3 = vec![2.0, 2.0, 2.0];
        let dones3 = vec![false, false, false];
        norm.update_and_normalize_all(&mut rewards3, &dones3);

        // Rewards should be normalized (not all equal to 2.0)
        // With variance-only normalization, sign is preserved
        assert!(rewards3[0] > 0.0);
    }

    #[test]
    fn test_return_normalizer_serialize_deserialize() {
        let mut norm = ReturnNormalizer::new(2, 2, 0.99, 10.0);

        // Build up some state
        norm.update_return(0, 0, 5.0);
        norm.update_variance_stats(0, 0);
        norm.update_return(1, 1, 3.0);
        norm.update_variance_stats(1, 1);

        // Serialize and deserialize
        let json = serde_json::to_string(&norm).unwrap();
        let loaded: ReturnNormalizer = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.returns[0][0], norm.returns[0][0]);
        assert_eq!(loaded.count, norm.count);
        assert!((loaded.var - norm.var).abs() < 1e-10);
    }

    // ======== PopArtNormalizer Tests ========

    #[test]
    fn test_popart_creation() {
        let norm = PopArtNormalizer::new();
        assert!(!norm.is_initialized());
        // Returns 1.0 std when not initialized (to avoid rescaling)
        assert_eq!(norm.std(), 1.0);
    }

    #[test]
    fn test_popart_update() {
        let mut norm = PopArtNormalizer::new();

        // First sample - not yet initialized (need at least 2)
        let (_old_mean, old_std) = norm.update(&[10.0]);
        assert!(!norm.is_initialized());
        assert_eq!(old_std, 1.0); // Old std was 1.0 (uninitialized)

        // Second sample - now initialized
        let (_old_mean, _old_std) = norm.update(&[20.0]);
        assert!(norm.is_initialized());

        // Mean should be 15.0
        assert!((norm.mean() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_popart_multi_sample_update() {
        let mut norm = PopArtNormalizer::new();

        // [0, 10, 20, 30, 40] -> mean=20, population variance=200
        norm.update(&[0.0, 10.0, 20.0, 30.0, 40.0]);

        assert!((norm.mean() - 20.0).abs() < 0.01);
        // std = sqrt(200 + epsilon) ≈ 14.14
        let std = norm.std();
        assert!(std > 14.0 && std < 15.0, "std was {std}");
    }

    #[test]
    fn test_popart_normalize() {
        let mut norm = PopArtNormalizer::new();

        // [0, 10, 20, 30, 40] -> mean=20, variance=200, std~=14.14
        norm.update(&[0.0, 10.0, 20.0, 30.0, 40.0]);

        // Normalizing the mean should give ~0
        let normalized = norm.normalize(&[20.0]);
        assert!(normalized[0].abs() < 0.1);

        // Normalizing value above mean should give positive
        let normalized = norm.normalize(&[34.14]); // ~1 std above mean
        assert!(normalized[0] > 0.5 && normalized[0] < 1.5);
    }

    #[test]
    fn test_popart_returns_old_stats() {
        let mut norm = PopArtNormalizer::new();

        // Add initial samples
        norm.update(&[0.0, 10.0]);
        let mean_before = norm.mean();
        let std_before = norm.std();

        // Update returns OLD stats
        let (old_mean, old_std) = norm.update(&[20.0]);

        // Old stats should match what we captured
        assert!((old_mean - mean_before).abs() < 1e-10);
        assert!((old_std - std_before).abs() < 1e-10);

        // New stats should be different
        assert!(norm.mean() != mean_before);
    }

    #[test]
    fn test_popart_serialization() {
        let mut norm = PopArtNormalizer::new();
        norm.update(&[1.0, 2.0, 100.0, 3.0, 200.0]);

        let json = serde_json::to_string(&norm).unwrap();
        let loaded: PopArtNormalizer = serde_json::from_str(&json).unwrap();

        assert!((loaded.mean() - norm.mean()).abs() < 1e-10);
        assert!((loaded.std() - norm.std()).abs() < 1e-10);
    }

    #[test]
    fn test_popart_normalize_before_initialized() {
        let mut norm = PopArtNormalizer::new();

        // Only one sample - not initialized
        norm.update(&[10.0]);
        assert!(!norm.is_initialized());

        // Should return identity (unnormalized)
        let result = norm.normalize(&[5.0]);
        assert_eq!(result[0], 5.0);
    }

    #[test]
    fn test_popart_denormalize_inverse_of_normalize() {
        let mut norm = PopArtNormalizer::new();
        norm.update(&[10.0, 20.0, 100.0, 200.0]);

        let original = vec![15.0, 150.0];

        // Normalize then denormalize should recover original
        let normalized = norm.normalize(&original);
        let mut denormalized = normalized;
        norm.denormalize(&mut denormalized);

        for (o, d) in original.iter().zip(denormalized.iter()) {
            assert!((o - d).abs() < 1e-4, "Expected {o}, got {d}");
        }
    }

    #[test]
    fn test_popart_denormalize_before_initialized() {
        let mut norm = PopArtNormalizer::new();

        // Only one sample - not initialized
        norm.update(&[10.0]);
        assert!(!norm.is_initialized());

        // Should return identity (unmodified)
        let mut values = vec![5.0];
        norm.denormalize(&mut values);
        assert_eq!(values[0], 5.0);
    }
}
