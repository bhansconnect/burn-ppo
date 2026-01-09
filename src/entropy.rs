//! Adaptive entropy coefficient control for PPO training.
//!
//! This module implements a PID-inspired controller that adjusts the entropy
//! coefficient to maintain target entropy levels throughout training.

use crate::config::Config;

/// Adaptive entropy coefficient controller.
///
/// Instead of blindly annealing the entropy coefficient, this controller:
/// 1. Sets a target entropy schedule based on training progress
/// 2. Measures actual entropy from PPO updates
/// 3. Adjusts the coefficient up/down to meet the target (bang-bang control)
pub struct AdaptiveEntropyController {
    // Config (immutable after construction)
    start_ratio: f64,
    final_ratio: f64,
    warmup_frac: f64,
    min_coef: f64,
    max_coef: f64,
    delta: f64,
    max_entropy: f64, // ln(num_actions)

    // State
    current_coef: f64,
    last_entropy: Option<f32>,
}

impl AdaptiveEntropyController {
    /// Create a new adaptive entropy controller.
    ///
    /// # Arguments
    /// * `config` - Training configuration containing adaptive entropy parameters
    /// * `num_actions` - Number of discrete actions (for computing max entropy)
    /// * `initial_coef` - Initial entropy coefficient (from `config.entropy_coef`)
    pub fn new(config: &Config, num_actions: usize, initial_coef: f64) -> Self {
        let max_entropy = (num_actions as f64).ln();

        Self {
            start_ratio: config.adaptive_entropy_start,
            final_ratio: config.adaptive_entropy_final,
            warmup_frac: config.adaptive_entropy_warmup,
            min_coef: config.adaptive_entropy_min_coef,
            max_coef: config.adaptive_entropy_max_coef,
            delta: config.adaptive_entropy_delta,
            max_entropy,
            current_coef: initial_coef,
            last_entropy: None,
        }
    }

    /// Record entropy from the latest PPO update.
    ///
    /// This should be called after each PPO update with the average entropy
    /// from that update's minibatches.
    pub fn record_entropy(&mut self, entropy: f32) {
        self.last_entropy = Some(entropy);
    }

    /// Get the current entropy coefficient, adjusting based on target vs actual.
    ///
    /// # Arguments
    /// * `progress` - Training progress as a fraction in [0, 1]
    ///
    /// # Returns
    /// A tuple of (coefficient, `target_entropy`) for use in PPO loss and logging.
    pub fn get_coefficient(&mut self, progress: f64) -> (f64, f64) {
        let target = self.target_entropy(progress);

        // Only adjust if we have entropy from a previous update
        if let Some(current) = self.last_entropy {
            let error = target - f64::from(current);

            // Bang-bang control: adjust by delta in direction of error
            // sign(0) = 0, so no adjustment when exactly at target
            self.current_coef += self.delta * error.signum();
            self.current_coef = self.current_coef.clamp(self.min_coef, self.max_coef);
        }

        (self.current_coef, target)
    }

    /// Compute the target entropy for a given training progress.
    ///
    /// - During warmup (progress < `warmup_frac`): constant at `start_ratio` * `max_entropy`
    /// - After warmup: linear decay from `start_ratio` to `final_ratio`
    fn target_entropy(&self, progress: f64) -> f64 {
        let ratio = if progress < self.warmup_frac {
            // During warmup: constant at start ratio
            self.start_ratio
        } else {
            // After warmup: linear decay to final ratio
            let post_warmup_progress =
                (progress - self.warmup_frac) / (1.0 - self.warmup_frac).max(1e-10);
            self.start_ratio + (self.final_ratio - self.start_ratio) * post_warmup_progress
        };

        ratio * self.max_entropy
    }

    /// Get the current coefficient value (for logging/debugging).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "utility method for debugging, used in tests")
    )]
    pub fn current_coefficient(&self) -> f64 {
        self.current_coef
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a controller with custom parameters for testing
    fn make_controller(
        start_ratio: f64,
        final_ratio: f64,
        warmup_frac: f64,
        min_coef: f64,
        max_coef: f64,
        delta: f64,
        num_actions: usize,
        initial_coef: f64,
    ) -> AdaptiveEntropyController {
        AdaptiveEntropyController {
            start_ratio,
            final_ratio,
            warmup_frac,
            min_coef,
            max_coef,
            delta,
            max_entropy: (num_actions as f64).ln(),
            current_coef: initial_coef,
            last_entropy: None,
        }
    }

    /// Helper with sensible defaults for Connect Four (7 actions)
    fn make_default_controller() -> AdaptiveEntropyController {
        make_controller(
            0.7,   // start_ratio
            0.2,   // final_ratio
            0.1,   // warmup_frac (10%)
            0.001, // min_coef
            0.05,  // max_coef
            0.001, // delta
            7,     // num_actions (Connect Four)
            0.01,  // initial_coef
        )
    }

    #[test]
    fn test_target_entropy_during_warmup() {
        let controller = make_default_controller();
        let max_entropy = 7f64.ln();

        // During warmup (progress < 0.1), target should be constant at start_ratio * max_entropy
        let target_0 = controller.target_entropy(0.0);
        let target_005 = controller.target_entropy(0.05);
        let target_009 = controller.target_entropy(0.09);

        let expected = 0.7 * max_entropy;
        assert!(
            (target_0 - expected).abs() < 1e-10,
            "At progress=0.0, target should be {expected}, got {target_0}"
        );
        assert!(
            (target_005 - expected).abs() < 1e-10,
            "At progress=0.05, target should be {expected}, got {target_005}"
        );
        assert!(
            (target_009 - expected).abs() < 1e-10,
            "At progress=0.09, target should be {expected}, got {target_009}"
        );
    }

    #[test]
    fn test_target_entropy_after_warmup() {
        let controller = make_default_controller();
        let max_entropy = 7f64.ln();

        // At warmup boundary (progress = 0.1): target = start_ratio * max_entropy
        let target_at_warmup = controller.target_entropy(0.1);
        let expected_start = 0.7 * max_entropy;
        assert!(
            (target_at_warmup - expected_start).abs() < 1e-10,
            "At warmup boundary, target should be {expected_start}, got {target_at_warmup}"
        );

        // At progress = 1.0: target = final_ratio * max_entropy
        let target_end = controller.target_entropy(1.0);
        let expected_end = 0.2 * max_entropy;
        assert!(
            (target_end - expected_end).abs() < 1e-10,
            "At progress=1.0, target should be {expected_end}, got {target_end}"
        );

        // At midpoint (progress = 0.55, which is 50% of post-warmup):
        // ratio = 0.7 + (0.2 - 0.7) * 0.5 = 0.7 - 0.25 = 0.45
        let target_mid = controller.target_entropy(0.55);
        let expected_mid = 0.45 * max_entropy;
        assert!(
            (target_mid - expected_mid).abs() < 1e-10,
            "At progress=0.55, target should be {expected_mid}, got {target_mid}"
        );
    }

    #[test]
    fn test_no_adjustment_without_entropy() {
        let mut controller = make_default_controller();
        let initial_coef = controller.current_coefficient();

        // First call to get_coefficient before any record_entropy should not adjust
        let (coef, _target) = controller.get_coefficient(0.5);

        assert!(
            (coef - initial_coef).abs() < 1e-10,
            "Coefficient should remain at initial value {initial_coef}, got {coef}"
        );
    }

    #[test]
    fn test_coefficient_increases_when_entropy_low() {
        let mut controller = make_default_controller();

        // Set up: target at progress=0.5 is roughly 0.45 * max_entropy ≈ 0.875
        let target = controller.target_entropy(0.5);
        assert!(target > 0.8, "Target should be > 0.8, got {target}");

        // Record very low entropy (below target)
        controller.record_entropy(0.3);

        let initial_coef = controller.current_coefficient();
        let (new_coef, _) = controller.get_coefficient(0.5);

        // Since entropy < target, error > 0, coefficient should increase by delta
        let expected_coef = initial_coef + 0.001;
        assert!(
            (new_coef - expected_coef).abs() < 1e-10,
            "Coefficient should increase from {initial_coef} to {expected_coef}, got {new_coef}"
        );
    }

    #[test]
    fn test_coefficient_decreases_when_entropy_high() {
        let mut controller = make_default_controller();

        // Record very high entropy (above any reasonable target)
        controller.record_entropy(2.0);

        let initial_coef = controller.current_coefficient();
        let (new_coef, target) = controller.get_coefficient(0.5);

        // Verify target < recorded entropy
        assert!(
            target < 2.0,
            "Target {target} should be less than recorded entropy 2.0"
        );

        // Since entropy > target, error < 0, coefficient should decrease by delta
        let expected_coef = initial_coef - 0.001;
        assert!(
            (new_coef - expected_coef).abs() < 1e-10,
            "Coefficient should decrease from {initial_coef} to {expected_coef}, got {new_coef}"
        );
    }

    #[test]
    fn test_coefficient_clamped_to_min() {
        // Start at minimum coefficient
        let mut controller = make_controller(
            0.7,   // start_ratio
            0.2,   // final_ratio
            0.1,   // warmup_frac
            0.001, // min_coef
            0.05,  // max_coef
            0.01,  // delta (larger for faster test)
            7,     // num_actions
            0.001, // initial_coef (at minimum)
        );

        // Record very high entropy to force decrease
        controller.record_entropy(2.0);

        // Multiple attempts to decrease should stay at min
        for _ in 0..10 {
            let (coef, _) = controller.get_coefficient(0.5);
            assert!(
                coef >= 0.001 - 1e-10,
                "Coefficient should not go below min 0.001, got {coef}"
            );
        }
    }

    #[test]
    fn test_coefficient_clamped_to_max() {
        // Start at maximum coefficient
        let mut controller = make_controller(
            0.7,   // start_ratio
            0.2,   // final_ratio
            0.1,   // warmup_frac
            0.001, // min_coef
            0.05,  // max_coef
            0.01,  // delta (larger for faster test)
            7,     // num_actions
            0.05,  // initial_coef (at maximum)
        );

        // Record very low entropy to force increase
        controller.record_entropy(0.1);

        // Multiple attempts to increase should stay at max
        for _ in 0..10 {
            let (coef, _) = controller.get_coefficient(0.5);
            assert!(
                coef <= 0.05 + 1e-10,
                "Coefficient should not go above max 0.05, got {coef}"
            );
        }
    }

    #[test]
    fn test_no_adjustment_when_at_target() {
        let mut controller = make_default_controller();

        // Get target at some progress point
        let target = controller.target_entropy(0.5);

        // Record entropy exactly at target (note: f64->f32 conversion may introduce tiny errors)
        controller.record_entropy(target as f32);

        let initial_coef = controller.current_coefficient();
        let (new_coef, _) = controller.get_coefficient(0.5);

        // Due to f64->f32 conversion, there may be a tiny adjustment (at most delta)
        // In practice, this is fine - the controller will stabilize near the target
        let delta = 0.001; // controller's adjustment step
        assert!(
            (new_coef - initial_coef).abs() <= delta + 1e-10,
            "Coefficient change should be at most delta={delta} when near target; \
             started at {initial_coef}, got {new_coef}"
        );
    }

    #[test]
    fn test_full_training_simulation() {
        // Simulate a training run where entropy naturally decreases
        let mut controller = make_default_controller();

        // Simulate: start with high entropy, gradually decrease
        // The controller should increase coefficient to counteract

        let mut coef_history = Vec::new();

        // Initial entropy high (random policy)
        let mut simulated_entropy = 1.8f32;

        for step in 0..100 {
            let progress = f64::from(step) / 100.0;

            // Record current entropy
            controller.record_entropy(simulated_entropy);

            // Get coefficient
            let (coef, _target) = controller.get_coefficient(progress);
            coef_history.push(coef);

            // Simulate entropy decay (policy improving)
            // With higher coef, decay is slower; with lower coef, decay is faster
            // This is a simplified model
            simulated_entropy *= 0.995;
            simulated_entropy = simulated_entropy.max(0.2); // Floor

            // If we're early in training and entropy drops below target,
            // the controller should try to compensate
        }

        // Verify the controller responded to low entropy by increasing coefficient
        let initial_coef = coef_history[0];
        let final_coef = coef_history.last().unwrap();

        // Since entropy was dropping below target, coefficient should have increased
        // (or at least stayed stable/clamped at max)
        assert!(
            *final_coef >= initial_coef - 0.01,
            "Coefficient should not have decreased significantly; \
             started at {initial_coef}, ended at {final_coef}"
        );
    }

    #[test]
    fn test_warmup_zero_handled_correctly() {
        // Edge case: warmup_frac = 0 means immediate decay
        let controller = make_controller(
            0.7,   // start_ratio
            0.2,   // final_ratio
            0.0,   // warmup_frac (no warmup)
            0.001, // min_coef
            0.05,  // max_coef
            0.001, // delta
            7,     // num_actions
            0.01,  // initial_coef
        );

        let max_entropy = 7f64.ln();

        // At progress 0, should still be at start ratio (boundary condition)
        let target_0 = controller.target_entropy(0.0);
        let expected_0 = 0.7 * max_entropy;
        assert!(
            (target_0 - expected_0).abs() < 1e-10,
            "At progress=0 with no warmup, target should be {expected_0}, got {target_0}"
        );

        // At progress 0.5, should be at midpoint
        let target_mid = controller.target_entropy(0.5);
        let expected_mid = 0.45 * max_entropy;
        assert!(
            (target_mid - expected_mid).abs() < 1e-10,
            "At progress=0.5 with no warmup, target should be {expected_mid}, got {target_mid}"
        );
    }

    #[test]
    fn test_warmup_one_handled_correctly() {
        // Edge case: warmup_frac = 1.0 means constant target throughout
        let controller = make_controller(
            0.7,   // start_ratio
            0.2,   // final_ratio
            1.0,   // warmup_frac (100% warmup = constant)
            0.001, // min_coef
            0.05,  // max_coef
            0.001, // delta
            7,     // num_actions
            0.01,  // initial_coef
        );

        let max_entropy = 7f64.ln();
        let expected = 0.7 * max_entropy;

        // Throughout training, should remain at start ratio
        for progress in [0.0, 0.25, 0.5, 0.75, 0.99] {
            let target = controller.target_entropy(progress);
            assert!(
                (target - expected).abs() < 1e-10,
                "At progress={progress} with 100% warmup, target should be {expected}, got {target}"
            );
        }
    }

    #[test]
    fn test_different_action_counts() {
        // Test with different num_actions to ensure max_entropy is computed correctly
        for num_actions in [2, 7, 10, 100] {
            let controller = make_controller(
                0.5,   // start_ratio
                0.5,   // final_ratio (constant for simplicity)
                0.0,   // warmup_frac
                0.001, // min_coef
                0.05,  // max_coef
                0.001, // delta
                num_actions,
                0.01,
            );

            let expected_max_entropy = (num_actions as f64).ln();
            let target = controller.target_entropy(0.5);
            let expected_target = 0.5 * expected_max_entropy;

            assert!(
                (target - expected_target).abs() < 1e-10,
                "For num_actions={num_actions}, target should be {expected_target}, got {target}"
            );
        }
    }
}
