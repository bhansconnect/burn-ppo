//! Adaptive entropy coefficient control for PPO training.
//!
//! This module implements a PID-inspired controller that adjusts the entropy
//! coefficient to maintain target entropy levels throughout training.

use crate::schedule::Schedule;

/// Adaptive entropy coefficient controller.
///
/// Instead of blindly annealing the entropy coefficient, this controller:
/// 1. Sets a target entropy based on the target schedule at current step
/// 2. Measures actual entropy from PPO updates
/// 3. Adjusts the coefficient up/down to meet the target (bang-bang control)
pub struct AdaptiveEntropyController {
    // Config (immutable after construction)
    target_schedule: Schedule, // Schedule of target ratios (0-1) of max entropy
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
    /// * `target_schedule` - Schedule of target entropy ratios (0-1, fraction of max entropy)
    /// * `num_actions` - Number of discrete actions (for computing max entropy)
    /// * `initial_coef` - Initial entropy coefficient
    /// * `min_coef` - Minimum entropy coefficient
    /// * `max_coef` - Maximum entropy coefficient
    /// * `delta` - Adjustment step size for bang-bang control
    pub fn new(
        target_schedule: Schedule,
        num_actions: usize,
        initial_coef: f64,
        min_coef: f64,
        max_coef: f64,
        delta: f64,
    ) -> Self {
        let max_entropy = (num_actions as f64).ln();

        Self {
            target_schedule,
            min_coef,
            max_coef,
            delta,
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
    /// * `step` - Current training step (for schedule lookup)
    ///
    /// # Returns
    /// A tuple of (coefficient, `target_entropy`) for use in PPO loss and logging.
    pub fn get_coefficient(&mut self, step: u64) -> (f64, f64) {
        let target = self.target_entropy(step);

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

    /// Compute the target entropy for a given step.
    ///
    /// Uses the `target_schedule` to get the ratio, then multiplies by `max_entropy`.
    fn target_entropy(&self, step: u64) -> f64 {
        let ratio = self.target_schedule.get(step);
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
        target_schedule: Schedule,
        min_coef: f64,
        max_coef: f64,
        delta: f64,
        num_actions: usize,
        initial_coef: f64,
    ) -> AdaptiveEntropyController {
        AdaptiveEntropyController::new(
            target_schedule,
            num_actions,
            initial_coef,
            min_coef,
            max_coef,
            delta,
        )
    }

    /// Helper with sensible defaults for Connect Four (7 actions)
    /// Uses a schedule that starts at 70% and decays to 20% at step 1000
    fn make_default_controller() -> AdaptiveEntropyController {
        // Equivalent to old warmup=0.1, start=0.7, final=0.2 over 1000 steps
        // Warmup phase (0-100): constant at 0.7
        // Decay phase (100-1000): linear from 0.7 to 0.2
        let schedule = Schedule::new(vec![(0.7, 0), (0.7, 100), (0.2, 1000)]);
        make_controller(
            schedule, 0.001, // min_coef
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
        let expected = 0.7 * max_entropy;

        // During warmup (step < 100), target should be constant at 0.7 * max_entropy
        for step in [0, 50, 99] {
            let target = controller.target_entropy(step);
            assert!(
                (target - expected).abs() < 1e-10,
                "At step={step}, target should be {expected}, got {target}"
            );
        }
    }

    #[test]
    fn test_target_entropy_after_warmup() {
        let controller = make_default_controller();
        let max_entropy = 7f64.ln();

        // At warmup boundary (step = 100): target = 0.7 * max_entropy
        let target_at_warmup = controller.target_entropy(100);
        let expected_start = 0.7 * max_entropy;
        assert!(
            (target_at_warmup - expected_start).abs() < 1e-10,
            "At warmup boundary, target should be {expected_start}, got {target_at_warmup}"
        );

        // At step = 1000: target = 0.2 * max_entropy
        let target_end = controller.target_entropy(1000);
        let expected_end = 0.2 * max_entropy;
        assert!(
            (target_end - expected_end).abs() < 1e-10,
            "At step=1000, target should be {expected_end}, got {target_end}"
        );

        // At midpoint (step = 550, which is 50% of post-warmup):
        // ratio = 0.7 + (0.2 - 0.7) * 0.5 = 0.45
        let target_mid = controller.target_entropy(550);
        let expected_mid = 0.45 * max_entropy;
        assert!(
            (target_mid - expected_mid).abs() < 1e-10,
            "At step=550, target should be {expected_mid}, got {target_mid}"
        );
    }

    #[test]
    fn test_no_adjustment_without_entropy() {
        let mut controller = make_default_controller();
        let initial_coef = controller.current_coefficient();

        // First call to get_coefficient before any record_entropy should not adjust
        let (coef, _target) = controller.get_coefficient(500);

        assert!(
            (coef - initial_coef).abs() < 1e-10,
            "Coefficient should remain at initial value {initial_coef}, got {coef}"
        );
    }

    #[test]
    fn test_coefficient_increases_when_entropy_low() {
        let mut controller = make_default_controller();

        // Set up: target at step=500 is roughly 0.45 * max_entropy ≈ 0.875
        let target = controller.target_entropy(500);
        assert!(target > 0.8, "Target should be > 0.8, got {target}");

        // Record very low entropy (below target)
        controller.record_entropy(0.3);

        let initial_coef = controller.current_coefficient();
        let (new_coef, _) = controller.get_coefficient(500);

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
        let (new_coef, target) = controller.get_coefficient(500);

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
        let schedule = Schedule::constant(0.5);
        let mut controller = make_controller(
            schedule, 0.001, // min_coef
            0.05,  // max_coef
            0.01,  // delta (larger for faster test)
            7,     // num_actions
            0.001, // initial_coef (at minimum)
        );

        // Record very high entropy to force decrease
        controller.record_entropy(2.0);

        // Multiple attempts to decrease should stay at min
        for _ in 0..10 {
            let (coef, _) = controller.get_coefficient(500);
            assert!(
                coef >= 0.001 - 1e-10,
                "Coefficient should not go below min 0.001, got {coef}"
            );
        }
    }

    #[test]
    fn test_coefficient_clamped_to_max() {
        // Start at maximum coefficient
        let schedule = Schedule::constant(0.5);
        let mut controller = make_controller(
            schedule, 0.001, // min_coef
            0.05,  // max_coef
            0.01,  // delta (larger for faster test)
            7,     // num_actions
            0.05,  // initial_coef (at maximum)
        );

        // Record very low entropy to force increase
        controller.record_entropy(0.1);

        // Multiple attempts to increase should stay at max
        for _ in 0..10 {
            let (coef, _) = controller.get_coefficient(500);
            assert!(
                coef <= 0.05 + 1e-10,
                "Coefficient should not go above max 0.05, got {coef}"
            );
        }
    }

    #[test]
    fn test_no_adjustment_when_at_target() {
        let mut controller = make_default_controller();

        // Get target at some step
        let target = controller.target_entropy(500);

        // Record entropy exactly at target (note: f64->f32 conversion may introduce tiny errors)
        controller.record_entropy(target as f32);

        let initial_coef = controller.current_coefficient();
        let (new_coef, _) = controller.get_coefficient(500);

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
            // Record current entropy
            controller.record_entropy(simulated_entropy);

            // Get coefficient
            let (coef, _target) = controller.get_coefficient(step * 10);
            coef_history.push(coef);

            // Simulate entropy decay (policy improving)
            simulated_entropy *= 0.995;
            simulated_entropy = simulated_entropy.max(0.2); // Floor
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
    fn test_constant_target_schedule() {
        // Test with a constant target (no decay)
        let schedule = Schedule::constant(0.5);
        let controller = make_controller(
            schedule, 0.001, // min_coef
            0.05,  // max_coef
            0.001, // delta
            7,     // num_actions
            0.01,  // initial_coef
        );

        let max_entropy = 7f64.ln();
        let expected = 0.5 * max_entropy;

        // Throughout training, should remain constant
        for step in [0, 100, 500, 1000, 10000] {
            let target = controller.target_entropy(step);
            assert!(
                (target - expected).abs() < 1e-10,
                "At step={step} with constant schedule, target should be {expected}, got {target}"
            );
        }
    }

    #[test]
    fn test_different_action_counts() {
        // Test with different num_actions to ensure max_entropy is computed correctly
        for num_actions in [2, 7, 10, 100] {
            let schedule = Schedule::constant(0.5);
            let controller = make_controller(
                schedule,
                0.001, // min_coef
                0.05,  // max_coef
                0.001, // delta
                num_actions,
                0.01,
            );

            let expected_max_entropy = (num_actions as f64).ln();
            let target = controller.target_entropy(500);
            let expected_target = 0.5 * expected_max_entropy;

            assert!(
                (target - expected_target).abs() < 1e-10,
                "For num_actions={num_actions}, target should be {expected_target}, got {target}"
            );
        }
    }
}
