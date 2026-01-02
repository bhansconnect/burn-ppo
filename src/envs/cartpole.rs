/// CartPole environment - classic control task
///
/// Physics based on OpenAI Gym CartPole-v1
/// Goal: Balance a pole on a cart by pushing left or right

use rand::Rng;

use crate::env::Environment;
use crate::profile::profile_function;

/// CartPole physics constants (matching OpenAI Gym)
const GRAVITY: f32 = 9.8;
const CART_MASS: f32 = 1.0;
const POLE_MASS: f32 = 0.1;
const TOTAL_MASS: f32 = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH: f32 = 0.5;
const POLE_MASS_LENGTH: f32 = POLE_MASS * POLE_HALF_LENGTH;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02; // Time step

/// Termination thresholds
const X_THRESHOLD: f32 = 2.4;
const THETA_THRESHOLD: f32 = 12.0 * std::f32::consts::PI / 180.0; // 12 degrees
const MAX_STEPS: usize = 500;

/// CartPole state
#[derive(Debug, Clone)]
pub struct CartPole {
    /// Cart position
    x: f32,
    /// Cart velocity
    x_dot: f32,
    /// Pole angle (radians, 0 = upright)
    theta: f32,
    /// Pole angular velocity
    theta_dot: f32,
    /// Steps taken in current episode
    steps: usize,
    /// RNG for initial state randomization
    rng: rand::rngs::StdRng,
}

impl CartPole {
    /// Get current state for rendering: [x, x_dot, theta, theta_dot]
    pub fn state(&self) -> [f32; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }

    /// Physics step using semi-implicit Euler integration
    fn physics_step(&mut self, force: f32) {
        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        // Equations of motion (derived from Lagrangian mechanics)
        let temp = (force + POLE_MASS_LENGTH * self.theta_dot.powi(2) * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_theta.powi(2) / TOTAL_MASS));
        let x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        // Semi-implicit Euler
        self.x_dot += TAU * x_acc;
        self.x += TAU * self.x_dot;
        self.theta_dot += TAU * theta_acc;
        self.theta += TAU * self.theta_dot;
    }

    /// Check if episode should terminate
    fn is_terminal(&self) -> bool {
        self.x.abs() > X_THRESHOLD
            || self.theta.abs() > THETA_THRESHOLD
            || self.steps >= MAX_STEPS
    }

    /// Get state as observation vector
    fn get_obs(&self) -> Vec<f32> {
        vec![self.x, self.x_dot, self.theta, self.theta_dot]
    }
}

impl Environment for CartPole {
    const OBSERVATION_DIM: usize = 4;
    const ACTION_COUNT: usize = 2;
    const NAME: &'static str = "cartpole";

    fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut env = Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            rng,
        };
        // Initialize to random state
        let _ = env.reset();
        env
    }

    fn render(&self) -> Option<String> {
        Some(format!(
            "x={:.3} | v={:.3} | \u{03b8}={:.1}\u{00b0} | \u{03c9}={:.2}",
            self.x,
            self.x_dot,
            self.theta.to_degrees(),
            self.theta_dot
        ))
    }

    fn reset(&mut self) -> Vec<f32> {
        profile_function!();
        // Random initial state in [-0.05, 0.05]
        self.x = self.rng.gen_range(-0.05..0.05);
        self.x_dot = self.rng.gen_range(-0.05..0.05);
        self.theta = self.rng.gen_range(-0.05..0.05);
        self.theta_dot = self.rng.gen_range(-0.05..0.05);
        self.steps = 0;
        self.get_obs()
    }

    fn step(&mut self, action: usize) -> (Vec<f32>, Vec<f32>, bool) {
        profile_function!();
        // Action: 0 = push left, 1 = push right
        let force = if action == 0 { -FORCE_MAG } else { FORCE_MAG };

        self.physics_step(force);
        self.steps += 1;

        let done = self.is_terminal();

        // Reward: +1 for each step the pole stays up
        let reward = if done && self.steps < MAX_STEPS {
            0.0 // Terminal due to failure
        } else {
            1.0
        };

        (self.get_obs(), vec![reward], done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartpole_reset() {
        let mut env = CartPole::new(42);
        let obs = env.reset();

        assert_eq!(obs.len(), 4);
        // All initial values should be small
        for val in &obs {
            assert!(val.abs() < 0.1);
        }
    }

    #[test]
    fn test_cartpole_step() {
        let mut env = CartPole::new(42);
        env.reset();

        let (obs, rewards, done) = env.step(1); // Push right

        assert_eq!(obs.len(), 4);
        assert_eq!(rewards, vec![1.0]);
        assert!(!done);
    }

    #[test]
    fn test_cartpole_termination_angle() {
        let mut env = CartPole::new(42);
        env.reset();

        // Force pole to extreme angle
        env.theta = THETA_THRESHOLD + 0.1;

        let (_, _, done) = env.step(0);
        assert!(done);
    }

    #[test]
    fn test_cartpole_termination_position() {
        let mut env = CartPole::new(42);
        env.reset();

        // Force cart to extreme position
        env.x = X_THRESHOLD + 0.1;

        let (_, _, done) = env.step(0);
        assert!(done);
    }

    #[test]
    fn test_cartpole_max_steps() {
        let mut env = CartPole::new(42);
        env.reset();

        // Run for max steps without terminating
        // (unlikely without physics, but test the counter)
        env.steps = MAX_STEPS - 1;
        let (_, _, done) = env.step(0);

        assert!(done);
    }

    #[test]
    fn test_cartpole_physics_pushes() {
        let mut env = CartPole::new(42);
        env.reset();
        env.x = 0.0;
        env.x_dot = 0.0;

        // Push right
        env.step(1);
        let x_after_right = env.x;

        env.reset();
        env.x = 0.0;
        env.x_dot = 0.0;

        // Push left
        env.step(0);
        let x_after_left = env.x;

        // Right push should move cart right (positive x)
        // Left push should move cart left (negative x)
        assert!(x_after_right > x_after_left);
    }

    #[test]
    fn test_cartpole_reproducible() {
        let mut env1 = CartPole::new(42);
        let mut env2 = CartPole::new(42);

        let obs1 = env1.reset();
        let obs2 = env2.reset();
        assert_eq!(obs1, obs2);

        let (obs1, rewards1, d1) = env1.step(1);
        let (obs2, rewards2, d2) = env2.step(1);
        assert_eq!(obs1, obs2);
        assert_eq!(rewards1, rewards2);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_cartpole_state() {
        let mut env = CartPole::new(42);
        env.reset();

        // Set known state values
        env.x = 0.5;
        env.x_dot = 1.0;
        env.theta = 0.1;
        env.theta_dot = -0.5;

        let state = env.state();
        assert_eq!(state, [0.5, 1.0, 0.1, -0.5]);
    }

    #[test]
    fn test_cartpole_render() {
        let mut env = CartPole::new(42);
        env.reset();

        // Set known state for predictable output
        env.x = 1.5;
        env.x_dot = 0.25;
        env.theta = 0.1; // ~5.7 degrees
        env.theta_dot = -0.3;

        let rendered = env.render();
        assert!(rendered.is_some());

        let output = rendered.unwrap();
        // Check that state values appear in output
        assert!(output.contains("x=1.500"));
        assert!(output.contains("v=0.250"));
        // Theta should be converted to degrees (~5.7)
        assert!(output.contains("θ="));
        assert!(output.contains("ω="));
    }

    #[test]
    fn test_cartpole_render_format() {
        let mut env = CartPole::new(42);
        env.reset();

        env.x = 0.0;
        env.x_dot = 0.0;
        env.theta = std::f32::consts::PI / 6.0; // 30 degrees
        env.theta_dot = 0.0;

        let output = env.render().unwrap();
        // Should show ~30 degrees
        assert!(output.contains("30.0°") || output.contains("30°"));
    }
}
