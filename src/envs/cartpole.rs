/// `CartPole` environment - classic control task
///
/// Physics based on `OpenAI` Gym CartPole-v1
/// Goal: Balance a pole on a cart by pushing left or right
use rand::Rng;

use crate::env::Environment;
use crate::profile::profile_function;

/// `CartPole` physics constants (matching `OpenAI` Gym)
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

/// `CartPole` state
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
    /// Get current state for rendering: [x, `x_dot`, theta, `theta_dot`]
    #[cfg(test)]
    pub const fn state(&self) -> [f32; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }

    /// Physics step using semi-implicit Euler integration
    fn physics_step(&mut self, force: f32) {
        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        // Equations of motion (derived from Lagrangian mechanics)
        let temp =
            (POLE_MASS_LENGTH * self.theta_dot.powi(2)).mul_add(sin_theta, force) / TOTAL_MASS;
        let theta_acc = GRAVITY.mul_add(sin_theta, -(cos_theta * temp))
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
        self.x.abs() > X_THRESHOLD || self.theta.abs() > THETA_THRESHOLD || self.steps >= MAX_STEPS
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
        // ASCII visualization of CartPole
        //
        // Layout (60 chars wide):
        //   Row 0: Header with numeric values
        //   Row 1: Empty
        //   Rows 2-6: Pole (5 chars tall, bottom to top)
        //   Row 7: Cart top edge
        //   Row 8: Cart bottom edge
        //   Row 9: Track with wheels
        //   Row 10: Position scale

        const WIDTH: usize = 60;
        const CART_WIDTH: usize = 9;
        const POLE_HEIGHT: usize = 5;

        // Buffer rows: 0=pole_top, 4=pole_bottom, 5=cart_top, 6=cart_bottom, 7=track
        let mut buffer = vec![vec![' '; WIDTH]; 8];

        // Row indices in buffer
        const CART_TOP: usize = 5;
        const CART_BOTTOM: usize = 6;
        const TRACK: usize = 7;

        // Calculate cart center position
        // x ranges from -2.4 to 2.4, map to screen coordinates
        let margin = CART_WIDTH / 2 + 3;
        let usable_width = WIDTH - 2 * margin;
        let x_clamped = self.x.clamp(-2.4, 2.4);
        let x_normalized = (x_clamped + 2.4) / 4.8; // 0.0 to 1.0
        let cart_center = margin + (x_normalized * usable_width as f32) as usize;

        // Draw track
        buffer[TRACK].fill('═');

        // Draw cart body
        let cart_left = cart_center.saturating_sub(CART_WIDTH / 2);
        let cart_right = (cart_center + CART_WIDTH / 2).min(WIDTH - 1);

        // Top of cart: ┌───────┐
        if cart_left < WIDTH {
            buffer[CART_TOP][cart_left] = '┌';
        }
        if cart_right < WIDTH {
            buffer[CART_TOP][cart_right] = '┐';
        }
        buffer[CART_TOP][(cart_left + 1)..cart_right].fill('─');

        // Bottom of cart: └───────┘
        if cart_left < WIDTH {
            buffer[CART_BOTTOM][cart_left] = '└';
        }
        if cart_right < WIDTH {
            buffer[CART_BOTTOM][cart_right] = '┘';
        }
        buffer[CART_BOTTOM][(cart_left + 1)..cart_right].fill('─');

        // Wheels on track: ═══○═══○═══
        let wheel_offset = 2;
        let wheel_left = cart_left + wheel_offset;
        let wheel_right = cart_right.saturating_sub(wheel_offset);
        if wheel_left < WIDTH {
            buffer[TRACK][wheel_left] = '○';
        }
        if wheel_right < WIDTH && wheel_right > wheel_left {
            buffer[TRACK][wheel_right] = '○';
        }

        // Draw pole from cart center going upward
        // Pivot point is top-center of cart
        let pivot_col = cart_center as f32;

        // Character aspect ratio: terminal chars are ~2x taller than wide
        // Horizontal offset = vertical_distance * tan(theta) * aspect_ratio
        let aspect_ratio = 2.0_f32;

        // Track column positions for each pole segment (bottom to top)
        let mut pole_cols: Vec<i32> = Vec::with_capacity(POLE_HEIGHT);
        for i in 0..POLE_HEIGHT {
            let vertical_dist = (i + 1) as f32;
            let h_offset = vertical_dist * self.theta.tan() * aspect_ratio;
            pole_cols.push((pivot_col + h_offset).round() as i32);
        }

        // Draw pole segments from bottom (near cart) to top
        for (i, &col) in pole_cols.iter().enumerate() {
            let row = CART_TOP - 1 - i; // Start just above cart top

            if col >= 0 && (col as usize) < WIDTH {
                let col_usize = col as usize;

                let pole_char = if i == POLE_HEIGHT - 1 {
                    // Top of pole - ball
                    '●'
                } else {
                    // Determine character based on direction to next segment
                    let next_col = pole_cols[i + 1];
                    let delta = next_col - col;

                    if delta > 0 {
                        '╱' // Moving right going up = / slope (bottom-left to top-right)
                    } else if delta < 0 {
                        '╲' // Moving left going up = \ slope (bottom-right to top-left)
                    } else {
                        '│' // Next segment is directly above (vertical)
                    }
                };

                buffer[row][col_usize] = pole_char;
            }
        }

        // Build output string
        let mut lines = Vec::with_capacity(11);

        // Header with numeric values
        lines.push(format!(
            " Step:{:4} │ x:{:6.2}  v:{:6.2} │ θ:{:5.1}°  ω:{:6.2}",
            self.steps,
            self.x,
            self.x_dot,
            self.theta.to_degrees(),
            self.theta_dot
        ));
        lines.push(String::new());

        // Add visual buffer rows
        for row in &buffer {
            lines.push(row.iter().collect());
        }

        // Scale line showing position bounds
        let mut scale = vec![' '; WIDTH];
        // Left bound: -2.4 at x=0 position
        let left_pos = margin;
        for (i, c) in "-2.4".chars().enumerate() {
            if left_pos + i < WIDTH {
                scale[left_pos + i] = c;
            }
        }
        // Center: 0
        let center_pos = margin + usable_width / 2;
        if center_pos < WIDTH {
            scale[center_pos] = '0';
        }
        // Right bound: 2.4 at x=2.4 position
        let right_pos = margin + usable_width;
        if right_pos >= 3 {
            for (i, c) in "2.4".chars().enumerate() {
                let pos = right_pos - 3 + i;
                if pos < WIDTH {
                    scale[pos] = c;
                }
            }
        }
        lines.push(scale.iter().collect());

        Some(lines.join("\n"))
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

    fn describe_action(&self, action: usize) -> String {
        match action {
            0 => "Push left".to_string(),
            1 => "Push right".to_string(),
            _ => format!("Action {action}"),
        }
    }

    fn parse_action(&self, input: &str) -> Result<usize, String> {
        match input.trim().to_lowercase().as_str() {
            "left" | "l" | "0" => Ok(0),
            "right" | "r" | "1" => Ok(1),
            _ => Err("Enter 'left' or 'right' (or 'l'/'r')".to_string()),
        }
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

    // =========================================================================
    // ASCII Render Tests
    // =========================================================================
    //
    // The render output looks like:
    //
    //  Step:   0 │ x:  0.00  v:  0.00 │ θ:  0.0°  ω:  0.00
    //
    //                            ●
    //                            │
    //                            │
    //                            │
    //                            │
    //                       ┌─────────┐
    // ══════════════════════╪══○═══○══╪═════════════════════════════════════
    //                       └─────────┘
    //       -2.4                 0                                      2.4

    /// Helper to set up an environment with specific state for render testing
    fn setup_render_env(x: f32, theta: f32) -> CartPole {
        let mut env = CartPole::new(42);
        env.reset();
        env.x = x;
        env.x_dot = 0.0;
        env.theta = theta;
        env.theta_dot = 0.0;
        env.steps = 0;
        env
    }

    /// Helper to find the column of a character in a specific line
    fn find_char_col(output: &str, line_idx: usize, ch: char) -> Option<usize> {
        output
            .lines()
            .nth(line_idx)
            .and_then(|line| line.chars().position(|c| c == ch))
    }

    /// Helper to check if a line contains a character
    fn line_contains(output: &str, line_idx: usize, ch: char) -> bool {
        output
            .lines()
            .nth(line_idx)
            .is_some_and(|line| line.contains(ch))
    }

    #[test]
    fn test_render_header_contains_values() {
        // Test: Header line shows all state values
        //
        // Expected format:
        //  Step:  42 │ x:  1.50  v:  0.25 │ θ:  5.7°  ω: -0.30

        let mut env = setup_render_env(1.5, 0.1); // 0.1 rad ≈ 5.7°
        env.x_dot = 0.25;
        env.theta_dot = -0.3;
        env.steps = 42;

        let output = env.render().unwrap();
        let header = output.lines().next().unwrap();

        assert!(
            header.contains("Step:  42"),
            "Header should show step count"
        );
        assert!(header.contains("x:  1.50"), "Header should show x position");
        assert!(header.contains("v:  0.25"), "Header should show velocity");
        assert!(
            header.contains("θ:  5.7°"),
            "Header should show angle in degrees"
        );
        assert!(
            header.contains("ω: -0.30"),
            "Header should show angular velocity"
        );
    }

    #[test]
    fn test_render_cart_centered_when_x_zero() {
        // Test: Cart is centered when x = 0
        //
        // Visual check: Cart center (middle of ┌─────────┐) should be at column ~30
        //               (WIDTH=60, so center is 30)

        let env = setup_render_env(0.0, 0.0);
        let output = env.render().unwrap();

        // Find cart top left corner (┌) and right corner (┐)
        let cart_line_idx = 7; // Cart top is line 7 (0=header, 1=blank, 2-6=pole, 7=cart)
        let left = find_char_col(&output, cart_line_idx, '┌').expect("Cart left corner not found");
        let right =
            find_char_col(&output, cart_line_idx, '┐').expect("Cart right corner not found");

        let cart_center = usize::midpoint(left, right);
        let expected_center: usize = 30; // WIDTH/2

        assert!(
            (cart_center as i32 - expected_center as i32).abs() <= 2,
            "Cart should be centered. Found center at {cart_center}, expected ~{expected_center}"
        );
    }

    #[test]
    fn test_render_cart_at_left_edge() {
        // Test: Cart moves left when x = -2.4
        //
        // Expected: Cart is near left side of display

        let env = setup_render_env(-2.4, 0.0);
        let output = env.render().unwrap();

        let cart_line_idx = 7;
        let left = find_char_col(&output, cart_line_idx, '┌').expect("Cart left corner not found");

        // Cart should be near left margin (around column 7-10)
        assert!(
            left < 15,
            "Cart at x=-2.4 should be on left side. Found at column {left}"
        );
    }

    #[test]
    fn test_render_cart_at_right_edge() {
        // Test: Cart moves right when x = 2.4
        //
        // Expected: Cart is near right side of display

        let env = setup_render_env(2.4, 0.0);
        let output = env.render().unwrap();

        let cart_line_idx = 7;
        let right =
            find_char_col(&output, cart_line_idx, '┐').expect("Cart right corner not found");

        // Cart should be near right side (column > 45 for WIDTH=60)
        assert!(
            right > 45,
            "Cart at x=2.4 should be on right side. Found at column {right}"
        );
    }

    #[test]
    fn test_render_pole_vertical_when_theta_zero() {
        // Test: Pole uses vertical character (│) when theta = 0
        //
        // Expected: All pole segments except top (●) are │

        let env = setup_render_env(0.0, 0.0);
        let output = env.render().unwrap();

        // Pole is in lines 2-6 (0=header, 1=blank, 2=top with ●, 3-6=pole body)
        // Line 2 should have ● (ball at top)
        assert!(
            line_contains(&output, 2, '●'),
            "Top of pole should have ball (●)"
        );

        // Lines 3-6 should have │ (vertical pole)
        for line_idx in 3..=6 {
            assert!(
                line_contains(&output, line_idx, '│'),
                "Line {line_idx} should have vertical pole segment (│)"
            );
        }
    }

    #[test]
    fn test_render_pole_leans_right_positive_theta() {
        // Test: Pole uses ╲ character when leaning right (theta > 0)
        //
        // At theta = 10° ≈ 0.175 rad, pole should visibly lean right

        let theta = 10.0_f32.to_radians();
        let env = setup_render_env(0.0, theta);
        let output = env.render().unwrap();

        // Should have at least one ╱ character in the pole (/ = bottom-left to top-right)
        let pole_area: String = output.lines().skip(2).take(5).collect();
        assert!(
            pole_area.contains('╱'),
            "Pole leaning right (θ=+10°) should have ╱ character.\nPole area:\n{pole_area}"
        );

        // Should still have ball at top
        assert!(
            line_contains(&output, 2, '●'),
            "Top of pole should have ball (●)"
        );
    }

    #[test]
    fn test_render_pole_leans_left_negative_theta() {
        // Test: Pole uses ╱ character when leaning left (theta < 0)
        //
        // At theta = -10° ≈ -0.175 rad, pole should visibly lean left

        let theta = (-10.0_f32).to_radians();
        let env = setup_render_env(0.0, theta);
        let output = env.render().unwrap();

        // Should have at least one ╲ character in the pole (\ = bottom-right to top-left)
        let pole_area: String = output.lines().skip(2).take(5).collect();
        assert!(
            pole_area.contains('╲'),
            "Pole leaning left (θ=-10°) should have ╲ character.\nPole area:\n{pole_area}"
        );

        // Should still have ball at top
        assert!(
            line_contains(&output, 2, '●'),
            "Top of pole should have ball (●)"
        );
    }

    #[test]
    fn test_render_track_and_wheels_present() {
        // Test: Track (═) and wheels (○) are rendered
        //
        // Expected: Track line has ═ characters and two ○ wheels

        let env = setup_render_env(0.0, 0.0);
        let output = env.render().unwrap();

        let track_line_idx = 9; // Track is line 9
        let track_line = output
            .lines()
            .nth(track_line_idx)
            .expect("Track line missing");

        // Track should have ═ characters
        assert!(
            track_line.contains('═'),
            "Track line should have ═ character"
        );

        // Should have exactly 2 wheel characters
        let wheel_count = track_line.chars().filter(|&c| c == '○').count();
        assert_eq!(wheel_count, 2, "Should have exactly 2 wheels");
    }

    #[test]
    fn test_render_scale_markers_present() {
        // Test: Scale line shows position markers -2.4, 0, 2.4
        //
        // Expected: Last line contains these markers

        let env = setup_render_env(0.0, 0.0);
        let output = env.render().unwrap();

        let scale_line = output.lines().last().expect("Scale line missing");

        assert!(scale_line.contains("-2.4"), "Scale should show -2.4 marker");
        assert!(scale_line.contains('0'), "Scale should show 0 marker");
        assert!(scale_line.contains("2.4"), "Scale should show 2.4 marker");
    }

    #[test]
    fn test_render_combined_position_and_angle() {
        // Test: Cart position and pole angle render correctly together
        //
        // Cart at x=1.0 (right of center), pole at theta=+5° (slight lean right)

        let theta = 5.0_f32.to_radians();
        let env = setup_render_env(1.0, theta);
        let output = env.render().unwrap();

        // Cart should be right of center
        let cart_line_idx = 7;
        let left = find_char_col(&output, cart_line_idx, '┌').expect("Cart left corner not found");
        let right =
            find_char_col(&output, cart_line_idx, '┐').expect("Cart right corner not found");
        let cart_center = usize::midpoint(left, right);

        assert!(
            cart_center > 30,
            "Cart at x=1.0 should be right of center. Found at column {cart_center}"
        );

        // Ball should be present
        assert!(
            line_contains(&output, 2, '●'),
            "Top of pole should have ball (●)"
        );
    }

    #[test]
    fn test_render_output_dimensions() {
        // Test: Output has expected dimensions
        //
        // Expected: 11 lines, each 60 characters wide (except header may vary)

        let env = setup_render_env(0.0, 0.0);
        let output = env.render().unwrap();

        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 11, "Output should have 11 lines");

        // Visual lines (not header) should be 60 chars
        for (i, line) in lines.iter().enumerate().skip(2) {
            assert_eq!(
                line.chars().count(),
                60,
                "Line {} should be 60 chars, got {}",
                i,
                line.chars().count()
            );
        }
    }
}
