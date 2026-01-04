//! Human player input handling for interactive play.
//!
//! This module provides functions to prompt human players for actions
//! during evaluation/play sessions.

// Uses unwrap for terminal I/O which should not fail

use crate::env::Environment;
use rand::Rng;
use std::io::{self, Write};

/// Prompt a human player for their action.
///
/// Handles:
/// - Displaying the prompt with player name and turn info
/// - Special commands: help, render, random, hint, quit
/// - Parsing input via environment's `parse_action()`
/// - Validating against action mask
/// - Retrying on invalid input
///
/// # Arguments
/// * `env` - The environment (for rendering, action masks, parsing)
/// * `player_index` - Which player is being prompted (0-indexed)
/// * `player_count` - Total number of players in the game
/// * `player_name` - Human's name for the prompt
/// * `get_hint` - Optional closure returning network's suggested action
/// * `rng` - RNG for random move option
///
/// # Returns
/// The valid action index chosen by the human
pub fn prompt_human_action<E: Environment>(
    env: &E,
    player_index: usize,
    player_count: usize,
    player_name: &str,
    get_hint: Option<&dyn Fn() -> usize>,
    rng: &mut impl Rng,
) -> usize {
    loop {
        // Prompt with help hint
        if player_count == 1 {
            print!("{player_name}'s turn [? for help]: ");
        } else {
            print!(
                "{}'s turn (Player {} of {}) [? for help]: ",
                player_name,
                player_index + 1,
                player_count
            );
        }
        io::stdout().flush().expect("stdout flush");

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Error reading input. Try again.");
            continue;
        }
        let input = input.trim().to_lowercase();

        match input.as_str() {
            "help" | "h" | "?" => {
                show_actions(env);
            }
            "render" | "r" | "board" => {
                if let Some(render) = env.render() {
                    println!("{render}");
                } else {
                    println!("(No render available for this environment)");
                }
            }
            "random" | "rand" => {
                let action = random_valid_action(env, rng);
                println!("Playing random: {}", env.describe_action(action));
                return action;
            }
            "hint" | "suggest" => {
                if let Some(hint_fn) = get_hint {
                    let action = hint_fn();
                    println!("Network suggests: {}", env.describe_action(action));
                } else {
                    println!("No network available for hints");
                }
            }
            "quit" | "q" => {
                println!("Goodbye!");
                std::process::exit(0);
            }
            _ => match env.parse_action(&input) {
                Ok(action) => {
                    if is_valid_action::<E>(env, action) {
                        return action;
                    }
                    println!("That move is not valid. Type 'help' to see valid moves.");
                }
                Err(msg) => {
                    println!("{msg}");
                }
            },
        }
    }
}

/// Display all actions with validity indicators.
fn show_actions<E: Environment>(env: &E) {
    let mask = env.action_mask();
    println!("\nAvailable actions:");
    for action in 0..E::ACTION_COUNT {
        let valid = mask.as_ref().is_none_or(|m| m[action]);
        let marker = if valid { "  " } else { "X " };
        println!("  {}[{}] {}", marker, action, env.describe_action(action));
    }
    println!("  (X = currently invalid)");
    println!("\nCommands: help, render, random, hint, quit\n");
}

/// Pick a random valid action.
pub fn random_valid_action<E: Environment>(env: &E, rng: &mut impl Rng) -> usize {
    let mask = env.action_mask();
    let valid: Vec<usize> = (0..E::ACTION_COUNT)
        .filter(|&a| mask.as_ref().is_none_or(|m| m[a]))
        .collect();

    if valid.is_empty() {
        // Fallback: return 0 if somehow no valid actions (shouldn't happen)
        0
    } else {
        valid[rng.gen_range(0..valid.len())]
    }
}

/// Check if an action is valid according to the action mask.
fn is_valid_action<E: Environment>(env: &E, action: usize) -> bool {
    if action >= E::ACTION_COUNT {
        return false;
    }
    env.action_mask().is_none_or(|m| m[action])
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: prompt_human_action is difficult to unit test since it requires stdin.
    // These tests cover the helper functions.

    struct MockEnv {
        mask: Option<Vec<bool>>,
    }

    impl Environment for MockEnv {
        const OBSERVATION_DIM: usize = 1;
        const ACTION_COUNT: usize = 4;
        const NAME: &'static str = "mock";

        fn new(_seed: u64) -> Self {
            Self { mask: None }
        }

        fn reset(&mut self) -> Vec<f32> {
            vec![0.0]
        }

        fn step(&mut self, _action: usize) -> (Vec<f32>, Vec<f32>, bool) {
            (vec![0.0], vec![0.0], false)
        }

        fn action_mask(&self) -> Option<Vec<bool>> {
            self.mask.clone()
        }
    }

    #[test]
    fn test_random_valid_action_no_mask() {
        let env = MockEnv { mask: None };
        let mut rng = rand::thread_rng();

        // With no mask, any action should be valid
        for _ in 0..10 {
            let action = random_valid_action(&env, &mut rng);
            assert!(action < MockEnv::ACTION_COUNT);
        }
    }

    #[test]
    fn test_random_valid_action_with_mask() {
        let env = MockEnv {
            mask: Some(vec![false, true, false, true]), // Only actions 1 and 3 valid
        };
        let mut rng = rand::thread_rng();

        for _ in 0..20 {
            let action = random_valid_action(&env, &mut rng);
            assert!(action == 1 || action == 3, "Got invalid action {action}");
        }
    }

    #[test]
    fn test_is_valid_action() {
        let env = MockEnv {
            mask: Some(vec![true, false, true, false]),
        };

        assert!(is_valid_action::<MockEnv>(&env, 0));
        assert!(!is_valid_action::<MockEnv>(&env, 1));
        assert!(is_valid_action::<MockEnv>(&env, 2));
        assert!(!is_valid_action::<MockEnv>(&env, 3));
        assert!(!is_valid_action::<MockEnv>(&env, 4)); // Out of bounds
    }

    #[test]
    fn test_is_valid_action_no_mask() {
        let env = MockEnv { mask: None };

        // All actions valid when no mask
        for action in 0..MockEnv::ACTION_COUNT {
            assert!(is_valid_action::<MockEnv>(&env, action));
        }
        // Out of bounds still invalid
        assert!(!is_valid_action::<MockEnv>(&env, 10));
    }

    #[test]
    fn test_random_valid_action_single_valid() {
        // Edge case: only one action is valid
        let env = MockEnv {
            mask: Some(vec![false, false, true, false]), // Only action 2 valid
        };
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let action = random_valid_action(&env, &mut rng);
            assert_eq!(action, 2, "Should always pick the only valid action");
        }
    }

    #[test]
    fn test_random_valid_action_all_invalid() {
        // Edge case: no valid actions (shouldn't happen in practice, but test fallback)
        let env = MockEnv {
            mask: Some(vec![false, false, false, false]),
        };
        let mut rng = rand::thread_rng();

        let action = random_valid_action(&env, &mut rng);
        assert_eq!(action, 0, "Fallback should return 0");
    }
}
