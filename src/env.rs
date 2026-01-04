/// Environment trait and `VecEnv` parallel wrapper
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::profile::{profile_function, profile_scope};

/// Game outcome for evaluation - who won/placed where
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameOutcome {
    /// Single winner (player index)
    Winner(usize),
    /// All players tied
    Tie,
    /// N-player rankings (1-indexed: 1=first, 2=second, etc.)
    Placements(Vec<usize>),
}

/// Minimal environment interface - just what PPO needs.
/// Uses write-buffer API for zero allocations in hot paths.
pub trait Environment: Send + Sync + Sized + 'static {
    /// Dimension of observation vector
    const OBSERVATION_DIM: usize;

    /// Number of discrete actions
    const ACTION_COUNT: usize;

    /// Environment name for logging
    const NAME: &'static str;

    /// Number of players (1 for single-agent, 2+ for multi-player)
    const NUM_PLAYERS: usize = 1;

    /// Create a new environment with the given seed.
    /// For deterministic games, the seed may be ignored.
    fn new(seed: u64) -> Self;

    /// Reset environment and write initial observation to buffer.
    /// Buffer must have length `OBSERVATION_DIM`.
    fn reset(&mut self, obs: &mut [f32]);

    /// Take action and write results to buffers. Returns done flag.
    ///
    /// For multi-player games:
    /// - obs: player-agnostic observation (length `OBSERVATION_DIM`)
    /// - rewards: reward for EACH player this step (length `NUM_PLAYERS`)
    /// - returns: true if episode ended
    fn step(&mut self, action: usize, obs: &mut [f32], rewards: &mut [f32]) -> bool;

    /// Current player index (`0..NUM_PLAYERS`). Default 0 for single-agent.
    fn current_player(&self) -> usize {
        0
    }

    /// Write action mask to buffer. Always writes.
    /// Buffer must have length `ACTION_COUNT`.
    /// Environments without dynamic masking should write all `true`.
    fn action_mask(&self, mask: &mut [bool]);

    /// Return explicit game outcome when episode is done.
    /// Override this if using reward shaping where `total_rewards` doesn't reflect outcome.
    /// Default: None (infer from `total_rewards` via argmax)
    fn game_outcome(&self) -> Option<GameOutcome> {
        None
    }

    /// Render the current state as a string for visualization.
    /// Used in eval watch mode. Default: None (no rendering support).
    fn render(&self) -> Option<String> {
        None
    }

    /// Human-readable description of an action (e.g., "Column 3" for Connect Four).
    /// Used for human player input prompts.
    fn describe_action(&self, action: usize) -> String {
        format!("Action {action}")
    }

    /// Parse human input string to action index.
    /// Returns Ok(action) or Err(help message).
    fn parse_action(&self, input: &str) -> Result<usize, String> {
        input
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Enter action number 0-{}", Self::ACTION_COUNT - 1))
    }
}

/// Episode statistics for completed episodes
#[derive(Debug, Clone)]
pub struct EpisodeStats {
    /// Total reward per player [`NUM_PLAYERS`]
    pub total_rewards: Vec<f32>,
    pub length: usize,
    /// Which environment index this episode came from
    pub env_index: usize,
    /// Game outcome (for multiplayer games)
    pub outcome: Option<GameOutcome>,
}

impl EpisodeStats {
    /// Get total reward for player 0 (backward compatibility for single-agent)
    pub fn total_reward(&self) -> f32 {
        self.total_rewards.first().copied().unwrap_or(0.0)
    }
}

/// Compute win rates and draw rate from a collection of game outcomes
///
/// Returns (`win_rates`[player], `draw_rate`) where `win_rates`[i] is the fraction
/// of games won by player i, and `draw_rate` is the fraction of games that were draws.
pub fn compute_outcome_rates(
    outcomes: &VecDeque<GameOutcome>,
    num_players: usize,
) -> (Vec<f32>, f32) {
    let total = outcomes.len() as f32;
    if total == 0.0 {
        return (vec![0.0; num_players], 0.0);
    }

    let mut wins = vec![0usize; num_players];
    let mut draws = 0usize;

    for outcome in outcomes {
        match outcome {
            GameOutcome::Winner(w) => wins[*w] += 1,
            GameOutcome::Tie => draws += 1,
            GameOutcome::Placements(places) => {
                // Count first-place finishes
                let first_count = places.iter().filter(|&&p| p == 1).count();
                if first_count == places.len() {
                    // All tied for first = draw
                    draws += 1;
                } else {
                    // Sole first place = win
                    for (i, &place) in places.iter().enumerate() {
                        if place == 1 && first_count == 1 {
                            wins[i] += 1;
                        }
                    }
                }
            }
        }
    }

    let win_rates: Vec<f32> = wins.iter().map(|&w| w as f32 / total).collect();
    let draw_rate = draws as f32 / total;
    (win_rates, draw_rate)
}

/// Vectorized environment wrapper for parallel execution
///
/// Steps N environments in parallel using rayon, automatically resetting
/// on episode termination. Uses preallocated buffers for zero allocations
/// in the hot path.
pub struct VecEnv<E: Environment> {
    envs: Vec<E>,
    /// Pre-allocated flat observation buffer [`num_envs` * `OBSERVATION_DIM`]
    obs_buffer: Vec<f32>,
    /// Pre-allocated flat reward buffer [`num_envs` * `NUM_PLAYERS`]
    reward_buffer: Vec<f32>,
    /// Pre-allocated flat action mask buffer [`num_envs` * `ACTION_COUNT`]
    mask_buffer: Vec<bool>,
    /// Pre-allocated done buffer [`num_envs`]
    done_buffer: Vec<bool>,
    /// Pre-allocated current player buffer [`num_envs`]
    player_buffer: Vec<usize>,
    /// Accumulated rewards per env, per player [`num_envs`][`NUM_PLAYERS`] (reset on episode end)
    episode_rewards: Vec<Vec<f32>>,
    /// Steps taken per env (reset on episode end)
    episode_lengths: Vec<usize>,
    /// Envs in terminal state (won't step or reset) [`num_envs`]
    terminal: Vec<bool>,
}

impl<E: Environment> VecEnv<E> {
    /// Create `VecEnv` from a factory function
    pub fn new<F>(num_envs: usize, factory: F) -> Self
    where
        F: Fn(usize) -> E,
    {
        let mut envs: Vec<E> = (0..num_envs).map(factory).collect();

        // Pre-allocate all buffers
        let mut obs_buffer = vec![0.0; num_envs * E::OBSERVATION_DIM];
        let reward_buffer = vec![0.0; num_envs * E::NUM_PLAYERS];
        let mut mask_buffer = vec![false; num_envs * E::ACTION_COUNT];
        let done_buffer = vec![false; num_envs];
        let mut player_buffer = vec![0; num_envs];

        // Initialize with reset observations, masks, and players
        for (i, env) in envs.iter_mut().enumerate() {
            let obs_slice = &mut obs_buffer[i * E::OBSERVATION_DIM..][..E::OBSERVATION_DIM];
            env.reset(obs_slice);

            let mask_slice = &mut mask_buffer[i * E::ACTION_COUNT..][..E::ACTION_COUNT];
            env.action_mask(mask_slice);

            player_buffer[i] = env.current_player();
        }

        Self {
            envs,
            obs_buffer,
            reward_buffer,
            mask_buffer,
            done_buffer,
            player_buffer,
            episode_rewards: vec![vec![0.0; E::NUM_PLAYERS]; num_envs],
            episode_lengths: vec![0; num_envs],
            terminal: vec![false; num_envs],
        }
    }

    /// Number of parallel environments
    pub const fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Zero-copy access to observations [`num_envs` * `OBSERVATION_DIM`]
    pub fn observations(&self) -> &[f32] {
        &self.obs_buffer
    }

    /// Zero-copy access to rewards [`num_envs` * `NUM_PLAYERS`]
    pub fn rewards(&self) -> &[f32] {
        &self.reward_buffer
    }

    /// Zero-copy access to action masks [`num_envs` * `ACTION_COUNT`]
    pub fn action_masks(&self) -> &[bool] {
        &self.mask_buffer
    }

    /// Zero-copy access to done flags [`num_envs`]
    pub fn dones(&self) -> &[bool] {
        &self.done_buffer
    }

    /// Zero-copy access to current players [`num_envs`]
    pub fn current_players(&self) -> &[usize] {
        &self.player_buffer
    }

    /// Mark an env as terminal (won't step or reset)
    pub fn set_terminal(&mut self, idx: usize) {
        self.terminal[idx] = true;
    }

    /// Get mask of terminal envs
    pub fn terminal_mask(&self) -> &[bool] {
        &self.terminal
    }

    /// Count of non-terminal (active) envs
    pub fn active_count(&self) -> usize {
        self.terminal.iter().filter(|&&t| !t).count()
    }

    /// Step all environments with given actions (parallel, zero-allocation)
    ///
    /// Results are written to internal buffers accessible via:
    /// - `observations()` - new observations
    /// - `rewards()` - step rewards
    /// - `dones()` - done flags
    ///
    /// Returns completed episode stats (only allocation is for completed episodes)
    pub fn step(&mut self, actions: &[usize]) -> Vec<EpisodeStats> {
        profile_function!();
        assert_eq!(actions.len(), self.envs.len());

        // Parallel step all environments - writes directly to preallocated buffers
        let completed: Vec<Option<EpisodeStats>> = {
            profile_scope!("parallel_env_step");

            self.envs
                .par_iter_mut()
                .zip(actions.par_iter())
                .zip(self.obs_buffer.par_chunks_mut(E::OBSERVATION_DIM))
                .zip(self.reward_buffer.par_chunks_mut(E::NUM_PLAYERS))
                .zip(self.done_buffer.par_iter_mut())
                .zip(self.mask_buffer.par_chunks_mut(E::ACTION_COUNT))
                .zip(self.player_buffer.par_iter_mut())
                .zip(self.episode_rewards.par_iter_mut())
                .zip(self.episode_lengths.par_iter_mut())
                .zip(self.terminal.par_iter())
                .enumerate()
                .map(
                    |(
                        env_idx,
                        (
                            (
                                (
                                    ((((((env, &action), obs), rewards), done), mask), player),
                                    ep_rewards,
                                ),
                                length,
                            ),
                            &is_terminal,
                        ),
                    )| {
                        #[cfg(feature = "tracy")]
                        let _span = tracy_client::span!("env_step");

                        // Skip terminal envs entirely
                        if is_terminal {
                            rewards.fill(0.0);
                            *done = true;
                            return None;
                        }

                        // Step writes directly to preallocated slices - no allocation!
                        *done = env.step(action, obs, rewards);

                        // Accumulate per-player rewards
                        for (i, &r) in rewards.iter().enumerate() {
                            ep_rewards[i] += r;
                        }
                        *length += 1;

                        // Update mask and player after step
                        env.action_mask(mask);
                        *player = env.current_player();

                        if *done {
                            // Capture game outcome BEFORE reset (state is lost after reset)
                            let outcome = env.game_outcome();
                            let stats = EpisodeStats {
                                total_rewards: ep_rewards.clone(),
                                length: *length,
                                env_index: env_idx,
                                outcome,
                            };
                            // Reset writes to same obs buffer
                            env.reset(obs);
                            env.action_mask(mask);
                            *player = env.current_player();
                            ep_rewards.fill(0.0);
                            *length = 0;
                            Some(stats)
                        } else {
                            None
                        }
                    },
                )
                .collect()
        };

        // Only allocation: collect completed episodes
        completed.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test environment: counter that terminates at `MAX_STEPS`
    struct CounterEnv<const MAX_STEPS: usize> {
        count: usize,
    }

    impl<const MAX_STEPS: usize> Environment for CounterEnv<MAX_STEPS> {
        const OBSERVATION_DIM: usize = 1;
        const ACTION_COUNT: usize = 2;
        const NAME: &'static str = "counter";

        fn new(_seed: u64) -> Self {
            Self { count: 0 }
        }

        fn reset(&mut self, obs: &mut [f32]) {
            self.count = 0;
            obs[0] = 0.0;
        }

        fn step(&mut self, _action: usize, obs: &mut [f32], rewards: &mut [f32]) -> bool {
            self.count += 1;
            let done = self.count >= MAX_STEPS;
            obs[0] = self.count as f32;
            rewards[0] = 1.0;
            done
        }

        fn action_mask(&self, mask: &mut [bool]) {
            mask.fill(true); // All actions valid
        }
    }

    #[test]
    fn test_vec_env_creation() {
        let vec_env: VecEnv<CounterEnv<3>> = VecEnv::new(4, |i| CounterEnv::<3>::new(i as u64));

        assert_eq!(vec_env.num_envs(), 4);
        assert_eq!(CounterEnv::<3>::OBSERVATION_DIM, 1);
        assert_eq!(CounterEnv::<3>::ACTION_COUNT, 2);
    }

    #[test]
    fn test_vec_env_step() {
        let mut vec_env: VecEnv<CounterEnv<3>> = VecEnv::new(2, |i| CounterEnv::<3>::new(i as u64));

        let actions = vec![0, 1];
        let completed = vec_env.step(&actions);

        // Access results via buffer accessors
        assert_eq!(vec_env.observations(), &[1.0, 1.0]); // Both envs stepped to count=1
        assert_eq!(vec_env.rewards(), &[1.0, 1.0]); // Rewards for each env
        assert_eq!(vec_env.dones(), &[false, false]);
        assert!(completed.is_empty());
    }

    #[test]
    fn test_vec_env_auto_reset() {
        let mut vec_env: VecEnv<CounterEnv<2>> = VecEnv::new(1, |i| CounterEnv::<2>::new(i as u64));

        // Step 1
        vec_env.step(&[0]);
        // Step 2 - should terminate
        let completed = vec_env.step(&[0]);

        assert_eq!(vec_env.dones(), &[true]);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].total_reward(), 2.0);
        assert_eq!(completed[0].length, 2);

        // After auto-reset, observations should be reset state
        assert_eq!(vec_env.observations(), &[0.0]);
    }

    #[test]
    fn test_episode_stats() {
        let mut vec_env: VecEnv<CounterEnv<5>> = VecEnv::new(1, |i| CounterEnv::<5>::new(i as u64));

        // Run 5 steps to complete episode
        for _ in 0..4 {
            vec_env.step(&[0]);
            assert_eq!(vec_env.dones(), &[false]);
        }

        let completed = vec_env.step(&[0]);
        assert_eq!(vec_env.dones(), &[true]);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].total_reward(), 5.0);
        assert_eq!(completed[0].length, 5);
    }

    // =========================================
    // compute_outcome_rates Tests
    // =========================================

    #[test]
    fn test_compute_outcome_rates_empty() {
        let outcomes: VecDeque<GameOutcome> = VecDeque::new();
        let (win_rates, draw_rate) = compute_outcome_rates(&outcomes, 2);

        assert_eq!(win_rates, vec![0.0, 0.0]);
        assert_eq!(draw_rate, 0.0);
    }

    #[test]
    fn test_compute_outcome_rates_all_wins_p0() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        for _ in 0..10 {
            outcomes.push_back(GameOutcome::Winner(0));
        }
        let (win_rates, draw_rate) = compute_outcome_rates(&outcomes, 2);

        assert_eq!(win_rates, vec![1.0, 0.0]);
        assert_eq!(draw_rate, 0.0);
    }

    #[test]
    fn test_compute_outcome_rates_all_wins_p1() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        for _ in 0..10 {
            outcomes.push_back(GameOutcome::Winner(1));
        }
        let (win_rates, draw_rate) = compute_outcome_rates(&outcomes, 2);

        assert_eq!(win_rates, vec![0.0, 1.0]);
        assert_eq!(draw_rate, 0.0);
    }

    #[test]
    fn test_compute_outcome_rates_all_ties() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        for _ in 0..10 {
            outcomes.push_back(GameOutcome::Tie);
        }
        let (win_rates, draw_rate) = compute_outcome_rates(&outcomes, 2);

        assert_eq!(win_rates, vec![0.0, 0.0]);
        assert_eq!(draw_rate, 1.0);
    }

    #[test]
    fn test_compute_outcome_rates_mixed() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        // 4 wins for P0, 3 wins for P1, 3 draws = 10 games
        for _ in 0..4 {
            outcomes.push_back(GameOutcome::Winner(0));
        }
        for _ in 0..3 {
            outcomes.push_back(GameOutcome::Winner(1));
        }
        for _ in 0..3 {
            outcomes.push_back(GameOutcome::Tie);
        }
        let (win_rates, draw_rate) = compute_outcome_rates(&outcomes, 2);

        assert!((win_rates[0] - 0.4).abs() < 0.001);
        assert!((win_rates[1] - 0.3).abs() < 0.001);
        assert!((draw_rate - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_compute_outcome_rates_placements_sole_winner() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        // 3-player game: P0 wins, P1 second, P2 third
        outcomes.push_back(GameOutcome::Placements(vec![1, 2, 3]));
        // P1 wins
        outcomes.push_back(GameOutcome::Placements(vec![2, 1, 3]));
        // P2 wins
        outcomes.push_back(GameOutcome::Placements(vec![3, 2, 1]));

        let (win_rates, draw_rate) = compute_outcome_rates(&outcomes, 3);

        // Each player won once out of 3 games
        assert!((win_rates[0] - 1.0 / 3.0).abs() < 0.001);
        assert!((win_rates[1] - 1.0 / 3.0).abs() < 0.001);
        assert!((win_rates[2] - 1.0 / 3.0).abs() < 0.001);
        assert_eq!(draw_rate, 0.0);
    }

    #[test]
    fn test_compute_outcome_rates_placements_all_tied() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        // 3-player game: all tied for first
        outcomes.push_back(GameOutcome::Placements(vec![1, 1, 1]));
        outcomes.push_back(GameOutcome::Placements(vec![1, 1, 1]));

        let (win_rates, draw_rate) = compute_outcome_rates(&outcomes, 3);

        assert_eq!(win_rates, vec![0.0, 0.0, 0.0]);
        assert_eq!(draw_rate, 1.0); // All games were ties
    }

    // =========================================
    // VecEnv Terminal Operations Tests
    // =========================================

    #[test]
    fn test_vec_env_set_terminal() {
        let mut vec_env: VecEnv<CounterEnv<10>> =
            VecEnv::new(3, |i| CounterEnv::<10>::new(i as u64));

        // Initially all active
        assert_eq!(vec_env.terminal_mask(), &[false, false, false]);

        // Mark env 1 as terminal
        vec_env.set_terminal(1);
        assert_eq!(vec_env.terminal_mask(), &[false, true, false]);

        // Mark env 0 as terminal
        vec_env.set_terminal(0);
        assert_eq!(vec_env.terminal_mask(), &[true, true, false]);
    }

    #[test]
    fn test_vec_env_active_count() {
        let mut vec_env: VecEnv<CounterEnv<10>> =
            VecEnv::new(5, |i| CounterEnv::<10>::new(i as u64));

        assert_eq!(vec_env.active_count(), 5);

        vec_env.set_terminal(0);
        assert_eq!(vec_env.active_count(), 4);

        vec_env.set_terminal(2);
        vec_env.set_terminal(4);
        assert_eq!(vec_env.active_count(), 2);
    }

    #[test]
    fn test_vec_env_terminal_skips_step() {
        let mut vec_env: VecEnv<CounterEnv<10>> =
            VecEnv::new(2, |i| CounterEnv::<10>::new(i as u64));

        // Mark env 0 as terminal
        vec_env.set_terminal(0);

        // Step - env 0 should be skipped
        vec_env.step(&[0, 0]);

        // Env 0 was terminal: returns zeroed rewards and done=true
        assert_eq!(vec_env.rewards()[0], 0.0);
        assert!(vec_env.dones()[0]);

        // Env 1 was active: stepped normally
        assert_eq!(vec_env.rewards()[1], 1.0);
        assert!(!vec_env.dones()[1]);
        assert_eq!(vec_env.observations()[1], 1.0); // Env 1 advanced to count=1
    }

    // =========================================
    // Pre-refactor VecEnv coverage tests
    // =========================================

    #[test]
    fn test_vec_env_action_masks() {
        use crate::envs::ConnectFour;

        let vec_env: VecEnv<ConnectFour> = VecEnv::new(2, |i| ConnectFour::new(i as u64));
        let masks = vec_env.action_masks();

        // ConnectFour has action masking - verify shape
        assert_eq!(masks.len(), 2 * ConnectFour::ACTION_COUNT); // 2 envs * 7 actions

        // Initially all columns valid (empty board)
        assert!(masks.iter().all(|&m| m));
    }

    #[test]
    fn test_vec_env_multi_player() {
        use crate::envs::ConnectFour;

        let mut vec_env: VecEnv<ConnectFour> = VecEnv::new(2, |i| ConnectFour::new(i as u64));

        // Check current players - both should start with P0
        let players = vec_env.current_players();
        assert_eq!(players.len(), 2);
        assert!(players.iter().all(|&p| p == 0));

        // Step and verify player switches
        vec_env.step(&[0, 0]); // Both drop in column 0
        let players = vec_env.current_players();
        assert!(players.iter().all(|&p| p == 1)); // Both now P1
    }

    #[test]
    fn test_vec_env_obs_after_auto_reset() {
        // CounterEnv<3> terminates after 3 steps
        let mut vec_env: VecEnv<CounterEnv<3>> = VecEnv::new(2, |i| CounterEnv::<3>::new(i as u64));

        // Step 3 times to trigger termination and auto-reset
        vec_env.step(&[0, 0]);
        vec_env.step(&[0, 0]);
        vec_env.step(&[0, 0]);

        // Both envs should have terminated
        assert!(vec_env.dones()[0]);
        assert!(vec_env.dones()[1]);

        // After auto-reset, observations should be reset state (0.0)
        assert_eq!(vec_env.observations()[0], 0.0);
        assert_eq!(vec_env.observations()[1], 0.0);
    }

    #[test]
    fn test_vec_env_four_player() {
        use crate::envs::LiarsDice;

        let vec_env: VecEnv<LiarsDice> = VecEnv::new(2, |i| LiarsDice::new(i as u64));

        // Verify observation shape
        assert_eq!(vec_env.observations().len(), 2 * LiarsDice::OBSERVATION_DIM); // 2 * 78 = 156

        // Verify action masks shape
        assert_eq!(vec_env.action_masks().len(), 2 * LiarsDice::ACTION_COUNT); // 2 * 49 = 98

        // Verify current players
        assert_eq!(vec_env.current_players().len(), 2);
    }
}
