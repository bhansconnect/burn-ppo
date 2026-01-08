/// Environment trait and `VecEnv` parallel wrapper
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::profile::{profile_function, profile_scope};

/// Game outcome representing final placements for all players.
///
/// Uses standard competition ranking (1224 ranking):
/// - Placements are 1-indexed (1 = first place)
/// - Tied players share the same rank
/// - After a tie, the next rank is skipped (e.g., `[1, 1, 3, 4]` not `[1, 1, 2, 3]`)
/// - Full tie: all players get 1 (e.g., `[1, 1, 1, 1]`)
///
/// Examples:
/// - Clear winner: `GameOutcome(vec![1, 2, 3, 4])` (player 0 won, player 3 last)
/// - Two-way tie for 1st: `GameOutcome(vec![1, 1, 3, 4])`
/// - Three-way tie for 2nd: `GameOutcome(vec![1, 2, 2, 2])`
/// - Full tie: `GameOutcome(vec![1, 1, 1, 1])`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameOutcome(pub Vec<usize>);

/// Minimal environment interface - just what PPO needs
pub trait Environment: Send + Sync + Sized + 'static {
    /// Dimension of observation vector
    const OBSERVATION_DIM: usize;

    /// Number of discrete actions
    const ACTION_COUNT: usize;

    /// Environment name for logging
    const NAME: &'static str;

    /// Number of players (1 for single-agent, 2+ for multi-player)
    const NUM_PLAYERS: usize = 1;

    /// Spatial observation shape for CNN networks: (height, width, channels)
    /// If None, CNN cannot be used (only MLP).
    ///
    /// **Layout convention**: The observation vector is structured as:
    ///   `[spatial_data..., extra_features...]`
    /// where:
    ///   - `spatial_data` has size H * W * C (can be reshaped to [H, W, C])
    ///   - `extra_features` has size `OBSERVATION_DIM` - (H * W * C)
    ///
    /// For CNN: `spatial_data` goes through conv layers, `extra_features`
    /// are concatenated after flattening the conv output.
    const OBSERVATION_SHAPE: Option<(usize, usize, usize)> = None;

    /// Default starting temperature for evaluation/tournament play.
    /// Higher values = more exploration/randomness.
    /// Stochastic games (e.g., bluffing games) should override to 1.0.
    const EVAL_TEMP: f32 = 0.3;

    /// Optional temperature cutoff for evaluation: (`move_number`, `final_temp`).
    /// After `move_number` moves, temperature drops to `final_temp`.
    /// None = use constant temperature throughout.
    const EVAL_TEMP_CUTOFF: Option<(usize, f32)> = None;

    /// Create a new environment with the given seed.
    /// For deterministic games, the seed may be ignored.
    fn new(seed: u64) -> Self;

    /// Reset environment and return initial observation
    fn reset(&mut self) -> Vec<f32>;

    /// Take action and return (observation, rewards[`NUM_PLAYERS`], done)
    ///
    /// For multi-player games:
    /// - observation: player-agnostic (same encoding for all players)
    /// - rewards: reward for EACH player this step (not just actor)
    /// - done: true if episode ended
    fn step(&mut self, action: usize) -> (Vec<f32>, Vec<f32>, bool);

    /// Current player index (`0..NUM_PLAYERS`). Default 0 for single-agent.
    fn current_player(&self) -> usize {
        0
    }

    /// Return mask of valid actions (true = valid). None = all valid.
    fn action_mask(&self) -> Option<Vec<bool>> {
        None
    }

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

/// Compute average Swiss points per player and draw rate from game outcomes
///
/// Swiss points formula: `points = num_players - avg_position`
/// - 2-player: win=1.0, draw=0.5, loss=0.0 (equivalent to `win_rate` + 0.5*`draw_rate`)
/// - 4-player: 1st=3.0, 2nd=2.0, 3rd=1.0, 4th=0.0
/// - Ties: shared positions get averaged points
///
/// Returns (`avg_points[player]`, `draw_rate`) where draw = all tied for 1st
pub fn compute_avg_points(outcomes: &VecDeque<GameOutcome>, num_players: usize) -> (Vec<f32>, f32) {
    let total = outcomes.len() as f32;
    if total == 0.0 {
        return (vec![0.0; num_players], 0.0);
    }

    let mut total_points = vec![0.0f64; num_players];
    let mut draws = 0usize;

    for outcome in outcomes {
        let placements = &outcome.0;

        // Check if all tied for first (draw)
        if placements.iter().all(|&p| p == 1) {
            draws += 1;
        }

        // Calculate Swiss points using fractional ranking
        // points = num_players - avg_position_for_that_rank
        let n = placements.len();
        for (player, &place) in placements.iter().enumerate() {
            // Count how many players share this placement
            let tied_count = placements.iter().filter(|&&p| p == place).count();
            // Average position for tied players: place, place+1, ..., place+tied-1
            // avg = place + (tied_count - 1) / 2
            let avg_position = place as f64 + (tied_count as f64 - 1.0) / 2.0;
            // Points = num_players - avg_position
            let points = n as f64 - avg_position;
            total_points[player] += points;
        }
    }

    let avg_points: Vec<f32> = total_points
        .iter()
        .map(|&p| (p / f64::from(total)) as f32)
        .collect();
    let draw_rate = draws as f32 / total;
    (avg_points, draw_rate)
}

/// Vectorized environment wrapper for parallel execution
///
/// Steps N environments in parallel, automatically resetting
/// on episode termination.
pub struct VecEnv<E: Environment> {
    envs: Vec<E>,
    /// Pre-allocated flat observation buffer [`num_envs` * `obs_dim`]
    obs_buffer: Vec<f32>,
    /// Accumulated rewards per env, per player [`num_envs`][num_players] (reset on episode end)
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

        // Pre-allocate flat observation buffer and initialize with reset observations
        let mut obs_buffer = vec![0.0; num_envs * E::OBSERVATION_DIM];
        for (i, env) in envs.iter_mut().enumerate() {
            let obs = env.reset();
            let offset = i * E::OBSERVATION_DIM;
            obs_buffer[offset..offset + E::OBSERVATION_DIM].copy_from_slice(&obs);
        }

        Self {
            envs,
            obs_buffer,
            episode_rewards: vec![vec![0.0; E::NUM_PLAYERS]; num_envs],
            episode_lengths: vec![0; num_envs],
            terminal: vec![false; num_envs],
        }
    }

    /// Number of parallel environments
    pub const fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Get current observations as flat array [`num_envs` * `obs_dim`]
    pub fn get_observations(&self) -> Vec<f32> {
        self.obs_buffer.clone()
    }

    /// Get current player index for each environment (`0..NUM_PLAYERS`)
    pub fn get_current_players(&self) -> Vec<usize> {
        self.envs.iter().map(Environment::current_player).collect()
    }

    /// Get action masks for all environments, flattened [`num_envs` * `action_count`]
    /// Returns None if environment doesn't support action masking.
    pub fn get_action_masks(&self) -> Option<Vec<bool>> {
        // Check first env for masking support
        self.envs[0].action_mask()?;

        let action_count = E::ACTION_COUNT;
        let mut masks = Vec::with_capacity(self.envs.len() * action_count);
        for env in &self.envs {
            if let Some(mask) = env.action_mask() {
                masks.extend(mask);
            }
        }
        Some(masks)
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

    /// Step all environments with given actions
    ///
    /// Returns:
    /// - observations: [`num_envs`, `obs_dim`] flattened
    /// - `all_rewards`: [`num_envs`][num_players] - rewards for ALL players per env
    /// - dones: [`num_envs`]
    /// - `completed_episodes`: stats for any episodes that finished this step
    pub fn step(
        &mut self,
        actions: &[usize],
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<bool>, Vec<EpisodeStats>) {
        profile_function!();
        assert_eq!(actions.len(), self.envs.len());

        let num_players = E::NUM_PLAYERS;

        // Parallel step all environments
        let results: Vec<_> = {
            profile_scope!("parallel_env_step");

            self.envs
                .par_iter_mut()
                .zip(actions.par_iter())
                .zip(self.episode_rewards.par_iter_mut())
                .zip(self.episode_lengths.par_iter_mut())
                .zip(self.obs_buffer.par_chunks_mut(E::OBSERVATION_DIM))
                .zip(self.terminal.par_iter())
                .enumerate()
                .map(
                    |(
                        env_idx,
                        (((((env, &action), ep_rewards), length), obs_chunk), &is_terminal),
                    )| {
                        #[cfg(feature = "tracy")]
                        let _span = tracy_client::span!("env_step");

                        // Skip terminal envs entirely
                        if is_terminal {
                            return (vec![0.0; num_players], true, None);
                        }

                        let (new_obs, step_rewards, done) = env.step(action);

                        // Accumulate per-player rewards
                        for (i, &r) in step_rewards.iter().enumerate() {
                            ep_rewards[i] += r;
                        }
                        *length += 1;

                        let completed = if done {
                            // Capture game outcome BEFORE reset (state is lost after reset)
                            let outcome = env.game_outcome();
                            let stats = EpisodeStats {
                                total_rewards: ep_rewards.clone(),
                                length: *length,
                                env_index: env_idx,
                                outcome,
                            };
                            let reset_obs = env.reset();
                            obs_chunk.copy_from_slice(&reset_obs);
                            for r in ep_rewards.iter_mut() {
                                *r = 0.0;
                            }
                            *length = 0;
                            Some(stats)
                        } else {
                            obs_chunk.copy_from_slice(&new_obs);
                            None
                        };

                        (step_rewards, done, completed)
                    },
                )
                .collect()
        };

        // Collect results (sequential, fast)
        let mut all_rewards = Vec::with_capacity(self.envs.len());
        let mut dones = Vec::with_capacity(self.envs.len());
        let mut completed_episodes = Vec::new();

        for (step_rewards, done, completed) in results {
            // Ensure proper size even if env returned wrong size
            let mut rewards = step_rewards;
            rewards.resize(num_players, 0.0);
            all_rewards.push(rewards);
            dones.push(done);
            if let Some(stats) = completed {
                completed_episodes.push(stats);
            }
        }

        let flat_obs = self.get_observations();
        (flat_obs, all_rewards, dones, completed_episodes)
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

        fn reset(&mut self) -> Vec<f32> {
            self.count = 0;
            vec![0.0]
        }

        fn step(&mut self, _action: usize) -> (Vec<f32>, Vec<f32>, bool) {
            self.count += 1;
            let done = self.count >= MAX_STEPS;
            (vec![self.count as f32], vec![1.0], done)
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
        let (obs, all_rewards, dones, completed) = vec_env.step(&actions);

        assert_eq!(obs, vec![1.0, 1.0]); // Both envs stepped to count=1
        assert_eq!(all_rewards, vec![vec![1.0], vec![1.0]]);
        assert_eq!(dones, vec![false, false]);
        assert!(completed.is_empty());
    }

    #[test]
    fn test_vec_env_auto_reset() {
        let mut vec_env: VecEnv<CounterEnv<2>> = VecEnv::new(1, |i| CounterEnv::<2>::new(i as u64));

        // Step 1
        vec_env.step(&[0]);
        // Step 2 - should terminate
        let (obs, _all_rewards, dones, completed) = vec_env.step(&[0]);

        assert_eq!(dones, vec![true]);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].total_reward(), 2.0);
        assert_eq!(completed[0].length, 2);

        // After auto-reset, observations should be reset state
        assert_eq!(obs, vec![0.0]);
    }

    #[test]
    fn test_episode_stats() {
        let mut vec_env: VecEnv<CounterEnv<5>> = VecEnv::new(1, |i| CounterEnv::<5>::new(i as u64));

        // Run 5 steps to complete episode
        for _ in 0..4 {
            let (_, _, dones, _) = vec_env.step(&[0]);
            assert_eq!(dones, vec![false]);
        }

        let (_, _, dones, completed) = vec_env.step(&[0]);
        assert_eq!(dones, vec![true]);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].total_reward(), 5.0);
        assert_eq!(completed[0].length, 5);
    }

    // =========================================
    // compute_avg_points Tests
    // =========================================

    #[test]
    fn test_compute_avg_points_empty() {
        let outcomes: VecDeque<GameOutcome> = VecDeque::new();
        let (avg_points, draw_rate) = compute_avg_points(&outcomes, 2);

        assert_eq!(avg_points, vec![0.0, 0.0]);
        assert_eq!(draw_rate, 0.0);
    }

    #[test]
    fn test_compute_avg_points_2player_all_wins_p0() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        for _ in 0..10 {
            outcomes.push_back(GameOutcome(vec![1, 2])); // P0 wins
        }
        let (avg_points, draw_rate) = compute_avg_points(&outcomes, 2);

        // P0 always gets 1st (1.0 pts), P1 always gets 2nd (0.0 pts)
        assert!((avg_points[0] - 1.0).abs() < 0.001);
        assert!((avg_points[1] - 0.0).abs() < 0.001);
        assert_eq!(draw_rate, 0.0);
    }

    #[test]
    fn test_compute_avg_points_2player_all_ties() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        for _ in 0..10 {
            outcomes.push_back(GameOutcome(vec![1, 1])); // Tie
        }
        let (avg_points, draw_rate) = compute_avg_points(&outcomes, 2);

        // Both tied for 1st, avg position = 1.5, points = 2 - 1.5 = 0.5
        assert!((avg_points[0] - 0.5).abs() < 0.001);
        assert!((avg_points[1] - 0.5).abs() < 0.001);
        assert_eq!(draw_rate, 1.0);
    }

    #[test]
    fn test_compute_avg_points_2player_mixed() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        // 4 wins for P0, 4 wins for P1, 2 draws = 10 games
        for _ in 0..4 {
            outcomes.push_back(GameOutcome(vec![1, 2])); // P0 wins
        }
        for _ in 0..4 {
            outcomes.push_back(GameOutcome(vec![2, 1])); // P1 wins
        }
        for _ in 0..2 {
            outcomes.push_back(GameOutcome(vec![1, 1])); // Tie
        }
        let (avg_points, draw_rate) = compute_avg_points(&outcomes, 2);

        // P0: 4 wins (4.0 pts) + 4 losses (0.0 pts) + 2 draws (1.0 pts) = 5.0 / 10 = 0.5
        // P1: 4 losses (0.0 pts) + 4 wins (4.0 pts) + 2 draws (1.0 pts) = 5.0 / 10 = 0.5
        assert!((avg_points[0] - 0.5).abs() < 0.001);
        assert!((avg_points[1] - 0.5).abs() < 0.001);
        assert!((draw_rate - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_compute_avg_points_4player_placements() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        // 4-player game: P0 always wins, P1 2nd, P2 3rd, P3 4th
        outcomes.push_back(GameOutcome(vec![1, 2, 3, 4]));

        let (avg_points, draw_rate) = compute_avg_points(&outcomes, 4);

        // 4-player: max_points = 3.0 (1st place)
        // P0: 1st = 4-1 = 3.0 pts
        // P1: 2nd = 4-2 = 2.0 pts
        // P2: 3rd = 4-3 = 1.0 pts
        // P3: 4th = 4-4 = 0.0 pts
        assert!((avg_points[0] - 3.0).abs() < 0.001);
        assert!((avg_points[1] - 2.0).abs() < 0.001);
        assert!((avg_points[2] - 1.0).abs() < 0.001);
        assert!((avg_points[3] - 0.0).abs() < 0.001);
        assert_eq!(draw_rate, 0.0);
    }

    #[test]
    fn test_compute_avg_points_4player_tied_for_2nd() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        // P0 wins, P1 and P2 tied for 2nd, P3 4th
        outcomes.push_back(GameOutcome(vec![1, 2, 2, 4]));

        let (avg_points, draw_rate) = compute_avg_points(&outcomes, 4);

        // P0: 1st = 4-1 = 3.0 pts
        // P1: tied 2nd, avg position = (2+3)/2 = 2.5, pts = 4-2.5 = 1.5
        // P2: tied 2nd, avg position = 2.5, pts = 1.5
        // P3: 4th = 4-4 = 0.0 pts
        assert!((avg_points[0] - 3.0).abs() < 0.001);
        assert!((avg_points[1] - 1.5).abs() < 0.001);
        assert!((avg_points[2] - 1.5).abs() < 0.001);
        assert!((avg_points[3] - 0.0).abs() < 0.001);
        assert_eq!(draw_rate, 0.0);
    }

    #[test]
    fn test_compute_avg_points_4player_all_tied() {
        let mut outcomes: VecDeque<GameOutcome> = VecDeque::new();
        // All 4 players tied for 1st
        outcomes.push_back(GameOutcome(vec![1, 1, 1, 1]));

        let (avg_points, draw_rate) = compute_avg_points(&outcomes, 4);

        // All tied for 1st, avg position = (1+2+3+4)/4 = 2.5, pts = 4-2.5 = 1.5
        for pts in &avg_points {
            assert!((pts - 1.5).abs() < 0.001);
        }
        assert_eq!(draw_rate, 1.0);
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
        let (obs, all_rewards, dones, _) = vec_env.step(&[0, 0]);

        // Env 0 was terminal: returns zeroed rewards and done=true
        assert_eq!(all_rewards[0], vec![0.0]);
        assert!(dones[0]);

        // Env 1 was active: stepped normally
        assert_eq!(all_rewards[1], vec![1.0]);
        assert!(!dones[1]);
        assert_eq!(obs[1], 1.0); // Env 1 advanced to count=1
    }
}
