/// Environment trait and VecEnv parallel wrapper

use rayon::prelude::*;

use crate::profile::{profile_function, profile_scope};

/// Game outcome for evaluation - who won/placed where
#[derive(Debug, Clone, PartialEq)]
pub enum GameOutcome {
    /// Single winner (player index)
    Winner(usize),
    /// All players tied
    Tie,
    /// N-player rankings (1-indexed: 1=first, 2=second, etc.)
    Placements(Vec<usize>),
}

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

    /// Create a new environment with the given seed.
    /// For deterministic games, the seed may be ignored.
    fn new(seed: u64) -> Self;

    /// Reset environment and return initial observation
    fn reset(&mut self) -> Vec<f32>;

    /// Take action and return (observation, rewards[NUM_PLAYERS], done)
    ///
    /// For multi-player games:
    /// - observation: player-agnostic (same encoding for all players)
    /// - rewards: reward for EACH player this step (not just actor)
    /// - done: true if episode ended
    fn step(&mut self, action: usize) -> (Vec<f32>, Vec<f32>, bool);

    /// Current player index (0..NUM_PLAYERS). Default 0 for single-agent.
    fn current_player(&self) -> usize {
        0
    }

    /// Return mask of valid actions (true = valid). None = all valid.
    fn action_mask(&self) -> Option<Vec<bool>> {
        None
    }

    /// Return explicit game outcome when episode is done.
    /// Override this if using reward shaping where total_rewards doesn't reflect outcome.
    /// Default: None (infer from total_rewards via argmax)
    fn game_outcome(&self) -> Option<GameOutcome> {
        None
    }

    /// Render the current state as a string for visualization.
    /// Used in eval watch mode. Default: None (no rendering support).
    fn render(&self) -> Option<String> {
        None
    }
}

/// Episode statistics for completed episodes
#[derive(Debug, Clone)]
pub struct EpisodeStats {
    /// Total reward per player [NUM_PLAYERS]
    pub total_rewards: Vec<f32>,
    pub length: usize,
    /// Which environment index this episode came from
    pub env_index: usize,
}

impl EpisodeStats {
    /// Get total reward for player 0 (backward compatibility for single-agent)
    pub fn total_reward(&self) -> f32 {
        self.total_rewards.first().copied().unwrap_or(0.0)
    }
}

/// Vectorized environment wrapper for parallel execution
///
/// Steps N environments in parallel, automatically resetting
/// on episode termination.
pub struct VecEnv<E: Environment> {
    envs: Vec<E>,
    /// Pre-allocated flat observation buffer [num_envs * obs_dim]
    obs_buffer: Vec<f32>,
    /// Accumulated rewards per env, per player [num_envs][num_players] (reset on episode end)
    episode_rewards: Vec<Vec<f32>>,
    /// Steps taken per env (reset on episode end)
    episode_lengths: Vec<usize>,
    /// Envs in terminal state (won't step or reset) [num_envs]
    terminal: Vec<bool>,
}

impl<E: Environment> VecEnv<E> {
    /// Create VecEnv from a factory function
    pub fn new<F>(num_envs: usize, factory: F) -> Self
    where
        F: Fn(usize) -> E,
    {
        let mut envs: Vec<E> = (0..num_envs).map(|i| factory(i)).collect();

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
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Get current observations as flat array [num_envs * obs_dim]
    pub fn get_observations(&self) -> Vec<f32> {
        self.obs_buffer.clone()
    }

    /// Get current player index for each environment (0..NUM_PLAYERS)
    pub fn get_current_players(&self) -> Vec<usize> {
        self.envs.iter().map(|e| e.current_player()).collect()
    }

    /// Get action masks for all environments, flattened [num_envs * action_count]
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
    /// - observations: [num_envs, obs_dim] flattened
    /// - all_rewards: [num_envs][num_players] - rewards for ALL players per env
    /// - dones: [num_envs]
    /// - completed_episodes: stats for any episodes that finished this step
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
                    |(env_idx, (((((env, &action), ep_rewards), length), obs_chunk), &is_terminal))| {
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
                            let stats = EpisodeStats {
                                total_rewards: ep_rewards.clone(),
                                length: *length,
                                env_index: env_idx,
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

    /// Simple test environment: counter that terminates at MAX_STEPS
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
}
