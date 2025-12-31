/// Environment trait and VecEnv parallel wrapper

/// Minimal environment interface - just what PPO needs
pub trait Environment: Send + Sync + 'static {
    /// Reset environment and return initial observation
    fn reset(&mut self) -> Vec<f32>;

    /// Take action and return (observation, reward, done)
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool);

    /// Dimension of observation vector
    fn observation_dim(&self) -> usize;

    /// Number of discrete actions
    fn action_count(&self) -> usize;

    /// Environment name for logging
    fn name(&self) -> &'static str;
}

/// Episode statistics for completed episodes
#[derive(Debug, Clone)]
pub struct EpisodeStats {
    pub total_reward: f32,
    pub length: usize,
}

/// Vectorized environment wrapper for parallel execution
///
/// Steps N environments in parallel, automatically resetting
/// on episode termination.
pub struct VecEnv<E: Environment> {
    envs: Vec<E>,
    /// Current observations for each env
    observations: Vec<Vec<f32>>,
    /// Accumulated reward per env (reset on episode end)
    episode_rewards: Vec<f32>,
    /// Steps taken per env (reset on episode end)
    episode_lengths: Vec<usize>,
}

impl<E: Environment> VecEnv<E> {
    /// Create VecEnv from a factory function
    pub fn new<F>(num_envs: usize, factory: F) -> Self
    where
        F: Fn(usize) -> E,
    {
        let mut envs: Vec<E> = (0..num_envs).map(|i| factory(i)).collect();
        let observations: Vec<Vec<f32>> = envs.iter_mut().map(|e| e.reset()).collect();

        Self {
            envs,
            observations,
            episode_rewards: vec![0.0; num_envs],
            episode_lengths: vec![0; num_envs],
        }
    }

    /// Number of parallel environments
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Observation dimension (all envs must match)
    pub fn observation_dim(&self) -> usize {
        self.envs[0].observation_dim()
    }

    /// Action count (all envs must match)
    pub fn action_count(&self) -> usize {
        self.envs[0].action_count()
    }

    /// Get current observations as flat array [num_envs * obs_dim]
    pub fn get_observations(&self) -> Vec<f32> {
        self.observations.iter().flatten().copied().collect()
    }

    /// Step all environments with given actions
    ///
    /// Returns:
    /// - observations: [num_envs, obs_dim] flattened
    /// - rewards: [num_envs]
    /// - dones: [num_envs]
    /// - completed_episodes: stats for any episodes that finished this step
    pub fn step(&mut self, actions: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<bool>, Vec<EpisodeStats>) {
        assert_eq!(actions.len(), self.envs.len());

        let mut rewards = Vec::with_capacity(self.envs.len());
        let mut dones = Vec::with_capacity(self.envs.len());
        let mut completed_episodes = Vec::new();

        for (i, (env, &action)) in self.envs.iter_mut().zip(actions.iter()).enumerate() {
            let (obs, reward, done) = env.step(action);

            // Track episode stats
            self.episode_rewards[i] += reward;
            self.episode_lengths[i] += 1;

            if done {
                // Record completed episode
                completed_episodes.push(EpisodeStats {
                    total_reward: self.episode_rewards[i],
                    length: self.episode_lengths[i],
                });

                // Auto-reset and store new initial observation
                self.observations[i] = env.reset();
                self.episode_rewards[i] = 0.0;
                self.episode_lengths[i] = 0;
            } else {
                self.observations[i] = obs;
            }

            rewards.push(reward);
            dones.push(done);
        }

        let flat_obs = self.get_observations();
        (flat_obs, rewards, dones, completed_episodes)
    }

    /// Reset all environments
    pub fn reset_all(&mut self) {
        for (i, env) in self.envs.iter_mut().enumerate() {
            self.observations[i] = env.reset();
            self.episode_rewards[i] = 0.0;
            self.episode_lengths[i] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test environment: counter that terminates at max_steps
    struct CounterEnv {
        count: usize,
        max_steps: usize,
    }

    impl CounterEnv {
        fn new(max_steps: usize) -> Self {
            Self { count: 0, max_steps }
        }
    }

    impl Environment for CounterEnv {
        fn reset(&mut self) -> Vec<f32> {
            self.count = 0;
            vec![0.0]
        }

        fn step(&mut self, _action: usize) -> (Vec<f32>, f32, bool) {
            self.count += 1;
            let done = self.count >= self.max_steps;
            (vec![self.count as f32], 1.0, done)
        }

        fn observation_dim(&self) -> usize {
            1
        }

        fn action_count(&self) -> usize {
            2
        }

        fn name(&self) -> &'static str {
            "counter"
        }
    }

    #[test]
    fn test_vec_env_creation() {
        let vec_env: VecEnv<CounterEnv> = VecEnv::new(4, |_| CounterEnv::new(3));

        assert_eq!(vec_env.num_envs(), 4);
        assert_eq!(vec_env.observation_dim(), 1);
        assert_eq!(vec_env.action_count(), 2);
    }

    #[test]
    fn test_vec_env_step() {
        let mut vec_env: VecEnv<CounterEnv> = VecEnv::new(2, |_| CounterEnv::new(3));

        let actions = vec![0, 1];
        let (obs, rewards, dones, completed) = vec_env.step(&actions);

        assert_eq!(obs, vec![1.0, 1.0]); // Both envs stepped to count=1
        assert_eq!(rewards, vec![1.0, 1.0]);
        assert_eq!(dones, vec![false, false]);
        assert!(completed.is_empty());
    }

    #[test]
    fn test_vec_env_auto_reset() {
        let mut vec_env: VecEnv<CounterEnv> = VecEnv::new(1, |_| CounterEnv::new(2));

        // Step 1
        vec_env.step(&[0]);
        // Step 2 - should terminate
        let (obs, rewards, dones, completed) = vec_env.step(&[0]);

        assert_eq!(dones, vec![true]);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].total_reward, 2.0);
        assert_eq!(completed[0].length, 2);

        // After auto-reset, observations should be reset state
        assert_eq!(obs, vec![0.0]);
    }

    #[test]
    fn test_episode_stats() {
        let mut vec_env: VecEnv<CounterEnv> = VecEnv::new(1, |_| CounterEnv::new(5));

        // Run 5 steps to complete episode
        for _ in 0..4 {
            let (_, _, dones, _) = vec_env.step(&[0]);
            assert_eq!(dones, vec![false]);
        }

        let (_, _, dones, completed) = vec_env.step(&[0]);
        assert_eq!(dones, vec![true]);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].total_reward, 5.0);
        assert_eq!(completed[0].length, 5);
    }
}
