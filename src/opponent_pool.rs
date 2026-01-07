//! Historical Opponent Pool for OpenAI-Five-style training
//!
//! Implements opponent pool training to prevent strategy collapse and improve generalization.
//!
//! A configurable fraction of training games are played against historical checkpoints,
//! with opponents sampled using rating-weighted probabilities.
//!
//! Key features:
//! - Rating-based sampling: harder opponents sampled more frequently
//! - Per-slot diversity: each opponent player slot gets independently sampled opponent
//! - Lazy loading: only load models when actively playing
//! - Graceful rotation: wait for games to complete before swapping opponents

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use skillratings::weng_lin::{weng_lin_multi_team, WengLinConfig, WengLinRating};
use skillratings::MultiTeamOutcome;

use crate::checkpoint::{load_metadata, load_normalizer, CheckpointManager, CheckpointMetadata};
use crate::config::Config;
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;

/// Default Weng-Lin uncertainty for new opponents
const DEFAULT_UNCERTAINTY: f64 = 25.0 / 3.0;

/// Persisted per-opponent rating data (saved in `pool_ratings.json`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpponentRating {
    /// Checkpoint name (e.g., `step_00010000`)
    pub checkpoint_name: String,
    /// Weng-Lin mu (skill estimate)
    pub rating_mu: f64,
    /// Weng-Lin sigma (uncertainty)
    pub rating_sigma: f64,
    /// Total games played against current network
    pub games_played: usize,
    /// Times current network beat this opponent
    pub wins_vs_current: usize,
    /// Training step when rating was last updated
    pub last_updated_step: usize,
}

impl OpponentRating {
    /// Create a new opponent rating with initial values from checkpoint metadata
    pub fn new(checkpoint_name: String, initial_rating: f64) -> Self {
        Self {
            checkpoint_name,
            rating_mu: initial_rating,
            rating_sigma: DEFAULT_UNCERTAINTY,
            games_played: 0,
            wins_vs_current: 0,
            last_updated_step: 0,
        }
    }
}

/// In-memory loaded opponent with model and normalizer
#[expect(dead_code, reason = "fields reserved for pool evaluation (Phase 6)")]
pub struct LoadedOpponent<B: Backend> {
    /// Path to checkpoint directory
    pub checkpoint_path: PathBuf,
    /// Loaded model
    pub model: ActorCritic<B>,
    /// Observation normalizer (if checkpoint had one)
    pub normalizer: Option<ObsNormalizer>,
    /// Rating data for this opponent
    pub rating: OpponentRating,
    /// Index in the available pool
    pub pool_index: usize,
}

/// Per-environment state for opponent pool training
///
/// Tracks which seat the learner sits in and which opponents are assigned
/// to each seat in the game.
#[derive(Debug, Clone)]
pub struct EnvState {
    /// Which seat the learner sits in (0 to num_players-1)
    pub learner_position: usize,
    /// Maps position (seat) to pool index. None means learner sits there.
    /// `position_to_opponent[seat] = Some(pool_idx)` for opponents
    pub position_to_opponent: Vec<Option<usize>>,
    /// Pool indices of assigned opponents (constant until rotation)
    /// Length = `num_players` - 1
    pub assigned_opponents: Vec<usize>,
    /// New opponents waiting for episode end (graceful rotation)
    pub pending_opponents: Option<Vec<usize>>,
}

impl EnvState {
    /// Create a new env state with random initial position assignment
    pub fn new(num_players: usize, assigned_opponents: Vec<usize>, rng: &mut impl Rng) -> Self {
        let mut state = Self {
            learner_position: 0,
            position_to_opponent: vec![None; num_players],
            assigned_opponents,
            pending_opponents: None,
        };
        state.shuffle_positions(num_players, rng);
        state
    }

    /// Shuffle positions at episode start (same opponents, different seats)
    pub fn shuffle_positions(&mut self, num_players: usize, rng: &mut impl Rng) {
        // Pick random seat for learner
        self.learner_position = rng.gen_range(0..num_players);

        // Get non-learner seats and shuffle
        let mut other_seats: Vec<usize> = (0..num_players)
            .filter(|&p| p != self.learner_position)
            .collect();
        other_seats.shuffle(rng);

        // Map opponents to shuffled seats
        self.position_to_opponent = vec![None; num_players];
        for (i, &seat) in other_seats.iter().enumerate() {
            self.position_to_opponent[seat] = Some(self.assigned_opponents[i]);
        }
    }

    /// Apply pending rotation (call at episode end)
    ///
    /// Returns true if rotation was applied
    pub fn apply_pending_rotation(&mut self) -> bool {
        if let Some(new_opponents) = self.pending_opponents.take() {
            self.assigned_opponents = new_opponents;
            true
        } else {
            false
        }
    }

    /// Check if this env has a pending rotation
    #[cfg(test)]
    pub fn has_pending_rotation(&self) -> bool {
        self.pending_opponents.is_some()
    }
}

/// Main opponent pool manager
pub struct OpponentPool<B: Backend> {
    /// All available opponents (checkpoint path + rating metadata)
    available: Vec<(PathBuf, OpponentRating)>,

    /// Currently loaded opponents by pool index
    /// Key = pool index, Value = loaded opponent
    loaded: HashMap<usize, LoadedOpponent<B>>,

    /// Number of opponent slots per game (`num_players` - 1)
    num_opponent_slots: usize,

    /// Directory containing checkpoints
    checkpoints_dir: PathBuf,

    /// Sampling temperature for softmax
    temperature: f32,

    /// Config for model loading
    config: Config,

    /// Device for model loading
    device: B::Device,

    /// RNG for sampling
    rng: rand::rngs::StdRng,

    /// Learner's rating (persisted across evaluations within a training run)
    /// Initialized from parent checkpoint if forked, otherwise starts at 25.0
    learner_rating: WengLinRating,
}

/// Persisted pool ratings file format (`pool_ratings.json`)
#[derive(Debug, Serialize, Deserialize)]
struct PoolRatingsFile {
    /// Learner's current rating (optional for backwards compatibility)
    #[serde(default)]
    learner: Option<LearnerRatingData>,
    /// Opponent ratings
    opponents: Vec<OpponentRating>,
}

/// Learner rating data for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearnerRatingData {
    rating_mu: f64,
    rating_sigma: f64,
}

impl<B: Backend> OpponentPool<B> {
    /// Create a new opponent pool
    ///
    /// Scans `checkpoints_dir` for available checkpoints and loads ratings.
    ///
    /// `initial_learner_rating`: Optional (mu, sigma) from parent checkpoint if forked/resumed.
    /// If None, learner starts at 25.0 with default uncertainty.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "checkpoints_dir is stored in struct, taking ownership is intentional"
    )]
    pub fn new(
        checkpoints_dir: PathBuf,
        num_players: usize,
        temperature: f32,
        config: Config,
        device: B::Device,
        seed: u64,
        initial_learner_rating: Option<(f64, f64)>,
    ) -> Result<Self> {
        // Initialize learner rating:
        // - If forked/resumed: use parent's rating with reset uncertainty
        // - Otherwise: start fresh at 25.0
        let learner_rating = initial_learner_rating.map_or(
            WengLinRating {
                rating: 25.0,
                uncertainty: DEFAULT_UNCERTAINTY,
            },
            |(mu, _sigma)| WengLinRating {
                rating: mu,
                uncertainty: DEFAULT_UNCERTAINTY, // Reset uncertainty for new training context
            },
        );

        let mut pool = Self {
            available: Vec::new(),
            loaded: HashMap::new(),
            num_opponent_slots: num_players.saturating_sub(1),
            checkpoints_dir: checkpoints_dir.clone(),
            temperature,
            config,
            device,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            learner_rating,
        };

        // Load existing ratings (may override learner_rating if resuming same run)
        pool.load_ratings()?;

        // Scan for checkpoints and merge with loaded ratings
        pool.scan_checkpoints()?;

        Ok(pool)
    }

    /// Get number of available opponents
    pub fn num_available(&self) -> usize {
        self.available.len()
    }

    /// Get number of opponent slots per game
    #[expect(dead_code, reason = "reserved for pool evaluation")]
    pub fn num_opponent_slots(&self) -> usize {
        self.num_opponent_slots
    }

    /// Check if pool has enough opponents for training
    pub fn has_opponents(&self) -> bool {
        !self.available.is_empty()
    }

    /// Get the learner's current rating
    pub fn learner_rating(&self) -> &WengLinRating {
        &self.learner_rating
    }

    /// Update the learner's rating (called after pool evaluation)
    pub fn set_learner_rating(&mut self, rating: WengLinRating) {
        self.learner_rating = rating;
    }

    /// Get ratings file path
    fn ratings_path(&self) -> PathBuf {
        self.checkpoints_dir.join("pool_ratings.json")
    }

    /// Load ratings from disk
    ///
    /// Supports both old format (`Vec<OpponentRating>`) and new format (`PoolRatingsFile`)
    pub fn load_ratings(&mut self) -> Result<()> {
        let path = self.ratings_path();
        if !path.exists() {
            return Ok(());
        }

        let json = fs::read_to_string(&path).context("Failed to read pool_ratings.json")?;

        // Try new format first, fall back to old format for backwards compatibility
        let (learner_data, opponent_ratings) =
            if let Ok(file) = serde_json::from_str::<PoolRatingsFile>(&json) {
                (file.learner, file.opponents)
            } else if let Ok(ratings) = serde_json::from_str::<Vec<OpponentRating>>(&json) {
                // Old format: just opponent ratings array
                (None, ratings)
            } else {
                return Err(anyhow::anyhow!("Failed to parse pool_ratings.json"));
            };

        // Load learner rating if present (resuming same run)
        if let Some(learner) = learner_data {
            self.learner_rating = WengLinRating {
                rating: learner.rating_mu,
                uncertainty: learner.rating_sigma,
            };
        }

        // Convert opponent ratings to available list (paths will be filled in by scan_checkpoints)
        for rating in opponent_ratings {
            let checkpoint_path = self.checkpoints_dir.join(&rating.checkpoint_name);
            self.available.push((checkpoint_path, rating));
        }

        Ok(())
    }

    /// Save ratings to disk (new format with learner rating)
    pub fn save_ratings(&self) -> Result<()> {
        let file = PoolRatingsFile {
            learner: Some(LearnerRatingData {
                rating_mu: self.learner_rating.rating,
                rating_sigma: self.learner_rating.uncertainty,
            }),
            opponents: self.available.iter().map(|(_, r)| r.clone()).collect(),
        };

        let json = serde_json::to_string_pretty(&file)?;
        fs::write(self.ratings_path(), json).context("Failed to write pool_ratings.json")?;

        Ok(())
    }

    /// Scan checkpoints directory and add new checkpoints to pool
    #[expect(clippy::unnecessary_wraps, reason = "Result for future error handling")]
    pub fn scan_checkpoints(&mut self) -> Result<()> {
        // Get existing checkpoint names for deduplication
        let existing: std::collections::HashSet<String> = self
            .available
            .iter()
            .map(|(_, r)| r.checkpoint_name.clone())
            .collect();

        // Scan for checkpoint directories (step_XXXXXXXX format)
        let Ok(entries) = fs::read_dir(&self.checkpoints_dir) else {
            return Ok(()); // Directory doesn't exist yet
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            // Skip non-checkpoint directories
            if !name.starts_with("step_") {
                continue;
            }

            // Skip already known checkpoints
            if existing.contains(&name) {
                continue;
            }

            // Load metadata to get initial rating
            let Ok(metadata) = load_metadata(&path) else {
                continue; // Skip invalid checkpoints
            };

            // Create rating with training_rating as initial mu
            let rating = OpponentRating::new(name, metadata.training_rating);
            self.available.push((path, rating));
        }

        // Sort by checkpoint name (step number) for deterministic ordering
        self.available
            .sort_by(|a, b| a.1.checkpoint_name.cmp(&b.1.checkpoint_name));

        Ok(())
    }

    /// Add a newly saved checkpoint to the pool
    pub fn add_checkpoint(&mut self, checkpoint_path: PathBuf, metadata: &CheckpointMetadata) {
        let name = checkpoint_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Check if already in pool
        if self
            .available
            .iter()
            .any(|(_, r)| r.checkpoint_name == name)
        {
            return;
        }

        let rating = OpponentRating::new(name, metadata.training_rating);
        self.available.push((checkpoint_path, rating));
    }

    /// Sample a single opponent, excluding already-assigned indices
    pub fn sample_opponent(&mut self, exclude: &[usize]) -> Option<usize> {
        if self.available.is_empty() {
            return None;
        }

        // Get eligible opponents
        let eligible: Vec<(usize, f64)> = self
            .available
            .iter()
            .enumerate()
            .filter(|(i, _)| !exclude.contains(i))
            .map(|(i, (_, rating))| (i, rating.rating_mu))
            .collect();

        if eligible.is_empty() {
            // Fall back to any opponent if all are excluded
            return Some(self.rng.gen_range(0..self.available.len()));
        }

        // Compute softmax probabilities
        let max_rating = eligible
            .iter()
            .map(|(_, r)| *r)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_ratings: Vec<f64> = eligible
            .iter()
            .map(|(_, r)| ((r - max_rating) / f64::from(self.temperature)).exp())
            .collect();

        let sum: f64 = exp_ratings.iter().sum();
        if sum == 0.0 {
            // Uniform fallback
            let idx = self.rng.gen_range(0..eligible.len());
            return Some(eligible[idx].0);
        }

        // Sample from distribution
        let sample: f64 = self.rng.r#gen();
        let mut cumsum = 0.0;
        for (i, &exp) in exp_ratings.iter().enumerate() {
            cumsum += exp / sum;
            if sample < cumsum {
                return Some(eligible[i].0);
            }
        }

        Some(
            eligible
                .last()
                .expect("eligible list is non-empty at this point")
                .0,
        )
    }

    /// Sample opponents for all slots (`num_players` - 1 unique opponents)
    pub fn sample_all_slots(&mut self) -> Vec<usize> {
        let mut assigned = Vec::with_capacity(self.num_opponent_slots);

        for _ in 0..self.num_opponent_slots {
            if let Some(idx) = self.sample_opponent(&assigned) {
                assigned.push(idx);
            }
        }

        assigned
    }

    /// Get or load a model by pool index
    #[expect(
        dead_code,
        reason = "API flexibility - use get_model_and_normalizer for combined access"
    )]
    pub fn get_model(&mut self, pool_index: usize) -> Result<&ActorCritic<B>> {
        if !self.loaded.contains_key(&pool_index) {
            self.load_opponent(pool_index)?;
        }
        Ok(&self.loaded.get(&pool_index).expect("just loaded").model)
    }

    /// Get normalizer for a loaded opponent
    #[expect(
        dead_code,
        reason = "API flexibility - use get_model_and_normalizer for combined access"
    )]
    pub fn get_normalizer(&self, pool_index: usize) -> Option<&ObsNormalizer> {
        self.loaded
            .get(&pool_index)
            .and_then(|opp| opp.normalizer.as_ref())
    }

    /// Get both model and normalizer in a single borrow (avoids borrow checker issues)
    pub fn get_model_and_normalizer(
        &mut self,
        pool_index: usize,
    ) -> Result<(&ActorCritic<B>, Option<&ObsNormalizer>)> {
        if !self.loaded.contains_key(&pool_index) {
            self.load_opponent(pool_index)?;
        }
        let loaded = self.loaded.get(&pool_index).expect("just loaded");
        Ok((&loaded.model, loaded.normalizer.as_ref()))
    }

    /// Load an opponent model from checkpoint
    fn load_opponent(&mut self, pool_index: usize) -> Result<()> {
        let (path, rating) = &self.available[pool_index];

        let (model, _metadata) = CheckpointManager::load::<B>(path, &self.config, &self.device)
            .with_context(|| format!("Failed to load opponent from {}", path.display()))?;

        let normalizer = load_normalizer(path)?;

        let loaded = LoadedOpponent {
            checkpoint_path: path.clone(),
            model,
            normalizer,
            rating: rating.clone(),
            pool_index,
        };

        self.loaded.insert(pool_index, loaded);
        Ok(())
    }

    /// Unload opponents that are no longer needed
    pub fn unload_unused(&mut self, in_use: &[usize]) {
        let to_remove: Vec<usize> = self
            .loaded
            .keys()
            .filter(|k| !in_use.contains(k))
            .copied()
            .collect();

        for idx in to_remove {
            self.loaded.remove(&idx);
        }
    }

    /// Update ratings after a game completes
    ///
    /// `placements[i]` = 1 for 1st place, 2 for 2nd, etc.
    /// `learner_position` = which seat the learner was in
    /// `opponents` = pool indices of opponents at each non-learner position
    pub fn update_ratings_after_game(
        &mut self,
        placements: &[usize],
        learner_position: usize,
        env_state: &EnvState,
        current_step: usize,
    ) {
        let num_players = placements.len();

        // Build teams array in position order
        // Use learner's current rating from pool evaluations to prevent opponent deflation
        let learner_rating = WengLinRating {
            rating: self.learner_rating.rating,
            uncertainty: self.learner_rating.uncertainty,
        };

        // Collect ratings for all positions
        let mut team_ratings: Vec<WengLinRating> = Vec::with_capacity(num_players);
        for pos in 0..num_players {
            if pos == learner_position {
                team_ratings.push(learner_rating);
            } else {
                let pool_idx = env_state.position_to_opponent[pos]
                    .expect("non-learner position must have opponent assigned");
                let (_, rating) = &self.available[pool_idx];
                team_ratings.push(WengLinRating {
                    rating: rating.rating_mu,
                    uncertainty: rating.rating_sigma,
                });
            }
        }

        // Build rating groups for weng_lin_multi_team
        let rating_arrs: Vec<[WengLinRating; 1]> = team_ratings.iter().map(|r| [*r]).collect();
        let rating_groups: Vec<(&[WengLinRating], MultiTeamOutcome)> = rating_arrs
            .iter()
            .enumerate()
            .map(|(i, arr)| (arr.as_slice(), MultiTeamOutcome::new(placements[i])))
            .collect();

        let wl_config = WengLinConfig::new();
        let new_ratings = weng_lin_multi_team(&rating_groups, &wl_config);

        // Update opponent ratings (not learner)
        let learner_placement = placements[learner_position];
        #[expect(
            clippy::needless_range_loop,
            reason = "need position index for multiple lookups"
        )]
        for pos in 0..num_players {
            if pos == learner_position {
                continue;
            }

            let pool_idx = env_state.position_to_opponent[pos]
                .expect("non-learner position must have opponent assigned");
            let (_, rating) = &mut self.available[pool_idx];

            // new_ratings[pos] is a Vec with one element (single-player team)
            if let Some(new_rating) = new_ratings.get(pos).and_then(|v| v.first()) {
                rating.rating_mu = new_rating.rating;
                rating.rating_sigma = new_rating.uncertainty;
            }
            rating.games_played += 1;
            rating.last_updated_step = current_step;

            // Track if learner beat this opponent
            let opponent_placement = placements[pos];
            if learner_placement < opponent_placement {
                rating.wins_vs_current += 1;
            }

            // Also update loaded opponent if present
            if let Some(loaded) = self.loaded.get_mut(&pool_idx) {
                loaded.rating = rating.clone();
            }
        }
    }

    /// Get rating info for logging
    #[expect(dead_code, reason = "reserved for pool evaluation metrics")]
    pub fn get_rating_stats(&self) -> (f64, f64, f64) {
        if self.available.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let ratings: Vec<f64> = self.available.iter().map(|(_, r)| r.rating_mu).collect();
        let min = ratings.iter().copied().fold(f64::INFINITY, f64::min);
        let max = ratings.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let avg = ratings.iter().sum::<f64>() / ratings.len() as f64;

        (min, avg, max)
    }

    /// Get loaded opponent indices
    #[expect(dead_code, reason = "reserved for debugging and metrics")]
    pub fn loaded_indices(&self) -> Vec<usize> {
        self.loaded.keys().copied().collect()
    }

    /// Get the highest rated opponent index
    pub fn highest_rated(&self) -> Option<usize> {
        self.available
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1 .1
                    .rating_mu
                    .partial_cmp(&b.1 .1.rating_mu)
                    .expect("ratings should not be NaN")
            })
            .map(|(i, _)| i)
    }

    /// Get rating for a specific opponent
    #[expect(dead_code, reason = "API for future pool inspection features")]
    pub fn get_rating(&self, pool_index: usize) -> Option<&OpponentRating> {
        self.available.get(pool_index).map(|(_, r)| r)
    }

    /// Get mutable reference to RNG (for creating `EnvStates`)
    pub fn rng_mut(&mut self) -> &mut rand::rngs::StdRng {
        &mut self.rng
    }

    /// Sample opponents for evaluation
    ///
    /// Samples `num_opponents` using rating-weighted sampling, always including
    /// the highest-rated opponent (the "best").
    pub fn sample_eval_opponents(&mut self, num_opponents: usize) -> Vec<usize> {
        if self.available.is_empty() {
            return Vec::new();
        }

        let mut selected = Vec::with_capacity(num_opponents);

        // Always include highest-rated opponent
        if let Some(best_idx) = self.highest_rated() {
            selected.push(best_idx);
        }

        // Sample remaining opponents
        while selected.len() < num_opponents && selected.len() < self.available.len() {
            if let Some(idx) = self.sample_opponent(&selected) {
                selected.push(idx);
            } else {
                break;
            }
        }

        selected
    }

    /// Get available opponent paths and ratings for evaluation
    pub fn get_opponents_for_eval(&self, indices: &[usize]) -> Vec<(PathBuf, OpponentRating)> {
        indices
            .iter()
            .filter_map(|&idx| self.available.get(idx).cloned())
            .collect()
    }

    /// Update multiple opponent ratings after evaluation games
    ///
    /// Takes a list of (`pool_index`, `new_mu`, `new_sigma`, `games_played_delta`)
    pub fn update_ratings_from_eval(
        &mut self,
        updates: &[(usize, f64, f64, usize)],
        current_step: usize,
    ) {
        for &(pool_idx, new_mu, new_sigma, games_delta) in updates {
            if let Some((_, rating)) = self.available.get_mut(pool_idx) {
                rating.rating_mu = new_mu;
                rating.rating_sigma = new_sigma;
                rating.games_played += games_delta;
                rating.last_updated_step = current_step;

                // Also update loaded opponent if present
                if let Some(loaded) = self.loaded.get_mut(&pool_idx) {
                    loaded.rating = rating.clone();
                }
            }
        }
    }
}

/// Result of periodic pool evaluation
#[derive(Debug)]
pub struct PoolEvalResult {
    /// Current network's average Swiss points across all eval games
    pub current_avg_points: f64,
    /// Average point margin vs best opponent (`current_points` - `best_points` per game)
    /// Positive means outperforming best, negative means underperforming
    pub vs_best_margin: f64,
    /// Current network's Weng-Lin rating after evaluation
    pub current_rating: f64,
    /// Current network's rating uncertainty
    pub current_uncertainty: f64,
    /// Draw rate across all games
    pub draw_rate: f64,
    /// Number of games played
    pub games_played: usize,
    /// Number of unique opponents faced
    pub num_opponents: usize,
    /// Evaluation time in milliseconds
    pub elapsed_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opponent_rating_new() {
        let rating = OpponentRating::new("step_00010000".to_string(), 30.0);
        assert_eq!(rating.checkpoint_name, "step_00010000");
        assert!((rating.rating_mu - 30.0).abs() < f64::EPSILON);
        assert!((rating.rating_sigma - DEFAULT_UNCERTAINTY).abs() < f64::EPSILON);
        assert_eq!(rating.games_played, 0);
        assert_eq!(rating.wins_vs_current, 0);
    }

    #[test]
    fn test_env_state_shuffle_positions() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut state = EnvState::new(4, vec![0, 1, 2], &mut rng);

        // Should have 3 opponents assigned
        assert_eq!(state.assigned_opponents.len(), 3);

        // Learner should be in one position
        assert!(state.learner_position < 4);

        // Position map should have exactly one None (learner) and 3 Some
        let learner_count = state
            .position_to_opponent
            .iter()
            .filter(|p| p.is_none())
            .count();
        assert_eq!(learner_count, 1);

        // Shuffle and verify positions change
        let old_position = state.learner_position;
        let mut changed = false;
        for _ in 0..10 {
            state.shuffle_positions(4, &mut rng);
            if state.learner_position != old_position {
                changed = true;
                break;
            }
        }
        // With 4 positions and 10 shuffles, very likely to change
        assert!(changed, "Position should change after shuffles");
    }

    #[test]
    fn test_env_state_pending_rotation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut state = EnvState::new(4, vec![0, 1, 2], &mut rng);

        // No pending rotation initially
        assert!(!state.has_pending_rotation());
        assert!(!state.apply_pending_rotation());

        // Set pending rotation
        state.pending_opponents = Some(vec![3, 4, 5]);
        assert!(state.has_pending_rotation());

        // Apply it
        assert!(state.apply_pending_rotation());
        assert_eq!(state.assigned_opponents, vec![3, 4, 5]);
        assert!(!state.has_pending_rotation());
    }

    #[test]
    fn test_rating_serialization() {
        let rating = OpponentRating {
            checkpoint_name: "step_00010000".to_string(),
            rating_mu: 35.5,
            rating_sigma: 7.0,
            games_played: 100,
            wins_vs_current: 45,
            last_updated_step: 50000,
        };

        let json = serde_json::to_string(&rating).unwrap();
        let loaded: OpponentRating = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.checkpoint_name, rating.checkpoint_name);
        assert!((loaded.rating_mu - rating.rating_mu).abs() < f64::EPSILON);
        assert_eq!(loaded.games_played, rating.games_played);
    }

    #[test]
    fn test_pool_ratings_file_serialization() {
        // Test new format with learner rating
        let file = PoolRatingsFile {
            learner: Some(LearnerRatingData {
                rating_mu: 30.5,
                rating_sigma: 6.0,
            }),
            opponents: vec![
                OpponentRating::new("step_00010000".to_string(), 25.0),
                OpponentRating::new("step_00020000".to_string(), 28.0),
            ],
        };

        let json = serde_json::to_string_pretty(&file).expect("serialize");
        let loaded: PoolRatingsFile = serde_json::from_str(&json).expect("deserialize");

        assert!(loaded.learner.is_some());
        let learner = loaded.learner.unwrap();
        assert!((learner.rating_mu - 30.5).abs() < f64::EPSILON);
        assert!((learner.rating_sigma - 6.0).abs() < f64::EPSILON);
        assert_eq!(loaded.opponents.len(), 2);
    }

    #[test]
    fn test_pool_ratings_file_backwards_compatible() {
        // Test loading old format (just array of opponent ratings)
        let old_format = r#"[
            {"checkpoint_name": "step_00010000", "rating_mu": 25.0, "rating_sigma": 8.33, "games_played": 10, "wins_vs_current": 5, "last_updated_step": 1000}
        ]"#;

        // Old format should fail to parse as PoolRatingsFile
        let result = serde_json::from_str::<PoolRatingsFile>(old_format);
        assert!(
            result.is_err(),
            "Old array format should not parse as PoolRatingsFile"
        );

        // But should parse as Vec<OpponentRating>
        let ratings: Vec<OpponentRating> =
            serde_json::from_str(old_format).expect("should parse as vec");
        assert_eq!(ratings.len(), 1);
        assert_eq!(ratings[0].checkpoint_name, "step_00010000");
    }

    #[test]
    fn test_pool_ratings_file_missing_learner() {
        // Test new format where learner field is missing (should default to None)
        let json_no_learner = r#"{
            "opponents": [
                {"checkpoint_name": "step_00010000", "rating_mu": 25.0, "rating_sigma": 8.33, "games_played": 0, "wins_vs_current": 0, "last_updated_step": 0}
            ]
        }"#;

        let loaded: PoolRatingsFile =
            serde_json::from_str(json_no_learner).expect("should parse without learner");
        assert!(loaded.learner.is_none());
        assert_eq!(loaded.opponents.len(), 1);
    }

    #[test]
    fn test_learner_rating_initialization_fresh() {
        // Fresh run (no initial rating) should start at 25.0
        let rating = None::<(f64, f64)>.map_or(
            WengLinRating {
                rating: 25.0,
                uncertainty: DEFAULT_UNCERTAINTY,
            },
            |(mu, _sigma)| WengLinRating {
                rating: mu,
                uncertainty: DEFAULT_UNCERTAINTY,
            },
        );

        assert!((rating.rating - 25.0).abs() < f64::EPSILON);
        assert!((rating.uncertainty - DEFAULT_UNCERTAINTY).abs() < f64::EPSILON);
    }

    #[test]
    fn test_learner_rating_initialization_forked() {
        // Forked run should inherit parent's mu but reset uncertainty
        let parent_rating = (35.0, 5.0); // Parent had rating 35, uncertainty 5
        let rating = Some(parent_rating).map_or(
            WengLinRating {
                rating: 25.0,
                uncertainty: DEFAULT_UNCERTAINTY,
            },
            |(mu, _sigma)| WengLinRating {
                rating: mu,
                uncertainty: DEFAULT_UNCERTAINTY, // Reset uncertainty
            },
        );

        assert!(
            (rating.rating - 35.0).abs() < f64::EPSILON,
            "Should inherit parent's mu"
        );
        assert!(
            (rating.uncertainty - DEFAULT_UNCERTAINTY).abs() < f64::EPSILON,
            "Should reset uncertainty to default"
        );
    }

    #[test]
    fn test_weng_lin_rating_update() {
        use skillratings::weng_lin::{weng_lin_multi_team, WengLinConfig};
        use skillratings::MultiTeamOutcome;

        // Test that rating updates correctly after a win
        let learner = WengLinRating {
            rating: 25.0,
            uncertainty: DEFAULT_UNCERTAINTY,
        };
        let opponent = WengLinRating {
            rating: 25.0,
            uncertainty: DEFAULT_UNCERTAINTY,
        };

        // Build rating groups: (rating slice, outcome)
        // Placement 1 = 1st place (winner), 2 = 2nd place (loser)
        let learner_arr = [learner];
        let opponent_arr = [opponent];
        let rating_groups: Vec<(&[WengLinRating], MultiTeamOutcome)> = vec![
            (&learner_arr[..], MultiTeamOutcome::new(1)), // learner wins
            (&opponent_arr[..], MultiTeamOutcome::new(2)), // opponent loses
        ];

        let new_ratings = weng_lin_multi_team(&rating_groups, &WengLinConfig::default());

        let new_learner = new_ratings[0][0];

        assert!(new_learner.rating > 25.0, "Winner's rating should increase");
        assert!(
            new_learner.uncertainty < DEFAULT_UNCERTAINTY,
            "Uncertainty should decrease after game"
        );
    }
}
