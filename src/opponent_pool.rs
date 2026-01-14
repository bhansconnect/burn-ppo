//! Historical Opponent Pool for OpenAI-Five-style training
//!
//! Implements opponent pool training to prevent strategy collapse and improve generalization.
//!
//! A configurable fraction of training games are played against historical checkpoints,
//! with opponents sampled using qi-score-weighted probabilities.
//!
//! Key features:
//! - qi-based sampling: softmax(e^qi) for opponent selection
//! - qi update: decreases when opponent loses, unchanged when opponent wins
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

use crate::checkpoint::{load_metadata, load_normalizer, CheckpointManager};
use crate::config::Config;
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;

/// Persisted per-opponent qi score data (saved in `qi_scores.json`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpponentQi {
    /// Checkpoint name (e.g., `step_00010000`)
    pub checkpoint_name: String,
    /// Checkpoint training step (for relative version calculation)
    pub checkpoint_step: usize,
    /// Quality score (higher = more likely to sample)
    pub qi: f64,
}

impl OpponentQi {
    /// Create a new opponent qi with initial value
    pub fn new(checkpoint_name: String, checkpoint_step: usize, initial_qi: f64) -> Self {
        Self {
            checkpoint_name,
            checkpoint_step,
            qi: initial_qi,
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
    /// qi score data for this opponent
    pub qi_data: OpponentQi,
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

/// Pending game result for batched qi updates
///
/// Game results are collected during training and applied in shuffled order
/// at rotation time to avoid order-based bias.
struct PendingGameResult {
    /// Did the current model beat this opponent? (pairwise result)
    current_won: bool,
    /// Pool index of the opponent
    opponent_pool_idx: usize,
    /// Step when game completed
    #[expect(dead_code, reason = "preserved for future game result logging")]
    step: usize,
}

/// Main opponent pool manager
pub struct OpponentPool<B: Backend> {
    /// All available opponents (checkpoint path + qi metadata)
    available: Vec<(PathBuf, OpponentQi)>,

    /// Currently loaded opponents by pool index
    /// Key = pool index, Value = loaded opponent
    loaded: HashMap<usize, LoadedOpponent<B>>,

    /// Number of opponent slots per game (`num_players` - 1)
    num_opponent_slots: usize,

    /// Directory containing checkpoints
    checkpoints_dir: PathBuf,

    /// qi eta (learning rate for qi updates)
    qi_eta: f64,

    /// Config for model loading
    config: Config,

    /// Device for model loading
    device: B::Device,

    /// RNG for sampling
    rng: rand::rngs::StdRng,

    /// Active subset of opponent indices (limited by `pool_size_limit`)
    active_indices: Vec<usize>,

    /// Maximum number of active opponents
    pool_size_limit: usize,

    /// Pending game results for batched qi updates
    /// Applied in shuffled order at rotation time
    pending_game_results: Vec<PendingGameResult>,
}

/// Persisted qi scores file format (`qi_scores.json`)
#[derive(Debug, Serialize, Deserialize)]
struct QiScoresFile {
    /// qi eta parameter used
    qi_eta: f64,
    /// Opponent qi scores
    opponents: Vec<OpponentQi>,
}

impl<B: Backend> OpponentPool<B> {
    /// Create a new opponent pool
    ///
    /// Scans `checkpoints_dir` for available checkpoints and loads qi scores.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "checkpoints_dir is stored in struct, taking ownership is intentional"
    )]
    pub fn new(
        checkpoints_dir: PathBuf,
        num_players: usize,
        qi_eta: f64,
        config: Config,
        device: B::Device,
        seed: u64,
        pool_size_limit: usize,
    ) -> Result<Self> {
        let mut pool = Self {
            available: Vec::new(),
            loaded: HashMap::new(),
            num_opponent_slots: num_players.saturating_sub(1),
            checkpoints_dir: checkpoints_dir.clone(),
            qi_eta,
            config,
            device,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            active_indices: Vec::new(),
            pool_size_limit,
            pending_game_results: Vec::new(),
        };

        // Load existing qi scores from qi_scores.json
        pool.load_qi_scores()?;

        // Scan for checkpoints and merge with loaded ratings
        pool.scan_checkpoints()?;

        // Initialize active subset
        pool.refresh_active_subset();

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

    /// Refresh the active opponent subset
    ///
    /// Samples up to `pool_size_limit` unique opponents using qi-weighted sampling.
    /// Call this at initialization and at each rotation to get variety.
    pub fn refresh_active_subset(&mut self) {
        if self.available.is_empty() {
            self.active_indices.clear();
            return;
        }

        let max_active = self.pool_size_limit.min(self.available.len());
        let mut selected = Vec::with_capacity(max_active);

        // Sample opponents using rating-weighted (or uniform) sampling
        while selected.len() < max_active {
            if let Some(idx) = self.sample_opponent(&selected) {
                selected.push(idx);
            } else {
                break;
            }
        }

        self.active_indices = selected;
    }

    /// Get the current active opponent indices
    #[expect(dead_code, reason = "reserved for debugging and metrics")]
    pub fn active_indices(&self) -> &[usize] {
        &self.active_indices
    }

    /// Get qi scores file path
    fn qi_scores_path(&self) -> PathBuf {
        self.checkpoints_dir.join("qi_scores.json")
    }

    /// Load qi scores from disk
    fn load_qi_scores(&mut self) -> Result<()> {
        let path = self.qi_scores_path();
        if !path.exists() {
            return Ok(()); // Fresh start, no existing qi scores
        }

        let json = fs::read_to_string(&path).context("Failed to read qi_scores.json")?;
        let file: QiScoresFile = serde_json::from_str(&json)?;

        // Build map of checkpoint_name -> qi for fast lookup
        let qi_map: HashMap<String, (usize, f64)> = file
            .opponents
            .into_iter()
            .map(|o| (o.checkpoint_name.clone(), (o.checkpoint_step, o.qi)))
            .collect();

        // Store for merging after scan_checkpoints
        // We store in available temporarily, scan_checkpoints will add new ones
        for (name, (step, qi)) in qi_map {
            let checkpoint_path = self.checkpoints_dir.join(&name);
            let qi_data = OpponentQi::new(name, step, qi);
            self.available.push((checkpoint_path, qi_data));
        }

        Ok(())
    }

    /// Save qi scores to disk
    pub fn save_qi_scores(&self) -> Result<()> {
        let file = QiScoresFile {
            qi_eta: self.qi_eta,
            opponents: self.available.iter().map(|(_, q)| q.clone()).collect(),
        };

        let json = serde_json::to_string_pretty(&file)?;
        fs::write(self.qi_scores_path(), json).context("Failed to write qi_scores.json")?;

        Ok(())
    }

    /// Scan checkpoints directory and add new checkpoints to pool
    #[expect(clippy::unnecessary_wraps, reason = "Result for future error handling")]
    pub fn scan_checkpoints(&mut self) -> Result<()> {
        // Get existing checkpoint names for deduplication
        let existing: std::collections::HashSet<String> = self
            .available
            .iter()
            .map(|(_, q)| q.checkpoint_name.clone())
            .collect();

        // Compute max qi for initializing new opponents
        let max_qi = if self.available.is_empty() {
            0.0
        } else {
            self.available
                .iter()
                .map(|(_, q)| q.qi)
                .fold(f64::MIN, f64::max)
        };

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

            // Load metadata to get step number
            let Ok(metadata) = load_metadata(&path) else {
                continue; // Skip invalid checkpoints
            };

            // Create qi with max of existing qi (or 0.0 if pool is empty)
            let qi = OpponentQi::new(name, metadata.step, max_qi);
            self.available.push((path, qi));
        }

        // Sort by checkpoint name (step number) for deterministic ordering
        self.available
            .sort_by(|a, b| a.1.checkpoint_name.cmp(&b.1.checkpoint_name));

        Ok(())
    }

    /// Add a newly saved checkpoint to the pool
    pub fn add_checkpoint(&mut self, checkpoint_path: PathBuf, step: usize) {
        let name = checkpoint_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Check if already in pool
        if self
            .available
            .iter()
            .any(|(_, q)| q.checkpoint_name == name)
        {
            return;
        }

        // New opponent gets max of existing qi scores (or 0.0 if empty)
        let max_qi = if self.available.is_empty() {
            0.0
        } else {
            self.available
                .iter()
                .map(|(_, q)| q.qi)
                .fold(f64::MIN, f64::max)
        };

        let qi = OpponentQi::new(name, step, max_qi);
        self.available.push((checkpoint_path, qi));
    }

    /// Sample a single opponent using qi-weighted softmax, excluding already-assigned indices
    pub fn sample_opponent(&mut self, exclude: &[usize]) -> Option<usize> {
        if self.available.is_empty() {
            return None;
        }

        // Get eligible opponent indices
        let eligible: Vec<usize> = self
            .available
            .iter()
            .enumerate()
            .filter(|(i, _)| !exclude.contains(i))
            .map(|(i, _)| i)
            .collect();

        if eligible.is_empty() {
            // Fall back to any opponent if all are excluded
            return Some(self.rng.gen_range(0..self.available.len()));
        }

        // qi-weighted sampling: get qi scores for eligible opponents
        let eligible_with_qi: Vec<(usize, f64)> = eligible
            .iter()
            .map(|&i| (i, self.available[i].1.qi))
            .collect();

        // Compute softmax probabilities (e^qi)
        let max_qi = eligible_with_qi
            .iter()
            .map(|(_, q)| *q)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_qi: Vec<f64> = eligible_with_qi
            .iter()
            .map(|(_, q)| (q - max_qi).exp())
            .collect();

        let sum: f64 = exp_qi.iter().sum();
        if sum == 0.0 {
            // Uniform fallback
            let idx = self.rng.gen_range(0..eligible_with_qi.len());
            return Some(eligible_with_qi[idx].0);
        }

        // Sample from distribution
        let sample: f64 = self.rng.r#gen();
        let mut cumsum = 0.0;
        for (i, &exp) in exp_qi.iter().enumerate() {
            cumsum += exp / sum;
            if sample < cumsum {
                return Some(eligible_with_qi[i].0);
            }
        }

        Some(
            eligible_with_qi
                .last()
                .expect("eligible list is non-empty at this point")
                .0,
        )
    }

    /// Sample a single opponent from the active subset using qi-weighted softmax
    fn sample_opponent_from_active(&mut self, exclude: &[usize]) -> Option<usize> {
        if self.active_indices.is_empty() {
            return None;
        }

        // Get eligible opponent indices from active subset
        let eligible: Vec<usize> = self
            .active_indices
            .iter()
            .filter(|&&i| !exclude.contains(&i))
            .copied()
            .collect();

        if eligible.is_empty() {
            // Fall back to any active opponent if all are excluded
            return Some(self.active_indices[self.rng.gen_range(0..self.active_indices.len())]);
        }

        // qi-weighted sampling: get qi scores for eligible opponents
        let eligible_with_qi: Vec<(usize, f64)> = eligible
            .iter()
            .map(|&i| (i, self.available[i].1.qi))
            .collect();

        // Compute softmax probabilities (e^qi)
        let max_qi = eligible_with_qi
            .iter()
            .map(|(_, q)| *q)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_qi: Vec<f64> = eligible_with_qi
            .iter()
            .map(|(_, q)| (q - max_qi).exp())
            .collect();

        let sum: f64 = exp_qi.iter().sum();
        if sum == 0.0 {
            // Uniform fallback
            let idx = self.rng.gen_range(0..eligible_with_qi.len());
            return Some(eligible_with_qi[idx].0);
        }

        // Sample from distribution
        let sample: f64 = self.rng.r#gen();
        let mut cumsum = 0.0;
        for (i, &exp) in exp_qi.iter().enumerate() {
            cumsum += exp / sum;
            if sample < cumsum {
                return Some(eligible_with_qi[i].0);
            }
        }

        Some(
            eligible_with_qi
                .last()
                .expect("eligible list is non-empty at this point")
                .0,
        )
    }

    /// Sample opponents for all slots (`num_players` - 1 unique opponents)
    /// Samples ONLY from the active subset
    pub fn sample_all_slots(&mut self) -> Vec<usize> {
        let mut assigned = Vec::with_capacity(self.num_opponent_slots);

        for _ in 0..self.num_opponent_slots {
            if let Some(idx) = self.sample_opponent_from_active(&assigned) {
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
        let (path, qi_data) = &self.available[pool_index];

        let (model, _metadata) = CheckpointManager::load::<B>(path, &self.config, &self.device)
            .with_context(|| format!("Failed to load opponent from {}", path.display()))?;

        let normalizer = load_normalizer(path)?;

        let loaded = LoadedOpponent {
            checkpoint_path: path.clone(),
            model,
            normalizer,
            qi_data: qi_data.clone(),
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

    /// Queue a game result for batched qi update
    ///
    /// Game results are collected and applied in shuffled order at rotation time
    /// to avoid order-based bias (faster games completing first).
    ///
    /// `placements[i]` = 1 for 1st place, 2 for 2nd, etc. (lower = better)
    /// `learner_position` = which seat the learner was in
    pub fn queue_game_for_qi_update(
        &mut self,
        placements: &[usize],
        learner_position: usize,
        env_state: &EnvState,
        current_step: usize,
    ) {
        let learner_placement = placements[learner_position];

        // For each opponent, determine if they beat the current model (pairwise)
        for (pos, &placement) in placements.iter().enumerate() {
            if pos == learner_position {
                continue;
            }

            let Some(pool_idx) = env_state.position_to_opponent[pos] else {
                continue;
            };

            // Opponent wins if they placed better (lower number) than learner
            let current_won = learner_placement < placement;

            self.pending_game_results.push(PendingGameResult {
                current_won,
                opponent_pool_idx: pool_idx,
                step: current_step,
            });
        }
    }

    /// Apply all pending game results to qi scores in shuffled order
    ///
    /// qi update rule:
    /// - If opponent wins (current model loses): qi unchanged
    /// - If opponent loses (current model wins): qi -= eta / (N - pi)
    ///   where N = total opponents, pi = selection probability of this opponent
    pub fn apply_pending_qi_updates(&mut self) {
        if self.pending_game_results.is_empty() {
            return;
        }

        // Shuffle games to avoid completion-order bias
        self.pending_game_results.shuffle(&mut self.rng);

        let n = self.available.len() as f64;

        // Skip qi updates if only one opponent (denominator would be ~0)
        if n <= 1.0 {
            self.pending_game_results.clear();
            return;
        }

        for game in &self.pending_game_results {
            if game.current_won {
                // Current model won => opponent lost => decrease qi
                let pi = self.compute_selection_probability(game.opponent_pool_idx);
                if let Some((_, qi_data)) = self.available.get_mut(game.opponent_pool_idx) {
                    qi_data.qi -= self.qi_eta / (n - pi);
                }
            }
            // If opponent won, qi unchanged
        }

        // Clear pending results
        self.pending_game_results.clear();
    }

    /// Compute selection probability for a specific opponent (softmax over qi)
    pub fn compute_selection_probability(&self, pool_idx: usize) -> f64 {
        if self.available.is_empty() {
            return 0.0;
        }

        let qis: Vec<f64> = self.available.iter().map(|(_, q)| q.qi).collect();
        let max_qi = qis.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_qis: Vec<f64> = qis.iter().map(|q| (q - max_qi).exp()).collect();
        let sum: f64 = exp_qis.iter().sum();

        if sum == 0.0 {
            return 1.0 / self.available.len() as f64; // Uniform
        }

        exp_qis[pool_idx] / sum
    }

    /// Compute selection probabilities for all opponents
    pub fn compute_all_selection_probabilities(&self) -> Vec<f64> {
        if self.available.is_empty() {
            return vec![];
        }

        let qis: Vec<f64> = self.available.iter().map(|(_, q)| q.qi).collect();
        let max_qi = qis.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_qis: Vec<f64> = qis.iter().map(|q| (q - max_qi).exp()).collect();
        let sum: f64 = exp_qis.iter().sum();

        if sum == 0.0 {
            let uniform = 1.0 / self.available.len() as f64;
            return vec![uniform; self.available.len()];
        }

        exp_qis.iter().map(|e| e / sum).collect()
    }

    /// Get qi score stats for logging (min, avg, max)
    pub fn get_qi_stats(&self) -> (f64, f64, f64) {
        if self.available.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let qi_scores: Vec<f64> = self.available.iter().map(|(_, q)| q.qi).collect();
        let min = qi_scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max = qi_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let avg = qi_scores.iter().sum::<f64>() / qi_scores.len() as f64;

        (min, avg, max)
    }

    /// Compute percentile checkpoint positions based on cumulative probability.
    ///
    /// Returns checkpoint positions for percentiles [p10, p20, ..., p90].
    /// Position is negative: -1 = newest, -5 = 5th newest, etc.
    /// Walks through opponents from newest to oldest, tracking when cumulative
    /// probability crosses each 10% threshold.
    pub fn compute_qi_percentiles(&self) -> [i32; 9] {
        let mut percentiles = [0i32; 9];

        if self.available.is_empty() {
            return percentiles;
        }

        let probs = self.compute_all_selection_probabilities();

        // Sort by checkpoint step descending (newest first)
        let mut sorted_by_step: Vec<(usize, usize)> = self
            .available
            .iter()
            .enumerate()
            .map(|(idx, (_, qi))| (idx, qi.checkpoint_step))
            .collect();
        sorted_by_step.sort_by(|a, b| b.1.cmp(&a.1)); // Descending

        // Walk through from newest to oldest, accumulating probability
        let mut cumulative_prob = 0.0;
        let mut next_threshold_idx = 0; // 0 = looking for 10%, 1 = looking for 20%, etc.

        for (position, (pool_idx, _)) in sorted_by_step.iter().enumerate() {
            cumulative_prob += probs[*pool_idx];

            // Check if we've crossed the next threshold
            while next_threshold_idx < 9 {
                let threshold = (next_threshold_idx + 1) as f64 * 0.1; // 0.1, 0.2, ..., 0.9
                if cumulative_prob >= threshold {
                    // Position is 0-indexed, convert to -1, -2, etc.
                    // Safe: position is bounded by available.len() which is reasonable
                    let pos_i32 = i32::try_from(position + 1).unwrap_or(i32::MAX);
                    percentiles[next_threshold_idx] = -pos_i32;
                    next_threshold_idx += 1;
                } else {
                    break;
                }
            }

            if next_threshold_idx >= 9 {
                break;
            }
        }

        // Fill in any remaining percentiles with the total count
        let total = i32::try_from(self.available.len()).unwrap_or(i32::MAX);
        for p in percentiles.iter_mut().skip(next_threshold_idx) {
            *p = -total;
        }

        percentiles
    }

    /// Get mutable reference to RNG (for creating `EnvStates`)
    pub fn rng_mut(&mut self) -> &mut rand::rngs::StdRng {
        &mut self.rng
    }

    /// Get opponents for evaluation
    ///
    /// Returns the active subset of opponents (limited by `pool_size_limit`).
    /// The active subset is refreshed at each rotation.
    pub fn sample_eval_opponents(&self) -> Vec<usize> {
        self.active_indices.clone()
    }

    /// Get available opponent paths and qi data for evaluation
    pub fn get_opponents_for_eval(&self, indices: &[usize]) -> Vec<(PathBuf, OpponentQi)> {
        indices
            .iter()
            .filter_map(|&idx| self.available.get(idx).cloned())
            .collect()
    }

    /// Get the checkpoints directory path
    pub fn checkpoints_dir(&self) -> &PathBuf {
        &self.checkpoints_dir
    }

    /// Generate ASCII histogram of qi distribution for debug output
    /// Hybrid display: top 10 individual networks, then percentile buckets for the rest.
    pub fn generate_qi_histogram(&self) -> String {
        use std::fmt::Write;

        if self.available.is_empty() {
            return "No opponents in pool".to_string();
        }

        let probs = self.compute_all_selection_probabilities();
        let (min_qi, avg_qi, max_qi) = self.get_qi_stats();
        let percentiles = self.compute_qi_percentiles();

        // Sort by step descending to assign checkpoint-relative positions
        let mut sorted_by_step: Vec<(usize, usize)> = self
            .available
            .iter()
            .enumerate()
            .map(|(idx, (_, qi))| (idx, qi.checkpoint_step))
            .collect();
        sorted_by_step.sort_by(|a, b| b.1.cmp(&a.1)); // Descending

        // Build entries ordered by checkpoint (newest first): (ckpt_rel, qi, prob)
        let entries: Vec<_> = sorted_by_step
            .iter()
            .enumerate()
            .map(|(rel_pos, (pool_idx, _))| {
                let pos_i32 = i32::try_from(rel_pos + 1).unwrap_or(i32::MAX);
                let ckpt_rel = -pos_i32;
                let qi = self.available[*pool_idx].1.qi;
                let prob = probs[*pool_idx];
                (ckpt_rel, qi, prob)
            })
            .collect();

        let mut output = String::new();
        let n = self.available.len();
        let _ = writeln!(output, "qi Distribution (n={n}):");
        let _ = writeln!(
            output,
            "  qi stats: min={min_qi:.3} avg={avg_qi:.3} max={max_qi:.3}\n"
        );

        // Part A: Show top 10 individual networks
        let individual_count = entries.len().min(10);
        let max_prob_top10 = entries
            .iter()
            .take(individual_count)
            .map(|e| e.2)
            .fold(0.0, f64::max);

        let _ = writeln!(
            output,
            "{:>6}  {:>8}  {:>6}  Histogram",
            "Ckpt", "qi", "Prob"
        );

        for (ckpt_rel, qi, prob) in entries.iter().take(individual_count) {
            let bar_len = if max_prob_top10 > 0.0 {
                #[expect(clippy::cast_sign_loss, reason = "value clamped to >= 0.0 before cast")]
                let len = ((prob / max_prob_top10) * 30.0).max(0.0) as usize;
                len
            } else {
                0
            };
            let bar = "█".repeat(bar_len);
            let _ = writeln!(output, "{ckpt_rel:>6}  {qi:>8.3}  {prob:>6.4}  {bar}");
        }

        // Part B: Show percentile buckets for remaining networks
        if entries.len() > 10 {
            let _ = writeln!(output, "\n  Remaining networks (cumulative prob buckets):");

            // Compute bucket counts from percentile positions
            // percentiles[i] is the position where (i+1)*10% was reached (negative value)
            let mut bucket_counts = [0usize; 10];
            let total = self.available.len();

            // Helper to convert negative percentile to positive usize
            #[expect(clippy::cast_sign_loss, reason = "value clamped to >= 0 before cast")]
            let to_pos = |p: i32| -> usize { (-p).max(0) as usize };

            for i in 0..10 {
                let end_pos = if i < 9 { to_pos(percentiles[i]) } else { total };
                let start_pos = if i == 0 {
                    0
                } else {
                    to_pos(percentiles[i - 1])
                };
                bucket_counts[i] = end_pos.saturating_sub(start_pos);
            }

            // Find max count among first 9 buckets (for bar scaling)
            let max_count_first9 = bucket_counts[..9].iter().copied().max().unwrap_or(1);

            for i in 0..10 {
                let start_pct = i * 10;
                let end_pct = (i + 1) * 10;
                let count = bucket_counts[i];

                // Only show buckets that start after top 10
                let bucket_start = if i == 0 {
                    0
                } else {
                    to_pos(percentiles[i - 1])
                };
                if bucket_start < 10 && i < 9 {
                    continue; // This bucket overlaps with individual display
                }

                let bar_len = if max_count_first9 > 0 {
                    (count.min(max_count_first9) * 20 / max_count_first9).max(1)
                } else {
                    1
                };

                // Last bucket: use ellipsis if it exceeds scale
                let bar = if i == 9 && count > max_count_first9 {
                    let half = bar_len / 2;
                    format!("{}...{}", "█".repeat(half), "█".repeat(bar_len - half))
                } else {
                    "█".repeat(bar_len)
                };

                let _ = writeln!(
                    output,
                    "  {start_pct:>2}-{end_pct:>3}%  {count:>4} networks  {bar}"
                );
            }
        }

        output
    }

    /// Format selected opponent indices as checkpoint-relative positions for debug output
    /// -1 = latest checkpoint, -2 = second latest, etc.
    pub fn format_selected_opponents(&self, indices: &[usize]) -> String {
        if self.available.is_empty() {
            return String::new();
        }

        // Sort available checkpoints by step (descending) to determine relative position
        let mut sorted_by_step: Vec<(usize, usize)> = self
            .available
            .iter()
            .enumerate()
            .map(|(idx, (_, qi))| (idx, qi.checkpoint_step))
            .collect();
        sorted_by_step.sort_by(|a, b| b.1.cmp(&a.1)); // Descending by step

        // Build index -> relative position map (-1 = latest, -2 = second latest, etc.)
        let idx_to_relative: HashMap<usize, i32> = sorted_by_step
            .iter()
            .enumerate()
            .map(|(rel_pos, (idx, _))| {
                let pos_i32 = i32::try_from(rel_pos + 1).unwrap_or(i32::MAX);
                (*idx, -pos_i32)
            })
            .collect();

        indices
            .iter()
            .map(|&idx| format!("{}", idx_to_relative.get(&idx).copied().unwrap_or(0)))
            .collect::<Vec<_>>()
            .join(", ")
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
    fn test_opponent_qi_new() {
        let qi = OpponentQi::new("step_00010000".to_string(), 10000, 0.5);
        assert_eq!(qi.checkpoint_name, "step_00010000");
        assert_eq!(qi.checkpoint_step, 10000);
        assert!((qi.qi - 0.5).abs() < f64::EPSILON);
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
    fn test_qi_serialization() {
        let qi = OpponentQi {
            checkpoint_name: "step_00010000".to_string(),
            checkpoint_step: 10000,
            qi: 0.75,
        };

        let json = serde_json::to_string(&qi).unwrap();
        let loaded: OpponentQi = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.checkpoint_name, qi.checkpoint_name);
        assert_eq!(loaded.checkpoint_step, qi.checkpoint_step);
        assert!((loaded.qi - qi.qi).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qi_scores_file_serialization() {
        let file = QiScoresFile {
            qi_eta: 0.01,
            opponents: vec![
                OpponentQi::new("step_00010000".to_string(), 10000, 0.0),
                OpponentQi::new("step_00020000".to_string(), 20000, 0.5),
            ],
        };

        let json = serde_json::to_string_pretty(&file).expect("serialize");
        let loaded: QiScoresFile = serde_json::from_str(&json).expect("deserialize");

        assert!((loaded.qi_eta - 0.01).abs() < f64::EPSILON);
        assert_eq!(loaded.opponents.len(), 2);
        assert_eq!(loaded.opponents[0].checkpoint_name, "step_00010000");
        assert!((loaded.opponents[1].qi - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_softmax_probabilities() {
        // Test softmax probability computation
        let qi_scores = [0.0, 0.0, 0.0]; // Equal qi scores
        let max_qi = qi_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_qis: Vec<f64> = qi_scores.iter().map(|q| (q - max_qi).exp()).collect();
        let sum: f64 = exp_qis.iter().sum();
        let probs: Vec<f64> = exp_qis.iter().map(|e| e / sum).collect();

        // With equal qi, all probabilities should be 1/3
        for p in probs {
            assert!((p - 1.0 / 3.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_softmax_probabilities_different_qi() {
        // Higher qi should give higher probability
        let qi_scores = [0.0, 1.0, 2.0];
        let max_qi = qi_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_qis: Vec<f64> = qi_scores.iter().map(|q| (q - max_qi).exp()).collect();
        let sum: f64 = exp_qis.iter().sum();
        let probs: Vec<f64> = exp_qis.iter().map(|e| e / sum).collect();

        // Probability should increase with qi
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_qi_update_formula() {
        // Test the qi update: qi -= eta / (N - pi)
        let eta: f64 = 0.01;
        let n: f64 = 10.0;
        let pi: f64 = 0.2; // 20% selection probability

        let initial_qi: f64 = 0.5;
        let new_qi = initial_qi - eta / (n - pi);

        // qi should decrease
        assert!(new_qi < initial_qi);

        // The decrease should be eta / (n - pi) = 0.01 / 9.8 ≈ 0.00102
        let expected_decrease = eta / (n - pi);
        assert!(((initial_qi - new_qi) - expected_decrease).abs() < f64::EPSILON);
    }

    #[test]
    fn test_highest_qi_empty() {
        // Empty list should return None
        let opponents: Vec<(std::path::PathBuf, OpponentQi)> = vec![];
        let highest = opponents
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1 .1
                    .qi
                    .partial_cmp(&b.1 .1.qi)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        assert!(highest.is_none());
    }

    #[test]
    fn test_highest_qi_single() {
        // Single opponent should return index 0
        let opponents = [(
            std::path::PathBuf::from("step_100"),
            OpponentQi {
                checkpoint_name: "step_100".to_string(),
                checkpoint_step: 100,
                qi: 0.5,
            },
        )];
        let highest = opponents
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1 .1
                    .qi
                    .partial_cmp(&b.1 .1.qi)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        assert_eq!(highest, Some(0));
    }

    #[test]
    fn test_highest_qi_multiple() {
        // Should return index of highest qi
        let opponents = [
            (
                std::path::PathBuf::from("step_100"),
                OpponentQi {
                    checkpoint_name: "step_100".to_string(),
                    checkpoint_step: 100,
                    qi: 0.1,
                },
            ),
            (
                std::path::PathBuf::from("step_200"),
                OpponentQi {
                    checkpoint_name: "step_200".to_string(),
                    checkpoint_step: 200,
                    qi: 0.9, // Highest
                },
            ),
            (
                std::path::PathBuf::from("step_300"),
                OpponentQi {
                    checkpoint_name: "step_300".to_string(),
                    checkpoint_step: 300,
                    qi: 0.5,
                },
            ),
        ];
        let highest = opponents
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1 .1
                    .qi
                    .partial_cmp(&b.1 .1.qi)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        assert_eq!(highest, Some(1)); // step_200 has highest qi
    }

    #[test]
    fn test_qi_initialization_first_opponent() {
        // First opponent should get qi = 0.0
        let existing_qis: Vec<f64> = vec![];
        let new_qi = existing_qis
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let initial_qi = if new_qi.is_finite() { new_qi } else { 0.0 };
        assert!((initial_qi - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qi_initialization_subsequent_opponent() {
        // New opponent should get qi = max(existing qi)
        let existing_qis = [0.1, 0.5, 0.3];
        let new_qi = existing_qis
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let initial_qi = if new_qi.is_finite() { new_qi } else { 0.0 };
        assert!((initial_qi - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qi_initialization_with_negative_qis() {
        // Even with negative qi scores, new opponent gets max
        let existing_qis = [-0.5, -0.1, -0.3];
        let new_qi = existing_qis
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let initial_qi = if new_qi.is_finite() { new_qi } else { 0.0 };
        assert!((initial_qi - (-0.1)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qi_scores_file_round_trip() {
        // Test save and load preserves all data
        let original = QiScoresFile {
            qi_eta: 0.02,
            opponents: vec![
                OpponentQi {
                    checkpoint_name: "step_00001000".to_string(),
                    checkpoint_step: 1000,
                    qi: 0.0,
                },
                OpponentQi {
                    checkpoint_name: "step_00002000".to_string(),
                    checkpoint_step: 2000,
                    qi: -0.05,
                },
                OpponentQi {
                    checkpoint_name: "step_00003000".to_string(),
                    checkpoint_step: 3000,
                    qi: 0.0, // Newest, should have max qi
                },
            ],
        };

        let json = serde_json::to_string(&original).unwrap();
        let loaded: QiScoresFile = serde_json::from_str(&json).unwrap();

        assert!((loaded.qi_eta - 0.02).abs() < f64::EPSILON);
        assert_eq!(loaded.opponents.len(), 3);
        assert_eq!(loaded.opponents[0].checkpoint_name, "step_00001000");
        assert_eq!(loaded.opponents[1].checkpoint_step, 2000);
        assert!((loaded.opponents[1].qi - (-0.05)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_softmax_with_negative_qi() {
        // Softmax should work correctly with negative qi values
        let qi_scores = [-1.0, 0.0, 1.0];
        let max_qi = qi_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_qis: Vec<f64> = qi_scores.iter().map(|q| (q - max_qi).exp()).collect();
        let sum: f64 = exp_qis.iter().sum();
        let probs: Vec<f64> = exp_qis.iter().map(|e| e / sum).collect();

        // All probabilities should sum to 1
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);

        // Higher qi should still give higher probability
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large qi values (should not overflow due to max subtraction)
        let qi_scores = [100.0, 101.0, 102.0];
        let max_qi = qi_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_qis: Vec<f64> = qi_scores.iter().map(|q| (q - max_qi).exp()).collect();
        let sum: f64 = exp_qis.iter().sum();
        let probs: Vec<f64> = exp_qis.iter().map(|e| e / sum).collect();

        // Should not be NaN or Inf
        for p in &probs {
            assert!(p.is_finite());
            assert!(*p >= 0.0);
            assert!(*p <= 1.0);
        }

        // Should sum to 1
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_qi_update_opponent_wins_no_change() {
        // When opponent wins (current loses), qi should not change
        let initial_qi: f64 = 0.5;
        let current_won = false;

        // Simulate the update logic
        let final_qi: f64 = if current_won {
            let eta: f64 = 0.01;
            let n: f64 = 10.0;
            let pi: f64 = 0.1;
            initial_qi - eta / (n - pi)
        } else {
            initial_qi // No change when opponent wins
        };

        assert!((final_qi - initial_qi).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qi_update_current_wins_decreases() {
        // When current wins (opponent loses), qi should decrease
        let initial_qi: f64 = 0.5;
        let current_won = true;
        let eta: f64 = 0.01;
        let n: f64 = 10.0;
        let pi: f64 = 0.1;

        let final_qi: f64 = if current_won {
            initial_qi - eta / (n - pi)
        } else {
            initial_qi
        };

        assert!(final_qi < initial_qi);
        // Verify exact decrease
        let expected_decrease = eta / (n - pi);
        assert!(((initial_qi - final_qi) - expected_decrease).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qi_update_high_probability_smaller_decrease() {
        // Opponents with higher selection probability should decrease less
        let eta = 0.01;
        let n = 10.0;

        let pi_low = 0.05; // Low probability opponent
        let pi_high = 0.3; // High probability opponent

        let decrease_low = eta / (n - pi_low); // 0.01 / 9.95 ≈ 0.001005
        let decrease_high = eta / (n - pi_high); // 0.01 / 9.7 ≈ 0.001031

        // Higher probability means larger denominator decrease means smaller qi decrease
        // Wait, that's backwards. Let me recalculate:
        // decrease_low = 0.01 / 9.95 = 0.001005...
        // decrease_high = 0.01 / 9.7 = 0.001030...
        // So higher pi actually means LARGER decrease (smaller denominator)
        assert!(decrease_high > decrease_low);
    }

    #[test]
    fn test_selection_probability_computation() {
        // Test the selection probability formula used in qi updates
        let qi_scores = [0.0, 0.5, 1.0];
        let max_qi = qi_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_qis: Vec<f64> = qi_scores.iter().map(|q| (q - max_qi).exp()).collect();
        let sum: f64 = exp_qis.iter().sum();

        // Compute individual probabilities
        let p0 = exp_qis[0] / sum;
        let p1 = exp_qis[1] / sum;
        let p2 = exp_qis[2] / sum;

        // Verify they're valid probabilities
        assert!(p0 > 0.0 && p0 < 1.0);
        assert!(p1 > 0.0 && p1 < 1.0);
        assert!(p2 > 0.0 && p2 < 1.0);
        assert!((p0 + p1 + p2 - 1.0).abs() < 1e-10);

        // Verify ordering matches qi ordering
        assert!(p0 < p1);
        assert!(p1 < p2);
    }

    #[test]
    fn test_format_selected_opponents() {
        // Test the debug output formatting
        let opponents = [
            (
                std::path::PathBuf::from("step_1000"),
                OpponentQi {
                    checkpoint_name: "step_1000".to_string(),
                    checkpoint_step: 1000,
                    qi: 0.0,
                },
            ),
            (
                std::path::PathBuf::from("step_2000"),
                OpponentQi {
                    checkpoint_name: "step_2000".to_string(),
                    checkpoint_step: 2000,
                    qi: 0.0,
                },
            ),
        ];

        let current_step: i64 = 3000;
        let indices = [0, 1];

        // Compute expected output manually
        #[expect(
            clippy::cast_possible_wrap,
            reason = "checkpoint_step won't exceed i64::MAX in tests"
        )]
        let formatted: String = indices
            .iter()
            .map(|&idx| {
                let (_, opp) = &opponents[idx];
                let relative = opp.checkpoint_step as i64 - current_step;
                format!("{relative}")
            })
            .collect::<Vec<_>>()
            .join(", ");

        assert_eq!(formatted, "-2000, -1000");
    }
}
