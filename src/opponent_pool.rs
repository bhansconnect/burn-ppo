//! Historical Opponent Pool for Self-Play Training
//!
//! Implements opponent pool training to prevent strategy collapse and improve generalization.
//!
//! A configurable fraction of training games are played against historical checkpoints,
//! with opponents sampled using win-rate-weighted probabilities.
//!
//! Key features:
//! - Win-rate-based sampling: `P(opponent) ∝ (1 - win_rate)^p` (focus on hard opponents)
//! - Batch EMA updates: win rates updated once per rotation, not per game
//! - Per-slot diversity: each opponent player slot gets independently sampled opponent
//! - Lazy loading: only load models when actively playing
//! - Graceful rotation: wait for games to complete before swapping opponents

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use plotters::backend::BitMapBackend;
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::checkpoint::{load_metadata, load_normalizer, CheckpointManager};
use crate::config::Config;
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;

/// Persisted per-opponent statistics (saved in `opponent_stats.json`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpponentStats {
    /// Checkpoint name (e.g., `step_00010000`)
    pub checkpoint_name: String,
    /// Checkpoint training step (for relative version calculation)
    pub checkpoint_step: usize,
    /// Learner's win rate against this opponent (EMA, 0.0 to 1.0)
    pub win_rate: f64,
    /// Total games played against this opponent (for debugging/confidence)
    pub games_played: u32,
}

impl OpponentStats {
    /// Create a new opponent stats with default win rate (0.5 = neutral)
    pub fn new(checkpoint_name: String, checkpoint_step: usize) -> Self {
        Self {
            checkpoint_name,
            checkpoint_step,
            win_rate: 0.5, // Neutral initial assumption
            games_played: 0,
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
    /// Stats data for this opponent
    pub stats: OpponentStats,
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
    /// Pool indices of assigned opponents (resampled after each game)
    /// Length = `num_players` - 1
    pub assigned_opponents: Vec<usize>,
}

impl EnvState {
    /// Create a new env state with random initial position assignment
    pub fn new(num_players: usize, assigned_opponents: Vec<usize>, rng: &mut impl Rng) -> Self {
        let mut state = Self {
            learner_position: 0,
            position_to_opponent: vec![None; num_players],
            assigned_opponents,
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
}

// Note: PendingGameResult removed - we now use rotation-level tracking with
// pending_rotation_stats: HashMap<usize, (u32, u32)> for (wins, games) per opponent

/// Main opponent pool manager
pub struct OpponentPool<B: Backend> {
    /// All available opponents (checkpoint path + stats)
    available: Vec<(PathBuf, OpponentStats)>,

    /// Currently loaded opponents by pool index
    /// Key = pool index, Value = loaded opponent
    loaded: HashMap<usize, LoadedOpponent<B>>,

    /// Number of opponent slots per game (`num_players` - 1)
    num_opponent_slots: usize,

    /// Directory containing checkpoints
    checkpoints_dir: PathBuf,

    /// EMA alpha for win rate smoothing (applied once per rotation)
    opponent_select_alpha: f64,

    /// Exponent p for (1-win_rate)^p selection probability
    opponent_select_exponent: f64,

    /// Config for model loading
    config: Config,

    /// Device for model loading
    device: B::Device,

    /// RNG for sampling
    rng: rand::rngs::StdRng,

    /// Current opponent indices shared by all envs (length = `num_opponent_slots`)
    /// Refreshed after each policy update for optimal batching
    current_opponents: Vec<usize>,

    /// Pending rotation stats for batched win rate updates
    /// Key = pool index, Value = (wins, games) accumulated this rotation
    pending_rotation_stats: HashMap<usize, (u32, u32)>,
}

/// Persisted opponent stats file format (`opponent_stats.json`)
#[derive(Debug, Serialize, Deserialize)]
struct OpponentStatsFile {
    /// File format version
    version: u32,
    /// Config parameters used
    config: OpponentStatsConfig,
    /// Per-opponent statistics
    opponents: Vec<OpponentStats>,
}

/// Config section of opponent stats file
#[derive(Debug, Serialize, Deserialize)]
struct OpponentStatsConfig {
    opponent_select_alpha: f64,
    opponent_select_exponent: f64,
}

impl<B: Backend> OpponentPool<B> {
    /// Create a new opponent pool
    ///
    /// Scans `checkpoints_dir` for available checkpoints and loads opponent stats.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "checkpoints_dir is stored in struct, taking ownership is intentional"
    )]
    pub fn new(
        checkpoints_dir: PathBuf,
        num_players: usize,
        opponent_select_alpha: f64,
        opponent_select_exponent: f64,
        config: Config,
        device: B::Device,
        seed: u64,
    ) -> Result<Self> {
        let mut pool = Self {
            available: Vec::new(),
            loaded: HashMap::new(),
            num_opponent_slots: num_players.saturating_sub(1),
            checkpoints_dir: checkpoints_dir.clone(),
            opponent_select_alpha,
            opponent_select_exponent,
            config,
            device,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            current_opponents: Vec::new(),
            pending_rotation_stats: HashMap::new(),
        };

        // Load existing opponent stats (with migration from old qi_scores.json)
        pool.load_opponent_stats()?;

        // Scan for checkpoints and merge with loaded stats
        pool.scan_checkpoints()?;

        // Initialize current opponents
        pool.refresh_current_opponents();

        Ok(pool)
    }

    /// Refresh the current opponent set for all envs to share
    ///
    /// Samples `num_opponent_slots` unique opponents using win-rate-weighted sampling.
    /// All opponent envs share these opponents for optimal forward pass batching.
    /// Call this after each policy update.
    pub fn refresh_current_opponents(&mut self) {
        self.current_opponents.clear();

        if self.available.is_empty() {
            return;
        }

        // Sample num_opponent_slots unique opponents
        for _ in 0..self.num_opponent_slots {
            if let Some(idx) = self.sample_opponent(&self.current_opponents.clone()) {
                self.current_opponents.push(idx);
            }
        }
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

    /// Get opponent stats file path (in run folder, not checkpoints folder)
    fn opponent_stats_path(&self) -> PathBuf {
        self.checkpoints_dir
            .parent()
            .expect("checkpoints_dir has parent")
            .join("opponent_stats.json")
    }

    /// Get legacy qi scores file path (for migration)
    fn legacy_qi_scores_path(&self) -> PathBuf {
        self.checkpoints_dir
            .parent()
            .expect("checkpoints_dir has parent")
            .join("qi_scores.json")
    }

    /// Load opponent stats from disk (with migration from legacy `qi_scores.json`)
    fn load_opponent_stats(&mut self) -> Result<()> {
        // Legacy structs for migration from old qi_scores.json format
        #[derive(Deserialize)]
        struct LegacyQiScoresFile {
            #[expect(dead_code, reason = "field exists in legacy format but not used")]
            qi_eta: f64,
            opponents: Vec<LegacyOpponentQi>,
        }
        #[derive(Deserialize)]
        struct LegacyOpponentQi {
            checkpoint_name: String,
            checkpoint_step: usize,
            #[expect(dead_code, reason = "field exists in legacy format but not used")]
            qi: f64,
        }

        let path = self.opponent_stats_path();

        if path.exists() {
            // Load new format
            let json = fs::read_to_string(&path).context("Failed to read opponent_stats.json")?;
            let file: OpponentStatsFile = serde_json::from_str(&json)?;

            for stats in file.opponents {
                let checkpoint_path = self.checkpoints_dir.join(&stats.checkpoint_name);
                self.available.push((checkpoint_path, stats));
            }
        } else {
            // Try to migrate from legacy qi_scores.json
            let legacy_path = self.legacy_qi_scores_path();
            if legacy_path.exists() {
                eprintln!(
                    "Migrating from legacy qi_scores.json to opponent_stats.json (all win_rate = 0.5)"
                );
                let json =
                    fs::read_to_string(&legacy_path).context("Failed to read qi_scores.json")?;

                let legacy_file: LegacyQiScoresFile = serde_json::from_str(&json)?;

                for legacy in legacy_file.opponents {
                    let stats =
                        OpponentStats::new(legacy.checkpoint_name.clone(), legacy.checkpoint_step);
                    let checkpoint_path = self.checkpoints_dir.join(&legacy.checkpoint_name);
                    self.available.push((checkpoint_path, stats));
                }
            }
        }

        Ok(())
    }

    /// Save opponent stats to disk (atomic write via temp file + rename)
    pub fn save_opponent_stats(&self) -> Result<()> {
        let file = OpponentStatsFile {
            version: 1,
            config: OpponentStatsConfig {
                opponent_select_alpha: self.opponent_select_alpha,
                opponent_select_exponent: self.opponent_select_exponent,
            },
            opponents: self.available.iter().map(|(_, s)| s.clone()).collect(),
        };

        let json = serde_json::to_string_pretty(&file)?;

        // Atomic write: temp file + rename
        let path = self.opponent_stats_path();
        let temp_path = path.with_extension("json.tmp");
        fs::write(&temp_path, json).context("Failed to write temp opponent_stats file")?;
        fs::rename(&temp_path, &path).context("Failed to rename opponent_stats file")?;

        Ok(())
    }

    /// Scan checkpoints directory and add new checkpoints to pool
    #[expect(clippy::unnecessary_wraps, reason = "Result for future error handling")]
    pub fn scan_checkpoints(&mut self) -> Result<()> {
        // Get existing checkpoint names for deduplication
        let existing: std::collections::HashSet<String> = self
            .available
            .iter()
            .map(|(_, s)| s.checkpoint_name.clone())
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

            // Load metadata to get step number
            let Ok(metadata) = load_metadata(&path) else {
                continue; // Skip invalid checkpoints
            };

            // Create stats with default win_rate = 0.5 (neutral)
            let stats = OpponentStats::new(name, metadata.step);
            self.available.push((path, stats));
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
            .any(|(_, s)| s.checkpoint_name == name)
        {
            return;
        }

        // New opponent starts with win_rate = 0.5 (neutral assumption)
        let stats = OpponentStats::new(name, step);
        self.available.push((checkpoint_path, stats));
    }

    /// Sample a single opponent using win-rate-weighted sampling, excluding already-assigned indices
    ///
    /// Probability is proportional to `(1 - win_rate)^p`, prioritizing opponents the learner loses to.
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

        // Compute (1 - win_rate)^p weights for eligible opponents
        let weights: Vec<f64> = eligible
            .iter()
            .map(|&i| {
                let win_rate = self.available[i].1.win_rate;
                (1.0 - win_rate).powf(self.opponent_select_exponent)
            })
            .collect();

        let sum: f64 = weights.iter().sum();
        if sum == 0.0 {
            // All opponents have win_rate = 1.0, uniform fallback
            let idx = self.rng.gen_range(0..eligible.len());
            return Some(eligible[idx]);
        }

        // Sample from distribution
        let sample: f64 = self.rng.r#gen();
        let mut cumsum = 0.0;
        for (i, &weight) in weights.iter().enumerate() {
            cumsum += weight / sum;
            if sample < cumsum {
                return Some(eligible[i]);
            }
        }

        Some(
            *eligible
                .last()
                .expect("eligible list is non-empty at this point"),
        )
    }

    /// Get the current opponent set for a game
    ///
    /// Returns a copy of the current opponents (refreshed after each policy update).
    /// All envs share the same opponent set for optimal forward pass batching.
    pub fn sample_all_slots(&self) -> Vec<usize> {
        self.current_opponents.clone()
    }

    /// Get checkpoint name for a pool index
    pub fn get_checkpoint_name(&self, pool_index: usize) -> String {
        self.available.get(pool_index).map_or_else(
            || format!("unknown_{pool_index}"),
            |(_, stats)| stats.checkpoint_name.clone(),
        )
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
        let (path, stats) = &self.available[pool_index];

        let (model, _metadata) = CheckpointManager::load::<B>(path, &self.config, &self.device)
            .with_context(|| format!("Failed to load opponent from {}", path.display()))?;

        let normalizer = load_normalizer(path)?;

        let loaded = LoadedOpponent {
            checkpoint_path: path.clone(),
            model,
            normalizer,
            stats: stats.clone(),
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

    /// Queue a game result for batched win rate update
    ///
    /// Game results are accumulated per opponent and applied once per rotation
    /// using batch EMA updates.
    ///
    /// `placements[i]` = 1 for 1st place, 2 for 2nd, etc. (lower = better)
    /// `learner_position` = which seat the learner was in
    /// `position_to_opponent` = map from seat position to opponent pool index (captured before shuffle)
    pub fn queue_game_result(
        &mut self,
        placements: &[usize],
        learner_position: usize,
        position_to_opponent: &[Option<usize>],
    ) {
        let learner_placement = placements[learner_position];

        // For each opponent, determine if they beat the current model (pairwise)
        for (pos, &placement) in placements.iter().enumerate() {
            if pos == learner_position {
                continue;
            }

            let Some(pool_idx) = position_to_opponent[pos] else {
                continue;
            };

            // Learner wins against this opponent if learner placed better (lower number)
            let learner_won = learner_placement < placement;

            // Accumulate stats for this opponent
            let entry = self
                .pending_rotation_stats
                .entry(pool_idx)
                .or_insert((0, 0));
            entry.1 += 1; // games
            if learner_won {
                entry.0 += 1; // wins
            }
        }
    }

    /// Apply all pending game results to win rates using batch EMA
    ///
    /// For each opponent with pending stats:
    /// - `rotation_win_rate = wins / games`
    /// - `win_rate = win_rate * (1 - alpha) + rotation_win_rate * alpha`
    ///
    /// This updates once per rotation (not per game) to avoid overwriting
    /// the historical win rate too quickly.
    pub fn apply_pending_win_rate_updates(&mut self) {
        if self.pending_rotation_stats.is_empty() {
            return;
        }

        let alpha = self.opponent_select_alpha;

        for (pool_idx, (wins, games)) in self.pending_rotation_stats.drain() {
            if games == 0 {
                continue;
            }

            if let Some((_, stats)) = self.available.get_mut(pool_idx) {
                let rotation_win_rate = f64::from(wins) / f64::from(games);
                stats.win_rate = stats.win_rate * (1.0 - alpha) + rotation_win_rate * alpha;
                stats.games_played += games;
            }
        }
    }

    /// Compute selection probability for a specific opponent using `(1 - win_rate)^p`
    #[expect(dead_code, reason = "API for debugging/introspection")]
    pub fn compute_selection_probability(&self, pool_idx: usize) -> f64 {
        if self.available.is_empty() {
            return 0.0;
        }

        // Compute (1 - win_rate)^p weights for all opponents
        let weights: Vec<f64> = self
            .available
            .iter()
            .map(|(_, stats)| (1.0 - stats.win_rate).powf(self.opponent_select_exponent))
            .collect();

        let sum: f64 = weights.iter().sum();
        if sum == 0.0 {
            // All opponents have win_rate = 1.0, uniform fallback
            return 1.0 / self.available.len() as f64;
        }

        weights[pool_idx] / sum
    }

    /// Compute selection probabilities for all opponents using `(1 - win_rate)^p`
    pub fn compute_all_selection_probabilities(&self) -> Vec<f64> {
        if self.available.is_empty() {
            return vec![];
        }

        // Compute (1 - win_rate)^p weights for all opponents
        let weights: Vec<f64> = self
            .available
            .iter()
            .map(|(_, stats)| (1.0 - stats.win_rate).powf(self.opponent_select_exponent))
            .collect();

        let sum: f64 = weights.iter().sum();
        if sum == 0.0 {
            // All opponents have win_rate = 1.0, uniform fallback
            let uniform = 1.0 / self.available.len() as f64;
            return vec![uniform; self.available.len()];
        }

        weights.iter().map(|w| w / sum).collect()
    }

    /// Get mutable reference to RNG (for creating `EnvStates`)
    pub fn rng_mut(&mut self) -> &mut rand::rngs::StdRng {
        &mut self.rng
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
            .map(|(idx, (_, stats))| (idx, stats.checkpoint_step))
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

    /// Save selection probability distribution graph to the checkpoint directory.
    /// Also creates a symlink at the run root pointing to `checkpoints/latest/selection_probability.png`.
    pub fn save_selection_probability_graph(&self, checkpoint_path: &Path) -> Result<()> {
        if self.available.is_empty() {
            return Ok(());
        }

        let probs = self.compute_all_selection_probabilities();

        // Sort by step descending to get checkpoint-relative positions
        let mut sorted_by_step: Vec<(usize, usize)> = self
            .available
            .iter()
            .enumerate()
            .map(|(idx, (_, stats))| (idx, stats.checkpoint_step))
            .collect();
        sorted_by_step.sort_by(|a, b| b.1.cmp(&a.1));

        // Build data points: (checkpoint_relative_position, probability, opponent_win_rate)
        // Flip win_rate to show opponent's perspective (1 - learner_win_rate)
        let data: Vec<(i32, f64, f64)> = sorted_by_step
            .iter()
            .enumerate()
            .map(|(rel_pos, (pool_idx, _))| {
                let pos_i32 = i32::try_from(rel_pos + 1).unwrap_or(i32::MAX);
                let prob = probs[*pool_idx];
                let opponent_win_rate = 1.0 - self.available[*pool_idx].1.win_rate;
                (-pos_i32, prob, opponent_win_rate)
            })
            .collect();

        // Calculate games played stats for annotation
        let games: Vec<u32> = sorted_by_step
            .iter()
            .map(|(pool_idx, _)| self.available[*pool_idx].1.games_played)
            .collect();
        let min_games = games.iter().copied().min().unwrap_or(0);
        let max_games = games.iter().copied().max().unwrap_or(0);
        let total_games: u32 = games.iter().sum();

        // Find max probability for Y-axis scaling
        let max_prob = data.iter().map(|d| d.1).fold(0.0, f64::max);
        let y_max = (max_prob * 1.1).max(0.01);

        // X-axis: from -n to 0 (oldest to newest)
        let x_min = data.iter().map(|d| d.0).min().unwrap_or(-1) - 1;
        let x_max = 0;

        let graph_path = checkpoint_path.join("selection_probability.png");

        let root = BitMapBackend::new(&graph_path, (800, 450)).into_drawing_area();
        root.fill(&WHITE).context("Failed to fill background")?;

        // Build dual-axis chart: primary (left) for probability, secondary (right) for win rate
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Selection Probability & Win Rate by Checkpoint",
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(50)
            .right_y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, 0.0..y_max)
            .context("Failed to build chart")?
            .set_secondary_coord(x_min..x_max, 0.0..1.0f64);

        chart
            .configure_mesh()
            .x_desc("Checkpoint (relative)")
            .y_desc("Probability")
            .draw()
            .context("Failed to draw mesh")?;

        chart
            .configure_secondary_axes()
            .y_desc("Opponent Win Rate")
            .draw()
            .context("Failed to draw secondary mesh")?;

        // Draw probability bars (blue, primary axis)
        chart
            .draw_series(data.iter().map(|(x, prob, _)| {
                let x0 = *x;
                let x1 = x + 1;
                Rectangle::new([(x0, 0.0), (x1, *prob)], BLUE.mix(0.7).filled())
            }))
            .context("Failed to draw probability bars")?
            .label("Probability")
            .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], BLUE.mix(0.7).filled()));

        // Draw win rate line (red, secondary axis)
        chart
            .draw_secondary_series(LineSeries::new(
                data.iter().map(|(x, _, win_rate)| (*x, *win_rate)),
                RED.stroke_width(2),
            ))
            .context("Failed to draw win rate line")?
            .label("Opp. Win Rate")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 15, y)], RED.stroke_width(2)));

        // Draw win rate points (red circles, secondary axis)
        chart
            .draw_secondary_series(
                data.iter()
                    .map(|(x, _, win_rate)| Circle::new((*x, *win_rate), 4, RED.filled())),
            )
            .context("Failed to draw win rate points")?;

        // Draw legend
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()
            .context("Failed to draw legend")?;

        // Draw games played annotation at bottom
        let annotation =
            format!("Games played: min={min_games}, max={max_games}, total={total_games}");
        root.draw(&Text::new(
            annotation,
            (400, 430),
            ("sans-serif", 12).into_font().color(&BLACK),
        ))
        .context("Failed to draw annotation")?;

        root.present().context("Failed to present chart")?;

        // Create symlink at run root pointing to checkpoints/latest/selection_probability.png (once)
        // checkpoint_path is like: run_dir/checkpoints/step_XXXX
        // We want: run_dir/selection_probability.png -> checkpoints/latest/selection_probability.png
        if let Some(checkpoints_dir) = checkpoint_path.parent() {
            if let Some(run_dir) = checkpoints_dir.parent() {
                let symlink_path = run_dir.join("selection_probability.png");
                if !symlink_path.exists() && !symlink_path.is_symlink() {
                    #[cfg(unix)]
                    {
                        let _ = std::os::unix::fs::symlink(
                            "checkpoints/latest/selection_probability.png",
                            &symlink_path,
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== OpponentStats Tests ====================

    #[test]
    fn test_opponent_stats_new() {
        let stats = OpponentStats::new("step_00010000".to_string(), 10000);
        assert_eq!(stats.checkpoint_name, "step_00010000");
        assert_eq!(stats.checkpoint_step, 10000);
        assert!((stats.win_rate - 0.5).abs() < f64::EPSILON); // Default win_rate = 0.5
        assert_eq!(stats.games_played, 0);
    }

    #[test]
    fn test_opponent_stats_serialization() {
        let stats = OpponentStats {
            checkpoint_name: "step_00010000".to_string(),
            checkpoint_step: 10000,
            win_rate: 0.75,
            games_played: 100,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let loaded: OpponentStats = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.checkpoint_name, stats.checkpoint_name);
        assert_eq!(loaded.checkpoint_step, stats.checkpoint_step);
        assert!((loaded.win_rate - stats.win_rate).abs() < f64::EPSILON);
        assert_eq!(loaded.games_played, stats.games_played);
    }

    #[test]
    fn test_opponent_stats_file_serialization() {
        let file = OpponentStatsFile {
            version: 1,
            config: OpponentStatsConfig {
                opponent_select_alpha: 0.1,
                opponent_select_exponent: 1.0,
            },
            opponents: vec![
                OpponentStats::new("step_00010000".to_string(), 10000),
                OpponentStats {
                    checkpoint_name: "step_00020000".to_string(),
                    checkpoint_step: 20000,
                    win_rate: 0.65,
                    games_played: 50,
                },
            ],
        };

        let json = serde_json::to_string_pretty(&file).expect("serialize");
        let loaded: OpponentStatsFile = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(loaded.version, 1);
        assert!((loaded.config.opponent_select_alpha - 0.1).abs() < f64::EPSILON);
        assert!((loaded.config.opponent_select_exponent - 1.0).abs() < f64::EPSILON);
        assert_eq!(loaded.opponents.len(), 2);
        assert_eq!(loaded.opponents[0].checkpoint_name, "step_00010000");
        assert!((loaded.opponents[1].win_rate - 0.65).abs() < f64::EPSILON);
    }

    // ==================== EnvState Tests ====================

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

    // ==================== Win Rate EMA Tests ====================

    #[test]
    fn test_win_rate_ema_basic() {
        // win_rate = 0.5, rotation_win_rate = 1.0, alpha = 0.1
        // Expected: 0.5 * 0.9 + 1.0 * 0.1 = 0.55
        let initial_win_rate: f64 = 0.5;
        let rotation_win_rate: f64 = 1.0;
        let alpha: f64 = 0.1;

        let new_win_rate = initial_win_rate * (1.0 - alpha) + rotation_win_rate * alpha;
        assert!((new_win_rate - 0.55).abs() < f64::EPSILON);
    }

    #[test]
    fn test_win_rate_ema_zero_rotation() {
        // win_rate = 0.5, rotation_win_rate = 0.0, alpha = 0.1
        // Expected: 0.5 * 0.9 + 0.0 * 0.1 = 0.45
        let initial_win_rate: f64 = 0.5;
        let rotation_win_rate: f64 = 0.0;
        let alpha: f64 = 0.1;

        let new_win_rate = initial_win_rate * (1.0 - alpha) + rotation_win_rate * alpha;
        assert!((new_win_rate - 0.45).abs() < f64::EPSILON);
    }

    #[test]
    fn test_win_rate_ema_one_alpha() {
        // alpha = 1.0 should fully replace with rotation_win_rate
        let initial_win_rate: f64 = 0.3;
        let rotation_win_rate: f64 = 0.8;
        let alpha: f64 = 1.0;

        let new_win_rate = initial_win_rate * (1.0 - alpha) + rotation_win_rate * alpha;
        assert!((new_win_rate - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_win_rate_convergence() {
        // Simulate multiple rotations with constant results
        // Should converge toward the rotation win rate
        let mut win_rate: f64 = 0.5;
        let rotation_win_rate: f64 = 0.9;
        let alpha: f64 = 0.1;

        for _ in 0..100 {
            win_rate = win_rate * (1.0 - alpha) + rotation_win_rate * alpha;
        }

        // Should be very close to 0.9 after 100 updates
        assert!((win_rate - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_win_rate_bounds() {
        // After many wins: win_rate should approach 1.0 but never exceed
        let mut win_rate = 0.5;
        let alpha = 0.1;

        for _ in 0..1000 {
            win_rate = win_rate * (1.0 - alpha) + 1.0 * alpha;
        }
        assert!(win_rate <= 1.0);
        assert!(win_rate >= 0.999); // Should be very close to 1.0

        // After many losses: win_rate should approach 0.0 but never go negative
        win_rate = 0.5;
        for _ in 0..1000 {
            win_rate = win_rate * (1.0 - alpha) + 0.0 * alpha;
        }
        assert!(win_rate >= 0.0);
        assert!(win_rate < 0.001); // Should be very close to 0.0
    }

    // ==================== Selection Probability Tests ====================

    #[test]
    fn test_selection_probability_basic() {
        // win_rates = [0.2, 0.5, 0.8], exponent = 1.0
        // weights = [0.8, 0.5, 0.2], sum = 1.5
        // probs = [0.533, 0.333, 0.133]
        let win_rates: [f64; 3] = [0.2, 0.5, 0.8];
        let exponent: f64 = 1.0;

        let weights: Vec<f64> = win_rates.iter().map(|w| (1.0 - w).powf(exponent)).collect();
        let sum: f64 = weights.iter().sum();
        let probs: Vec<f64> = weights.iter().map(|w| w / sum).collect();

        assert!((probs[0] - 0.8 / 1.5).abs() < 1e-10);
        assert!((probs[1] - 0.5 / 1.5).abs() < 1e-10);
        assert!((probs[2] - 0.2 / 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_selection_probability_exponent_2() {
        // win_rates = [0.2, 0.5, 0.8], exponent = 2.0
        // weights = [0.64, 0.25, 0.04]
        let win_rates: [f64; 3] = [0.2, 0.5, 0.8];
        let exponent: f64 = 2.0;

        let weights: Vec<f64> = win_rates.iter().map(|w| (1.0 - w).powf(exponent)).collect();
        let sum: f64 = weights.iter().sum();
        let probs: Vec<f64> = weights.iter().map(|w| w / sum).collect();

        // Higher exponent should make the distribution more peaked toward hard opponents
        // Verify probability of hardest opponent is much higher than with p=1
        assert!(probs[0] > 0.6); // Should dominate
        assert!(probs[2] < 0.1); // Should be very small

        // Verify ordering: lower win_rate → higher probability
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_selection_probability_uniform_win_rates() {
        // All win_rates = 0.5 → uniform sampling
        let win_rates: [f64; 3] = [0.5, 0.5, 0.5];
        let exponent: f64 = 1.0;

        let weights: Vec<f64> = win_rates.iter().map(|w| (1.0 - w).powf(exponent)).collect();
        let sum: f64 = weights.iter().sum();
        let probs: Vec<f64> = weights.iter().map(|w| w / sum).collect();

        // All should be 1/3
        for p in &probs {
            assert!((*p - 1.0 / 3.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_selection_probability_edge_win_rate_1() {
        // win_rate = 1.0 → weight = 0 → excluded from sampling
        let win_rates: [f64; 3] = [0.5, 1.0, 0.3];
        let exponent: f64 = 1.0;

        let weights: Vec<f64> = win_rates.iter().map(|w| (1.0 - w).powf(exponent)).collect();
        let sum: f64 = weights.iter().sum();

        // Weight for win_rate=1.0 should be 0
        assert!((weights[1] - 0.0).abs() < f64::EPSILON);

        // Remaining probabilities should normalize correctly
        let probs: Vec<f64> = weights.iter().map(|w| w / sum).collect();
        assert!((probs[1] - 0.0).abs() < f64::EPSILON);
        assert!((probs[0] + probs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_selection_probability_sums_to_one() {
        // For any set of win_rates, probabilities sum to 1.0
        let test_cases: [Vec<f64>; 5] = [
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.9, 0.8, 0.7],
            vec![0.5],
            vec![0.0, 1.0],
            vec![0.33, 0.33, 0.34],
        ];

        for win_rates in test_cases {
            let exponent: f64 = 1.0;
            let weights: Vec<f64> = win_rates.iter().map(|w| (1.0 - w).powf(exponent)).collect();
            let sum: f64 = weights.iter().sum();

            if sum > 0.0 {
                let probs: Vec<f64> = weights.iter().map(|w| w / sum).collect();
                let total: f64 = probs.iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-10,
                    "Probabilities don't sum to 1 for {win_rates:?}"
                );
            }
        }
    }

    #[test]
    fn test_selection_probability_ordering() {
        // Lower win_rate → higher probability (for p > 0)
        let win_rates: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
        let exponent: f64 = 1.0;

        let weights: Vec<f64> = win_rates.iter().map(|w| (1.0 - w).powf(exponent)).collect();
        let sum: f64 = weights.iter().sum();
        let probs: Vec<f64> = weights.iter().map(|w| w / sum).collect();

        // Each probability should be greater than the next
        for i in 0..probs.len() - 1 {
            assert!(
                probs[i] > probs[i + 1],
                "Expected probs[{}] > probs[{}] but got {} <= {}",
                i,
                i + 1,
                probs[i],
                probs[i + 1]
            );
        }
    }

    // ==================== Numerical Stability Tests ====================

    #[test]
    fn test_no_nan_in_probabilities() {
        // Various edge cases should never produce NaN
        let test_cases: Vec<(Vec<f64>, f64)> = vec![
            (vec![0.0, 0.0, 0.0], 1.0),
            (vec![1.0, 1.0, 1.0], 1.0), // All weights 0 → uniform fallback
            (vec![0.5], 1.0),
            (vec![0.999_999, 0.000_001], 2.0),
        ];

        for (win_rates, exponent) in test_cases {
            let weights: Vec<f64> = win_rates.iter().map(|w| (1.0 - w).powf(exponent)).collect();
            let sum: f64 = weights.iter().sum();

            let probs: Vec<f64> = if sum == 0.0 {
                vec![1.0 / win_rates.len() as f64; win_rates.len()]
            } else {
                weights.iter().map(|w| w / sum).collect()
            };

            for p in &probs {
                assert!(
                    !p.is_nan(),
                    "NaN probability for {win_rates:?}, p={exponent}"
                );
                assert!(p.is_finite(), "Infinite probability for {win_rates:?}");
            }
        }
    }

    #[test]
    fn test_win_rate_exactly_zero() {
        // win_rate = 0.0 → weight = 1.0^p = 1.0
        let win_rate: f64 = 0.0;
        let exponent: f64 = 1.0;
        let weight = (1.0 - win_rate).powf(exponent);
        assert!((weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_win_rate_exactly_one() {
        // win_rate = 1.0 → weight = 0.0^p = 0.0
        let win_rate: f64 = 1.0;
        let exponent: f64 = 1.0;
        let weight = (1.0 - win_rate).powf(exponent);
        assert!((weight - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_large_exponent_numerical_stability() {
        // Large exponents with small (1-w) should not overflow
        let win_rate: f64 = 0.1; // (1-w) = 0.9
        let exponent: f64 = 100.0;
        let weight = (1.0 - win_rate).powf(exponent);

        assert!(weight.is_finite());
        assert!(weight >= 0.0);
        // 0.9^100 ≈ 2.66e-5
        assert!(weight < 0.001);
    }

    // ==================== Persistence Tests ====================

    #[test]
    fn test_opponent_stats_file_round_trip() {
        let original = OpponentStatsFile {
            version: 1,
            config: OpponentStatsConfig {
                opponent_select_alpha: 0.15,
                opponent_select_exponent: 2.0,
            },
            opponents: vec![
                OpponentStats {
                    checkpoint_name: "step_00001000".to_string(),
                    checkpoint_step: 1000,
                    win_rate: 0.3,
                    games_played: 100,
                },
                OpponentStats {
                    checkpoint_name: "step_00002000".to_string(),
                    checkpoint_step: 2000,
                    win_rate: 0.7,
                    games_played: 50,
                },
            ],
        };

        let json = serde_json::to_string(&original).unwrap();
        let loaded: OpponentStatsFile = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.version, 1);
        assert!((loaded.config.opponent_select_alpha - 0.15).abs() < f64::EPSILON);
        assert!((loaded.config.opponent_select_exponent - 2.0).abs() < f64::EPSILON);
        assert_eq!(loaded.opponents.len(), 2);
        assert_eq!(loaded.opponents[0].checkpoint_name, "step_00001000");
        assert!((loaded.opponents[0].win_rate - 0.3).abs() < f64::EPSILON);
        assert_eq!(loaded.opponents[1].games_played, 50);
    }

    // ==================== Initialization Tests ====================

    #[test]
    fn test_new_opponent_default_win_rate() {
        // New opponent should get win_rate = 0.5 (neutral assumption)
        let stats = OpponentStats::new("step_00100000".to_string(), 100_000);
        assert!((stats.win_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(stats.games_played, 0);
    }
}
