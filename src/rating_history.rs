//! Rating history tracking for training progress
//!
//! Records game results from opponent pool training and computes Elo ratings
//! using Plackett-Luce maximum likelihood estimation.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

use crate::plackett_luce::{self, GameResult, PlackettLuceConfig};

/// A single game result for rating (serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatingGame {
    /// Checkpoint name for the current network
    pub current: String,
    /// Checkpoint names for opponents
    pub opponents: Vec<String>,
    /// Placements: `[current_placement, opponent_placements...]`
    pub placements: Vec<usize>,
}

/// Metadata persisted alongside game history
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RatingMetadata {
    /// Checkpoint name -> step mapping
    checkpoint_steps: HashMap<String, usize>,
    /// Name of first checkpoint (anchor)
    first_checkpoint: Option<String>,
    /// Name of current checkpoint (for game attribution)
    current_checkpoint: Option<String>,
}

/// Result of rating computation
#[derive(Debug, Clone)]
pub struct RatingResult {
    /// Rating of the latest checkpoint
    pub current_elo: f64,
    /// Rating of currently highest-rated checkpoint
    pub best_elo: f64,
    /// Step of currently highest-rated checkpoint
    pub best_step: usize,
    /// Total games in rating history
    pub total_games: usize,
}

/// Persistent storage for rating game results
pub struct RatingHistory {
    /// All recorded games
    games: Vec<RatingGame>,
    /// Checkpoint name -> player index mapping
    checkpoint_to_idx: HashMap<String, usize>,
    /// Player index -> checkpoint name (for reverse lookup)
    idx_to_checkpoint: Vec<String>,
    /// Player index -> checkpoint step
    idx_to_step: Vec<usize>,
    /// Index of first checkpoint (anchor at 1000)
    first_checkpoint_idx: Option<usize>,
    /// Current checkpoint name (games being accumulated)
    current_checkpoint: Option<String>,
    /// Run directory (for persistence paths)
    run_dir: PathBuf,
    /// Path to persist games
    games_path: PathBuf,
    /// Path to persist metadata
    metadata_path: PathBuf,
    /// Cached ratings (recomputed when needed)
    cached_ratings: Option<Vec<f64>>,
}

impl RatingHistory {
    /// Create a new rating history for a run directory
    pub fn new(run_dir: &Path) -> Self {
        Self {
            games: Vec::new(),
            checkpoint_to_idx: HashMap::new(),
            idx_to_checkpoint: Vec::new(),
            idx_to_step: Vec::new(),
            first_checkpoint_idx: None,
            current_checkpoint: None,
            run_dir: run_dir.to_path_buf(),
            games_path: run_dir.join("rating_games.jsonl"),
            metadata_path: run_dir.join("rating_metadata.json"),
            cached_ratings: None,
        }
    }

    /// Load rating history from disk
    pub fn load(run_dir: &Path) -> Result<Self> {
        let mut history = Self::new(run_dir);

        // Load metadata first (contains checkpoint steps)
        if history.metadata_path.exists() {
            let metadata_str = fs::read_to_string(&history.metadata_path)
                .context("Failed to read rating_metadata.json")?;
            let metadata: RatingMetadata = serde_json::from_str(&metadata_str)
                .context("Failed to parse rating_metadata.json")?;

            // Restore checkpoint steps
            for (name, step) in metadata.checkpoint_steps {
                history.ensure_checkpoint_registered(&name, step);
            }

            // Restore first checkpoint
            if let Some(ref first) = metadata.first_checkpoint {
                if let Some(&idx) = history.checkpoint_to_idx.get(first) {
                    history.first_checkpoint_idx = Some(idx);
                }
            }

            // Restore current checkpoint
            history.current_checkpoint = metadata.current_checkpoint;
        }

        // Load games
        if history.games_path.exists() {
            let file =
                File::open(&history.games_path).context("Failed to open rating_games.jsonl")?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line.context("Failed to read line from rating_games.jsonl")?;
                if line.trim().is_empty() {
                    continue;
                }
                let game: RatingGame = serde_json::from_str(&line)
                    .context("Failed to parse game from rating_games.jsonl")?;

                // Register checkpoints (with step 0 if not already known from metadata)
                history.ensure_checkpoint_registered(&game.current, 0);
                for opponent in &game.opponents {
                    history.ensure_checkpoint_registered(opponent, 0);
                }

                history.games.push(game);
            }
        }

        // Set first checkpoint if we have checkpoints but no first was set
        if history.first_checkpoint_idx.is_none() && !history.idx_to_checkpoint.is_empty() {
            history.first_checkpoint_idx = Some(0);
        }

        Ok(history)
    }

    /// Save metadata to disk
    fn save_metadata(&self) -> Result<()> {
        let metadata = RatingMetadata {
            checkpoint_steps: self
                .idx_to_checkpoint
                .iter()
                .zip(self.idx_to_step.iter())
                .map(|(name, &step)| (name.clone(), step))
                .collect(),
            first_checkpoint: self
                .first_checkpoint_idx
                .and_then(|idx| self.idx_to_checkpoint.get(idx).cloned()),
            current_checkpoint: self.current_checkpoint.clone(),
        };

        let json =
            serde_json::to_string_pretty(&metadata).context("Failed to serialize metadata")?;

        // Atomic write via temp file
        let temp_path = self.run_dir.join(".rating_metadata.json.tmp");
        fs::write(&temp_path, &json).context("Failed to write temp metadata file")?;
        fs::rename(&temp_path, &self.metadata_path).context("Failed to rename metadata file")?;

        Ok(())
    }

    /// Ensure a checkpoint is registered in the index mappings
    fn ensure_checkpoint_registered(&mut self, checkpoint_name: &str, step: usize) -> usize {
        if let Some(&idx) = self.checkpoint_to_idx.get(checkpoint_name) {
            // Update step if provided
            if step > 0 && self.idx_to_step[idx] == 0 {
                self.idx_to_step[idx] = step;
            }
            return idx;
        }

        let idx = self.idx_to_checkpoint.len();
        self.checkpoint_to_idx
            .insert(checkpoint_name.to_string(), idx);
        self.idx_to_checkpoint.push(checkpoint_name.to_string());
        self.idx_to_step.push(step);
        idx
    }

    /// Record a game result (called after each episode vs pool opponent)
    pub fn record_game(&mut self, current: &str, opponents: &[String], placements: Vec<usize>) {
        // Ensure all participants are registered
        self.ensure_checkpoint_registered(current, 0);
        for opponent in opponents {
            self.ensure_checkpoint_registered(opponent, 0);
        }

        let game = RatingGame {
            current: current.to_string(),
            opponents: opponents.to_vec(),
            placements,
        };

        self.games.push(game.clone());
        self.cached_ratings = None; // Invalidate cache

        // Persist immediately (append)
        if let Err(e) = self.append_game(&game) {
            eprintln!("Warning: Failed to persist rating game: {e}");
        }
    }

    /// Append a single game to the storage file
    fn append_game(&self, game: &RatingGame) -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.games_path)
            .context("Failed to open rating_games.jsonl for append")?;

        let line = serde_json::to_string(game).context("Failed to serialize game")?;
        writeln!(file, "{line}").context("Failed to write game to file")?;

        Ok(())
    }

    /// Called when a new checkpoint is saved
    pub fn on_checkpoint_saved(&mut self, checkpoint_name: &str, step: usize) {
        let idx = self.ensure_checkpoint_registered(checkpoint_name, step);

        // Set as first checkpoint if this is the first one
        if self.first_checkpoint_idx.is_none() {
            self.first_checkpoint_idx = Some(idx);
        }

        // Update step
        self.idx_to_step[idx] = step;

        // Set as current checkpoint for future games
        self.current_checkpoint = Some(checkpoint_name.to_string());

        // Invalidate cache since checkpoint list changed
        self.cached_ratings = None;

        // Persist metadata
        if let Err(e) = self.save_metadata() {
            eprintln!("Warning: Failed to save rating metadata: {e}");
        }
    }

    /// Get the current checkpoint name (for recording games)
    pub fn current_checkpoint(&self) -> Option<&str> {
        self.current_checkpoint.as_deref()
    }

    /// Set the current checkpoint (used when resuming)
    #[expect(dead_code, reason = "API for future use when resuming training")]
    pub fn set_current_checkpoint(&mut self, checkpoint_name: &str, step: usize) {
        self.ensure_checkpoint_registered(checkpoint_name, step);
        self.current_checkpoint = Some(checkpoint_name.to_string());
    }

    /// Compute all ratings
    pub fn compute_ratings(&mut self) -> RatingResult {
        let num_checkpoints = self.idx_to_checkpoint.len();

        // Handle edge cases
        if num_checkpoints == 0 || self.games.is_empty() {
            self.cached_ratings = Some(vec![]);
            return RatingResult {
                current_elo: 1000.0,
                best_elo: 1000.0,
                best_step: 0,
                total_games: 0,
            };
        }

        // Convert to Plackett-Luce format
        let pl_games: Vec<GameResult> = self
            .games
            .iter()
            .map(|game| {
                let current_idx = self.checkpoint_to_idx[&game.current];
                let mut players = vec![current_idx];
                let opponent_indices: Vec<usize> = game
                    .opponents
                    .iter()
                    .map(|o| self.checkpoint_to_idx[o])
                    .collect();
                players.extend(opponent_indices);
                GameResult::new(players, game.placements.clone())
            })
            .collect();

        // Compute ratings
        let config = PlackettLuceConfig::default();
        let result = plackett_luce::compute_ratings(num_checkpoints, &pl_games, &config);

        // Extract raw ratings
        let raw_ratings: Vec<f64> = result.ratings.iter().map(|r| r.rating).collect();

        // Anchor first checkpoint to 1000
        let first_idx = self.first_checkpoint_idx.unwrap_or(0);
        let first_raw = raw_ratings.get(first_idx).copied().unwrap_or(1000.0);
        let shift = 1000.0 - first_raw;

        // Apply shift and find best
        let adjusted_ratings: Vec<f64> = raw_ratings.iter().map(|r| r + shift).collect();

        let mut best_idx = 0;
        let mut best_rating = f64::NEG_INFINITY;
        for (idx, &rating) in adjusted_ratings.iter().enumerate() {
            if rating > best_rating {
                best_rating = rating;
                best_idx = idx;
            }
        }

        // Current = second-to-last checkpoint (the one that has played games)
        // The latest checkpoint was just created and hasn't played rating games yet,
        // so we report the previous checkpoint's rating instead
        let current_idx = num_checkpoints.saturating_sub(2);
        let current_rating = adjusted_ratings.get(current_idx).copied().unwrap_or(1000.0);

        // Cache ratings
        self.cached_ratings = Some(adjusted_ratings);

        RatingResult {
            current_elo: current_rating,
            best_elo: best_rating,
            best_step: self.idx_to_step.get(best_idx).copied().unwrap_or(0),
            total_games: self.games.len(),
        }
    }

    /// Get checkpoint step for an index
    #[expect(dead_code, reason = "API for potential future use")]
    pub fn get_step_for_idx(&self, idx: usize) -> usize {
        self.idx_to_step.get(idx).copied().unwrap_or(0)
    }

    /// Generate PNG graph of Elo over time
    pub fn generate_graph(&mut self, output_path: &Path) -> Result<()> {
        // Ensure ratings are computed
        if self.cached_ratings.is_none() {
            self.compute_ratings();
        }

        let ratings = self
            .cached_ratings
            .as_ref()
            .expect("cached_ratings should be set after compute_ratings()");

        if ratings.is_empty() {
            return Ok(()); // Nothing to plot
        }

        // Collect (step, elo) pairs, sorted by step
        let mut data: Vec<(usize, f64)> = self
            .idx_to_step
            .iter()
            .zip(ratings.iter())
            .map(|(&step, &elo)| (step, elo))
            .filter(|(step, _)| *step > 0) // Only include checkpoints with known steps
            .collect();

        data.sort_by_key(|(step, _)| *step);

        // Skip the latest checkpoint - it was just created and hasn't played
        // rating games yet, so its Elo is not meaningful
        if data.len() > 1 {
            data.pop();
        }

        if data.is_empty() {
            return Ok(());
        }

        // Find best checkpoint for marking
        let (best_step, best_elo) = data
            .iter()
            .copied()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 1000.0));

        // Compute axis ranges
        let max_step = data.iter().map(|(s, _)| *s).max().unwrap_or(1);
        let min_elo = data.iter().map(|(_, e)| *e).fold(f64::INFINITY, f64::min) - 50.0;
        let max_elo = data
            .iter()
            .map(|(_, e)| *e)
            .fold(f64::NEG_INFINITY, f64::max)
            + 50.0;

        // Create the plot
        let root = BitMapBackend::new(output_path, (800, 400)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Training Progress (Elo)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0usize..max_step, min_elo..max_elo)?;

        chart
            .configure_mesh()
            .x_desc("Training Step")
            .y_desc("Elo Rating")
            .draw()?;

        // Draw the line
        chart.draw_series(LineSeries::new(data.clone(), &BLUE))?;

        // Draw points for each checkpoint
        chart.draw_series(
            data.iter()
                .map(|&(x, y)| Circle::new((x, y), 3, BLUE.filled())),
        )?;

        // Mark current best with a red point
        chart.draw_series(std::iter::once(Circle::new(
            (best_step, best_elo),
            6,
            RED.filled(),
        )))?;

        // Add legend for best
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .draw()?;

        root.present()?;

        Ok(())
    }

    /// Get number of games recorded
    pub fn num_games(&self) -> usize {
        self.games.len()
    }

    /// Get number of checkpoints
    pub fn num_checkpoints(&self) -> usize {
        self.idx_to_checkpoint.len()
    }

    /// Check if we have any games
    #[expect(dead_code, reason = "API for potential future use")]
    pub fn has_games(&self) -> bool {
        !self.games.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_new_rating_history() {
        let dir = tempdir().unwrap();
        let history = RatingHistory::new(dir.path());
        assert_eq!(history.num_games(), 0);
        assert_eq!(history.num_checkpoints(), 0);
    }

    #[test]
    fn test_record_game() {
        let dir = tempdir().unwrap();
        let mut history = RatingHistory::new(dir.path());

        history.on_checkpoint_saved("step_00000000", 0);
        history.on_checkpoint_saved("step_00010000", 10000);

        history.record_game(
            "step_00010000",
            &["step_00000000".to_string()],
            vec![1, 2], // Current won
        );

        assert_eq!(history.num_games(), 1);
        assert_eq!(history.num_checkpoints(), 2);
    }

    #[test]
    fn test_compute_ratings_simple() {
        let dir = tempdir().unwrap();
        let mut history = RatingHistory::new(dir.path());

        // Create 3 checkpoints: 0, 10000, 20000
        // current_elo reports second-to-last (10000) since latest (20000) has no games yet
        history.on_checkpoint_saved("step_00000000", 0);
        history.on_checkpoint_saved("step_00010000", 10000);
        history.on_checkpoint_saved("step_00020000", 20000);

        // Checkpoint 10000 beats checkpoint 0 consistently
        for _ in 0..10 {
            history.record_game("step_00010000", &["step_00000000".to_string()], vec![1, 2]);
        }

        let result = history.compute_ratings();

        // First checkpoint should be anchored at 1000
        // current_elo reports second-to-last checkpoint (step 10000) which won all games
        assert!(result.current_elo > 1000.0);
        assert_eq!(result.best_step, 10000);
        assert_eq!(result.total_games, 10);
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();

        // Create and populate history
        {
            let mut history = RatingHistory::new(dir.path());
            history.on_checkpoint_saved("step_00000000", 0);
            history.record_game("step_00000000", &["step_00000000".to_string()], vec![1, 2]);
        }

        // Load and verify
        let history = RatingHistory::load(dir.path()).unwrap();
        assert_eq!(history.num_games(), 1);
    }
}
