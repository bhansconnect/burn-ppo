//! Evaluation subcommand for assessing trained models
//!
//! Features:
//! - Watch mode: visualize games with ASCII rendering
//! - Stats mode: parallel game execution with win/loss/draw statistics
//! - Temperature-based sampling with schedule support
//! - ELO delta calculation for 2-player games

// Evaluation uses unwrap/expect for:
// - Tensor data extraction (cannot fail with correct shapes)
// - Internal data structure invariants

use std::io::{self, Write as IoWrite};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::checkpoint::{load_normalizer, CheckpointManager, CheckpointMetadata};
use crate::config::{Config, EvalArgs};
use crate::dispatch_env;
use crate::env::{Environment, GameOutcome, VecEnv};
use crate::human::{prompt_human_action, random_valid_action};
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;
use crate::profile::profile_function;

/// Source of actions for a player slot.
///
/// Used to configure each player in an evaluation session.
#[derive(Clone, Debug)]
pub enum PlayerSource {
    /// Network loaded from checkpoint path
    Checkpoint(std::path::PathBuf),
    /// Human player with interactive terminal input
    Human { name: String },
    /// Random valid actions (useful for baseline comparisons)
    Random,
}

impl PlayerSource {
    /// Returns true if this source requires terminal interaction
    pub const fn is_human(&self) -> bool {
        matches!(self, Self::Human { .. })
    }

    /// Display name for this player source
    pub fn display_name(&self) -> String {
        match self {
            Self::Checkpoint(path) => {
                // Extract meaningful name from path
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("checkpoint")
                    .to_string()
            }
            Self::Human { name } => name.clone(),
            Self::Random => "Random".to_string(),
        }
    }
}

/// Temperature schedule for action sampling
///
/// Supports:
/// - Constant temperature (cutoff = None): use initial temperature for all moves
/// - Hard cutoff: initial temperature until move N, then final
/// - Decay: linear interpolation from initial to final over N moves
#[derive(Debug, Clone)]
pub struct TempSchedule {
    initial: f32,
    final_temp: f32,
    cutoff: Option<usize>,
    decay: bool,
}

impl TempSchedule {
    pub const fn new(initial: f32, final_temp: f32, cutoff: Option<usize>, decay: bool) -> Self {
        Self {
            initial,
            final_temp,
            cutoff,
            decay,
        }
    }

    /// Create a `TempSchedule` from CLI arguments
    ///
    /// # Errors
    /// Returns an error if --temp-final or --temp-decay is used without --temp-cutoff
    pub fn from_args(args: &EvalArgs) -> Result<Self> {
        // Validate: temp_final and temp_decay require temp_cutoff
        if args.temp_cutoff.is_none() {
            if args.temp_final.is_some() {
                anyhow::bail!("--temp-final requires --temp-cutoff to be set");
            }
            if args.temp_decay {
                anyhow::bail!("--temp-decay requires --temp-cutoff to be set");
            }
        }

        Ok(Self::new(
            args.temperature,
            args.temp_final.unwrap_or(0.0),
            args.temp_cutoff,
            args.temp_decay,
        ))
    }

    /// Get temperature for a given move number
    pub fn get_temp(&self, move_num: usize) -> f32 {
        let Some(cutoff) = self.cutoff else {
            return self.initial; // No cutoff = always use initial temp
        };

        if move_num >= cutoff {
            return self.final_temp;
        }

        if !self.decay {
            return self.initial; // Hard cutoff: initial until cutoff
        }

        // Linear decay: initial → final over cutoff moves
        let t = move_num as f32 / cutoff as f32;
        t.mul_add(self.final_temp - self.initial, self.initial)
    }
}

/// Sample an action given logits and temperature
///
/// - temp = 0.0: deterministic argmax
/// - temp > 0.0: softmax(logits / temp) sampling
pub fn sample_with_temperature(
    logits: &[f32],
    mask: Option<&[bool]>,
    temperature: f32,
    rng: &mut impl Rng,
) -> usize {
    // Apply mask by setting invalid actions to very negative logits
    let masked_logits: Vec<f32> = if let Some(mask) = mask {
        logits
            .iter()
            .zip(mask.iter())
            .map(|(&l, &valid)| if valid { l } else { f32::NEG_INFINITY })
            .collect()
    } else {
        logits.to_vec()
    };

    if temperature == 0.0 {
        // Deterministic: argmax
        return masked_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
    }

    // Apply temperature: softmax(logits / temp)
    let scaled: Vec<f32> = masked_logits.iter().map(|x| x / temperature).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();

    if sum == 0.0 {
        // All actions masked or numerical issues, fall back to first valid
        return mask.and_then(|m| m.iter().position(|&v| v)).unwrap_or(0);
    }

    let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

    // Sample from distribution
    let mut cumulative = 0.0;
    let r: f32 = rng.gen();
    for (i, p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

/// Calculate ELO delta between two players based on match results
///
/// Returns (delta, stderr) where positive delta means p0 is stronger.
/// Standard error is computed using the delta method to propagate
/// uncertainty in the score proportion through the ELO formula.
pub fn elo_delta_2p(p0_wins: usize, p1_wins: usize, draws: usize) -> (f64, f64) {
    let n = (p0_wins + p1_wins + draws) as f64;
    if n == 0.0 {
        return (0.0, f64::INFINITY);
    }

    let score = 0.5f64.mul_add(draws as f64, p0_wins as f64) / n; // P0's score [0, 1]

    // ELO formula: delta = -400 * log10(1/score - 1)
    // Derived from: P(win) = 1 / (1 + 10^(-delta/400))
    let delta = if score <= 0.0 {
        f64::NEG_INFINITY
    } else if score >= 1.0 {
        f64::INFINITY
    } else {
        -400.0 * ((1.0 / score) - 1.0).log10()
    };

    // Standard error using delta method:
    // SE(score) = sqrt(score * (1 - score) / n)
    // d(delta)/d(score) = 400 / (ln(10) * score * (1 - score))
    // SE(delta) = |d(delta)/d(score)| * SE(score)
    //           = 400 / (ln(10) * sqrt(n * score * (1 - score)))
    let stderr = if score > 0.001 && score < 0.999 {
        let pq = score * (1.0 - score);
        400.0 / (std::f64::consts::LN_10 * (n * pq).sqrt())
    } else {
        // At extreme win rates, the standard error is very large/undefined
        f64::INFINITY
    };

    (delta, stderr)
}

/// Convert rewards to placements (1 = first, 2 = second, etc.)
/// Handles ties by giving the same placement to tied players
pub fn rewards_to_placements(rewards: &[f32]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = rewards.iter().copied().enumerate().collect();
    // Sort by reward descending
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut placements = vec![0; rewards.len()];
    let mut current_placement = 1;
    let mut i = 0;

    while i < indexed.len() {
        let current_reward = indexed[i].1;
        let mut tied_count = 0;

        // Count all players tied at this reward level
        while i + tied_count < indexed.len()
            && (indexed[i + tied_count].1 - current_reward).abs() < 1e-6
        {
            tied_count += 1;
        }

        // Assign same placement to all tied players
        for j in 0..tied_count {
            placements[indexed[i + j].0] = current_placement;
        }

        current_placement += tied_count;
        i += tied_count;
    }

    placements
}

/// Determine game outcome from environment and rewards
pub fn determine_outcome<E: Environment>(env: &E, total_rewards: &[f32]) -> GameOutcome {
    // 1. Check explicit outcome first
    if let Some(outcome) = env.game_outcome() {
        return outcome;
    }

    // 2. Infer from total_rewards
    let placements = rewards_to_placements(total_rewards);
    let first_place_count = placements.iter().filter(|&&p| p == 1).count();

    if first_place_count == placements.len() {
        GameOutcome::Tie
    } else if first_place_count == 1 {
        let winner = placements
            .iter()
            .position(|&p| p == 1)
            .expect("first_place_count == 1 guarantees a winner");
        GameOutcome::Winner(winner)
    } else {
        GameOutcome::Placements(placements)
    }
}

/// Statistics tracker for evaluation results
pub struct EvalStats {
    num_players: usize,
    /// Wins per checkpoint [`num_checkpoints`]
    wins: Vec<usize>,
    /// Losses per checkpoint [`num_checkpoints`]
    losses: Vec<usize>,
    /// Draws count (all tied)
    draws: usize,
    /// Placement counts per checkpoint [`num_checkpoints`][num_placements]
    placements: Vec<Vec<usize>>,
    /// Total games played
    total_games: usize,
    /// Total rewards per game [`game_idx`][player]
    game_rewards: Vec<Vec<f64>>,
    /// Episode lengths
    episode_lengths: Vec<usize>,
}

impl EvalStats {
    pub fn new(num_players: usize) -> Self {
        Self {
            num_players,
            wins: vec![0; num_players],
            losses: vec![0; num_players],
            draws: 0,
            placements: vec![vec![0; num_players]; num_players],
            total_games: 0,
            game_rewards: Vec::new(),
            episode_lengths: Vec::new(),
        }
    }

    /// Record a game outcome with optional rewards and length
    pub fn record_with_rewards(&mut self, outcome: &GameOutcome, rewards: &[f32], length: usize) {
        self.game_rewards
            .push(rewards.iter().map(|&r| f64::from(r)).collect());
        self.episode_lengths.push(length);
        self.record(outcome);
    }

    /// Record a game outcome (backward-compatible)
    pub fn record(&mut self, outcome: &GameOutcome) {
        self.total_games += 1;

        match outcome {
            GameOutcome::Winner(winner) => {
                self.wins[*winner] += 1;
                for i in 0..self.num_players {
                    if i != *winner {
                        self.losses[i] += 1;
                    }
                }
                // Record placements: winner gets 1st, others share 2nd
                self.placements[*winner][0] += 1;
                for i in 0..self.num_players {
                    if i != *winner {
                        self.placements[i][1] += 1;
                    }
                }
            }
            GameOutcome::Tie => {
                self.draws += 1;
                // All players get 1st place in a tie
                for i in 0..self.num_players {
                    self.placements[i][0] += 1;
                }
            }
            GameOutcome::Placements(places) => {
                for (i, &place) in places.iter().enumerate() {
                    if place > 0 && place <= self.num_players {
                        self.placements[i][place - 1] += 1;
                    }
                    // Win if got 1st place alone
                    let first_count = places.iter().filter(|&&p| p == 1).count();
                    if place == 1 && first_count == 1 {
                        self.wins[i] += 1;
                    } else if place == self.num_players {
                        self.losses[i] += 1;
                    }
                }
                // Check for tie (all got 1st)
                if places.iter().all(|&p| p == 1) {
                    self.draws += 1;
                }
            }
        }
    }

    /// Print statistics summary
    pub fn print_summary(&self, checkpoint_names: &[String]) {
        println!("\n=== Evaluation Results ({}-player) ===", self.num_players);
        println!("Total games: {}\n", self.total_games);

        if self.num_players == 1 {
            // Single-player: show full reward distribution
            self.print_single_player_summary(checkpoint_names);
        } else if self.num_players == 2 {
            // 2-player format with win/loss/draw and ELO
            self.print_two_player_summary(checkpoint_names);
        } else {
            // N-player format with placement percentages
            self.print_multi_player_summary(checkpoint_names);
        }

        // Print average rewards if available
        self.print_reward_summary();
    }

    fn print_single_player_summary(&self, checkpoint_names: &[String]) {
        if self.game_rewards.is_empty() {
            println!("No reward data collected.");
            return;
        }

        // Collect rewards for player 0 (only player in single-player)
        let mut rewards: Vec<f64> = self
            .game_rewards
            .iter()
            .filter_map(|r| r.first().copied())
            .collect();

        if rewards.is_empty() {
            return;
        }

        rewards.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = rewards.len() as f64;
        let mean = rewards.iter().sum::<f64>() / n;
        let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let min = rewards.first().copied().unwrap_or(0.0);
        let max = rewards.last().copied().unwrap_or(0.0);
        let p25 = rewards[(rewards.len() * 25 / 100).min(rewards.len() - 1)];
        let p50 = rewards[(rewards.len() * 50 / 100).min(rewards.len() - 1)];
        let p75 = rewards[(rewards.len() * 75 / 100).min(rewards.len() - 1)];

        println!(
            "Checkpoint: {}",
            checkpoint_names.first().unwrap_or(&String::new())
        );
        println!();
        println!("Reward Distribution:");
        println!("  Mean:   {mean:.1} ± {std:.1} (std)");
        println!("  Range:  [{min:.1}, {max:.1}]");
        println!("  25th:   {p25:.1}");
        println!("  Median: {p50:.1}");
        println!("  75th:   {p75:.1}");

        // Episode length stats if available
        if !self.episode_lengths.is_empty() {
            let avg_len: f64 = self.episode_lengths.iter().sum::<usize>() as f64
                / self.episode_lengths.len() as f64;
            let min_len = self.episode_lengths.iter().min().copied().unwrap_or(0);
            let max_len = self.episode_lengths.iter().max().copied().unwrap_or(0);
            println!();
            println!("Episode Length:");
            println!("  Mean:  {avg_len:.1}");
            println!("  Range: [{min_len}, {max_len}]");
        }
    }

    fn print_two_player_summary(&self, checkpoint_names: &[String]) {
        for (i, name) in checkpoint_names.iter().enumerate() {
            let win_pct = 100.0 * self.wins[i] as f64 / self.total_games as f64;
            let loss_pct = 100.0 * self.losses[i] as f64 / self.total_games as f64;
            let draw_pct = 100.0 * self.draws as f64 / self.total_games as f64;
            println!(
                "Checkpoint {} ({}): {} wins ({:.1}%), {} losses ({:.1}%), {} draws ({:.1}%)",
                i, name, self.wins[i], win_pct, self.losses[i], loss_pct, self.draws, draw_pct
            );
        }

        // ELO delta
        let (delta, stderr) = elo_delta_2p(self.wins[0], self.wins[1], self.draws);
        if delta.is_finite() {
            let stronger = if delta > 0.0 {
                "checkpoint 0 stronger"
            } else {
                "checkpoint 1 stronger"
            };
            println!("\nELO Delta: {delta:+.0} ± {stderr:.0} ({stronger})");
        }
    }

    fn print_multi_player_summary(&self, checkpoint_names: &[String]) {
        for (i, name) in checkpoint_names.iter().enumerate() {
            let placement_pcts: Vec<String> = self.placements[i]
                .iter()
                .enumerate()
                .map(|(p, &count)| {
                    let pct = 100.0 * count as f64 / self.total_games as f64;
                    format!("{:.0}% {}", pct, ordinal(p + 1))
                })
                .collect();

            let avg_placement: f64 = self.placements[i]
                .iter()
                .enumerate()
                .map(|(p, &count)| (p + 1) as f64 * count as f64)
                .sum::<f64>()
                / self.total_games as f64;

            println!(
                "Checkpoint {}: {} (avg placement: {:.2})",
                i,
                placement_pcts.join(", "),
                avg_placement
            );
            println!("  {name}");
        }
    }

    fn print_reward_summary(&self) {
        if self.game_rewards.is_empty() || self.num_players == 1 {
            return; // Already handled in single-player summary
        }

        println!();
        println!("Average Rewards per Player:");
        for player in 0..self.num_players {
            let rewards: Vec<f64> = self
                .game_rewards
                .iter()
                .filter_map(|r| r.get(player).copied())
                .collect();

            if !rewards.is_empty() {
                let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
                println!("  Player {player}: {mean:.3}");
            }
        }
    }
}

/// Result of challenger evaluation between current model and best checkpoint
#[derive(Debug, Clone)]
pub struct ChallengerResult {
    /// Wins for the current (challenger) model
    pub current_wins: usize,
    /// Wins for the best checkpoint model
    pub best_wins: usize,
    /// Draw count
    pub draws: usize,
    /// Win rate for current model (0.0 - 1.0)
    pub win_rate: f64,
    /// Whether current model should be promoted to best
    pub should_promote: bool,
    /// Time taken for evaluation in milliseconds
    pub elapsed_ms: u64,
}

/// Get ordinal suffix (1st, 2nd, 3rd, 4th, ...)
fn ordinal(n: usize) -> String {
    let suffix = match n % 10 {
        1 if n % 100 != 11 => "st",
        2 if n % 100 != 12 => "nd",
        3 if n % 100 != 13 => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

/// Load a model from a checkpoint directory
///
/// Returns the model, metadata, and optional normalizer (if the checkpoint was trained with
/// observation normalization enabled).
fn load_model_from_checkpoint<B: Backend>(
    checkpoint_path: &Path,
    device: &B::Device,
) -> Result<(ActorCritic<B>, CheckpointMetadata, Option<ObsNormalizer>)> {
    // Load metadata to get network dimensions
    let metadata_path = checkpoint_path.join("metadata.json");
    let metadata_json =
        std::fs::read_to_string(&metadata_path).context("Failed to read checkpoint metadata")?;
    let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

    // Create config with network architecture from metadata
    let config = Config {
        hidden_size: metadata.hidden_size,
        num_hidden: metadata.num_hidden,
        ..Config::default()
    };

    // Load model
    let (model, _) = CheckpointManager::load::<B>(checkpoint_path, &config, device)?;

    // Load normalizer if it exists (trained with normalize_obs=true)
    let normalizer = load_normalizer(checkpoint_path)?;

    Ok((model, metadata, normalizer))
}

/// Run challenger evaluation: current model vs best checkpoint
///
/// Plays `num_games` games between the current model and the best checkpoint,
/// with position swapping for fairness in 2-player games.
///
/// Returns a `ChallengerResult` indicating whether the current model should be promoted.
pub fn run_challenger_eval<B: Backend, E: Environment>(
    current_model: &ActorCritic<B>,
    current_normalizer: Option<&ObsNormalizer>,
    best_checkpoint_path: &Path,
    num_games: usize,
    threshold: f64,
    config: &Config,
    device: &B::Device,
    seed: u64,
) -> Result<ChallengerResult> {
    profile_function!();
    let start = Instant::now();
    let num_players = E::NUM_PLAYERS;

    // Load best checkpoint model and normalizer
    let best_config = Config {
        hidden_size: config.hidden_size,
        num_hidden: config.num_hidden,
        ..Config::default()
    };
    let (best_model, _) = CheckpointManager::load::<B>(best_checkpoint_path, &best_config, device)?;
    let best_normalizer = load_normalizer(best_checkpoint_path)?;

    // Setup models and checkpoint mapping
    // models = [new, best], checkpoint_to_model = [0, 1, 1, 1, ...]
    // This means: checkpoint 0 = new model, checkpoints 1..N-1 = best model
    let models = vec![current_model.clone(), best_model];
    let normalizers = vec![current_normalizer.cloned(), best_normalizer];
    let checkpoint_to_model: Vec<usize> = (0..num_players).map(|i| usize::from(i != 0)).collect();
    let names: Vec<String> = (0..num_players)
        .map(|i| if i == 0 { "new".into() } else { "best".into() })
        .collect();

    // Create temperature schedule from config
    let temp_schedule = TempSchedule::new(
        config.challenger_temperature,
        config.challenger_temp_final.unwrap_or(0.0),
        config.challenger_temp_cutoff,
        config.challenger_temp_decay,
    );

    // Run games - stats mode handles position permutations
    let num_envs = num_games.min(64);
    let mut rng = StdRng::seed_from_u64(seed);
    let stats = run_stats_mode_env::<B, E>(
        &models,
        &normalizers,
        &checkpoint_to_model,
        &names,
        num_games,
        num_envs,
        &temp_schedule,
        &mut rng,
        device,
        true, // silent
    );

    // Extract stats for checkpoint 0 (the "new" model)
    let current_wins = stats.wins[0];
    let best_wins = stats.losses[0];
    let draws = stats.draws;

    // Tie value = 1/N for N-player games
    let tie_value = 1.0 / num_players as f64;
    let total = (current_wins + best_wins + draws) as f64;
    let win_rate = if total > 0.0 {
        (current_wins as f64 + tie_value * draws as f64) / total
    } else {
        tie_value // Expected value if no games played
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(ChallengerResult {
        current_wins,
        best_wins,
        draws,
        win_rate,
        should_promote: win_rate > threshold,
        elapsed_ms,
    })
}

/// Run evaluation based on command-line arguments
pub fn run_evaluation<B: Backend>(args: &EvalArgs, device: &B::Device) -> Result<()> {
    use std::collections::HashMap;

    // Check if we have human or random players
    let has_human = !args.humans.is_empty();
    let has_random = args.random;
    let has_mixed_players = has_human || has_random;

    // Build player sources list
    let mut player_sources: Vec<PlayerSource> = Vec::new();

    // Add checkpoints as player sources
    for path in &args.checkpoints {
        player_sources.push(PlayerSource::Checkpoint(path.clone()));
    }

    // Add human players
    for name in &args.humans {
        player_sources.push(PlayerSource::Human { name: name.clone() });
    }

    // Add random player if requested
    if has_random {
        player_sources.push(PlayerSource::Random);
    }

    // If we have human/random players, use interactive mode
    if has_mixed_players {
        return run_interactive_evaluation::<B>(args, &player_sources, device);
    }

    // Original checkpoint-only flow below
    let mut models: Vec<ActorCritic<B>> = Vec::new();
    let mut normalizers: Vec<Option<ObsNormalizer>> = Vec::new();
    let mut path_to_model_idx: HashMap<std::path::PathBuf, usize> = HashMap::new();
    let mut checkpoint_to_model: Vec<usize> = Vec::new();
    let mut checkpoint_names = Vec::new();
    let mut metadata_opt: Option<CheckpointMetadata> = None;

    for path in &args.checkpoints {
        // Resolve symlinks (latest, best)
        let resolved = if path.is_symlink() {
            path.read_link().map_or_else(
                |_| path.clone(),
                |target| path.parent().unwrap_or(path).join(target),
            )
        } else {
            path.clone()
        };

        // Check if this path was already loaded (deduplication)
        let model_idx = if let Some(&idx) = path_to_model_idx.get(&resolved) {
            idx
        } else {
            let (model, metadata, normalizer) = load_model_from_checkpoint::<B>(&resolved, device)?;

            // Store first metadata for environment info
            if let Some(first) = &metadata_opt {
                // Verify all checkpoints are for the same environment
                if metadata.obs_dim != first.obs_dim
                    || metadata.action_count != first.action_count
                    || metadata.num_players != first.num_players
                {
                    anyhow::bail!(
                        "Checkpoint {} has different dimensions than first checkpoint",
                        path.display()
                    );
                }
            } else {
                metadata_opt = Some(metadata.clone());
            }

            let idx = models.len();
            models.push(model);
            normalizers.push(normalizer);
            path_to_model_idx.insert(resolved, idx);
            idx
        };

        checkpoint_to_model.push(model_idx);
        checkpoint_names.push(path.display().to_string());
    }

    let metadata = metadata_opt.context("No checkpoints provided")?;

    // Verify we have right number of checkpoints for environment
    if metadata.num_players > 1 && checkpoint_to_model.len() != metadata.num_players {
        if checkpoint_to_model.len() == 1 {
            // Self-play: all players use the same checkpoint (no reload needed)
            println!(
                "Self-play mode: using same checkpoint for all {} players",
                metadata.num_players
            );
            for _ in 1..metadata.num_players {
                checkpoint_to_model.push(0); // All point to first (and only) model
                checkpoint_names.push(checkpoint_names[0].clone());
            }
        } else {
            anyhow::bail!(
                "Expected {} checkpoints for {}-player game, got {}",
                metadata.num_players,
                metadata.num_players,
                checkpoint_to_model.len()
            );
        }
    }

    // Dispatch to watch or stats mode
    let watch = args.watch || args.step;
    let num_games = if watch && args.num_games == 100 {
        1 // Default to 1 game in watch mode
    } else {
        args.num_games
    };

    if watch {
        run_watch_mode::<B>(
            &models,
            &normalizers,
            &checkpoint_to_model,
            &checkpoint_names,
            &metadata,
            num_games,
            args,
            device,
        )
    } else {
        run_stats_mode::<B>(
            &models,
            &normalizers,
            &checkpoint_to_model,
            &checkpoint_names,
            &metadata,
            num_games,
            args,
            device,
        )
    }
}

/// Run interactive evaluation with mixed player sources (human, network, random)
fn run_interactive_evaluation<B: Backend>(
    args: &EvalArgs,
    player_sources: &[PlayerSource],
    device: &B::Device,
) -> Result<()> {
    use std::collections::HashMap;

    // Determine environment from first checkpoint or from --env arg
    let env_name = if let Some(first_checkpoint) = args.checkpoints.first() {
        let (_, metadata, _) = load_model_from_checkpoint::<B>(first_checkpoint, device)?;
        metadata.env_name
    } else if let Some(env) = &args.env_name {
        env.clone()
    } else {
        anyhow::bail!(
            "No checkpoint provided. Use --env to specify environment for human/random games.\n\
             Example: --env connect_four --human Alice --human Bob"
        );
    };

    // Load models and normalizers for checkpoint players
    let mut models: Vec<Option<ActorCritic<B>>> = Vec::new();
    let mut normalizers: Vec<Option<ObsNormalizer>> = Vec::new();
    let mut path_to_model: HashMap<std::path::PathBuf, (ActorCritic<B>, Option<ObsNormalizer>)> =
        HashMap::new();

    for source in player_sources {
        match source {
            PlayerSource::Checkpoint(path) => {
                // Resolve symlinks
                let resolved = if path.is_symlink() {
                    path.read_link().map_or_else(
                        |_| path.clone(),
                        |target| path.parent().unwrap_or(path).join(target),
                    )
                } else {
                    path.clone()
                };

                // Load model and normalizer (with deduplication)
                if !path_to_model.contains_key(&resolved) {
                    let (model, _, normalizer) =
                        load_model_from_checkpoint::<B>(&resolved, device)?;
                    path_to_model.insert(resolved.clone(), (model, normalizer));
                }
                let (model, normalizer) = path_to_model
                    .get(&resolved)
                    .expect("just inserted if missing");
                models.push(Some(model.clone()));
                normalizers.push(normalizer.clone());
            }
            PlayerSource::Human { .. } | PlayerSource::Random => {
                models.push(None);
                normalizers.push(None);
            }
        }
    }

    // Setup
    let seed = args.seed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed);
    let temp_schedule = TempSchedule::from_args(args)?;

    let num_games = if args.num_games == 100 {
        1
    } else {
        args.num_games
    };

    // Dispatch based on environment name
    dispatch_env!(env_name, {
        // Verify player count matches
        let expected_players = E::NUM_PLAYERS;
        if player_sources.len() != expected_players {
            anyhow::bail!(
                "{} requires {} players, but {} were specified.\n\
                 Use --checkpoint, --human, and --random to specify players.",
                E::NAME,
                expected_players,
                player_sources.len()
            );
        }

        run_interactive_game::<B, E>(
            player_sources,
            &models,
            &normalizers,
            num_games,
            &temp_schedule,
            &mut rng,
            device,
        );
        Ok(())
    })
}

/// Run evaluation in watch mode (sequential, with rendering)
fn run_watch_mode<B: Backend>(
    models: &[ActorCritic<B>],
    normalizers: &[Option<ObsNormalizer>],
    checkpoint_to_model: &[usize],
    checkpoint_names: &[String],
    metadata: &CheckpointMetadata,
    num_games: usize,
    args: &EvalArgs,
    device: &B::Device,
) -> Result<()> {
    let seed = args.seed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed);
    let temp_schedule = TempSchedule::from_args(args)?;

    // Dispatch based on env_name stored in checkpoint metadata
    crate::dispatch_env_ok!(metadata.env_name, {
        run_watch_mode_env::<B, E>(
            models,
            normalizers,
            checkpoint_to_model,
            checkpoint_names,
            num_games,
            &temp_schedule,
            args.step,
            args.animate,
            args.fps,
            &mut rng,
            device,
        );
    })
}

/// Watch mode implementation for a specific environment type
fn run_watch_mode_env<B: Backend, E: Environment>(
    models: &[ActorCritic<B>],
    normalizers: &[Option<ObsNormalizer>],
    checkpoint_to_model: &[usize],
    checkpoint_names: &[String],
    num_games: usize,
    temp_schedule: &TempSchedule,
    step_mode: bool,
    animate: bool,
    fps: u32,
    rng: &mut StdRng,
    device: &B::Device,
) {
    use rand::RngCore;
    let mut stats = EvalStats::new(E::NUM_PLAYERS);

    for game_idx in 0..num_games {
        println!("\n=== Game {} ===", game_idx + 1);
        for (i, name) in checkpoint_names.iter().enumerate() {
            println!("Player {i}: {name}");
        }
        println!();

        let mut env = E::new(rng.next_u64());
        let mut obs = env.reset();
        let mut total_rewards = vec![0.0f32; E::NUM_PLAYERS];
        let mut move_num = 0;
        let mut is_first_frame = true;
        let mut last_frame_lines = 0usize;
        let frame_duration = Duration::from_millis(1000 / u64::from(fps));

        loop {
            let frame_start = if animate { Some(Instant::now()) } else { None };
            let current_player = env.current_player();
            // In watch mode, checkpoint_idx == player_idx (no position swapping)
            let model_idx = checkpoint_to_model[current_player];
            let model = &models[model_idx];
            let normalizer = &normalizers[model_idx];
            let mask = env.action_mask();

            // Get action from model (normalize if checkpoint was trained with normalize_obs)
            let obs_for_model = if let Some(norm) = normalizer {
                norm.normalize(&obs)
            } else {
                obs.clone()
            };
            let obs_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(&obs_for_model[..], device)
                .reshape([1, E::OBSERVATION_DIM]);
            let (logits, _) = model.forward(obs_tensor);
            let logits_vec: Vec<f32> = logits
                .to_data()
                .to_vec()
                .expect("tensor data to vec conversion");

            let temp = temp_schedule.get_temp(move_num);
            let action = sample_with_temperature(&logits_vec, mask.as_deref(), temp, rng);

            // Render before step
            if let Some(rendered) = env.render() {
                if animate {
                    let frame_lines = rendered.lines().count();
                    // Move cursor up to overwrite previous frame
                    if !is_first_frame {
                        // +1 for the action line
                        print!("\x1b[{}A", last_frame_lines + 1);
                    }
                    println!("{rendered}");
                    // Trailing spaces clear any leftover characters from previous longer text
                    println!("Action: {action} (temp={temp:.2})                    ");
                    io::stdout().flush().ok();
                    last_frame_lines = frame_lines;
                    is_first_frame = false;
                    // Sleep only the remainder of frame time
                    if let Some(start) = frame_start {
                        let elapsed = start.elapsed();
                        if elapsed < frame_duration {
                            thread::sleep(frame_duration.saturating_sub(elapsed));
                        }
                    }
                } else {
                    println!("{rendered}");
                    println!("Player {current_player} selects action {action} (temp={temp:.2})");
                }
            } else {
                println!("Player {current_player} selects action {action} (temp={temp:.2})");
            }

            if step_mode {
                wait_for_enter();
            }

            // Step environment
            let (new_obs, rewards, done) = env.step(action);
            for (i, &r) in rewards.iter().enumerate() {
                total_rewards[i] += r;
            }
            obs = new_obs;
            move_num += 1;

            if done {
                // Show final state
                if let Some(rendered) = env.render() {
                    if animate {
                        // Overwrite the animation frame with final state
                        print!("\x1b[{}A", last_frame_lines + 1);
                        println!("{rendered}");
                        println!(); // Extra line to separate from outcome
                    } else {
                        println!("{rendered}");
                    }
                }

                let outcome = determine_outcome(&env, &total_rewards);
                stats.record(&outcome);

                match &outcome {
                    GameOutcome::Winner(w) => println!("\nWinner: Player {w}"),
                    GameOutcome::Tie => println!("\nGame ended in a tie"),
                    GameOutcome::Placements(p) => {
                        println!("\nFinal placements: {p:?}");
                    }
                }
                println!("Total rewards: {total_rewards:?}");
                println!("Game length: {move_num} moves");
                break;
            }
        }
    }

    if num_games > 1 {
        stats.print_summary(checkpoint_names);
    }
}

/// Wait for user to press Enter
fn wait_for_enter() {
    print!("Press Enter to continue...");
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok();
}

/// Run interactive game with mixed player sources (human, network, random).
///
/// This is used when any player is human or random.
/// Always runs in sequential mode (single env, one game at a time).
pub fn run_interactive_game<B: Backend, E: Environment>(
    player_sources: &[PlayerSource],
    models: &[Option<ActorCritic<B>>],
    normalizers: &[Option<ObsNormalizer>],
    num_games: usize,
    temp_schedule: &TempSchedule,
    rng: &mut StdRng,
    device: &B::Device,
) {
    use rand::RngCore;

    let num_players = E::NUM_PLAYERS;
    assert_eq!(
        player_sources.len(),
        num_players,
        "Expected {num_players} player sources for {num_players}-player game"
    );

    // Build display names for players
    let player_names: Vec<String> = player_sources
        .iter()
        .map(PlayerSource::display_name)
        .collect();

    let mut stats = EvalStats::new(num_players);

    for game_idx in 0..num_games {
        println!("\n=== Game {} of {} ===", game_idx + 1, num_games);
        for (i, name) in player_names.iter().enumerate() {
            let source_type = match &player_sources[i] {
                PlayerSource::Checkpoint(_) => "[Network]",
                PlayerSource::Human { .. } => "[Human]",
                PlayerSource::Random => "[Random]",
            };
            println!("Player {}: {} {}", i + 1, name, source_type);
        }
        println!("\nCommands: help (h), render (r), random, hint, quit (q)\n");

        let mut env = E::new(rng.next_u64());
        let obs = env.reset();
        let mut total_rewards = vec![0.0f32; num_players];
        let mut move_num = 0;

        // Store observation for network hints
        let mut current_obs = obs;

        loop {
            let current_player = env.current_player();
            let player_source = &player_sources[current_player];
            let mask = env.action_mask();

            // Render board before human turns (or always for visibility)
            let has_human = player_sources.iter().any(PlayerSource::is_human);
            if has_human {
                if let Some(rendered) = env.render() {
                    println!("{rendered}");
                }
            }

            // Get action based on player source
            let action = match player_source {
                PlayerSource::Human { name } => {
                    // Build hint closure from any available network
                    let hint: Option<Box<dyn Fn() -> usize>> = build_hint_closure::<B, E>(
                        models,
                        normalizers,
                        &current_obs,
                        mask.as_deref(),
                        device,
                    );

                    prompt_human_action(
                        &env,
                        current_player,
                        num_players,
                        name,
                        hint.as_ref().map(std::convert::AsRef::as_ref),
                        rng,
                    )
                }
                PlayerSource::Random => {
                    let action = random_valid_action(&env, rng);
                    println!(
                        "Player {} (Random) plays: {}",
                        current_player + 1,
                        env.describe_action(action)
                    );
                    action
                }
                PlayerSource::Checkpoint(_) => {
                    // Network action (normalize if checkpoint was trained with normalize_obs)
                    let model = models[current_player]
                        .as_ref()
                        .expect("Model should be loaded for checkpoint player");
                    let normalizer = &normalizers[current_player];

                    let obs_for_model = if let Some(norm) = normalizer {
                        norm.normalize(&current_obs)
                    } else {
                        current_obs.clone()
                    };
                    let obs_tensor: Tensor<B, 2> =
                        Tensor::<B, 1>::from_floats(&obs_for_model[..], device)
                            .reshape([1, E::OBSERVATION_DIM]);
                    let (logits, _) = model.forward(obs_tensor);
                    let logits_vec: Vec<f32> = logits
                        .to_data()
                        .to_vec()
                        .expect("tensor data to vec conversion");

                    let temp = temp_schedule.get_temp(move_num);
                    let action = sample_with_temperature(&logits_vec, mask.as_deref(), temp, rng);

                    println!(
                        "Player {} ({}) plays: {} (temp={:.2})",
                        current_player + 1,
                        player_names[current_player],
                        env.describe_action(action),
                        temp
                    );
                    action
                }
            };

            // Step environment
            let (new_obs, rewards, done) = env.step(action);
            for (i, &r) in rewards.iter().enumerate() {
                total_rewards[i] += r;
            }
            current_obs = new_obs;
            move_num += 1;

            if done {
                // Show final state
                if let Some(rendered) = env.render() {
                    println!("{rendered}");
                }

                let outcome = determine_outcome(&env, &total_rewards);
                stats.record(&outcome);

                match &outcome {
                    GameOutcome::Winner(w) => {
                        println!("\nWinner: Player {} ({})", w + 1, player_names[*w]);
                    }
                    GameOutcome::Tie => println!("\nGame ended in a tie"),
                    GameOutcome::Placements(p) => {
                        println!("\nFinal placements:");
                        for (i, &place) in p.iter().enumerate() {
                            println!("  {}: {} ({})", place, player_names[i], ordinal(place));
                        }
                    }
                }
                println!("Total rewards: {total_rewards:?}");
                println!("Game length: {move_num} moves");
                break;
            }
        }
    }

    if num_games > 1 {
        stats.print_summary(&player_names);
    }
}

/// Build a hint closure from any available network model
fn build_hint_closure<'a, B: Backend, E: Environment>(
    models: &'a [Option<ActorCritic<B>>],
    normalizers: &'a [Option<ObsNormalizer>],
    obs: &[f32],
    mask: Option<&[bool]>,
    device: &B::Device,
) -> Option<Box<dyn Fn() -> usize + 'a>> {
    // Find first available network and its normalizer
    let (model_idx, model) = models
        .iter()
        .enumerate()
        .find_map(|(i, m)| m.as_ref().map(|m| (i, m)))?;
    let normalizer = &normalizers[model_idx];

    // Compute action now (normalize if checkpoint was trained with normalize_obs)
    let obs_for_model = if let Some(norm) = normalizer {
        norm.normalize(obs)
    } else {
        obs.to_vec()
    };
    let obs_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(&obs_for_model[..], device).reshape([1, E::OBSERVATION_DIM]);
    let (logits, _) = model.forward(obs_tensor);
    let logits_vec: Vec<f32> = logits
        .to_data()
        .to_vec()
        .expect("tensor data to vec conversion");

    // Use temperature 0 for deterministic hint
    let mut temp_rng = StdRng::seed_from_u64(0);
    let action = sample_with_temperature(&logits_vec, mask, 0.0, &mut temp_rng);

    Some(Box::new(move || action))
}

/// Run evaluation in stats mode (parallel)
fn run_stats_mode<B: Backend>(
    models: &[ActorCritic<B>],
    normalizers: &[Option<ObsNormalizer>],
    checkpoint_to_model: &[usize],
    checkpoint_names: &[String],
    metadata: &CheckpointMetadata,
    num_games: usize,
    args: &EvalArgs,
    device: &B::Device,
) -> Result<()> {
    let seed = args.seed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed);
    let temp_schedule = TempSchedule::from_args(args)?;

    // Dispatch based on env_name stored in checkpoint metadata
    // Stats are printed inside run_stats_mode_env when not silent
    crate::dispatch_env_ok!(metadata.env_name, {
        run_stats_mode_env::<B, E>(
            models,
            normalizers,
            checkpoint_to_model,
            checkpoint_names,
            num_games,
            args.num_envs,
            &temp_schedule,
            &mut rng,
            device,
            false, // not silent - print progress and summary
        );
    })
}

/// Stats mode implementation for a specific environment type
///
/// When `silent` is true, suppresses output (used by challenger eval).
/// Returns `EvalStats` for further processing.
fn run_stats_mode_env<B: Backend, E: Environment>(
    models: &[ActorCritic<B>],
    normalizers: &[Option<ObsNormalizer>],
    checkpoint_to_model: &[usize],
    checkpoint_names: &[String],
    num_games: usize,
    num_envs: usize,
    temp_schedule: &TempSchedule,
    rng: &mut StdRng,
    device: &B::Device,
    silent: bool,
) -> EvalStats {
    use rand::RngCore;
    let num_players = E::NUM_PLAYERS;
    let num_checkpoints = checkpoint_to_model.len();
    let obs_dim = E::OBSERVATION_DIM;
    let action_count = E::ACTION_COUNT;

    // Create vectorized environment with unique seeds per env
    let base_seed = rng.next_u64();
    let mut vec_env: VecEnv<E> =
        VecEnv::new(num_envs, |i| E::new(base_seed.wrapping_add(i as u64)));
    let mut stats = EvalStats::new(num_checkpoints);

    // Track move counts per environment (for temperature schedule)
    let mut move_counts = vec![0usize; num_envs];

    // Track permutation offset per environment for position rotation
    // checkpoint i plays as player (i + perm_offset) % num_players
    // This rotates all checkpoints through all positions for fairness
    let mut perm_offsets: Vec<usize> = (0..num_envs).map(|env_idx| env_idx % num_players).collect();

    if !silent {
        println!("Running {num_games} games across {num_envs} parallel environments...");
    }

    let mut games_completed = 0;

    while games_completed < num_games {
        // Check if all envs are terminal
        if vec_env.active_count() == 0 {
            break;
        }

        let obs_flat = vec_env.get_observations();
        let current_players = vec_env.get_current_players();
        let masks_flat = vec_env.get_action_masks();
        let terminal_mask = vec_env.terminal_mask();

        // Build actions for each environment
        let mut actions = vec![0usize; num_envs];

        // Group environments by which model should act (based on position mapping)
        for (model_idx, model) in models.iter().enumerate() {
            // Find environments where this model's checkpoint is the current player
            let mut env_indices = Vec::new();
            let mut env_obs = Vec::new();
            let mut env_masks = Vec::new();

            for env_idx in 0..num_envs {
                // Skip terminal envs
                if terminal_mask[env_idx] {
                    continue;
                }

                let current_player = current_players[env_idx];
                // Which checkpoint controls this player position in this env?
                // checkpoint i plays as player (i + perm_offset) % num_players
                // So checkpoint_for_player = (player - perm_offset + num_players) % num_players
                let checkpoint_for_player =
                    (current_player + num_players - perm_offsets[env_idx]) % num_players;

                // Check if this checkpoint uses the current model (supports deduplication)
                if checkpoint_to_model[checkpoint_for_player] == model_idx {
                    env_indices.push(env_idx);

                    let offset = env_idx * obs_dim;
                    env_obs.extend_from_slice(&obs_flat[offset..offset + obs_dim]);

                    if let Some(ref masks) = masks_flat {
                        let mask_offset = env_idx * action_count;
                        env_masks
                            .extend_from_slice(&masks[mask_offset..mask_offset + action_count]);
                    }
                }
            }

            if env_indices.is_empty() {
                continue;
            }

            // Batch inference for this model (normalize if checkpoint was trained with normalize_obs)
            let batch_size = env_indices.len();
            let normalizer = &normalizers[model_idx];
            let obs_for_model = if let Some(norm) = normalizer {
                let mut obs_copy = env_obs.clone();
                norm.normalize_batch(&mut obs_copy, obs_dim);
                obs_copy
            } else {
                env_obs.clone()
            };
            let obs_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(&obs_for_model[..], device)
                .reshape([batch_size, obs_dim]);
            let (logits_tensor, _) = model.forward(obs_tensor.clone());
            let logits_flat: Vec<f32> = logits_tensor
                .to_data()
                .to_vec()
                .expect("tensor data to vec conversion");

            // Sample actions
            for (i, &env_idx) in env_indices.iter().enumerate() {
                let logit_offset = i * action_count;
                let logits = &logits_flat[logit_offset..logit_offset + action_count];

                let mask = if env_masks.is_empty() {
                    None
                } else {
                    let mask_offset = i * action_count;
                    Some(&env_masks[mask_offset..mask_offset + action_count])
                };

                let temp = temp_schedule.get_temp(move_counts[env_idx]);
                actions[env_idx] = sample_with_temperature(logits, mask, temp, rng);
            }
        }

        // Step all environments
        let (_, _, _, completed_episodes) = vec_env.step(&actions);

        // Update move counts
        for count in &mut move_counts {
            *count += 1;
        }

        // Process completed episodes (cap at num_games)
        let completed_env_indices: Vec<usize> =
            completed_episodes.iter().map(|ep| ep.env_index).collect();

        for ep in completed_episodes {
            // Only record up to num_games
            if games_completed >= num_games {
                continue;
            }

            games_completed += 1;

            // Use the env_index from the completed episode
            let env_idx = ep.env_index;
            let perm_offset = perm_offsets[env_idx];

            // Map player rewards to checkpoint rewards using permutation offset
            // checkpoint i was at player position (i + perm_offset) % num_players
            let checkpoint_rewards: Vec<f32> = (0..num_checkpoints)
                .map(|checkpoint_idx| {
                    let player_idx = (checkpoint_idx + perm_offset) % num_players;
                    ep.total_rewards[player_idx]
                })
                .collect();

            // Determine outcome from checkpoint rewards
            let outcome = {
                let placements = rewards_to_placements(&checkpoint_rewards);
                let first_count = placements.iter().filter(|&&p| p == 1).count();
                if first_count == placements.len() {
                    GameOutcome::Tie
                } else if first_count == 1 {
                    let winner = placements
                        .iter()
                        .position(|&p| p == 1)
                        .expect("first_count == 1 guarantees a winner");
                    GameOutcome::Winner(winner)
                } else {
                    GameOutcome::Placements(placements)
                }
            };

            stats.record_with_rewards(&outcome, &checkpoint_rewards, ep.length);

            // Reset move count and rotate to next permutation for fairness
            move_counts[env_idx] = 0;
            perm_offsets[env_idx] = (perm_offset + 1) % num_players;

            // Progress indicator
            if !silent && games_completed % 100 == 0 {
                println!("  {games_completed} / {num_games} games completed");
            }
        }

        // Freeze excess envs to hit exact target
        // This ensures slow games complete while hitting exact num_games
        let remaining = num_games.saturating_sub(games_completed);
        let active = vec_env.active_count();

        if active > remaining {
            let to_freeze = active - remaining;
            let mut frozen = 0;

            // Prefer freezing just-reset envs (preserves mid-game envs)
            for &env_idx in &completed_env_indices {
                if frozen >= to_freeze {
                    break;
                }
                if !vec_env.terminal_mask()[env_idx] {
                    vec_env.set_terminal(env_idx);
                    frozen += 1;
                }
            }

            // Fallback: freeze any remaining active
            for i in 0..num_envs {
                if frozen >= to_freeze {
                    break;
                }
                if !vec_env.terminal_mask()[i] {
                    vec_env.set_terminal(i);
                    frozen += 1;
                }
            }
        }
    }

    if !silent {
        stats.print_summary(checkpoint_names);
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temp_schedule_constant() {
        // No cutoff = use initial temp always
        let schedule = TempSchedule::new(0.5, 0.0, None, false);
        assert_eq!(schedule.get_temp(0), 0.5);
        assert_eq!(schedule.get_temp(100), 0.5);
    }

    #[test]
    fn test_temp_schedule_hard_cutoff() {
        let schedule = TempSchedule::new(1.0, 0.3, Some(10), false);
        assert_eq!(schedule.get_temp(0), 1.0);
        assert_eq!(schedule.get_temp(9), 1.0);
        assert_eq!(schedule.get_temp(10), 0.3);
        assert_eq!(schedule.get_temp(100), 0.3);
    }

    #[test]
    fn test_temp_schedule_decay() {
        let schedule = TempSchedule::new(1.0, 0.0, Some(10), true);
        assert_eq!(schedule.get_temp(0), 1.0);
        assert!((schedule.get_temp(5) - 0.5).abs() < 0.01);
        assert_eq!(schedule.get_temp(10), 0.0);
    }

    #[test]
    fn test_temp_schedule_cutoff_zero() {
        // Explicit cutoff of 0 = immediate switch to final temp
        let schedule = TempSchedule::new(1.0, 0.3, Some(0), false);
        assert_eq!(schedule.get_temp(0), 0.3);
        assert_eq!(schedule.get_temp(100), 0.3);
    }

    #[test]
    fn test_temp_schedule_from_args_valid() {
        use std::path::PathBuf;

        // No cutoff, no final/decay - valid
        let args = EvalArgs {
            checkpoints: vec![PathBuf::from("test")],
            humans: vec![],
            random: false,
            env_name: None,
            num_games: 100,
            num_envs: 64,
            watch: false,
            step: false,
            animate: false,
            fps: 10,
            seed: None,
            temperature: 0.5,
            temp_final: None,
            temp_cutoff: None,
            temp_decay: false,
        };
        let schedule = TempSchedule::from_args(&args).unwrap();
        assert_eq!(schedule.get_temp(0), 0.5);
        assert_eq!(schedule.get_temp(100), 0.5);

        // With cutoff and final - valid
        let args = EvalArgs {
            temp_cutoff: Some(10),
            temp_final: Some(0.1),
            ..args
        };
        let schedule = TempSchedule::from_args(&args).unwrap();
        assert_eq!(schedule.get_temp(0), 0.5);
        assert_eq!(schedule.get_temp(10), 0.1);
    }

    #[test]
    fn test_temp_schedule_from_args_final_without_cutoff() {
        use std::path::PathBuf;

        let args = EvalArgs {
            checkpoints: vec![PathBuf::from("test")],
            humans: vec![],
            random: false,
            env_name: None,
            num_games: 100,
            num_envs: 64,
            watch: false,
            step: false,
            animate: false,
            fps: 10,
            seed: None,
            temperature: 0.5,
            temp_final: Some(0.1), // final without cutoff = error
            temp_cutoff: None,
            temp_decay: false,
        };
        let result = TempSchedule::from_args(&args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("--temp-final requires --temp-cutoff"));
    }

    #[test]
    fn test_temp_schedule_from_args_decay_without_cutoff() {
        use std::path::PathBuf;

        let args = EvalArgs {
            checkpoints: vec![PathBuf::from("test")],
            humans: vec![],
            random: false,
            env_name: None,
            num_games: 100,
            num_envs: 64,
            watch: false,
            step: false,
            animate: false,
            fps: 10,
            seed: None,
            temperature: 0.5,
            temp_final: None,
            temp_cutoff: None,
            temp_decay: true, // decay without cutoff = error
        };
        let result = TempSchedule::from_args(&args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("--temp-decay requires --temp-cutoff"));
    }

    #[test]
    fn test_sample_deterministic() {
        let logits = vec![1.0, 2.0, 0.5];
        let mut rng = StdRng::seed_from_u64(42);
        let action = sample_with_temperature(&logits, None, 0.0, &mut rng);
        assert_eq!(action, 1); // argmax
    }

    #[test]
    fn test_sample_with_mask() {
        let logits = vec![10.0, 2.0, 0.5]; // First is highest but masked
        let mask = vec![false, true, true];
        let mut rng = StdRng::seed_from_u64(42);
        let action = sample_with_temperature(&logits, Some(&mask), 0.0, &mut rng);
        assert_eq!(action, 1); // argmax of valid actions
    }

    #[test]
    fn test_elo_delta_even() {
        let (delta, _) = elo_delta_2p(50, 50, 0);
        assert!(delta.abs() < 1.0); // Should be ~0
    }

    #[test]
    fn test_elo_delta_75_percent() {
        let (delta, _) = elo_delta_2p(75, 25, 0);
        assert!((delta - 191.0).abs() < 10.0); // ~191 ELO for 75% win rate
    }

    #[test]
    fn test_elo_delta_with_draws() {
        let (delta, _) = elo_delta_2p(40, 40, 20);
        assert!(delta.abs() < 1.0); // 50% score with draws = ~0 ELO
    }

    #[test]
    fn test_elo_stderr_reasonable() {
        // With 100 games at 50% win rate, stderr should be around 35 ELO
        // SE = 400 / (ln(10) * sqrt(n * 0.5 * 0.5)) = 400 / (2.303 * 5) = 34.7
        let (_, stderr) = elo_delta_2p(50, 50, 0);
        assert!((stderr - 35.0).abs() < 5.0);

        // With more games, stderr should decrease
        let (_, stderr_1000) = elo_delta_2p(500, 500, 0);
        assert!(stderr_1000 < stderr); // More games = smaller error

        // At extreme win rates, stderr should be large/infinite
        let (_, stderr_extreme) = elo_delta_2p(999, 1, 0);
        assert!(stderr_extreme > 100.0 || stderr_extreme.is_infinite());
    }

    #[test]
    fn test_rewards_to_placements() {
        assert_eq!(rewards_to_placements(&[1.0, 0.5, 0.0]), vec![1, 2, 3]);
        assert_eq!(rewards_to_placements(&[0.0, 1.0, 0.5]), vec![3, 1, 2]);
    }

    #[test]
    fn test_rewards_to_placements_tie() {
        assert_eq!(rewards_to_placements(&[1.0, 1.0]), vec![1, 1]);
        assert_eq!(rewards_to_placements(&[1.0, 0.5, 0.5]), vec![1, 2, 2]);
    }

    #[test]
    fn test_eval_stats_winner() {
        let mut stats = EvalStats::new(2);
        stats.record(&GameOutcome::Winner(0));
        assert_eq!(stats.wins[0], 1);
        assert_eq!(stats.losses[1], 1);
        assert_eq!(stats.total_games, 1);
    }

    #[test]
    fn test_eval_stats_tie() {
        let mut stats = EvalStats::new(2);
        stats.record(&GameOutcome::Tie);
        assert_eq!(stats.draws, 1);
        assert_eq!(stats.total_games, 1);
    }

    #[test]
    fn test_ordinal() {
        assert_eq!(ordinal(1), "1st");
        assert_eq!(ordinal(2), "2nd");
        assert_eq!(ordinal(3), "3rd");
        assert_eq!(ordinal(4), "4th");
        assert_eq!(ordinal(11), "11th");
        assert_eq!(ordinal(21), "21st");
    }

    #[test]
    fn test_elo_perfect_win_rate() {
        // 100% win rate should give infinite ELO delta
        let (delta, stderr) = elo_delta_2p(100, 0, 0);
        assert!(delta.is_infinite() && delta > 0.0);
        assert!(stderr.is_infinite());
    }

    #[test]
    fn test_elo_zero_win_rate() {
        // 0% win rate should give negative infinite ELO delta
        let (delta, stderr) = elo_delta_2p(0, 100, 0);
        assert!(delta.is_infinite() && delta < 0.0);
        assert!(stderr.is_infinite());
    }

    #[test]
    fn test_elo_no_games() {
        // No games should return 0 delta with infinite error
        let (delta, stderr) = elo_delta_2p(0, 0, 0);
        assert_eq!(delta, 0.0);
        assert!(stderr.is_infinite());
    }

    #[test]
    fn test_single_player_stats() {
        let mut stats = EvalStats::new(1);

        // Record some single-player games with rewards
        stats.record_with_rewards(&GameOutcome::Winner(0), &[100.0], 50);
        stats.record_with_rewards(&GameOutcome::Winner(0), &[200.0], 100);
        stats.record_with_rewards(&GameOutcome::Winner(0), &[150.0], 75);

        assert_eq!(stats.total_games, 3);
        assert_eq!(stats.game_rewards.len(), 3);
        assert_eq!(stats.episode_lengths.len(), 3);

        // Check recorded values
        assert_eq!(stats.game_rewards[0], vec![100.0]);
        assert_eq!(stats.game_rewards[1], vec![200.0]);
        assert_eq!(stats.episode_lengths[1], 100);
    }

    #[test]
    fn test_record_with_rewards_multiplayer() {
        let mut stats = EvalStats::new(2);

        stats.record_with_rewards(&GameOutcome::Winner(0), &[1.0, 0.0], 10);
        stats.record_with_rewards(&GameOutcome::Winner(1), &[0.0, 1.0], 15);

        assert_eq!(stats.total_games, 2);
        assert_eq!(stats.wins[0], 1);
        assert_eq!(stats.wins[1], 1);
        assert_eq!(stats.game_rewards.len(), 2);
        assert_eq!(stats.game_rewards[0], vec![1.0, 0.0]);
    }

    /// Test environment that returns explicit game outcome
    struct MockEnvWithOutcome {
        outcome: Option<GameOutcome>,
    }

    impl Environment for MockEnvWithOutcome {
        const OBSERVATION_DIM: usize = 1;
        const ACTION_COUNT: usize = 2;
        const NAME: &'static str = "mock";
        const NUM_PLAYERS: usize = 2;

        fn new(_seed: u64) -> Self {
            Self { outcome: None }
        }

        fn reset(&mut self) -> Vec<f32> {
            vec![0.0]
        }

        fn step(&mut self, _action: usize) -> (Vec<f32>, Vec<f32>, bool) {
            (vec![0.0], vec![0.0, 0.0], true)
        }

        fn game_outcome(&self) -> Option<GameOutcome> {
            self.outcome.clone()
        }
    }

    /// Test environment without explicit game outcome (uses reward fallback)
    struct MockEnvNoOutcome;

    impl Environment for MockEnvNoOutcome {
        const OBSERVATION_DIM: usize = 1;
        const ACTION_COUNT: usize = 2;
        const NAME: &'static str = "mock_no_outcome";
        const NUM_PLAYERS: usize = 2;

        fn new(_seed: u64) -> Self {
            Self
        }

        fn reset(&mut self) -> Vec<f32> {
            vec![0.0]
        }

        fn step(&mut self, _action: usize) -> (Vec<f32>, Vec<f32>, bool) {
            (vec![0.0], vec![0.0, 0.0], true)
        }
        // game_outcome() uses default impl returning None
    }

    #[test]
    fn test_determine_outcome_uses_explicit_outcome() {
        let env = MockEnvWithOutcome {
            outcome: Some(GameOutcome::Winner(1)),
        };
        // Even if rewards say player 0 wins, explicit outcome wins
        let outcome = determine_outcome(&env, &[10.0, 0.0]);
        assert_eq!(outcome, GameOutcome::Winner(1));
    }

    #[test]
    fn test_determine_outcome_falls_back_to_rewards() {
        let env = MockEnvNoOutcome;
        // No explicit outcome, should infer from rewards
        let outcome = determine_outcome(&env, &[1.0, 0.0]);
        assert_eq!(outcome, GameOutcome::Winner(0));

        let outcome = determine_outcome(&env, &[0.0, 1.0]);
        assert_eq!(outcome, GameOutcome::Winner(1));
    }

    #[test]
    fn test_determine_outcome_tie_from_rewards() {
        let env = MockEnvNoOutcome;
        let outcome = determine_outcome(&env, &[0.5, 0.5]);
        assert_eq!(outcome, GameOutcome::Tie);
    }

    #[test]
    fn test_sample_stochastic_distribution() {
        // With high temperature, sampling should produce varied results
        let logits = vec![1.0, 1.0, 1.0]; // Equal logits
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = [0usize; 3];

        for _ in 0..300 {
            let action = sample_with_temperature(&logits, None, 1.0, &mut rng);
            counts[action] += 1;
        }

        // With equal logits and temp=1.0, each action should get ~100 samples
        // Allow for statistical variance (should be within 50-150 each)
        for count in counts {
            assert!(count > 50, "Action count {count} too low");
            assert!(count < 150, "Action count {count} too high");
        }
    }

    #[test]
    fn test_sample_low_temp_more_deterministic() {
        // With low temp, sampling should favor the highest logit more strongly
        let logits = vec![2.0, 1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = [0usize; 3];

        for _ in 0..100 {
            let action = sample_with_temperature(&logits, None, 0.1, &mut rng);
            counts[action] += 1;
        }

        // With low temp, action 0 should dominate
        assert!(counts[0] > 90, "Action 0 should dominate with low temp");
    }

    #[test]
    fn test_sample_all_masked_fallback() {
        // Edge case: all actions masked (shouldn't happen in practice)
        // When all masked, logits become -inf, which produces NaN in softmax
        // The function falls through to return last action index
        let logits = vec![1.0, 2.0, 3.0];
        let mask = vec![false, false, false];
        let mut rng = StdRng::seed_from_u64(42);

        // Returns last index due to NaN arithmetic in edge case
        let action = sample_with_temperature(&logits, Some(&mask), 1.0, &mut rng);
        assert_eq!(action, 2); // probs.len() - 1
    }

    #[test]
    fn test_eval_stats_placements() {
        let mut stats = EvalStats::new(3);

        // Record outcomes using placements
        stats.record(&GameOutcome::Placements(vec![1, 2, 3])); // P0 wins, P1 second, P2 third
        stats.record(&GameOutcome::Placements(vec![3, 1, 2])); // P1 wins, P2 second, P0 third
        stats.record(&GameOutcome::Placements(vec![2, 3, 1])); // P2 wins, P0 second, P1 third

        assert_eq!(stats.total_games, 3);
        // Each player won once
        assert_eq!(stats.wins[0], 1);
        assert_eq!(stats.wins[1], 1);
        assert_eq!(stats.wins[2], 1);

        // Check placement counts: placements[player][place-1] = count
        assert_eq!(stats.placements[0][0], 1); // P0 got 1st once
        assert_eq!(stats.placements[0][1], 1); // P0 got 2nd once
        assert_eq!(stats.placements[0][2], 1); // P0 got 3rd once
    }
}
