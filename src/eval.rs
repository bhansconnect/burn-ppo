//! Evaluation subcommand for assessing trained models
//!
//! Features:
//! - Watch mode: visualize games with ASCII rendering
//! - Stats mode: parallel game execution with win/loss/draw statistics
//! - Temperature-based sampling with schedule support
//! - Plackett-Luce rating for skill comparison (in tournaments)

// Evaluation uses unwrap/expect for:
// - Tensor data extraction (cannot fail with correct shapes)
// - Internal data structure invariants

use std::io::{self, Write as IoWrite};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use crate::plackett_luce::{self, GameResult as PlGameResult, PlackettLuceConfig};
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
use crate::tournament::compute_display_names;

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

    /// Create a `TempSchedule` from CLI arguments with environment defaults
    ///
    /// Uses `E::EVAL_TEMP` and `E::EVAL_TEMP_CUTOFF` as defaults, allowing CLI to override.
    ///
    /// # Errors
    /// Returns an error if --temp-final or --temp-decay is used without --temp-cutoff
    pub fn from_args_with_env_defaults<E: Environment>(args: &EvalArgs) -> Result<Self> {
        // If --no-temp-cutoff is set, disable cutoff entirely
        if args.no_temp_cutoff {
            return Ok(Self::new(
                args.temp.unwrap_or(E::EVAL_TEMP),
                0.0,
                None,
                false,
            ));
        }

        // Get environment default cutoff
        let env_cutoff = E::EVAL_TEMP_CUTOFF;

        // Determine effective cutoff: CLI override > env default
        let effective_cutoff = if args.temp_cutoff.is_some() {
            args.temp_cutoff
        } else {
            env_cutoff.map(|(c, _)| c)
        };

        // Validate: temp_final and temp_decay require some cutoff
        if effective_cutoff.is_none() {
            if args.temp_final.is_some() {
                anyhow::bail!("--temp-final requires --temp-cutoff to be set (or env default)");
            }
            if args.temp_decay {
                anyhow::bail!("--temp-decay requires --temp-cutoff to be set (or env default)");
            }
        }

        // Determine final temp: CLI override > env default > 0.0
        let effective_final = args
            .temp_final
            .unwrap_or_else(|| env_cutoff.map_or(0.0, |(_, f)| f));

        Ok(Self::new(
            args.temp.unwrap_or(E::EVAL_TEMP),
            effective_final,
            effective_cutoff,
            args.temp_decay,
        ))
    }

    /// Create a `TempSchedule` from CLI arguments with a simple default temperature
    ///
    /// This is a simpler version that doesn't use environment defaults for cutoff.
    /// Used for testing.
    ///
    /// # Errors
    /// Returns an error if --temp-final or --temp-decay is used without --temp-cutoff
    #[cfg(test)]
    pub fn from_args_with_default(args: &EvalArgs, default_temp: f32) -> Result<Self> {
        // If --no-temp-cutoff is set, disable cutoff entirely
        if args.no_temp_cutoff {
            return Ok(Self::new(
                args.temp.unwrap_or(default_temp),
                0.0,
                None,
                false,
            ));
        }

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
            args.temp.unwrap_or(default_temp),
            args.temp_final.unwrap_or(0.0),
            args.temp_cutoff,
            args.temp_decay,
        ))
    }

    /// Describe the temperature schedule for display
    pub fn describe(&self) -> String {
        match self.cutoff {
            None => format!("temp={:.2} (constant)", self.initial),
            Some(cutoff) if self.decay => {
                format!(
                    "temp={:.2}→{:.2} (decay over {} moves)",
                    self.initial, self.final_temp, cutoff
                )
            }
            Some(cutoff) => {
                format!(
                    "temp={:.2}→{:.2} (cutoff at move {})",
                    self.initial, self.final_temp, cutoff
                )
            }
        }
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
    env.game_outcome()
        .unwrap_or_else(|| GameOutcome(rewards_to_placements(total_rewards)))
}

/// Statistics tracker for evaluation results
pub struct EvalStats {
    num_players: usize,
    /// Placement counts per checkpoint [`num_checkpoints`][num_placements]
    pub placements: Vec<Vec<usize>>,
    /// Total games played
    pub total_games: usize,
    /// Total rewards per game [`game_idx`][player]
    game_rewards: Vec<Vec<f64>>,
    /// Episode lengths
    episode_lengths: Vec<usize>,
    /// Individual game outcomes for rating calculations
    pub game_outcomes: Vec<GameOutcome>,
}

impl EvalStats {
    pub fn new(num_players: usize) -> Self {
        Self {
            num_players,
            placements: vec![vec![0; num_players]; num_players],
            total_games: 0,
            game_rewards: Vec::new(),
            episode_lengths: Vec::new(),
            game_outcomes: Vec::new(),
        }
    }

    /// Get wins for a player (1st place finishes, excluding draws) - for test assertions
    #[cfg(test)]
    pub fn wins(&self, player: usize) -> usize {
        // Count games where this player got 1st alone
        self.game_outcomes
            .iter()
            .filter(|outcome| {
                outcome.0[player] == 1 && outcome.0.iter().filter(|&&p| p == 1).count() == 1
            })
            .count()
    }

    /// Get losses for a player (last place finishes) - for test assertions
    #[cfg(test)]
    pub fn losses(&self, player: usize) -> usize {
        self.game_outcomes
            .iter()
            .filter(|outcome| outcome.0[player] == self.num_players)
            .count()
    }

    /// Get total draws (all players tied for 1st) - for test assertions
    #[cfg(test)]
    pub fn draws(&self) -> usize {
        self.game_outcomes
            .iter()
            .filter(|outcome| outcome.0.iter().all(|&p| p == 1))
            .count()
    }

    /// Record a game outcome with optional rewards and length
    pub fn record_with_rewards(&mut self, outcome: &GameOutcome, rewards: &[f32], length: usize) {
        self.game_rewards
            .push(rewards.iter().map(|&r| f64::from(r)).collect());
        self.episode_lengths.push(length);
        self.record(outcome);
    }

    /// Record a game outcome
    pub fn record(&mut self, outcome: &GameOutcome) {
        self.total_games += 1;
        self.game_outcomes.push(outcome.clone());

        // Update placement counts
        for (i, &place) in outcome.0.iter().enumerate() {
            if place > 0 && place <= self.num_players {
                self.placements[i][place - 1] += 1;
            }
        }
    }

    /// Print statistics summary
    ///
    /// If `slot_to_model` is provided, stats will be aggregated by unique model.
    /// This merges placements and rewards for slots sharing the same model index,
    /// but keeps ratings separate (per-slot).
    pub fn print_summary(&self, checkpoint_names: &[String], slot_to_model: Option<&[usize]>) {
        println!("\n=== Evaluation Results ({}-player) ===", self.num_players);
        println!("Total games: {}\n", self.total_games);

        if self.num_players == 1 {
            // Single-player: show full reward distribution
            self.print_single_player_summary(checkpoint_names);
        } else if self.num_players == 2 {
            // 2-player format with win/loss/draw and ELO
            self.print_two_player_summary(checkpoint_names, slot_to_model);
        } else {
            // N-player format with placement percentages
            self.print_multi_player_summary(checkpoint_names, slot_to_model);
        }

        // Print average rewards if available
        self.print_reward_summary(checkpoint_names, slot_to_model);
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

    /// Unified N-player summary with Swiss points and Weng-Lin ratings
    /// Works for 2-player, 4-player, or any N-player games
    ///
    /// If `slot_to_model` is provided, placements are aggregated by model index.
    fn print_n_player_summary(&self, checkpoint_names: &[String], slot_to_model: Option<&[usize]>) {
        let num_checkpoints = checkpoint_names.len();

        // Build aggregated placements if merging by model
        let (merged_names, merged_placements, num_slots_per_model): (
            Vec<String>,
            Vec<Vec<usize>>,
            Vec<usize>,
        ) = if let Some(mapping) = slot_to_model {
            // Find unique model indices and aggregate
            let num_models = mapping.iter().copied().max().map_or(0, |m| m + 1);
            let mut names: Vec<String> = vec![String::new(); num_models];
            let mut placements: Vec<Vec<usize>> = vec![vec![0; self.num_players]; num_models];
            let mut slot_counts: Vec<usize> = vec![0; num_models];

            for (slot, &model_idx) in mapping.iter().enumerate() {
                if model_idx < num_models {
                    // Use first slot's name for each model
                    if names[model_idx].is_empty() && slot < checkpoint_names.len() {
                        names[model_idx].clone_from(&checkpoint_names[slot]);
                    }
                    // Sum placements
                    if slot < self.placements.len() {
                        for (p, &count) in self.placements[slot].iter().enumerate() {
                            if p < placements[model_idx].len() {
                                placements[model_idx][p] += count;
                            }
                        }
                    }
                    slot_counts[model_idx] += 1;
                }
            }

            (names, placements, slot_counts)
        } else {
            // No merging - use original data
            let slot_counts = vec![1; num_checkpoints];
            (
                checkpoint_names.to_vec(),
                self.placements.clone(),
                slot_counts,
            )
        };

        // Compute draw counts per model (games where all players tied for 1st)
        let mut draw_counts: Vec<usize> = vec![0; merged_names.len()];
        for outcome in &self.game_outcomes {
            let is_draw = outcome.0.iter().all(|&p| p == 1);
            if is_draw {
                if let Some(mapping) = slot_to_model {
                    // Count draw for each model that participated
                    for (slot, &model_idx) in mapping.iter().enumerate() {
                        if slot < outcome.0.len() && model_idx < draw_counts.len() {
                            draw_counts[model_idx] += 1;
                        }
                    }
                } else {
                    // No merging - each slot is its own model
                    for (slot, _) in outcome.0.iter().enumerate() {
                        if slot < draw_counts.len() {
                            draw_counts[slot] += 1;
                        }
                    }
                }
            }
        }

        // Show placement distribution and Swiss points per checkpoint (or merged model)
        for (i, name) in merged_names.iter().enumerate() {
            if name.is_empty() {
                continue;
            }

            // Total games for this model = total_games * num_slots
            let total_observations = self.total_games * num_slots_per_model[i];
            if total_observations == 0 {
                continue;
            }

            // Build placement percentages, separating draws from solo 1st
            let mut placement_pcts: Vec<String> = Vec::new();
            for (p, &count) in merged_placements[i].iter().enumerate() {
                if p == 0 {
                    // 1st place: subtract draws to get solo wins
                    let solo_wins = count.saturating_sub(draw_counts[i]);
                    let solo_pct = 100.0 * solo_wins as f64 / total_observations as f64;
                    placement_pcts.push(format!("{solo_pct:.0}% 1st"));
                } else {
                    let pct = 100.0 * count as f64 / total_observations as f64;
                    placement_pcts.push(format!("{:.0}% {}", pct, ordinal(p + 1)));
                }
            }

            // Add draw percentage at the end
            let draw_pct = 100.0 * draw_counts[i] as f64 / total_observations as f64;
            placement_pcts.push(format!("{draw_pct:.0}% Draw"));

            // Compute Swiss points: num_players - avg_placement (higher = better)
            let avg_placement: f64 = merged_placements[i]
                .iter()
                .enumerate()
                .map(|(p, &count)| (p + 1) as f64 * count as f64)
                .sum::<f64>()
                / total_observations as f64;
            let swiss_points = self.num_players as f64 - avg_placement;

            println!(
                "{}: {} ({:.2} pts)",
                name,
                placement_pcts.join(", "),
                swiss_points
            );
        }

        // Compute Plackett-Luce ratings using all game outcomes
        // Ratings are NOT merged - computed per slot
        let pl_games: Vec<PlGameResult> = self
            .game_outcomes
            .iter()
            .map(|outcome| {
                let players: Vec<usize> = (0..outcome.0.len()).collect();
                PlGameResult::new(players, outcome.0.clone())
            })
            .collect();

        // Use first checkpoint as anchor (arbitrary choice for eval)
        let rating_result = plackett_luce::compute_ratings(
            num_checkpoints,
            &pl_games,
            0, // Anchor first checkpoint
            &PlackettLuceConfig::default(),
        );
        let ratings = &rating_result.ratings;

        // Find strongest checkpoint
        let (strongest_idx, _) = ratings
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.rating
                    .partial_cmp(&b.1.rating)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &ratings[0]));

        // Display computation statistics
        let stats = &rating_result.stats;
        let converge_status = if stats.converged {
            "converged"
        } else {
            "did not converge"
        };
        println!(
            "\nRating computation: {} in {} iterations ({:.1}ms), final delta: {:.2e}",
            converge_status, stats.iterations_used, stats.computation_time_ms, stats.final_delta
        );

        plackett_luce::print_rating_guide();
        println!("\nRatings:");
        for (i, (name, rating)) in checkpoint_names.iter().zip(ratings.iter()).enumerate() {
            let marker = if i == strongest_idx {
                " <- strongest"
            } else {
                ""
            };
            println!(
                "  {}: {:.0}±{:.0}{marker}",
                name, rating.rating, rating.uncertainty
            );
        }
    }

    // Keep old names as aliases for backward compatibility in tests
    fn print_two_player_summary(
        &self,
        checkpoint_names: &[String],
        slot_to_model: Option<&[usize]>,
    ) {
        self.print_n_player_summary(checkpoint_names, slot_to_model);
    }

    fn print_multi_player_summary(
        &self,
        checkpoint_names: &[String],
        slot_to_model: Option<&[usize]>,
    ) {
        self.print_n_player_summary(checkpoint_names, slot_to_model);
    }

    fn print_reward_summary(&self, checkpoint_names: &[String], slot_to_model: Option<&[usize]>) {
        if self.game_rewards.is_empty() || self.num_players == 1 {
            return; // Already handled in single-player summary
        }

        println!();
        println!("Average Rewards:");

        if let Some(mapping) = slot_to_model {
            // Aggregate rewards by model
            let num_models = mapping.iter().copied().max().map_or(0, |m| m + 1);
            let mut model_names: Vec<String> = vec![String::new(); num_models];
            let mut model_reward_sums: Vec<f64> = vec![0.0; num_models];
            let mut model_reward_counts: Vec<usize> = vec![0; num_models];

            for (slot, &model_idx) in mapping.iter().enumerate() {
                if model_idx < num_models {
                    // Use first slot's name for each model
                    if model_names[model_idx].is_empty() && slot < checkpoint_names.len() {
                        model_names[model_idx].clone_from(&checkpoint_names[slot]);
                    }
                    // Sum rewards for this slot
                    for game_rewards in &self.game_rewards {
                        if let Some(&reward) = game_rewards.get(slot) {
                            model_reward_sums[model_idx] += reward;
                            model_reward_counts[model_idx] += 1;
                        }
                    }
                }
            }

            for (model_idx, name) in model_names.iter().enumerate() {
                if !name.is_empty() && model_reward_counts[model_idx] > 0 {
                    let mean = model_reward_sums[model_idx] / model_reward_counts[model_idx] as f64;
                    println!("  {name}: {mean:.3}");
                }
            }
        } else {
            // No merging - original behavior
            for (i, name) in checkpoint_names.iter().enumerate() {
                let rewards: Vec<f64> = self
                    .game_rewards
                    .iter()
                    .filter_map(|r| r.get(i).copied())
                    .collect();

                if !rewards.is_empty() {
                    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
                    println!("  {name}: {mean:.3}");
                }
            }
        }
    }
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
pub fn load_model_from_checkpoint<B: Backend>(
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
        activation: metadata.activation.clone(),
        split_networks: metadata.split_networks,
        network_type: metadata.network_type.clone(),
        num_conv_layers: metadata.num_conv_layers,
        conv_channels: metadata.conv_channels.clone(),
        kernel_size: metadata.kernel_size,
        cnn_fc_hidden_size: metadata.cnn_fc_hidden_size,
        cnn_num_fc_layers: metadata.cnn_num_fc_layers,
        // Note: global_state_dim is read from metadata in CheckpointManager::load
        critic_hidden_size: metadata.critic_hidden_size,
        critic_num_hidden: metadata.critic_num_hidden,
        ..Config::default()
    };

    // Load model
    let (model, _) = CheckpointManager::load::<B>(checkpoint_path, &config, device)?;

    // Load normalizer if it exists (trained with normalize_obs=true)
    let normalizer = load_normalizer(checkpoint_path)?;

    Ok((model, metadata, normalizer))
}

/// Run evaluation based on command-line arguments
pub fn run_evaluation<B: Backend>(args: &EvalArgs, device: &B::Device) -> Result<()> {
    use std::collections::HashMap;

    // Check if we have human or random players
    let has_human = !args.humans.is_empty();
    let has_random = args.random;

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

    // Route to interactive evaluation only for human players
    // Random players can use stats mode (parallel)
    if has_human {
        return run_interactive_evaluation::<B>(args, &player_sources, device);
    }

    // Stats mode flow - supports checkpoints and random players
    let mut models: Vec<Option<ActorCritic<B>>> = Vec::new();
    let mut normalizers: Vec<Option<ObsNormalizer>> = Vec::new();
    let mut path_to_model_idx: HashMap<std::path::PathBuf, usize> = HashMap::new();
    let mut checkpoint_to_model: Vec<usize> = Vec::new();
    let mut checkpoint_names = Vec::new();
    let mut metadata_opt: Option<CheckpointMetadata> = None;

    // First, collect all checkpoint paths for display name computation
    let checkpoint_paths: Vec<std::path::PathBuf> = player_sources
        .iter()
        .filter_map(|s| match s {
            PlayerSource::Checkpoint(path) => Some(path.clone()),
            _ => None,
        })
        .collect();

    // Compute unique display names for checkpoints
    let display_names = compute_display_names(&checkpoint_paths);
    let mut display_name_iter = display_names.into_iter();

    for source in &player_sources {
        match source {
            PlayerSource::Checkpoint(path) => {
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
                    let (model, metadata, normalizer) =
                        load_model_from_checkpoint::<B>(&resolved, device)?;

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
                    models.push(Some(model));
                    normalizers.push(normalizer);
                    path_to_model_idx.insert(resolved, idx);
                    idx
                };

                checkpoint_to_model.push(model_idx);
                // Use computed display name
                checkpoint_names.push(
                    display_name_iter
                        .next()
                        .expect("display names should match checkpoint count"),
                );
            }
            PlayerSource::Random => {
                // Random player gets its own slot with None model
                let idx = models.len();
                models.push(None);
                normalizers.push(None);
                checkpoint_to_model.push(idx);
                checkpoint_names.push("Random".to_string());
            }
            PlayerSource::Human { .. } => {
                unreachable!("Human players should have been routed to interactive mode")
            }
        }
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

    // Default to 1 game for interactive human play or watch mode
    // Random-only games (without watch) should use default 100 games
    let has_human = player_sources.iter().any(PlayerSource::is_human);
    let watch = args.watch || args.step;
    let num_games = if (has_human || watch) && args.num_games == 100 {
        1 // Default to single game for interactive human play
    } else {
        args.num_games
    };

    // Dispatch based on environment name
    dispatch_env!(env_name, {
        // Create temp schedule with environment-specific default
        let temp_schedule = TempSchedule::from_args_with_env_defaults::<E>(args)?;
        println!("Temperature: {}", temp_schedule.describe());

        // Determine expected player count
        let expected_players = if E::VARIABLE_PLAYER_COUNT {
            // Variable-player games require --players flag
            args.players.ok_or_else(|| {
                anyhow::anyhow!(
                    "{} supports variable player counts. Use --players N to specify (e.g., --players 4)",
                    E::NAME
                )
            })?
        } else {
            // Fixed-player games use their constant
            E::NUM_PLAYERS
        };

        if player_sources.len() != expected_players {
            anyhow::bail!(
                "{} requires {} players, but {} were specified.\n\
                 Use --checkpoint, --human, and --random to specify players.",
                E::NAME,
                expected_players,
                player_sources.len()
            );
        }

        let watch = args.watch || args.step;
        run_interactive_game::<B, E>(
            player_sources,
            &models,
            &normalizers,
            num_games,
            &temp_schedule,
            &mut rng,
            device,
            watch,
            expected_players,
        );
        Ok(())
    })
}

/// Run evaluation in watch mode (sequential, with rendering)
///
/// Models can be `None` for random players.
fn run_watch_mode<B: Backend>(
    models: &[Option<ActorCritic<B>>],
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

    // Dispatch based on env_name stored in checkpoint metadata
    crate::dispatch_env!(metadata.env_name, {
        // Create temp schedule with environment-specific default
        let temp_schedule = TempSchedule::from_args_with_env_defaults::<E>(args)?;
        println!("Temperature: {}", temp_schedule.describe());

        // Determine player count
        let num_players = if E::VARIABLE_PLAYER_COUNT {
            args.players.ok_or_else(|| {
                anyhow::anyhow!(
                    "{} supports variable player counts. Use --players N to specify (e.g., --players 4)",
                    E::NAME
                )
            })?
        } else {
            E::NUM_PLAYERS
        };

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
            num_players,
        );
        Ok(())
    })
}

/// Watch mode implementation for a specific environment type
///
/// Models can be `None` for random players.
fn run_watch_mode_env<B: Backend, E: Environment>(
    models: &[Option<ActorCritic<B>>],
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
    num_players: usize,
) {
    use rand::RngCore;
    let mut stats = EvalStats::new(num_players);

    for game_idx in 0..num_games {
        println!("\n=== Game {} ===", game_idx + 1);
        for (i, name) in checkpoint_names.iter().enumerate() {
            println!("Player {i}: {name}");
        }
        println!();

        let mut env = E::new(rng.next_u64());
        env.set_num_players(num_players);
        let mut obs = env.reset();
        let mut total_rewards = vec![0.0f32; num_players];
        let mut move_num = 0;
        let mut is_first_frame = true;
        let mut last_frame_lines = 0usize;
        let frame_duration = Duration::from_millis(1000 / u64::from(fps));

        loop {
            let frame_start = if animate { Some(Instant::now()) } else { None };
            let current_player = env.current_player();
            // In watch mode, checkpoint_idx == player_idx (no position swapping)
            let model_idx = checkpoint_to_model[current_player];
            let model_opt = &models[model_idx];
            let normalizer = &normalizers[model_idx];
            let mask = env.action_mask();

            // Get logits - either from model or uniform for random players
            let logits_vec: Vec<f32> = if let Some(model) = model_opt {
                // Get action from model (normalize if checkpoint was trained with normalize_obs)
                let obs_for_model = if let Some(norm) = normalizer {
                    norm.normalize(&obs)
                } else {
                    obs.clone()
                };
                let obs_tensor: Tensor<B, 2> =
                    Tensor::<B, 1>::from_floats(&obs_for_model[..], device)
                        .reshape([1, E::OBSERVATION_DIM]);

                // Handle CTDE networks: use actor network only (don't need values for action selection)
                let logits = if model.is_ctde() {
                    model.forward_actor(obs_tensor)
                } else {
                    model.forward(obs_tensor).0
                };
                logits
                    .to_data()
                    .to_vec()
                    .expect("tensor data to vec conversion")
            } else {
                // Random player: uniform logits
                vec![0.0f32; E::ACTION_COUNT]
            };

            let temp = temp_schedule.get_temp(move_num);
            let action = sample_with_temperature(&logits_vec, mask.as_deref(), temp, rng);

            // Render before step
            if let Some(rendered) = env.render() {
                if animate {
                    let frame_lines = rendered.lines().count();
                    // Move cursor up to overwrite previous frame
                    if !is_first_frame {
                        // +1 for the action line; \r moves to column 1; \x1b[J clears to end of screen
                        print!("\x1b[{}A\r\x1b[J", last_frame_lines + 1);
                        io::stdout().flush().ok();
                    }
                    print!("{rendered}");
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
                        print!("\x1b[{}A\r\x1b[J", last_frame_lines + 1);
                        io::stdout().flush().ok();
                        print!("{rendered}");
                        println!(); // Extra line to separate from outcome
                    } else {
                        println!("{rendered}");
                    }
                }

                let outcome = determine_outcome(&env, &total_rewards);
                stats.record(&outcome);

                // Print outcome based on placements
                let winners: Vec<_> = outcome
                    .0
                    .iter()
                    .enumerate()
                    .filter(|(_, &place)| place == 1)
                    .map(|(i, _)| i)
                    .collect();
                if winners.len() == outcome.0.len() {
                    println!("\nGame ended in a tie");
                } else if winners.len() == 1 {
                    println!("\nWinner: Player {}", winners[0]);
                } else {
                    println!("\nFinal placements: {:?}", outcome.0);
                }
                println!("Total rewards: {total_rewards:?}");
                println!("Game length: {move_num} moves");
                break;
            }
        }
    }

    if num_games > 1 {
        stats.print_summary(checkpoint_names, Some(checkpoint_to_model));
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
/// Output is suppressed when there are no human players for batch evaluation,
/// unless `watch` is true for rendering games.
pub fn run_interactive_game<B: Backend, E: Environment>(
    player_sources: &[PlayerSource],
    models: &[Option<ActorCritic<B>>],
    normalizers: &[Option<ObsNormalizer>],
    num_games: usize,
    temp_schedule: &TempSchedule,
    rng: &mut StdRng,
    device: &B::Device,
    watch: bool,
    num_players: usize,
) {
    use rand::RngCore;

    assert_eq!(
        player_sources.len(),
        num_players,
        "Expected {num_players} player sources for {num_players}-player game"
    );

    // Check if there are any human players (affects verbosity)
    // Also show output in watch mode
    let has_human = player_sources.iter().any(PlayerSource::is_human);
    let verbose = has_human || watch;

    // Build display names for players
    let player_names: Vec<String> = player_sources
        .iter()
        .map(PlayerSource::display_name)
        .collect();

    let mut stats = EvalStats::new(num_players);

    if !verbose && num_games > 1 {
        println!("Running {num_games} games...");
    }

    for game_idx in 0..num_games {
        if verbose {
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
        } else if num_games > 1 && (game_idx + 1) % 10 == 0 {
            // Progress indicator for batch runs
            println!("  {} / {} games completed", game_idx + 1, num_games);
        }

        let mut env = E::new(rng.next_u64());
        // Set player count for variable-player games (no-op for fixed-player games)
        env.set_num_players(num_players);
        let obs = env.reset();
        let mut total_rewards = vec![0.0f32; num_players];
        let mut move_num = 0;

        // Store observation for network hints
        let mut current_obs = obs;

        loop {
            let current_player = env.current_player();
            let player_source = &player_sources[current_player];
            let mask = env.action_mask();

            // Render board before human turns
            if verbose {
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
                    if verbose {
                        println!(
                            "Player {} (Random) plays: {}",
                            current_player + 1,
                            env.describe_action(action)
                        );
                    }
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

                    // Handle CTDE networks: use actor network only
                    let logits = if model.is_ctde() {
                        model.forward_actor(obs_tensor)
                    } else {
                        model.forward(obs_tensor).0
                    };
                    let logits_vec: Vec<f32> = logits
                        .to_data()
                        .to_vec()
                        .expect("tensor data to vec conversion");

                    let temp = temp_schedule.get_temp(move_num);
                    let action = sample_with_temperature(&logits_vec, mask.as_deref(), temp, rng);

                    if verbose {
                        println!(
                            "Player {} ({}) plays: {} (temp={:.2})",
                            current_player + 1,
                            player_names[current_player],
                            env.describe_action(action),
                            temp
                        );
                    }
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
                let outcome = determine_outcome(&env, &total_rewards);
                stats.record(&outcome);

                if verbose {
                    // Show final state
                    if let Some(rendered) = env.render() {
                        println!("{rendered}");
                    }

                    // Print outcome based on placements
                    let winners: Vec<_> = outcome
                        .0
                        .iter()
                        .enumerate()
                        .filter(|(_, &place)| place == 1)
                        .map(|(i, _)| i)
                        .collect();
                    if winners.len() == outcome.0.len() {
                        println!("\nGame ended in a tie");
                    } else if winners.len() == 1 {
                        println!(
                            "\nWinner: Player {} ({})",
                            winners[0] + 1,
                            player_names[winners[0]]
                        );
                    } else {
                        println!("\nFinal placements:");
                        for (i, &place) in outcome.0.iter().enumerate() {
                            println!("  {}: {} ({})", place, player_names[i], ordinal(place));
                        }
                    }
                    println!("Total rewards: {total_rewards:?}");
                    println!("Game length: {move_num} moves");
                }
                break;
            }
        }
    }

    if num_games > 1 {
        stats.print_summary(&player_names, None);
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

    // Handle CTDE networks: use actor network only
    let logits = if model.is_ctde() {
        model.forward_actor(obs_tensor)
    } else {
        model.forward(obs_tensor).0
    };
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
///
/// Models can be `None` for random players.
fn run_stats_mode<B: Backend>(
    models: &[Option<ActorCritic<B>>],
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

    // Dispatch based on env_name stored in checkpoint metadata
    // Stats are printed inside run_stats_mode_env when not silent
    crate::dispatch_env!(metadata.env_name, {
        // Create temp schedule with environment-specific default
        let temp_schedule = TempSchedule::from_args_with_env_defaults::<E>(args)?;
        println!("Temperature: {}", temp_schedule.describe());

        // Determine player count
        let num_players = if E::VARIABLE_PLAYER_COUNT {
            args.players.ok_or_else(|| {
                anyhow::anyhow!(
                    "{} supports variable player counts. Use --players N to specify (e.g., --players 4)",
                    E::NAME
                )
            })?
        } else {
            E::NUM_PLAYERS
        };

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
            num_players,
        );
        Ok(())
    })
}

/// Generate all permutations of 0..n using Heap's algorithm
fn generate_permutations(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut arr: Vec<usize> = (0..n).collect();
    heap_permute(&mut arr, n, &mut result);
    result
}

fn heap_permute(arr: &mut [usize], k: usize, result: &mut Vec<Vec<usize>>) {
    if k == 1 {
        result.push(arr.to_vec());
        return;
    }
    heap_permute(arr, k - 1, result);
    for i in 0..k - 1 {
        if k.is_multiple_of(2) {
            arr.swap(i, k - 1);
        } else {
            arr.swap(0, k - 1);
        }
        heap_permute(arr, k - 1, result);
    }
}

/// Stats mode implementation for a specific environment type
///
/// When `silent` is true, suppresses output (used by pool eval).
/// Returns `EvalStats` for further processing.
///
/// Models can be `None` for random players - these will output uniform logits
/// which get sampled with the action mask for uniform random valid actions.
pub fn run_stats_mode_env<B: Backend, E: Environment>(
    models: &[Option<ActorCritic<B>>],
    normalizers: &[Option<ObsNormalizer>],
    checkpoint_to_model: &[usize],
    checkpoint_names: &[String],
    num_games: usize,
    num_envs: usize,
    temp_schedule: &TempSchedule,
    rng: &mut StdRng,
    device: &B::Device,
    silent: bool,
    num_players: usize,
) -> EvalStats {
    use rand::RngCore;
    let num_checkpoints = checkpoint_to_model.len();
    let obs_dim = E::OBSERVATION_DIM;
    let action_count = E::ACTION_COUNT;

    // Validate we have enough checkpoints for this N-player game
    assert!(
        num_checkpoints >= num_players,
        "Insufficient checkpoints: {num_players}-player game requires at least {num_players} checkpoints, got {num_checkpoints}"
    );

    // Create vectorized environment with unique seeds per env
    let base_seed = rng.next_u64();
    let mut vec_env: VecEnv<E> =
        VecEnv::new(num_envs, |i| E::new(base_seed.wrapping_add(i as u64)));
    // Set player count for variable-player games (no-op for fixed-player games)
    vec_env.set_all_num_players(num_players);
    let mut stats = EvalStats::new(num_checkpoints);

    // Track move counts per environment (for temperature schedule)
    let mut move_counts = vec![0usize; num_envs];

    // Pre-generate all permutations for fair position coverage
    // This cycles through all N! permutations instead of just N rotations
    let all_perms = generate_permutations(num_players);
    let num_perms = all_perms.len();

    // Track which permutation each env uses (cycles through all N!)
    let mut perm_indices: Vec<usize> = (0..num_envs).map(|env_idx| env_idx % num_perms).collect();

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
        for (model_idx, model_opt) in models.iter().enumerate() {
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
                // perm[player_idx] = checkpoint that plays as that player
                let perm = &all_perms[perm_indices[env_idx]];
                let checkpoint_for_player = perm[current_player];

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

            let batch_size = env_indices.len();

            // Get logits - either from model or uniform for random players
            let logits_flat: Vec<f32> = if let Some(model) = model_opt {
                // Batch inference for this model (normalize if checkpoint was trained with normalize_obs)
                let normalizer = &normalizers[model_idx];
                let obs_for_model = if let Some(norm) = normalizer {
                    let mut obs_copy = env_obs.clone();
                    norm.normalize_batch(&mut obs_copy, obs_dim);
                    obs_copy
                } else {
                    env_obs.clone()
                };
                let obs_tensor: Tensor<B, 2> =
                    Tensor::<B, 1>::from_floats(&obs_for_model[..], device)
                        .reshape([batch_size, obs_dim]);

                // Handle CTDE networks: use actor network only
                let logits_tensor = if model.is_ctde() {
                    model.forward_actor(obs_tensor)
                } else {
                    model.forward(obs_tensor.clone()).0
                };
                logits_tensor
                    .to_data()
                    .to_vec()
                    .expect("tensor data to vec conversion")
            } else {
                // Random player: uniform logits (all zeros = equal probability after softmax)
                vec![0.0f32; batch_size * action_count]
            };

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
            let perm = &all_perms[perm_indices[env_idx]];

            // Map player rewards to checkpoint rewards using permutation
            // perm[player_idx] = checkpoint that played as that player
            // We need reward for each checkpoint, so find which player they were
            let checkpoint_rewards: Vec<f32> = (0..num_checkpoints)
                .map(|checkpoint_idx| {
                    // Find which player position this checkpoint was at
                    let player_idx = perm
                        .iter()
                        .position(|&c| c == checkpoint_idx)
                        .expect("checkpoint must be in permutation");
                    ep.total_rewards[player_idx]
                })
                .collect();

            // Determine outcome: use game's outcome if available, otherwise infer from rewards
            let outcome = if let Some(game_outcome) = &ep.outcome {
                // Remap placements from player positions to checkpoint positions
                let checkpoint_placements: Vec<usize> = (0..num_checkpoints)
                    .map(|checkpoint_idx| {
                        let player_idx = perm
                            .iter()
                            .position(|&c| c == checkpoint_idx)
                            .expect("checkpoint must be in permutation");
                        game_outcome.0[player_idx]
                    })
                    .collect();
                GameOutcome(checkpoint_placements)
            } else {
                // Fall back to rewards-based placements
                GameOutcome(rewards_to_placements(&checkpoint_rewards))
            };

            stats.record_with_rewards(&outcome, &checkpoint_rewards, ep.length);

            // Reset move count and cycle to next permutation for fairness
            move_counts[env_idx] = 0;
            perm_indices[env_idx] = (perm_indices[env_idx] + 1) % num_perms;

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
        stats.print_summary(checkpoint_names, Some(checkpoint_to_model));
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_generate_permutations() {
        // Test 2-player: 2! = 2 permutations
        let perms_2 = generate_permutations(2);
        assert_eq!(perms_2.len(), 2);
        let unique_2: HashSet<_> = perms_2.iter().collect();
        assert_eq!(unique_2.len(), 2);

        // Test 3-player: 3! = 6 permutations
        let perms_3 = generate_permutations(3);
        assert_eq!(perms_3.len(), 6);
        let unique_3: HashSet<_> = perms_3.iter().collect();
        assert_eq!(unique_3.len(), 6);

        // Test 4-player: 4! = 24 permutations
        let perms_4 = generate_permutations(4);
        assert_eq!(perms_4.len(), 24);
        let unique_4: HashSet<_> = perms_4.iter().collect();
        assert_eq!(unique_4.len(), 24);

        // Verify each permutation contains 0..n exactly once
        for perm in &perms_4 {
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![0, 1, 2, 3]);
        }
    }

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
            backend: None,
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
            temp: Some(0.5),
            temp_final: None,
            temp_cutoff: None,
            no_temp_cutoff: false,
            temp_decay: false,
            players: None,
        };
        let schedule = TempSchedule::from_args_with_default(&args, 0.3).unwrap();
        assert_eq!(schedule.get_temp(0), 0.5);
        assert_eq!(schedule.get_temp(100), 0.5);

        // With cutoff and final - valid
        let args = EvalArgs {
            temp_cutoff: Some(10),
            temp_final: Some(0.1),
            ..args
        };
        let schedule = TempSchedule::from_args_with_default(&args, 0.3).unwrap();
        assert_eq!(schedule.get_temp(0), 0.5);
        assert_eq!(schedule.get_temp(10), 0.1);
    }

    #[test]
    fn test_temp_schedule_from_args_final_without_cutoff() {
        use std::path::PathBuf;

        let args = EvalArgs {
            checkpoints: vec![PathBuf::from("test")],
            backend: None,
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
            temp: Some(0.5),
            temp_final: Some(0.1), // final without cutoff = error
            temp_cutoff: None,
            no_temp_cutoff: false,
            temp_decay: false,
            players: None,
        };
        let result = TempSchedule::from_args_with_default(&args, 0.3);
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
            backend: None,
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
            temp: Some(0.5),
            temp_final: None,
            temp_cutoff: None,
            no_temp_cutoff: false,
            temp_decay: true, // decay without cutoff = error
            players: None,
        };
        let result = TempSchedule::from_args_with_default(&args, 0.3);
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
        stats.record(&GameOutcome(vec![1, 2])); // Player 0 wins
        assert_eq!(stats.wins(0), 1);
        assert_eq!(stats.losses(1), 1);
        assert_eq!(stats.total_games, 1);
    }

    #[test]
    fn test_eval_stats_tie() {
        let mut stats = EvalStats::new(2);
        stats.record(&GameOutcome(vec![1, 1])); // Tie
        assert_eq!(stats.draws(), 1);
        assert_eq!(stats.total_games, 1);
    }

    #[test]
    fn test_eval_stats_print_two_player_summary() {
        // Test that print_two_player_summary runs without panicking
        // and exercises the Weng-Lin rating calculation code path
        let mut stats = EvalStats::new(2);

        // Record some games: 60 wins for p0, 30 for p1, 10 draws
        for _ in 0..60 {
            stats.record(&GameOutcome(vec![1, 2])); // P0 wins
        }
        for _ in 0..30 {
            stats.record(&GameOutcome(vec![2, 1])); // P1 wins
        }
        for _ in 0..10 {
            stats.record(&GameOutcome(vec![1, 1])); // Tie
        }

        assert_eq!(stats.total_games, 100);
        assert_eq!(stats.wins(0), 60);
        assert_eq!(stats.wins(1), 30);
        assert_eq!(stats.draws(), 10);

        // This exercises the Weng-Lin code path in print_two_player_summary
        let names = vec!["checkpoint_a".to_string(), "checkpoint_b".to_string()];
        stats.print_two_player_summary(&names, None);
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
    fn test_single_player_stats() {
        let mut stats = EvalStats::new(1);

        // Record some single-player games with rewards
        stats.record_with_rewards(&GameOutcome(vec![1]), &[100.0], 50);
        stats.record_with_rewards(&GameOutcome(vec![1]), &[200.0], 100);
        stats.record_with_rewards(&GameOutcome(vec![1]), &[150.0], 75);

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

        stats.record_with_rewards(&GameOutcome(vec![1, 2]), &[1.0, 0.0], 10);
        stats.record_with_rewards(&GameOutcome(vec![2, 1]), &[0.0, 1.0], 15);

        assert_eq!(stats.total_games, 2);
        assert_eq!(stats.wins(0), 1);
        assert_eq!(stats.wins(1), 1);
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
            outcome: Some(GameOutcome(vec![2, 1])), // Player 1 wins
        };
        // Even if rewards say player 0 wins, explicit outcome wins
        let outcome = determine_outcome(&env, &[10.0, 0.0]);
        assert_eq!(outcome, GameOutcome(vec![2, 1]));
    }

    #[test]
    fn test_determine_outcome_falls_back_to_rewards() {
        let env = MockEnvNoOutcome;
        // No explicit outcome, should infer from rewards
        let outcome = determine_outcome(&env, &[1.0, 0.0]);
        assert_eq!(outcome, GameOutcome(vec![1, 2])); // P0 wins

        let outcome = determine_outcome(&env, &[0.0, 1.0]);
        assert_eq!(outcome, GameOutcome(vec![2, 1])); // P1 wins
    }

    #[test]
    fn test_determine_outcome_tie_from_rewards() {
        let env = MockEnvNoOutcome;
        let outcome = determine_outcome(&env, &[0.5, 0.5]);
        assert_eq!(outcome, GameOutcome(vec![1, 1])); // Tie
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
        stats.record(&GameOutcome(vec![1, 2, 3])); // P0 wins, P1 second, P2 third
        stats.record(&GameOutcome(vec![3, 1, 2])); // P1 wins, P2 second, P0 third
        stats.record(&GameOutcome(vec![2, 3, 1])); // P2 wins, P0 second, P1 third

        assert_eq!(stats.total_games, 3);
        // Each player won once
        assert_eq!(stats.wins(0), 1);
        assert_eq!(stats.wins(1), 1);
        assert_eq!(stats.wins(2), 1);

        // Check placement counts: placements[player][place-1] = count
        assert_eq!(stats.placements[0][0], 1); // P0 got 1st once
        assert_eq!(stats.placements[0][1], 1); // P0 got 2nd once
        assert_eq!(stats.placements[0][2], 1); // P0 got 3rd once
    }

    // =========================================
    // PlayerSource Tests
    // =========================================

    #[test]
    fn test_player_source_is_human() {
        let human = PlayerSource::Human {
            name: "Alice".to_string(),
        };
        let checkpoint = PlayerSource::Checkpoint(std::path::PathBuf::from("/path/to/model"));
        let random = PlayerSource::Random;

        assert!(human.is_human());
        assert!(!checkpoint.is_human());
        assert!(!random.is_human());
    }

    #[test]
    fn test_player_source_display_name_checkpoint() {
        let checkpoint =
            PlayerSource::Checkpoint(std::path::PathBuf::from("/runs/test/checkpoints/best"));
        assert_eq!(checkpoint.display_name(), "best");

        let step_checkpoint =
            PlayerSource::Checkpoint(std::path::PathBuf::from("/runs/test/checkpoints/step_1000"));
        assert_eq!(step_checkpoint.display_name(), "step_1000");
    }

    #[test]
    fn test_player_source_display_name_human() {
        let human = PlayerSource::Human {
            name: "Bob".to_string(),
        };
        assert_eq!(human.display_name(), "Bob");
    }

    #[test]
    fn test_player_source_display_name_random() {
        let random = PlayerSource::Random;
        assert_eq!(random.display_name(), "Random");
    }

    #[test]
    fn test_eval_stats_merge_placements() {
        // Test that placements are correctly aggregated when slot_to_model is provided
        let mut stats = EvalStats::new(4); // 4-player game

        // Record some game outcomes (GameOutcome is a newtype struct)
        // Game 1: placements [1, 2, 3, 4] for slots 0, 1, 2, 3
        stats.record(&GameOutcome(vec![1, 2, 3, 4]));
        // Game 2: placements [2, 1, 4, 3]
        stats.record(&GameOutcome(vec![2, 1, 4, 3]));
        // Game 3: placements [1, 3, 2, 4]
        stats.record(&GameOutcome(vec![1, 3, 2, 4]));

        // Verify raw placements per slot
        assert_eq!(stats.placements[0], vec![2, 1, 0, 0]); // slot 0: 2x 1st, 1x 2nd
        assert_eq!(stats.placements[1], vec![1, 1, 1, 0]); // slot 1: 1x 1st, 1x 2nd, 1x 3rd
        assert_eq!(stats.placements[2], vec![0, 1, 1, 1]); // slot 2: 1x 2nd, 1x 3rd, 1x 4th
        assert_eq!(stats.placements[3], vec![0, 0, 1, 2]); // slot 3: 1x 3rd, 2x 4th
    }

    #[test]
    fn test_eval_stats_slot_to_model_aggregation() {
        // Test the aggregation logic used in print_n_player_summary
        let mut stats = EvalStats::new(4);

        // Record outcomes where slots 0,1,2 use model 0 and slot 3 uses model 1
        stats.record(&GameOutcome(vec![1, 2, 3, 4]));
        stats.record(&GameOutcome(vec![2, 1, 4, 3]));

        // slot_to_model: slots 0,1,2 -> model 0, slot 3 -> model 1
        let slot_to_model = [0, 0, 0, 1];

        // Simulate the aggregation logic from print_n_player_summary
        let num_models = slot_to_model.iter().copied().max().map_or(0, |m| m + 1);
        let mut merged_placements: Vec<Vec<usize>> = vec![vec![0; 4]; num_models];
        let mut slot_counts: Vec<usize> = vec![0; num_models];

        for (slot, &model_idx) in slot_to_model.iter().enumerate() {
            for (p, &count) in stats.placements[slot].iter().enumerate() {
                merged_placements[model_idx][p] += count;
            }
            slot_counts[model_idx] += 1;
        }

        // Model 0 (slots 0,1,2): sum of their placements
        // Slot 0: [1,1,0,0], Slot 1: [1,1,0,0], Slot 2: [0,0,1,1]
        // Sum: [2,2,1,1]
        assert_eq!(merged_placements[0], vec![2, 2, 1, 1]);
        assert_eq!(slot_counts[0], 3);

        // Model 1 (slot 3): [0,0,1,1]
        assert_eq!(merged_placements[1], vec![0, 0, 1, 1]);
        assert_eq!(slot_counts[1], 1);
    }

    #[test]
    fn test_eval_stats_reward_aggregation() {
        // Test reward aggregation by model
        let mut stats = EvalStats::new(4);

        // Record game with rewards
        stats.record_with_rewards(&GameOutcome(vec![1, 2, 3, 4]), &[10.0, 5.0, 2.0, 1.0], 10);
        stats.record_with_rewards(&GameOutcome(vec![2, 1, 4, 3]), &[8.0, 6.0, 0.0, 3.0], 12);

        // Verify rewards are recorded
        assert_eq!(stats.game_rewards.len(), 2);
        assert_eq!(stats.game_rewards[0], vec![10.0, 5.0, 2.0, 1.0]);
        assert_eq!(stats.game_rewards[1], vec![8.0, 6.0, 0.0, 3.0]);

        // slot_to_model: slots 0,1 -> model 0, slots 2,3 -> model 1
        let slot_to_model = [0, 0, 1, 1];

        // Simulate reward aggregation from print_reward_summary
        let num_models = 2;
        let mut model_reward_sums: Vec<f64> = vec![0.0; num_models];
        let mut model_reward_counts: Vec<usize> = vec![0; num_models];

        for (slot, &model_idx) in slot_to_model.iter().enumerate() {
            for game_rewards in &stats.game_rewards {
                if let Some(&reward) = game_rewards.get(slot) {
                    model_reward_sums[model_idx] += reward;
                    model_reward_counts[model_idx] += 1;
                }
            }
        }

        // Model 0 (slots 0,1): rewards (10+8) + (5+6) = 29, count = 4
        assert_eq!(model_reward_sums[0], 29.0);
        assert_eq!(model_reward_counts[0], 4);
        let model_0_avg = model_reward_sums[0] / model_reward_counts[0] as f64;
        assert!((model_0_avg - 7.25).abs() < 0.001);

        // Model 1 (slots 2,3): rewards (2+0) + (1+3) = 6, count = 4
        assert_eq!(model_reward_sums[1], 6.0);
        assert_eq!(model_reward_counts[1], 4);
        let model_1_avg = model_reward_sums[1] / model_reward_counts[1] as f64;
        assert!((model_1_avg - 1.5).abs() < 0.001);
    }
}
