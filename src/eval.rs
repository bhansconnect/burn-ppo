/// Evaluation subcommand for assessing trained models
///
/// Features:
/// - Watch mode: visualize games with ASCII rendering
/// - Stats mode: parallel game execution with win/loss/draw statistics
/// - Temperature-based sampling with schedule support
/// - ELO delta calculation for 2-player games

use std::io::{self, Write as IoWrite};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::checkpoint::{CheckpointManager, CheckpointMetadata};
use crate::config::{Config, EvalArgs};
use crate::dispatch_env;
use crate::env::{Environment, GameOutcome, VecEnv};
use crate::network::ActorCritic;

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
    pub fn new(initial: f32, final_temp: f32, cutoff: Option<usize>, decay: bool) -> Self {
        Self {
            initial,
            final_temp,
            cutoff,
            decay,
        }
    }

    /// Create a TempSchedule from CLI arguments
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
        self.initial + t * (self.final_temp - self.initial)
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
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Apply temperature: softmax(logits / temp)
    let scaled: Vec<f32> = masked_logits.iter().map(|x| x / temperature).collect();
    let max = scaled
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();

    if sum == 0.0 {
        // All actions masked or numerical issues, fall back to first valid
        return mask
            .and_then(|m| m.iter().position(|&v| v))
            .unwrap_or(0);
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

    let score = (p0_wins as f64 + 0.5 * draws as f64) / n; // P0's score [0, 1]

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
    let mut indexed: Vec<(usize, f32)> = rewards.iter().cloned().enumerate().collect();
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
        let winner = placements.iter().position(|&p| p == 1).unwrap();
        GameOutcome::Winner(winner)
    } else {
        GameOutcome::Placements(placements)
    }
}

/// Statistics tracker for evaluation results
pub struct EvalStats {
    num_players: usize,
    /// Wins per checkpoint [num_checkpoints]
    wins: Vec<usize>,
    /// Losses per checkpoint [num_checkpoints]
    losses: Vec<usize>,
    /// Draws count (all tied)
    draws: usize,
    /// Placement counts per checkpoint [num_checkpoints][num_placements]
    placements: Vec<Vec<usize>>,
    /// Total games played
    total_games: usize,
    /// Total rewards per game [game_idx][player]
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
    pub fn record_with_rewards(
        &mut self,
        outcome: &GameOutcome,
        rewards: &[f32],
        length: usize,
    ) {
        self.game_rewards
            .push(rewards.iter().map(|&r| r as f64).collect());
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

        println!("Checkpoint: {}", checkpoint_names.first().unwrap_or(&String::new()));
        println!();
        println!("Reward Distribution:");
        println!("  Mean:   {:.1} ± {:.1} (std)", mean, std);
        println!("  Range:  [{:.1}, {:.1}]", min, max);
        println!("  25th:   {:.1}", p25);
        println!("  Median: {:.1}", p50);
        println!("  75th:   {:.1}", p75);

        // Episode length stats if available
        if !self.episode_lengths.is_empty() {
            let avg_len: f64 =
                self.episode_lengths.iter().sum::<usize>() as f64 / self.episode_lengths.len() as f64;
            let min_len = self.episode_lengths.iter().min().copied().unwrap_or(0);
            let max_len = self.episode_lengths.iter().max().copied().unwrap_or(0);
            println!();
            println!("Episode Length:");
            println!("  Mean:  {:.1}", avg_len);
            println!("  Range: [{}, {}]", min_len, max_len);
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
            println!(
                "\nELO Delta: {:+.0} ± {:.0} ({})",
                delta, stderr, stronger
            );
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
            println!("  {}", name);
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
                println!("  Player {}: {:.3}", player, mean);
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
    format!("{}{}", n, suffix)
}

/// Load a model from a checkpoint directory
fn load_model_from_checkpoint<B: Backend>(
    checkpoint_path: &Path,
    device: &B::Device,
) -> Result<(ActorCritic<B>, CheckpointMetadata)> {
    // Load metadata to get network dimensions
    let metadata_path = checkpoint_path.join("metadata.json");
    let metadata_json = std::fs::read_to_string(&metadata_path)
        .context("Failed to read checkpoint metadata")?;
    let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

    // Create config with network architecture from metadata
    let config = Config {
        hidden_size: metadata.hidden_size,
        num_hidden: metadata.num_hidden,
        ..Config::default()
    };

    // Load model
    let (model, _) = CheckpointManager::load::<B>(checkpoint_path, &config, device)?;

    Ok((model, metadata))
}

/// Run evaluation based on command-line arguments
pub fn run_evaluation<B: Backend>(args: &EvalArgs, device: &B::Device) -> Result<()> {
    // Load all checkpoints
    let mut models = Vec::new();
    let mut checkpoint_names = Vec::new();
    let mut metadata_opt: Option<CheckpointMetadata> = None;

    for path in &args.checkpoints {
        // Resolve symlinks (latest, best)
        let resolved = if path.is_symlink() {
            path.read_link()
                .map(|target| path.parent().unwrap_or(path).join(target))
                .unwrap_or_else(|_| path.clone())
        } else {
            path.clone()
        };

        let (model, metadata) = load_model_from_checkpoint::<B>(&resolved, device)?;

        // Store first metadata for environment info
        if metadata_opt.is_none() {
            metadata_opt = Some(metadata.clone());
        } else {
            // Verify all checkpoints are for the same environment
            let first = metadata_opt.as_ref().unwrap();
            if metadata.obs_dim != first.obs_dim
                || metadata.action_count != first.action_count
                || metadata.num_players != first.num_players
            {
                anyhow::bail!(
                    "Checkpoint {} has different dimensions than first checkpoint",
                    path.display()
                );
            }
        }

        models.push(model);
        checkpoint_names.push(path.display().to_string());
    }

    let metadata = metadata_opt.context("No checkpoints provided")?;

    // Verify we have right number of checkpoints for environment
    if metadata.num_players > 1 && models.len() != metadata.num_players {
        if models.len() == 1 {
            // Self-play: duplicate the model for all players
            println!("Self-play mode: using same checkpoint for all {} players", metadata.num_players);
            for _ in 1..metadata.num_players {
                let (model, _) = load_model_from_checkpoint::<B>(&args.checkpoints[0], device)?;
                models.push(model);
                checkpoint_names.push(checkpoint_names[0].clone());
            }
        } else {
            anyhow::bail!(
                "Expected {} checkpoints for {}-player game, got {}",
                metadata.num_players,
                metadata.num_players,
                models.len()
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
            &checkpoint_names,
            &metadata,
            num_games,
            args,
            device,
        )
    } else {
        run_stats_mode::<B>(
            &models,
            &checkpoint_names,
            &metadata,
            num_games,
            args,
            device,
        )
    }
}

/// Run evaluation in watch mode (sequential, with rendering)
fn run_watch_mode<B: Backend>(
    models: &[ActorCritic<B>],
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
    dispatch_env!(metadata.env_name, {
        run_watch_mode_env::<B, E>(
            models,
            checkpoint_names,
            num_games,
            &temp_schedule,
            args.step,
            args.animate,
            args.fps,
            &mut rng,
            device,
        )
    })
}

/// Watch mode implementation for a specific environment type
fn run_watch_mode_env<B: Backend, E: Environment>(
    models: &[ActorCritic<B>],
    checkpoint_names: &[String],
    num_games: usize,
    temp_schedule: &TempSchedule,
    step_mode: bool,
    animate: bool,
    fps: u32,
    rng: &mut StdRng,
    device: &B::Device,
) -> Result<()> {
    use rand::RngCore;
    let mut stats = EvalStats::new(E::NUM_PLAYERS);

    for game_idx in 0..num_games {
        println!("\n=== Game {} ===", game_idx + 1);
        for (i, name) in checkpoint_names.iter().enumerate() {
            println!("Player {}: {}", i, name);
        }
        println!();

        let mut env = E::new(rng.next_u64());
        let mut obs = env.reset();
        let mut total_rewards = vec![0.0f32; E::NUM_PLAYERS];
        let mut move_num = 0;
        let mut is_first_frame = true;
        let mut last_frame_lines = 0usize;
        let frame_duration = Duration::from_millis(1000 / fps as u64);

        loop {
            let frame_start = if animate { Some(Instant::now()) } else { None };
            let current_player = env.current_player();
            let model = &models[current_player];
            let mask = env.action_mask();

            // Get action from model
            let obs_tensor: Tensor<B, 2> =
                Tensor::<B, 1>::from_floats(&obs[..], device).reshape([1, E::OBSERVATION_DIM]);
            let (logits, _) = model.forward(obs_tensor);
            let logits_vec: Vec<f32> = logits.to_data().to_vec().unwrap();

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
                    print!("{}\n", rendered);
                    // Trailing spaces clear any leftover characters from previous longer text
                    println!("Action: {} (temp={:.2})                    ", action, temp);
                    io::stdout().flush().ok();
                    last_frame_lines = frame_lines;
                    is_first_frame = false;
                    // Sleep only the remainder of frame time
                    if let Some(start) = frame_start {
                        let elapsed = start.elapsed();
                        if elapsed < frame_duration {
                            thread::sleep(frame_duration - elapsed);
                        }
                    }
                } else {
                    println!("{}", rendered);
                    println!(
                        "Player {} selects action {} (temp={:.2})",
                        current_player, action, temp
                    );
                }
            } else {
                println!(
                    "Player {} selects action {} (temp={:.2})",
                    current_player, action, temp
                );
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
                        println!("{}", rendered);
                        println!(); // Extra line to separate from outcome
                    } else {
                        println!("{}", rendered);
                    }
                }

                let outcome = determine_outcome(&env, &total_rewards);
                stats.record(&outcome);

                match &outcome {
                    GameOutcome::Winner(w) => println!("\nWinner: Player {}", w),
                    GameOutcome::Tie => println!("\nGame ended in a tie"),
                    GameOutcome::Placements(p) => {
                        println!("\nFinal placements: {:?}", p);
                    }
                }
                println!("Total rewards: {:?}", total_rewards);
                println!("Game length: {} moves", move_num);
                break;
            }
        }
    }

    if num_games > 1 {
        stats.print_summary(checkpoint_names);
    }

    Ok(())
}

/// Wait for user to press Enter
fn wait_for_enter() {
    print!("Press Enter to continue...");
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok();
}

/// Run evaluation in stats mode (parallel)
fn run_stats_mode<B: Backend>(
    models: &[ActorCritic<B>],
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
    dispatch_env!(metadata.env_name, {
        run_stats_mode_env::<B, E>(
            models,
            checkpoint_names,
            num_games,
            args.num_envs,
            &temp_schedule,
            &mut rng,
            device,
        )
    })
}

/// Stats mode implementation for a specific environment type
fn run_stats_mode_env<B: Backend, E: Environment>(
    models: &[ActorCritic<B>],
    checkpoint_names: &[String],
    num_games: usize,
    num_envs: usize,
    temp_schedule: &TempSchedule,
    rng: &mut StdRng,
    device: &B::Device,
) -> Result<()> {
    use rand::RngCore;
    let num_players = E::NUM_PLAYERS;
    let obs_dim = E::OBSERVATION_DIM;
    let action_count = E::ACTION_COUNT;

    // For 2-player games, swap positions every other game for fairness
    let swap_positions = num_players == 2;

    // Create vectorized environment with unique seeds per env
    let base_seed = rng.next_u64();
    let mut vec_env: VecEnv<E> = VecEnv::new(num_envs, |i| E::new(base_seed.wrapping_add(i as u64)));
    let mut stats = EvalStats::new(num_players);

    // Track move counts per environment (for temperature schedule)
    let mut move_counts = vec![0usize; num_envs];

    // Track which position each checkpoint is in per environment
    // For position swapping: positions[env_idx][checkpoint_idx] = player_idx
    let mut positions: Vec<Vec<usize>> = (0..num_envs)
        .map(|env_idx| {
            if swap_positions && env_idx % 2 == 1 {
                vec![1, 0] // Swapped
            } else {
                (0..num_players).collect() // Normal
            }
        })
        .collect();

    println!(
        "Running {} games across {} parallel environments...",
        num_games, num_envs
    );

    let mut games_completed = 0;

    while games_completed < num_games {
        let obs_flat = vec_env.get_observations();
        let current_players = vec_env.get_current_players();
        let masks_flat = vec_env.get_action_masks();

        // Build actions for each environment
        let mut actions = vec![0usize; num_envs];

        // Group environments by which model should act (based on position mapping)
        for model_idx in 0..models.len() {
            // Find environments where this model's checkpoint is the current player
            let mut env_indices = Vec::new();
            let mut env_obs = Vec::new();
            let mut env_masks = Vec::new();

            for env_idx in 0..num_envs {
                let current_player = current_players[env_idx];
                // Which checkpoint controls this player position in this env?
                let checkpoint_for_player = positions[env_idx]
                    .iter()
                    .position(|&p| p == current_player)
                    .unwrap_or(current_player);

                if checkpoint_for_player == model_idx {
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

            // Batch inference for this model
            let batch_size = env_indices.len();
            let obs_tensor: Tensor<B, 2> =
                Tensor::<B, 1>::from_floats(&env_obs[..], device).reshape([batch_size, obs_dim]);
            let (logits_tensor, _) = models[model_idx].forward(obs_tensor);
            let logits_flat: Vec<f32> = logits_tensor.to_data().to_vec().unwrap();

            // Sample actions
            for (i, &env_idx) in env_indices.iter().enumerate() {
                let logit_offset = i * action_count;
                let logits = &logits_flat[logit_offset..logit_offset + action_count];

                let mask = if !env_masks.is_empty() {
                    let mask_offset = i * action_count;
                    Some(&env_masks[mask_offset..mask_offset + action_count])
                } else {
                    None
                };

                let temp = temp_schedule.get_temp(move_counts[env_idx]);
                actions[env_idx] = sample_with_temperature(logits, mask, temp, rng);
            }
        }

        // Step all environments
        let (_, _, _, completed_episodes) = vec_env.step(&actions);

        // Update move counts
        for count in move_counts.iter_mut() {
            *count += 1;
        }

        // Process completed episodes
        for ep in completed_episodes.into_iter() {
            games_completed += 1;

            // Use the env_index from the completed episode
            let env_idx = ep.env_index;

            let mut checkpoint_rewards = vec![0.0f32; num_players];
            for (player_idx, &reward) in ep.total_rewards.iter().enumerate() {
                // Which checkpoint was at this player position?
                let checkpoint_idx = positions[env_idx]
                    .iter()
                    .position(|&p| p == player_idx)
                    .unwrap_or(player_idx);
                checkpoint_rewards[checkpoint_idx] = reward;
            }

            // Create a dummy env to check game_outcome (won't have the actual state)
            let outcome = {
                let placements = rewards_to_placements(&checkpoint_rewards);
                let first_count = placements.iter().filter(|&&p| p == 1).count();
                if first_count == placements.len() {
                    GameOutcome::Tie
                } else if first_count == 1 {
                    let winner = placements.iter().position(|&p| p == 1).unwrap();
                    GameOutcome::Winner(winner)
                } else {
                    GameOutcome::Placements(placements)
                }
            };

            stats.record_with_rewards(&outcome, &checkpoint_rewards, ep.length);

            // Reset move count and swap positions for this env
            move_counts[env_idx] = 0;
            if swap_positions {
                // Toggle between normal and swapped
                positions[env_idx] = if positions[env_idx] == vec![0, 1] {
                    vec![1, 0]
                } else {
                    vec![0, 1]
                };
            }

            if games_completed >= num_games {
                break;
            }

            // Progress indicator
            if games_completed % 100 == 0 {
                println!("  {} / {} games completed", games_completed, num_games);
            }
        }
    }

    stats.print_summary(checkpoint_names);

    Ok(())
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
            num_games: 100,
            num_envs: 64,
            watch: false,
            step: false,
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
            num_games: 100,
            num_envs: 64,
            watch: false,
            step: false,
            seed: None,
            temperature: 0.5,
            temp_final: Some(0.1), // final without cutoff = error
            temp_cutoff: None,
            temp_decay: false,
        };
        let result = TempSchedule::from_args(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--temp-final requires --temp-cutoff"));
    }

    #[test]
    fn test_temp_schedule_from_args_decay_without_cutoff() {
        use std::path::PathBuf;

        let args = EvalArgs {
            checkpoints: vec![PathBuf::from("test")],
            num_games: 100,
            num_envs: 64,
            watch: false,
            step: false,
            seed: None,
            temperature: 0.5,
            temp_final: None,
            temp_cutoff: None,
            temp_decay: true, // decay without cutoff = error
        };
        let result = TempSchedule::from_args(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--temp-decay requires --temp-cutoff"));
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
            assert!(count > 50, "Action count {} too low", count);
            assert!(count < 150, "Action count {} too high", count);
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
