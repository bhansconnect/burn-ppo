#![cfg_attr(not(test), deny(warnings))]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::float_cmp,
        clippy::default_trait_access,
        clippy::single_range_in_vec_init,
        clippy::allow_attributes_without_reason,
    )
)]
#![recursion_limit = "256"]

mod backend;
mod checkpoint;
mod config;
mod env;
mod envs;
mod eval;
mod human;
mod metrics;
mod network;
mod normalization;
mod ppo;
mod profile;
mod progress;
mod tournament;
mod utils;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;
use clap::Parser;
use rand::SeedableRng;

use crate::checkpoint::{
    load_metadata, load_normalizer, load_optimizer, load_rng_state, save_normalizer,
    save_optimizer, save_rng_state, update_training_rating, CheckpointManager, CheckpointMetadata,
};
use crate::config::{Cli, CliArgs, Command, Config};
use crate::env::{compute_avg_points, Environment, GameOutcome, VecEnv};
use crate::envs::{CartPole, ConnectFour, LiarsDice};
use crate::eval::run_challenger_eval;
use crate::metrics::MetricsLogger;
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;
use crate::ppo::{
    collect_rollouts, compute_gae, compute_gae_multiplayer, ppo_update, RolloutBuffer,
};
use crate::progress::TrainingProgress;
use crate::tournament::print_rating_guide;
use std::collections::VecDeque;

/// Extract run name from a checkpoint path
///
/// e.g., "`runs/cartpole_001/checkpoints/best`" -> "`cartpole_001`"
fn extract_run_name_from_checkpoint_path(checkpoint_path: &std::path::Path) -> Option<String> {
    // Navigate up from checkpoint: checkpoints/<name> -> checkpoints -> run_dir
    let run_dir = checkpoint_path.parent()?.parent()?;
    run_dir.file_name()?.to_str().map(String::from)
}

/// Get the step number from the best checkpoint symlink
///
/// Reads the symlink target and extracts the step number from the directory name.
/// e.g., "`step_00010240`" -> 10240
fn get_best_checkpoint_step(best_path: &std::path::Path) -> Option<usize> {
    let target = std::fs::read_link(best_path).ok()?;
    let name = target.file_name()?.to_str()?;
    // Parse "step_XXXXXXXX" format
    name.strip_prefix("step_").and_then(|s| s.parse().ok())
}

/// Count checkpoint directories in a run's checkpoints folder
///
/// Returns the count of directories matching "step_*" pattern
fn count_checkpoints(run_dir: &std::path::Path) -> usize {
    let checkpoints_dir = run_dir.join("checkpoints");
    if !checkpoints_dir.exists() {
        return 0;
    }
    std::fs::read_dir(&checkpoints_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("step_"))
                })
                .count()
        })
        .unwrap_or(0)
}

/// Check if run directory exists and prompt user for deletion confirmation
///
/// Returns Ok(true) if we should proceed (either didn't exist or user confirmed deletion)
/// Returns Ok(false) if user declined deletion
fn check_run_exists_and_prompt(run_dir: &std::path::Path) -> Result<bool> {
    use std::io::Write;

    if !run_dir.exists() {
        return Ok(true);
    }

    let checkpoint_count = count_checkpoints(run_dir);

    eprintln!();
    eprintln!(
        "Warning: Run '{}' already exists at {}",
        run_dir
            .file_name()
            .expect("run_dir has filename")
            .to_string_lossy(),
        run_dir.display()
    );
    eprintln!("This run contains {checkpoint_count} checkpoint(s).");
    eprintln!();
    eprint!("Delete existing run and continue? [y/N]: ");

    // Flush stderr to ensure prompt is displayed
    std::io::stderr().flush()?;

    // Read user input
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    if input == "y" || input == "yes" {
        eprintln!("Deleting existing run...");
        std::fs::remove_dir_all(run_dir)?;
        Ok(true)
    } else {
        eprintln!("Aborting.");
        Ok(false)
    }
}

/// Training mode determined by CLI arguments
enum TrainingMode {
    /// Fresh training run
    Fresh,
    /// Resume from existing run (same config, continue in place)
    Resume {
        run_dir: PathBuf,
        checkpoint_dir: PathBuf,
    },
    /// Fork from checkpoint (new run, allows config changes)
    Fork { checkpoint_dir: PathBuf },
}

/// Run training with a specific environment type
///
/// This is the core training function, generic over the environment type and backend.
/// Full static dispatch - `VecEnv`<CartPole> and `VecEnv`<ConnectFour> are separate types.
fn run_training<TB, E, F>(
    mode: &TrainingMode,
    config: &Config,
    run_dir: &Path,
    resumed_metadata: Option<&CheckpointMetadata>,
    device: &TB::Device,
    running: &Arc<AtomicBool>,
    env_factory: F,
) -> Result<()>
where
    TB: burn::tensor::backend::AutodiffBackend,
    E: Environment,
    F: Fn(usize) -> E,
{
    // Initialize RNG
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    // Create vectorized environment
    let num_envs = config.num_envs();
    let mut vec_env = VecEnv::new(num_envs, env_factory);

    let obs_dim = E::OBSERVATION_DIM;
    let action_count = E::ACTION_COUNT;
    let num_players = E::NUM_PLAYERS as u8;
    println!(
        "Created {} {} environments (obs_dim={}, actions={}, players={})",
        num_envs,
        E::NAME,
        obs_dim,
        action_count,
        num_players
    );

    // Create observation normalizer if enabled
    let mut obs_normalizer: Option<ObsNormalizer> = if config.normalize_obs {
        Some(ObsNormalizer::new(obs_dim, 10.0))
    } else {
        None
    };

    // Create optimizer with gradient clipping
    let optimizer_config = AdamConfig::new()
        .with_epsilon(config.adam_epsilon as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            config.max_grad_norm as f32,
        )));

    // Initialize model and optimizer based on mode
    let (mut model, mut optimizer, mut global_step, mut recent_returns, best_return) = match mode {
        TrainingMode::Fresh => {
            let model: ActorCritic<TB> =
                ActorCritic::new(obs_dim, action_count, num_players as usize, config, device);
            let optimizer = optimizer_config.init();
            println!("Created ActorCritic network");
            (model, optimizer, 0, Vec::new(), f32::NEG_INFINITY)
        }
        TrainingMode::Resume { checkpoint_dir, .. } | TrainingMode::Fork { checkpoint_dir } => {
            let metadata = resumed_metadata.expect("resumed_metadata required for Resume/Fork");

            // Check for training completion
            if metadata.step >= config.total_timesteps {
                println!(
                    "Training already complete at step {}. Use --total-timesteps to extend.",
                    metadata.step
                );
                return Ok(());
            }

            // Load model from checkpoint
            let (model, _) = CheckpointManager::load::<TB>(checkpoint_dir, config, device)
                .with_context(|| {
                    format!(
                        "Failed to load checkpoint from {}",
                        checkpoint_dir.display()
                    )
                })?;

            // Create optimizer and try to load saved state
            let optimizer = optimizer_config.init();
            let optimizer =
                match load_optimizer::<TB, _, ActorCritic<TB>>(optimizer, checkpoint_dir, device) {
                    Ok(opt) => opt,
                    Err(e) => {
                        eprintln!("Warning: Failed to load optimizer state: {e}");
                        optimizer_config.init()
                    }
                };

            // Load observation normalizer if it was saved
            if config.normalize_obs {
                match load_normalizer(checkpoint_dir) {
                    Ok(Some(loaded_norm)) => {
                        obs_normalizer = Some(loaded_norm);
                        println!("Loaded observation normalizer from checkpoint");
                    }
                    Ok(None) => {
                        // No normalizer saved, keep the fresh one
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load normalizer: {e}");
                    }
                }
            }

            // Load RNG state if saved (for reproducible continuation)
            match load_rng_state(checkpoint_dir) {
                Ok(Some(loaded_rng)) => {
                    rng = loaded_rng;
                    println!("Loaded RNG state from checkpoint");
                }
                Ok(None) => {
                    // No RNG state saved, keep the fresh one
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load RNG state: {e}");
                }
            }

            let step = metadata.step;
            let recent_returns = metadata.recent_returns.clone();
            let best_return = metadata.best_avg_return.unwrap_or(f32::NEG_INFINITY);

            println!(
                "Loaded checkpoint from step {} (avg return: {:.1})",
                step, metadata.avg_return
            );

            (model, optimizer, step, recent_returns, best_return)
        }
    };

    // Create rollout buffer with inference backend (non-autodiff)
    // This prevents memory accumulation from autodiff graph during rollout
    let mut buffer: RolloutBuffer<TB::InnerBackend> =
        RolloutBuffer::new(config.num_steps, num_envs, obs_dim, num_players, device);

    // Create metrics logger
    let mut logger = MetricsLogger::new(run_dir)?;
    // Log hyperparameters for new runs (Fresh and Fork create new run directories)
    if matches!(mode, TrainingMode::Fresh | TrainingMode::Fork { .. }) {
        logger.log_hparams(config)?;
    }
    logger.flush()?;

    // Create checkpoint manager
    let mut checkpoint_manager = CheckpointManager::new(run_dir)?;
    checkpoint_manager.set_best_avg_return(best_return);
    let mut last_checkpoint_step = global_step;
    let mut last_checkpoint_time = std::time::Instant::now();
    let mut warned_challenger_eval_time = false;

    // Create initial checkpoint at step 0 for fresh training
    if matches!(mode, TrainingMode::Fresh) {
        let metadata = CheckpointMetadata {
            step: 0,
            avg_return: 0.0, // No episodes completed yet
            rng_seed: config.seed,
            best_avg_return: None,
            recent_returns: Vec::new(),
            forked_from: None,
            obs_dim,
            action_count,
            num_players: num_players as usize,
            hidden_size: config.hidden_size,
            num_hidden: config.num_hidden,
            activation: config.activation.clone(),
            split_networks: config.split_networks,
            env_name: E::NAME.to_string(),
            training_rating: 25.0,
            training_uncertainty: 25.0 / 3.0,
        };

        checkpoint_manager.save(&model, &metadata, true)?;

        // Log initial challenger metrics for aim_watcher
        logger.log_scalar("challenger/best_step", 0.0, 0)?;
        logger.log_scalar("challenger/current_rating", 25.0, 0)?;
        logger.log_scalar("challenger/best_rating", 25.0, 0)?;
        logger.flush()?;
    }

    // Training state
    let steps_per_update = config.num_steps * num_envs;
    let total_updates = config.total_timesteps / steps_per_update;
    let remaining_timesteps = config.total_timesteps.saturating_sub(global_step);
    let num_updates = remaining_timesteps / steps_per_update;

    // For LR annealing, we need to know how many updates have already happened
    let update_offset = global_step / steps_per_update;

    println!(
        "Training for {} timesteps ({} updates of {} steps each)",
        config.total_timesteps, num_updates, steps_per_update
    );
    println!("---");

    // Progress bar (with multiplayer support)
    let progress =
        TrainingProgress::new_with_players(config.total_timesteps as u64, num_players as usize);

    // Timing for SPS calculation
    let training_start = std::time::Instant::now();
    let mut last_log_time = training_start;
    let mut last_log_step = global_step;

    // Phase timing accumulators (reset on log)
    let mut rollout_time_acc = std::time::Duration::ZERO;
    let mut gae_time_acc = std::time::Duration::ZERO;
    let mut update_time_acc = std::time::Duration::ZERO;

    // Episode tracking for metrics (cleared on each log)
    let mut episodes_since_log: Vec<(f32, usize)> = Vec::new(); // (return, length)

    // Episode tracking for checkpoint selection (cleared on each checkpoint)
    let mut episodes_since_checkpoint: Vec<f32> = Vec::new();

    // Multiplayer tracking (only used when num_players > 1)
    let num_players_usize = num_players as usize;
    // Per-player rolling returns (last 100 episodes)
    let mut recent_returns_per_player: Vec<VecDeque<f32>> = (0..num_players_usize)
        .map(|_| VecDeque::with_capacity(100))
        .collect();
    // Rolling game outcomes for win rate calculation
    let mut recent_outcomes: VecDeque<GameOutcome> = VecDeque::with_capacity(100);
    // Per-player episode data for logging (cleared on each log)
    let mut episodes_since_log_mp: Vec<(Vec<f32>, Option<GameOutcome>)> = Vec::new();

    // Training loop
    for update in 0..num_updates {
        profile::profile_scope!("training_update");

        if !running.load(Ordering::SeqCst) {
            println!("\nInterrupted by user");
            break;
        }

        // Learning rate annealing (using total updates for proper continuation)
        let lr = if config.lr_anneal {
            let actual_update = update_offset + update;
            let progress_frac = actual_update as f64 / total_updates as f64;
            config.learning_rate * (1.0 - progress_frac)
        } else {
            config.learning_rate
        };

        // Entropy coefficient annealing (decay to 10% of initial value)
        let ent_coef = if config.entropy_anneal {
            let actual_update = update_offset + update;
            let progress_frac = actual_update as f64 / total_updates as f64;
            config.entropy_coef * progress_frac.mul_add(-0.9, 1.0) // Decay to 10% of initial
        } else {
            config.entropy_coef
        };

        // Collect rollouts using non-autodiff model for inference
        let rollout_start = std::time::Instant::now();
        let inference_model = model.valid();
        let completed_episodes = collect_rollouts(
            &inference_model,
            &mut vec_env,
            &mut buffer,
            config.num_steps,
            device,
            &mut rng,
            obs_normalizer.as_mut(),
        );
        rollout_time_acc += rollout_start.elapsed();

        // Track episode returns for metrics and checkpointing
        for ep in &completed_episodes {
            let ep_return = ep.total_reward(); // Player 0's return for single-agent, or overall
                                               // For logging (cleared each log)
            episodes_since_log.push((ep_return, ep.length));
            // For checkpoint best selection (cleared each checkpoint)
            episodes_since_checkpoint.push(ep_return);
            // For progress bar and checkpoint metadata (rolling window)
            recent_returns.push(ep_return);
            if recent_returns.len() > 100 {
                recent_returns.remove(0);
            }

            // Multiplayer tracking
            if num_players > 1 {
                // Per-player returns
                for (p, &reward) in ep.total_rewards.iter().enumerate() {
                    if p < recent_returns_per_player.len() {
                        recent_returns_per_player[p].push_back(reward);
                        if recent_returns_per_player[p].len() > 100 {
                            recent_returns_per_player[p].pop_front();
                        }
                    }
                }

                // Track outcome for win rate calculation
                if let Some(ref outcome) = ep.outcome {
                    recent_outcomes.push_back(outcome.clone());
                    if recent_outcomes.len() > 100 {
                        recent_outcomes.pop_front();
                    }
                }

                // Store for logging
                episodes_since_log_mp.push((ep.total_rewards.clone(), ep.outcome.clone()));
            }
        }

        // Compute bootstrap value using non-autodiff inference
        let mut obs_flat = vec_env.get_observations();
        if let Some(ref norm) = obs_normalizer {
            norm.normalize_batch(&mut obs_flat, obs_dim);
        }
        let obs_tensor: Tensor<TB::InnerBackend, 2> =
            Tensor::<TB::InnerBackend, 1>::from_floats(obs_flat.as_slice(), device)
                .reshape([num_envs, obs_dim]);
        let (_, all_values) = inference_model.forward(obs_tensor);

        // Compute GAE - dispatch based on number of players
        let gae_start = std::time::Instant::now();
        if num_players > 1 {
            // Multi-player: use full values tensor [num_envs, num_players]
            compute_gae_multiplayer(
                &mut buffer,
                all_values,
                config.gamma as f32,
                config.gae_lambda as f32,
                num_players,
                device,
            );
        } else {
            // Single-player: extract player 0's values [num_envs]
            let last_values: Tensor<TB::InnerBackend, 1> =
                all_values.slice([0..num_envs, 0..1]).flatten(0, 1);
            compute_gae(
                &mut buffer,
                last_values,
                config.gamma as f32,
                config.gae_lambda as f32,
                device,
            );
        }
        gae_time_acc += gae_start.elapsed();

        // PPO update
        let update_start = std::time::Instant::now();
        let (updated_model, metrics) = ppo_update(
            model,
            &buffer,
            &mut optimizer,
            config,
            lr,
            ent_coef,
            &mut rng,
        );
        update_time_acc += update_start.elapsed();
        model = updated_model;

        global_step += steps_per_update;

        // Update progress bar
        if num_players > 1 {
            // Multiplayer: show per-player returns and win rates
            let returns_per_player: Vec<f32> = recent_returns_per_player
                .iter()
                .map(|rp| {
                    if rp.is_empty() {
                        0.0
                    } else {
                        rp.iter().sum::<f32>() / rp.len() as f32
                    }
                })
                .collect();
            let (avg_points, draw_rate) = compute_avg_points(&recent_outcomes, num_players_usize);
            progress.update_multiplayer(
                global_step as u64,
                &returns_per_player,
                &avg_points,
                draw_rate,
            );
        } else {
            // Single-player: show average return
            let avg_return = if recent_returns.is_empty() {
                0.0
            } else {
                recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
            };
            progress.update(global_step as u64, avg_return);
        }

        // Log metrics (at least once per log_freq steps)
        let should_log = update == 0
            || global_step / config.log_freq > (global_step - steps_per_update) / config.log_freq;
        if should_log {
            logger.log_scalar("train/policy_loss", metrics.policy_loss, global_step)?;
            logger.log_scalar("train/value_loss", metrics.value_loss, global_step)?;
            logger.log_scalar("train/entropy", metrics.entropy, global_step)?;
            logger.log_scalar("train/approx_kl", metrics.approx_kl, global_step)?;
            logger.log_scalar("train/clip_fraction", metrics.clip_fraction, global_step)?;
            logger.log_scalar("train/learning_rate", lr as f32, global_step)?;
            logger.log_scalar(
                "train/explained_variance",
                metrics.explained_variance,
                global_step,
            )?;
            logger.log_scalar("train/total_loss", metrics.total_loss, global_step)?;
            logger.log_scalar("train/value_mean", metrics.value_mean, global_step)?;
            logger.log_scalar("train/returns_mean", metrics.returns_mean, global_step)?;

            // Steps per second (since last log)
            let now = std::time::Instant::now();
            let interval_elapsed = now.duration_since(last_log_time).as_secs_f32();
            let interval_steps = (global_step - last_log_step) as f32;
            let sps = if interval_elapsed > 0.0 {
                interval_steps / interval_elapsed
            } else {
                0.0
            };
            logger.log_scalar("perf/sps", sps, global_step)?;

            // Phase timing
            let rollout_secs = rollout_time_acc.as_secs_f32();
            let gae_secs = gae_time_acc.as_secs_f32();
            let update_secs = update_time_acc.as_secs_f32();
            let total_secs = rollout_secs + gae_secs + update_secs;

            logger.log_scalar("perf/rollout_time", rollout_secs, global_step)?;
            logger.log_scalar("perf/gae_time", gae_secs, global_step)?;
            logger.log_scalar("perf/update_time", update_secs, global_step)?;

            if total_secs > 0.0 {
                logger.log_scalar(
                    "perf/rollout_pct",
                    rollout_secs / total_secs * 100.0,
                    global_step,
                )?;
                logger.log_scalar(
                    "perf/update_pct",
                    update_secs / total_secs * 100.0,
                    global_step,
                )?;
            }

            // Reset timing accumulators
            rollout_time_acc = std::time::Duration::ZERO;
            gae_time_acc = std::time::Duration::ZERO;
            update_time_acc = std::time::Duration::ZERO;

            last_log_time = now;
            last_log_step = global_step;

            // Episode return metrics (since last log)
            if !episodes_since_log.is_empty() {
                let returns: Vec<f32> = episodes_since_log.iter().map(|(r, _)| *r).collect();
                let lengths: Vec<usize> = episodes_since_log.iter().map(|(_, l)| *l).collect();

                let return_mean = returns.iter().sum::<f32>() / returns.len() as f32;
                let return_max = returns.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let return_min = returns.iter().copied().fold(f32::INFINITY, f32::min);

                let length_mean = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
                let length_max = *lengths.iter().max().expect("lengths non-empty") as f32;
                let length_min = *lengths.iter().min().expect("lengths non-empty") as f32;

                logger.log_scalar("episode/return_mean", return_mean, global_step)?;
                logger.log_scalar("episode/return_max", return_max, global_step)?;
                logger.log_scalar("episode/return_min", return_min, global_step)?;
                logger.log_scalar("episode/length_mean", length_mean, global_step)?;
                logger.log_scalar("episode/length_max", length_max, global_step)?;
                logger.log_scalar("episode/length_min", length_min, global_step)?;
                logger.log_scalar(
                    "episode/count",
                    episodes_since_log.len() as f32,
                    global_step,
                )?;

                episodes_since_log.clear();
            }

            // Multiplayer metrics (per-player returns and win rates)
            if num_players > 1 && !episodes_since_log_mp.is_empty() {
                // Per-player return statistics
                for player in 0..num_players_usize {
                    let player_returns: Vec<f32> = episodes_since_log_mp
                        .iter()
                        .map(|(rewards, _)| rewards.get(player).copied().unwrap_or(0.0))
                        .collect();

                    if !player_returns.is_empty() {
                        let mean = player_returns.iter().sum::<f32>() / player_returns.len() as f32;
                        let max = player_returns
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        let min = player_returns.iter().copied().fold(f32::INFINITY, f32::min);

                        logger.log_scalar(
                            &format!("episode/return_mean_p{player}"),
                            mean,
                            global_step,
                        )?;
                        logger.log_scalar(
                            &format!("episode/return_max_p{player}"),
                            max,
                            global_step,
                        )?;
                        logger.log_scalar(
                            &format!("episode/return_min_p{player}"),
                            min,
                            global_step,
                        )?;
                    }
                }

                // Swiss points and draw rate
                let total_games = episodes_since_log_mp.len();
                let outcomes: VecDeque<GameOutcome> = episodes_since_log_mp
                    .iter()
                    .filter_map(|(_, o)| o.clone())
                    .collect();
                let (avg_points, draw_rate) = compute_avg_points(&outcomes, num_players_usize);

                for (player, &pts) in avg_points.iter().enumerate() {
                    logger.log_scalar(
                        &format!("episode/avg_points_p{player}"),
                        pts,
                        global_step,
                    )?;
                }

                logger.log_scalar("episode/draw_rate", draw_rate, global_step)?;
                logger.log_scalar("episode/games_completed", total_games as f32, global_step)?;

                episodes_since_log_mp.clear();
            }

            logger.flush()?;
        }

        // Checkpointing
        if global_step - last_checkpoint_step >= config.checkpoint_freq {
            // Use episodes since last checkpoint for best selection
            let avg_return = if episodes_since_checkpoint.is_empty() {
                // Fall back to rolling window if no episodes completed since checkpoint
                if recent_returns.is_empty() {
                    0.0
                } else {
                    recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
                }
            } else {
                episodes_since_checkpoint.iter().sum::<f32>()
                    / episodes_since_checkpoint.len() as f32
            };

            let metadata = CheckpointMetadata {
                step: global_step,
                avg_return,
                rng_seed: config.seed,
                best_avg_return: Some(checkpoint_manager.best_avg_return()),
                recent_returns: recent_returns.clone(),
                forked_from: config.forked_from.clone(),
                obs_dim,
                action_count,
                num_players: num_players as usize,
                hidden_size: config.hidden_size,
                num_hidden: config.num_hidden,
                activation: config.activation.clone(),
                split_networks: config.split_networks,
                env_name: E::NAME.to_string(),
                training_rating: 25.0, // Updated after challenger eval via update_training_rating
                training_uncertainty: 25.0 / 3.0,
            };

            // Disable auto-best when using challenger evaluation for multiplayer
            let use_auto_best = !(config.challenger_eval && E::NUM_PLAYERS > 1);
            let checkpoint_path = checkpoint_manager.save(&model, &metadata, use_auto_best)?;

            // Save optimizer state alongside model
            if let Err(e) = save_optimizer::<TB, _, ActorCritic<TB>>(&optimizer, &checkpoint_path) {
                eprintln!("Warning: Failed to save optimizer state: {e}");
            }

            // Save observation normalizer if enabled
            if let Some(ref norm) = obs_normalizer {
                if let Err(e) = save_normalizer(norm, &checkpoint_path) {
                    eprintln!("Warning: Failed to save normalizer: {e}");
                }
            }

            // Save RNG state for reproducible continuation
            if let Err(e) = save_rng_state(&mut rng, &checkpoint_path) {
                eprintln!("Warning: Failed to save RNG state: {e}");
            }

            // Challenger evaluation for multiplayer games
            let challenger_avg_points: Option<f64> = if config.challenger_eval && E::NUM_PLAYERS > 1
            {
                let best_path = checkpoint_manager.best_checkpoint_path();
                if best_path.exists() {
                    // Load best checkpoint's training rating for accumulating skill
                    let (best_rating, best_uncertainty) = load_metadata(&best_path)
                        .map(|m| (m.training_rating, m.training_uncertainty))
                        .unwrap_or((25.0, 25.0 / 3.0));

                    // Run challenger evaluation with inference backend (no autodiff)
                    let challenger_model = model.valid();
                    match run_challenger_eval::<TB::InnerBackend, E>(
                        &challenger_model,
                        obs_normalizer.as_ref(),
                        &best_path,
                        best_rating,
                        best_uncertainty,
                        config.challenger_games,
                        config.challenger_threshold,
                        config,
                        device,
                        config.seed.wrapping_add(global_step as u64),
                    ) {
                        Ok(result) => {
                            // Save this checkpoint's training rating
                            if let Err(e) = update_training_rating(
                                &checkpoint_path,
                                result.current_rating,
                                result.current_uncertainty,
                            ) {
                                eprintln!("Warning: Failed to save training rating: {e}");
                            }
                            // Log metrics
                            logger.log_scalar(
                                "challenger/eval_time_ms",
                                result.elapsed_ms as f32,
                                global_step,
                            )?;
                            logger.log_scalar(
                                "challenger/current_avg_points",
                                result.current_avg_points as f32,
                                global_step,
                            )?;
                            logger.log_scalar(
                                "challenger/best_avg_points",
                                result.best_avg_points as f32,
                                global_step,
                            )?;
                            logger.log_scalar(
                                "challenger/draw_rate",
                                result.draw_rate as f32,
                                global_step,
                            )?;
                            logger.log_scalar(
                                "challenger/current_rating",
                                result.current_rating as f32,
                                global_step,
                            )?;
                            logger.log_scalar(
                                "challenger/current_uncertainty",
                                result.current_uncertainty as f32,
                                global_step,
                            )?;
                            logger.log_scalar(
                                "challenger/best_rating",
                                result.best_rating as f32,
                                global_step,
                            )?;
                            logger.log_scalar(
                                "challenger/best_uncertainty",
                                result.best_uncertainty as f32,
                                global_step,
                            )?;

                            // Check if eval took >5% of checkpoint interval
                            let checkpoint_interval_ms =
                                last_checkpoint_time.elapsed().as_millis() as u64;
                            if checkpoint_interval_ms > 0 {
                                let eval_pct = (result.elapsed_ms as f64
                                    / checkpoint_interval_ms as f64)
                                    * 100.0;
                                if eval_pct > 5.0 && !warned_challenger_eval_time {
                                    progress.println(&format!(
                                        "Warning: Challenger evaluation took {eval_pct:.1}% of checkpoint interval (consider reducing challenger_games)"
                                    ));
                                    warned_challenger_eval_time = true;
                                }
                            }

                            // Promote if win rate exceeds threshold
                            if result.should_promote {
                                if let Err(e) = checkpoint_manager.promote_to_best(&checkpoint_path)
                                {
                                    eprintln!("Warning: Failed to promote checkpoint to best: {e}");
                                }
                            }

                            // Log current best step (after potential promotion)
                            if let Some(best_step) = get_best_checkpoint_step(&best_path) {
                                logger.log_scalar(
                                    "challenger/best_step",
                                    best_step as f32,
                                    global_step,
                                )?;
                            }

                            Some(result.current_avg_points)
                        }
                        Err(e) => {
                            eprintln!("Warning: Challenger evaluation failed: {e}");
                            None
                        }
                    }
                } else {
                    // First checkpoint - becomes best automatically
                    if let Err(e) = checkpoint_manager.promote_to_best(&checkpoint_path) {
                        eprintln!("Warning: Failed to set initial best checkpoint: {e}");
                    }
                    None
                }
            } else {
                None
            };

            // Log checkpoint save message
            let checkpoint_msg = if let Some(avg_pts) = challenger_avg_points {
                format!(
                    "Saved checkpoint at step {} (avg return: {:.1}, vs best: {:.2} pts) -> {}",
                    global_step,
                    avg_return,
                    avg_pts,
                    checkpoint_path
                        .file_name()
                        .expect("checkpoint has filename")
                        .to_string_lossy()
                )
            } else {
                format!(
                    "Saved checkpoint at step {} (avg return: {:.1}) -> {}",
                    global_step,
                    avg_return,
                    checkpoint_path
                        .file_name()
                        .expect("checkpoint has filename")
                        .to_string_lossy()
                )
            };
            progress.println(&checkpoint_msg);

            last_checkpoint_step = global_step;
            last_checkpoint_time = std::time::Instant::now();
            episodes_since_checkpoint.clear();
        }

        profile::profile_frame!();
    }

    // Final checkpoint - save if there have been steps since last checkpoint
    if global_step > last_checkpoint_step && running.load(Ordering::SeqCst) {
        // Use episodes since last checkpoint for best selection
        let avg_return = if episodes_since_checkpoint.is_empty() {
            // Fall back to rolling window if no episodes completed since checkpoint
            if recent_returns.is_empty() {
                0.0
            } else {
                recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
            }
        } else {
            episodes_since_checkpoint.iter().sum::<f32>() / episodes_since_checkpoint.len() as f32
        };

        let metadata = CheckpointMetadata {
            step: global_step,
            avg_return,
            rng_seed: config.seed,
            best_avg_return: Some(checkpoint_manager.best_avg_return()),
            recent_returns: recent_returns.clone(),
            forked_from: config.forked_from.clone(),
            obs_dim,
            action_count,
            num_players: num_players as usize,
            hidden_size: config.hidden_size,
            num_hidden: config.num_hidden,
            activation: config.activation.clone(),
            split_networks: config.split_networks,
            env_name: E::NAME.to_string(),
            training_rating: 25.0, // Updated after challenger eval via update_training_rating
            training_uncertainty: 25.0 / 3.0,
        };

        // Disable auto-best when using challenger evaluation for multiplayer
        let use_auto_best = !(config.challenger_eval && E::NUM_PLAYERS > 1);
        let checkpoint_path = checkpoint_manager.save(&model, &metadata, use_auto_best)?;

        // Save optimizer state alongside model
        if let Err(e) = save_optimizer::<TB, _, ActorCritic<TB>>(&optimizer, &checkpoint_path) {
            eprintln!("Warning: Failed to save optimizer state: {e}");
        }

        // Save observation normalizer if enabled
        if let Some(ref norm) = obs_normalizer {
            if let Err(e) = save_normalizer(norm, &checkpoint_path) {
                eprintln!("Warning: Failed to save normalizer: {e}");
            }
        }

        // Save RNG state for reproducible continuation
        if let Err(e) = save_rng_state(&mut rng, &checkpoint_path) {
            eprintln!("Warning: Failed to save RNG state: {e}");
        }

        // Challenger evaluation for multiplayer games
        let challenger_avg_points: Option<f64> = if config.challenger_eval && E::NUM_PLAYERS > 1 {
            let best_path = checkpoint_manager.best_checkpoint_path();
            if best_path.exists() {
                // Load best checkpoint's training rating for accumulating skill
                let (best_rating, best_uncertainty) = load_metadata(&best_path)
                    .map(|m| (m.training_rating, m.training_uncertainty))
                    .unwrap_or((25.0, 25.0 / 3.0));

                // Run challenger evaluation with inference backend (no autodiff)
                let challenger_model = model.valid();
                match run_challenger_eval::<TB::InnerBackend, E>(
                    &challenger_model,
                    obs_normalizer.as_ref(),
                    &best_path,
                    best_rating,
                    best_uncertainty,
                    config.challenger_games,
                    config.challenger_threshold,
                    config,
                    device,
                    config.seed.wrapping_add(global_step as u64),
                ) {
                    Ok(result) => {
                        // Save this checkpoint's training rating
                        if let Err(e) = update_training_rating(
                            &checkpoint_path,
                            result.current_rating,
                            result.current_uncertainty,
                        ) {
                            eprintln!("Warning: Failed to save training rating: {e}");
                        }
                        // Log metrics
                        logger.log_scalar(
                            "challenger/eval_time_ms",
                            result.elapsed_ms as f32,
                            global_step,
                        )?;
                        logger.log_scalar(
                            "challenger/current_avg_points",
                            result.current_avg_points as f32,
                            global_step,
                        )?;
                        logger.log_scalar(
                            "challenger/best_avg_points",
                            result.best_avg_points as f32,
                            global_step,
                        )?;
                        logger.log_scalar(
                            "challenger/draw_rate",
                            result.draw_rate as f32,
                            global_step,
                        )?;
                        logger.log_scalar(
                            "challenger/current_rating",
                            result.current_rating as f32,
                            global_step,
                        )?;
                        logger.log_scalar(
                            "challenger/current_uncertainty",
                            result.current_uncertainty as f32,
                            global_step,
                        )?;
                        logger.log_scalar(
                            "challenger/best_rating",
                            result.best_rating as f32,
                            global_step,
                        )?;
                        logger.log_scalar(
                            "challenger/best_uncertainty",
                            result.best_uncertainty as f32,
                            global_step,
                        )?;

                        // Promote if avg points exceeds best
                        if result.should_promote {
                            if let Err(e) = checkpoint_manager.promote_to_best(&checkpoint_path) {
                                eprintln!("Warning: Failed to promote checkpoint to best: {e}");
                            }
                        }

                        // Log current best step (after potential promotion)
                        if let Some(best_step) = get_best_checkpoint_step(&best_path) {
                            logger.log_scalar(
                                "challenger/best_step",
                                best_step as f32,
                                global_step,
                            )?;
                        }

                        Some(result.current_avg_points)
                    }
                    Err(e) => {
                        eprintln!("Warning: Challenger evaluation failed: {e}");
                        None
                    }
                }
            } else {
                // First checkpoint - becomes best automatically
                if let Err(e) = checkpoint_manager.promote_to_best(&checkpoint_path) {
                    eprintln!("Warning: Failed to set initial best checkpoint: {e}");
                }
                None
            }
        } else {
            None
        };

        // Log final checkpoint save message
        let checkpoint_msg = if let Some(avg_pts) = challenger_avg_points {
            format!(
                "Saved final checkpoint at step {} (avg return: {:.1}, vs best: {:.2} pts) -> {}",
                global_step,
                avg_return,
                avg_pts,
                checkpoint_path
                    .file_name()
                    .expect("checkpoint has filename")
                    .to_string_lossy()
            )
        } else {
            format!(
                "Saved final checkpoint at step {} (avg return: {:.1}) -> {}",
                global_step,
                avg_return,
                checkpoint_path
                    .file_name()
                    .expect("checkpoint has filename")
                    .to_string_lossy()
            )
        };
        progress.println(&checkpoint_msg);
    }

    // Finish progress bar appropriately based on whether we were interrupted
    if running.load(Ordering::SeqCst) {
        progress.finish();
        println!("---");
        println!("Training complete!");
    } else {
        progress.finish_interrupted();
    }

    if !recent_returns.is_empty() {
        let avg_return: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
        println!("Final average return (last 100 episodes): {avg_return:.1}");
    }

    Ok(())
}

fn main() -> Result<()> {
    // Initialize rayon thread pool with named threads for Tracy
    rayon::ThreadPoolBuilder::new()
        .thread_name(|idx| format!("Rayon-{idx}"))
        .build_global()
        .expect("Failed to initialize rayon thread pool");

    // Name the main thread for Tracy
    #[cfg(feature = "tracy")]
    tracy_client::set_thread_name!("Main");

    let cli = Cli::parse();

    // Dispatch based on subcommand
    match cli.command {
        Some(Command::Eval(eval_args)) => {
            let backend_name = eval_args
                .backend
                .as_deref()
                .unwrap_or_else(|| backend::default_backend());
            backend::warn_if_better_backend_available(backend_name);
            dispatch_backend!(backend_name, device, {
                eval::run_evaluation::<TB>(&eval_args, &device)
            })
        }
        Some(Command::Tournament(tournament_args)) => {
            let backend_name = tournament_args
                .backend
                .as_deref()
                .unwrap_or_else(|| backend::default_backend());
            backend::warn_if_better_backend_available(backend_name);
            dispatch_backend!(backend_name, device, {
                tournament::run_tournament::<TB>(&tournament_args, &device)
            })
        }
        Some(Command::Train(args)) => {
            // Training mode with explicit subcommand
            run_training_cli(&args)
        }
        None => {
            // Default: parse as training args
            let args = CliArgs::parse();
            run_training_cli(&args)
        }
    }
}

fn run_training_cli(args: &CliArgs) -> Result<()> {
    // Determine training mode
    let mode = if let Some(resume_path) = &args.resume {
        // Resume mode: continue existing run
        let run_dir = resume_path.clone();
        let checkpoint_dir = run_dir.join("checkpoints/latest");
        if !checkpoint_dir.exists() {
            bail!(
                "No checkpoint found at {}. Cannot resume.",
                checkpoint_dir.display()
            );
        }
        TrainingMode::Resume {
            run_dir,
            checkpoint_dir,
        }
    } else if let Some(fork_path) = &args.fork {
        // Fork mode: new run from checkpoint
        let checkpoint_dir = fork_path.clone();
        if !checkpoint_dir.exists() {
            bail!(
                "Checkpoint not found at {}. Cannot fork.",
                checkpoint_dir.display()
            );
        }
        TrainingMode::Fork { checkpoint_dir }
    } else {
        TrainingMode::Fresh
    };

    // Load config based on mode
    let (config, run_dir, resumed_metadata) = match &mode {
        TrainingMode::Fresh => {
            let config = Config::load(args, None)?;
            let run_dir = config.run_path();

            // Check if run directory already exists
            if !check_run_exists_and_prompt(&run_dir)? {
                return Ok(());
            }

            (config, run_dir, None)
        }
        TrainingMode::Resume {
            run_dir,
            checkpoint_dir,
        } => {
            // Load config from the run directory
            let config_path = run_dir.join("config.toml");
            let mut config = Config::load_from_path(&config_path)
                .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

            // Apply limited resume overrides (only total_timesteps allowed)
            config.apply_resume_overrides(args);

            // Load checkpoint metadata
            let metadata_path = checkpoint_dir.join("metadata.json");
            let metadata_json = std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("Failed to read {}", metadata_path.display()))?;
            let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

            (config, run_dir.clone(), Some(metadata))
        }
        TrainingMode::Fork { checkpoint_dir } => {
            // Extract parent run name from checkpoint path
            let parent_run_name = extract_run_name_from_checkpoint_path(checkpoint_dir);

            // Use new config from CLI with forked_from set
            let config = Config::load(args, parent_run_name.as_deref())?;
            let run_dir = config.run_path();

            // Check if run directory already exists
            if !check_run_exists_and_prompt(&run_dir)? {
                return Ok(());
            }

            // Load checkpoint metadata
            let metadata_path = checkpoint_dir.join("metadata.json");
            let metadata_json = std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("Failed to read {}", metadata_path.display()))?;
            let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

            (config, run_dir, Some(metadata))
        }
    };

    config.validate()?;

    println!("burn-ppo v{}", env!("CARGO_PKG_VERSION"));
    match &mode {
        TrainingMode::Fresh => println!("Mode: Fresh training"),
        TrainingMode::Resume { .. } => println!("Mode: Resuming from checkpoint"),
        TrainingMode::Fork { checkpoint_dir } => {
            println!("Mode: Forking from {}", checkpoint_dir.display());
        }
    }
    println!("Environment: {}", config.env);
    println!(
        "Num envs: {} (resolved: {})",
        match &config.num_envs {
            config::NumEnvs::Auto(_) => "auto".to_string(),
            config::NumEnvs::Explicit(n) => n.to_string(),
        },
        config.num_envs()
    );
    println!("Seed: {}", config.seed);
    println!(
        "Run: {}",
        config.run_name.as_ref().expect("run_name is set")
    );

    // Set up graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc_handler(r);

    // Get backend from args or use default
    let backend_name = args
        .backend
        .as_deref()
        .unwrap_or_else(|| backend::default_backend());
    println!(
        "Backend: {} ({})",
        backend::get_backend_display_name(backend_name),
        backend_name
    );
    backend::warn_if_better_backend_available(backend_name);

    // Print rating guide for understanding Openskill ratings
    print_rating_guide();

    // Create or validate run directory
    match &mode {
        TrainingMode::Fresh | TrainingMode::Fork { .. } => {
            std::fs::create_dir_all(&run_dir)?;
            std::fs::create_dir_all(run_dir.join("checkpoints"))?;

            // Save config snapshot for reproducibility
            let config_snapshot_path = run_dir.join("config.toml");
            std::fs::write(&config_snapshot_path, toml::to_string_pretty(&config)?)?;
        }
        TrainingMode::Resume { .. } => {
            // Run directory already exists
        }
    }

    // Dispatch to backend and environment
    // Full static dispatch for both backend and environment types
    dispatch_backend!(backend_name, device, {
        let seed = config.seed;
        match config.env.as_str() {
            "cartpole" => run_training::<TB, CartPole, _>(
                &mode,
                &config,
                &run_dir,
                resumed_metadata.as_ref(),
                &device,
                &running,
                move |i| CartPole::new(seed + i as u64),
            ),
            "connect_four" => run_training::<TB, ConnectFour, _>(
                &mode,
                &config,
                &run_dir,
                resumed_metadata.as_ref(),
                &device,
                &running,
                move |i| ConnectFour::new(seed + i as u64),
            ),
            "liars_dice" => run_training::<TB, LiarsDice, _>(
                &mode,
                &config,
                &run_dir,
                resumed_metadata.as_ref(),
                &device,
                &running,
                move |i| LiarsDice::new(seed + i as u64),
            ),
            _ => bail!(
                "Unknown environment '{}'. Supported: cartpole, connect_four, liars_dice",
                config.env
            ),
        }
    })
}

/// Set up Ctrl+C handler for graceful shutdown
fn ctrlc_handler(running: Arc<AtomicBool>) {
    ctrlc::set_handler(move || {
        running.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn test_extract_run_name_from_checkpoint_path_valid() {
        let path = Path::new("runs/cartpole_001/checkpoints/best");
        assert_eq!(
            extract_run_name_from_checkpoint_path(path),
            Some("cartpole_001".into())
        );
    }

    #[test]
    fn test_extract_run_name_from_checkpoint_path_latest() {
        let path = Path::new("runs/connect_four_42/checkpoints/latest");
        assert_eq!(
            extract_run_name_from_checkpoint_path(path),
            Some("connect_four_42".into())
        );
    }

    #[test]
    fn test_extract_run_name_from_checkpoint_path_step() {
        let path = Path::new("runs/my_run/checkpoints/step_00010240");
        assert_eq!(
            extract_run_name_from_checkpoint_path(path),
            Some("my_run".into())
        );
    }

    #[test]
    fn test_extract_run_name_from_checkpoint_path_too_short() {
        let path = Path::new("checkpoints/best");
        assert_eq!(extract_run_name_from_checkpoint_path(path), None);
    }

    #[test]
    fn test_extract_run_name_from_checkpoint_path_single() {
        let path = Path::new("best");
        assert_eq!(extract_run_name_from_checkpoint_path(path), None);
    }

    #[test]
    fn test_get_best_checkpoint_step_valid() {
        let dir = tempdir().unwrap();
        let best = dir.path().join("best");
        #[cfg(unix)]
        std::os::unix::fs::symlink("step_00010240", &best).unwrap();
        #[cfg(unix)]
        assert_eq!(get_best_checkpoint_step(&best), Some(10240));
    }

    #[test]
    fn test_get_best_checkpoint_step_different_step() {
        let dir = tempdir().unwrap();
        let best = dir.path().join("best");
        #[cfg(unix)]
        std::os::unix::fs::symlink("step_00000512", &best).unwrap();
        #[cfg(unix)]
        assert_eq!(get_best_checkpoint_step(&best), Some(512));
    }

    #[test]
    fn test_get_best_checkpoint_step_not_symlink() {
        let dir = tempdir().unwrap();
        let best = dir.path().join("best");
        std::fs::create_dir(&best).unwrap();
        assert_eq!(get_best_checkpoint_step(&best), None);
    }

    #[test]
    fn test_get_best_checkpoint_step_invalid_format() {
        let dir = tempdir().unwrap();
        let best = dir.path().join("best");
        #[cfg(unix)]
        std::os::unix::fs::symlink("invalid_name", &best).unwrap();
        #[cfg(unix)]
        assert_eq!(get_best_checkpoint_step(&best), None);
    }

    #[test]
    fn test_get_best_checkpoint_step_nonexistent() {
        let dir = tempdir().unwrap();
        let best = dir.path().join("nonexistent");
        assert_eq!(get_best_checkpoint_step(&best), None);
    }

    #[test]
    fn test_count_checkpoints_no_dir() {
        let dir = tempdir().unwrap();
        let run_dir = dir.path().join("nonexistent_run");
        assert_eq!(count_checkpoints(&run_dir), 0);
    }

    #[test]
    fn test_count_checkpoints_empty() {
        let dir = tempdir().unwrap();
        let run_dir = dir.path().join("run");
        std::fs::create_dir_all(run_dir.join("checkpoints")).unwrap();
        assert_eq!(count_checkpoints(&run_dir), 0);
    }

    #[test]
    fn test_count_checkpoints_with_steps() {
        let dir = tempdir().unwrap();
        let run_dir = dir.path().join("run");
        let checkpoints = run_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoints).unwrap();
        std::fs::create_dir(checkpoints.join("step_00001024")).unwrap();
        std::fs::create_dir(checkpoints.join("step_00002048")).unwrap();
        std::fs::create_dir(checkpoints.join("step_00003072")).unwrap();
        assert_eq!(count_checkpoints(&run_dir), 3);
    }

    #[test]
    fn test_count_checkpoints_ignores_non_step() {
        let dir = tempdir().unwrap();
        let run_dir = dir.path().join("run");
        let checkpoints = run_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoints).unwrap();
        std::fs::create_dir(checkpoints.join("step_00001024")).unwrap();
        std::fs::create_dir(checkpoints.join("best")).unwrap();
        std::fs::create_dir(checkpoints.join("latest")).unwrap();
        std::fs::write(checkpoints.join("metadata.json"), "{}").unwrap();
        assert_eq!(count_checkpoints(&run_dir), 1);
    }
}
