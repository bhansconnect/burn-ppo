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

// Memory allocation tracking (enabled with --features stats_alloc)
#[cfg(feature = "stats_alloc")]
#[global_allocator]
static GLOBAL: &stats_alloc::StatsAlloc<std::alloc::System> = &stats_alloc::INSTRUMENTED_SYSTEM;

mod backend;
mod checkpoint;
mod config;
mod entropy;
mod env;
mod envs;
mod eval;
mod human;
mod metrics;
mod network;
mod normalization;
mod opponent_pool;
mod plackett_luce;
mod ppo;
mod profile;
mod progress;
mod rating_history;
mod supervisor;
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
    load_normalizer, load_optimizer, load_return_normalizer, load_rng_state, save_normalizer,
    save_optimizer, save_return_normalizer, save_rng_state, CheckpointManager, CheckpointMetadata,
};
use crate::config::{Cli, CliArgs, Command, Config};
use crate::env::{compute_avg_points, Environment, GameOutcome, VecEnv};
use crate::envs::{CartPole, ConnectFour, LiarsDice};
use crate::metrics::MetricsLogger;
use crate::network::ActorCritic;
use crate::normalization::{ObsNormalizer, ReturnNormalizer};
use crate::opponent_pool::{EnvState, OpponentPool};
use crate::ppo::{
    collect_rollouts, collect_rollouts_with_opponents, compute_gae, compute_gae_multiplayer,
    ppo_update, RolloutBuffer,
};
use crate::progress::TrainingProgress;
use crate::rating_history::RatingHistory;
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
    elapsed_time_offset_ms: u64,
    max_checkpoints_this_run: usize,
    env_factory: F,
) -> Result<()>
where
    TB: burn::tensor::backend::AutodiffBackend,
    TB::FloatElem: Into<f32>,
    E: Environment,
    F: Fn(usize) -> E,
{
    // Quiet mode: suppress verbose output when running as subprocess
    let quiet = max_checkpoints_this_run > 0;

    // Initialize RNG
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    TB::seed(device, config.seed);
    <TB as burn::tensor::backend::AutodiffBackend>::InnerBackend::seed(device, config.seed);

    // Create vectorized environment
    let num_envs = config.num_envs();
    let mut vec_env = VecEnv::new(num_envs, env_factory);

    let obs_dim = E::OBSERVATION_DIM;
    let obs_shape = E::OBSERVATION_SHAPE;
    let action_count = E::ACTION_COUNT;
    let num_players = E::NUM_PLAYERS as u8;
    if !quiet {
        println!(
            "Created {} {} environments (obs_dim={}, obs_shape={:?}, actions={}, players={})",
            num_envs,
            E::NAME,
            obs_dim,
            obs_shape,
            action_count,
            num_players
        );
    }

    // Warning for conflicting entropy config
    if config.adaptive_entropy && config.entropy_anneal {
        eprintln!(
            "Warning: Both adaptive_entropy and entropy_anneal are enabled. \
             adaptive_entropy takes precedence; entropy_anneal will be ignored."
        );
    }

    // Create observation normalizer if enabled
    let mut obs_normalizer: Option<ObsNormalizer> = if config.normalize_obs {
        Some(ObsNormalizer::new(obs_dim, 10.0))
    } else {
        None
    };

    // Create return normalizer if enabled (default: true)
    let mut return_normalizer: Option<ReturnNormalizer> = if config.normalize_returns {
        Some(ReturnNormalizer::new(
            num_envs,
            num_players as usize,
            config.gamma,
            config.return_clip,
        ))
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
    let (
        mut model,
        mut optimizer,
        mut global_step,
        mut recent_returns,
        best_return,
        initial_avg_return,
    ) = match mode {
        TrainingMode::Fresh => {
            let model: ActorCritic<TB> = ActorCritic::new(
                obs_dim,
                obs_shape,
                action_count,
                num_players as usize,
                config,
                device,
            );
            let optimizer = optimizer_config.init();
            if !quiet {
                println!("Created {} ActorCritic network", config.network_type);
            }
            (model, optimizer, 0, Vec::new(), f32::NEG_INFINITY, None)
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
                        if !quiet {
                            println!("Loaded observation normalizer from checkpoint");
                        }
                    }
                    Ok(None) => {
                        // No normalizer saved, keep the fresh one
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load normalizer: {e}");
                    }
                }
            }

            // Load return normalizer if it was saved (backward compatible - old checkpoints won't have it)
            if config.normalize_returns {
                match load_return_normalizer(checkpoint_dir) {
                    Ok(Some(loaded_norm)) => {
                        return_normalizer = Some(loaded_norm);
                        if !quiet {
                            println!("Loaded return normalizer from checkpoint");
                        }
                    }
                    Ok(None) => {
                        // No return normalizer saved (old checkpoint), keep the fresh one
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load return normalizer: {e}");
                    }
                }
            }

            // Load RNG state if saved (for reproducible continuation)
            match load_rng_state(checkpoint_dir) {
                Ok(Some(loaded_rng)) => {
                    rng = loaded_rng;
                    if !quiet {
                        println!("Loaded RNG state from checkpoint");
                    }
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
            let avg_return = metadata.avg_return;

            if !quiet {
                println!("Loaded checkpoint from step {step} (avg return: {avg_return:.1})");
            }

            (
                model,
                optimizer,
                step,
                recent_returns,
                best_return,
                Some(avg_return),
            )
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
            network_type: config.network_type.clone(),
            num_conv_layers: config.num_conv_layers,
            conv_channels: config.conv_channels.clone(),
            kernel_size: config.kernel_size,
            cnn_fc_hidden_size: config.cnn_fc_hidden_size,
            cnn_num_fc_layers: config.cnn_num_fc_layers,
            obs_shape,
            env_name: E::NAME.to_string(),
        };

        checkpoint_manager.save(&model, &metadata, true)?;
    }

    // Training state
    let steps_per_update = config.num_steps * num_envs;
    let total_updates = config.total_timesteps / steps_per_update;
    let remaining_timesteps = config.total_timesteps.saturating_sub(global_step);
    let num_updates = remaining_timesteps / steps_per_update;

    // For LR annealing, we need to know how many updates have already happened
    let update_offset = global_step / steps_per_update;

    // Parse time limit (fail fast if invalid)
    let time_limit = config.max_training_duration()?;

    if !quiet {
        if let Some(ref limit_str) = config.max_training_time {
            println!(
                "Training for {} timesteps ({} updates) or {}, whichever comes first",
                config.total_timesteps, num_updates, limit_str
            );
        } else {
            println!(
                "Training for {} timesteps ({} updates of {} steps each)",
                config.total_timesteps, num_updates, steps_per_update
            );
        }
        println!("---");
    }

    // Progress bar (with multiplayer support and elapsed time offset for subprocess reloads)
    // In subprocess mode, move cursor up and clear line to overwrite the abandoned progress bar
    if quiet {
        use std::io::Write;
        // Move cursor up one line (where old bar is), clear it, move to start
        let _ = std::io::stderr().write_all(b"\x1b[A\x1b[2K\r");
        let _ = std::io::stderr().flush();
    }
    let elapsed_offset = std::time::Duration::from_millis(elapsed_time_offset_ms);
    let progress = TrainingProgress::new_with_offset_and_position(
        config.total_timesteps as u64,
        num_players as usize,
        elapsed_offset,
        global_step as u64, // Start at resumed position to avoid flash to 0
        initial_avg_return, // Pass checkpoint avg_return for immediate display on resume
    );

    // Track initial step to detect if we actually trained (for completion logic)
    let initial_global_step = global_step;

    // Timing for SPS calculation
    let training_start = std::time::Instant::now();
    let mut last_log_time = training_start;
    let mut last_log_step = global_step;

    // Phase timing accumulators (reset on log)
    let mut rollout_time_acc = std::time::Duration::ZERO;
    let mut gae_time_acc = std::time::Duration::ZERO;
    let mut update_time_acc = std::time::Duration::ZERO;

    // Last training metrics for checkpoint display
    let mut last_metrics: Option<ppo::UpdateMetrics> = None;

    // Episode tracking for metrics (cleared on each log)
    let mut episodes_since_log: Vec<(f32, usize)> = Vec::new(); // (return, length)

    // Episode tracking for checkpoint selection (cleared on each checkpoint)
    let mut episodes_since_checkpoint: Vec<f32> = Vec::new();

    // Checkpoint counter for subprocess reload mode (tracks saves in this process)
    let mut checkpoints_saved_this_run: usize = 0;
    // Track if we exited due to checkpoint limit (vs natural completion)
    let mut exited_for_reload = false;

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

    // Opponent pool initialization (if enabled for multiplayer training)
    let needs_opponent_pool = config.opponent_pool_enabled;
    let mut opponent_pool: Option<OpponentPool<TB::InnerBackend>> =
        if needs_opponent_pool && num_players > 1 {
            let checkpoints_dir = run_dir.join("checkpoints");

            match OpponentPool::new(
                checkpoints_dir,
                num_players_usize,
                config.qi_eta,
                config.clone(),
                device.clone(),
                config.seed,
                config.opponent_pool_size_limit,
            ) {
                Ok(pool) => {
                    progress.println(&format!(
                        "Opponent pool initialized with {} checkpoints",
                        pool.num_available()
                    ));
                    Some(pool)
                }
                Err(e) => {
                    progress.eprintln(&format!("Warning: Failed to initialize opponent pool: {e}"));
                    None
                }
            }
        } else {
            None
        };

    // Initialize rating history for opponent pool training
    let mut rating_history: Option<RatingHistory> = if opponent_pool.is_some() {
        match RatingHistory::load(run_dir) {
            Ok(history) => {
                if history.num_games() > 0 {
                    progress.println(&format!(
                        "Loaded rating history: {} games, {} checkpoints",
                        history.num_games(),
                        history.num_checkpoints()
                    ));
                }
                Some(history)
            }
            Err(e) => {
                progress.eprintln(&format!("Warning: Failed to load rating history: {e}"));
                Some(RatingHistory::new(run_dir))
            }
        }
    } else {
        None
    };

    // Calculate number of opponent envs (only when training with opponents is enabled)
    #[expect(
        clippy::cast_sign_loss,
        reason = "opponent_pool_fraction is always positive"
    )]
    let num_opponent_envs = if config.opponent_pool_enabled && opponent_pool.is_some() {
        let raw = num_envs as f32 * config.opponent_pool_fraction;
        if raw > 0.0 && raw < 1.0 {
            progress.eprintln(&format!(
                "Warning: opponent_pool_fraction ({:.1}%) of {} envs = {:.2} envs. \
                 Rounding up to 1 env.",
                config.opponent_pool_fraction * 100.0,
                num_envs,
                raw
            ));
            1
        } else {
            raw as usize
        }
    } else {
        0
    };

    // Initialize env states for opponent pool training (only when training is enabled)
    let mut env_states: Vec<EnvState> = if config.opponent_pool_enabled
        && opponent_pool
            .as_ref()
            .is_some_and(opponent_pool::OpponentPool::has_opponents)
    {
        let pool = opponent_pool.as_mut().expect("checked above");
        (0..num_opponent_envs)
            .map(|_| {
                let opponents = pool.sample_all_slots();
                EnvState::new(num_players_usize, opponents, pool.rng_mut())
            })
            .collect()
    } else {
        Vec::new()
    };

    // Track steps since last opponent rotation
    let mut steps_since_rotation: usize = 0;

    // Determine how "best" checkpoint should be selected
    // - Single-player: use avg_return (auto_update_best = true)
    // - Multiplayer: best is updated via rating system (auto_update_best = false)
    let use_avg_return_for_best = num_players == 1;

    // Adaptive entropy controller (if enabled)
    let mut entropy_controller = if config.adaptive_entropy {
        Some(entropy::AdaptiveEntropyController::new(
            config,
            action_count,
            config.entropy_coef,
        ))
    } else {
        None
    };

    // Memory tracking for stats_alloc feature
    // Track net memory = allocated + reallocated - deallocated (actual heap usage)
    #[cfg(feature = "stats_alloc")]
    let mut last_net_bytes: i64 = {
        let s = GLOBAL.stats();
        s.bytes_allocated as i64 + s.bytes_reallocated as i64 - s.bytes_deallocated as i64
    };

    // Training loop
    for update in 0..num_updates {
        profile::profile_scope!("training_update");

        if !running.load(Ordering::SeqCst) {
            println!("\nInterrupted by user");
            break;
        }

        // Check time limit
        if let Some(limit) = time_limit {
            if training_start.elapsed() >= limit {
                // Safe: time_limit is Some only when max_training_time is Some
                let limit_str = config
                    .max_training_time
                    .as_ref()
                    .expect("max_training_time set when time_limit is Some");
                println!("\nTime limit reached ({limit_str})");
                break;
            }
        }

        // Learning rate annealing (decay to final value, default 0)
        let lr = if config.lr_anneal {
            let actual_update = update_offset + update;
            let progress_frac = actual_update as f64 / total_updates as f64;
            config.learning_rate + (config.lr_final - config.learning_rate) * progress_frac
        } else {
            config.learning_rate
        };

        // Entropy coefficient calculation:
        // 1. Adaptive entropy: PID-inspired control targeting specific entropy levels
        // 2. Entropy annealing: linear decay to final value
        // 3. Constant: use entropy_coef as-is
        let actual_update = update_offset + update;
        let progress_frac = actual_update as f64 / total_updates as f64;

        let (ent_coef, entropy_target) = if let Some(ref mut controller) = entropy_controller {
            // Adaptive entropy takes precedence
            controller.get_coefficient(progress_frac)
        } else if config.entropy_anneal {
            // Linear annealing (default: decay to 50% of initial)
            let final_coef = config
                .entropy_coef_final
                .unwrap_or(config.entropy_coef * 0.5);
            let coef = config.entropy_coef + (final_coef - config.entropy_coef) * progress_frac;
            (coef, 0.0) // No target for annealing mode
        } else {
            // Constant coefficient
            (config.entropy_coef, 0.0)
        };

        // Collect rollouts using non-autodiff model for inference
        let rollout_start = std::time::Instant::now();
        let inference_model = model.valid();

        // Use opponent pool collection if pool is active with opponents
        let use_opponent_pool = opponent_pool
            .as_ref()
            .is_some_and(|p| p.has_opponents() && !env_states.is_empty());

        let completed_episodes = if use_opponent_pool {
            let pool = opponent_pool
                .as_mut()
                .expect("opponent_pool verified above");
            let (episodes, opponent_completions) = collect_rollouts_with_opponents(
                &inference_model,
                pool,
                &mut vec_env,
                &mut buffer,
                &mut env_states,
                num_opponent_envs,
                config.num_steps,
                device,
                &mut rng,
                obs_normalizer.as_mut(),
                return_normalizer.as_mut(),
                global_step,
            );

            // Queue opponent completions for batched qi updates and record for rating
            for completion in opponent_completions {
                if !completion.placements.is_empty() {
                    let env_state = &env_states[completion.env_idx];
                    pool.queue_game_for_qi_update(
                        &completion.placements,
                        env_state.learner_position,
                        env_state,
                        global_step,
                    );

                    // Record game for rating history
                    if let Some(ref mut history) = rating_history {
                        // Get current checkpoint name (fallback to "latest" if not yet set)
                        let current_name = history
                            .current_checkpoint()
                            .map_or_else(|| "step_00000000".to_string(), String::from);

                        // Get opponent checkpoint names
                        let opponent_names: Vec<String> = completion
                            .opponent_pool_indices
                            .iter()
                            .map(|&idx| pool.get_checkpoint_name(idx))
                            .collect();

                        // Rearrange placements: [learner_placement, opponent_placements...]
                        // completion.placements is indexed by position
                        let mut rating_placements =
                            vec![completion.placements[env_state.learner_position]];
                        // Get opponent positions from position_to_opponent mapping
                        for (pos, slot) in env_state.position_to_opponent.iter().enumerate() {
                            if slot.is_some() {
                                rating_placements.push(completion.placements[pos]);
                            }
                        }

                        history.record_game(&current_name, &opponent_names, rating_placements);
                    }
                }
            }

            // Handle rotation
            steps_since_rotation += steps_per_update;
            if steps_since_rotation >= config.opponent_pool_rotation_steps {
                steps_since_rotation = 0;

                // Apply pending qi updates
                pool.apply_pending_qi_updates();

                // Refresh pool (scan for new checkpoints)
                let _ = pool.scan_checkpoints();

                // Re-sample active subset (provides variety + includes new checkpoints)
                pool.refresh_active_subset();

                // Debug output for opponent selection (to stderr for debug output)
                if config.debug_opponents {
                    let mut active = pool.sample_eval_opponents();
                    active.sort_by(|a, b| b.cmp(a));
                    let formatted = pool.format_selected_opponents(&active);
                    progress.eprintln(&format!(
                        "[debug-opponents] Rotation at step {global_step}: active pool [{formatted}]"
                    ));
                }

                // Unload unused opponents
                let in_use: Vec<usize> = env_states
                    .iter()
                    .flat_map(|s| s.assigned_opponents.iter().copied())
                    .collect();
                pool.unload_unused(&in_use);
            }

            episodes
        } else {
            collect_rollouts(
                &inference_model,
                &mut vec_env,
                &mut buffer,
                config.num_steps,
                device,
                &mut rng,
                obs_normalizer.as_mut(),
                return_normalizer.as_mut(),
            )
        };
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

            // Per-player tracking (for all games, not just multiplayer)
            for (p, &reward) in ep.total_rewards.iter().enumerate() {
                if p < recent_returns_per_player.len() {
                    recent_returns_per_player[p].push_back(reward);
                    if recent_returns_per_player[p].len() > 100 {
                        recent_returns_per_player[p].pop_front();
                    }
                }
            }

            // Track outcome for win rate calculation (multiplayer)
            if let Some(ref outcome) = ep.outcome {
                recent_outcomes.push_back(outcome.clone());
                if recent_outcomes.len() > 100 {
                    recent_outcomes.pop_front();
                }
            }

            // Store for per-player logging
            episodes_since_log_mp.push((ep.total_rewards.clone(), ep.outcome.clone()));
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
        last_metrics = Some(metrics.clone());

        // Log memory stats (when stats_alloc feature enabled)
        // Net memory = allocated + reallocated - deallocated (actual heap usage)
        #[cfg(feature = "stats_alloc")]
        {
            let stats = GLOBAL.stats();
            let net_bytes = stats.bytes_allocated as i64 + stats.bytes_reallocated as i64
                - stats.bytes_deallocated as i64;
            let delta_bytes = net_bytes - last_net_bytes;
            let delta_mb = delta_bytes as f64 / (1024.0 * 1024.0);
            let net_mb = net_bytes as f64 / (1024.0 * 1024.0);
            progress.eprintln(&format!(
                "[mem] update {}: {:.2} MB (delta: {:+.2} MB)",
                update, net_mb, delta_mb
            ));
            last_net_bytes = net_bytes;
        }

        // Record entropy for adaptive controller (used in next iteration)
        if let Some(ref mut controller) = entropy_controller {
            controller.record_entropy(metrics.entropy);
        }

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
            // Log adaptive entropy metrics when enabled
            if config.adaptive_entropy {
                logger.log_scalar("train/entropy_target", entropy_target as f32, global_step)?;
                logger.log_scalar("train/entropy_coef", ent_coef as f32, global_step)?;
            }
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
            logger.log_scalar("train/adv_mean_raw", metrics.adv_mean_raw, global_step)?;
            logger.log_scalar("train/adv_std_raw", metrics.adv_std_raw, global_step)?;
            logger.log_scalar("train/adv_min_raw", metrics.adv_min_raw, global_step)?;
            logger.log_scalar("train/adv_max_raw", metrics.adv_max_raw, global_step)?;
            logger.log_scalar(
                "train/value_error_mean",
                metrics.value_error_mean,
                global_step,
            )?;
            logger.log_scalar(
                "train/value_error_std",
                metrics.value_error_std,
                global_step,
            )?;
            logger.log_scalar(
                "train/value_error_max",
                metrics.value_error_max,
                global_step,
            )?;

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

            // Episode length metrics (since last log)
            if !episodes_since_log.is_empty() {
                let lengths: Vec<usize> = episodes_since_log.iter().map(|(_, l)| *l).collect();

                let length_mean = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
                let length_max = *lengths.iter().max().expect("lengths non-empty") as f32;
                let length_min = *lengths.iter().min().expect("lengths non-empty") as f32;

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

            // Per-player return metrics (for all games)
            if !episodes_since_log_mp.is_empty() {
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

                // Swiss points and draw rate (multiplayer only)
                if num_players > 1 {
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
                    logger.log_scalar(
                        "episode/games_completed",
                        total_games as f32,
                        global_step,
                    )?;
                }

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
                network_type: config.network_type.clone(),
                num_conv_layers: config.num_conv_layers,
                conv_channels: config.conv_channels.clone(),
                kernel_size: config.kernel_size,
                cnn_fc_hidden_size: config.cnn_fc_hidden_size,
                cnn_num_fc_layers: config.cnn_num_fc_layers,
                obs_shape,
                env_name: E::NAME.to_string(),
            };

            // Determine whether to auto-update "best" symlink
            // - Single-player: use avg_return
            // - Multiplayer: manual control via pool eval (or skip if pool_eval disabled)
            let auto_update_best = use_avg_return_for_best;

            let checkpoint_path = checkpoint_manager.save(&model, &metadata, auto_update_best)?;

            // Save optimizer state alongside model
            if let Err(e) = save_optimizer::<TB, _, ActorCritic<TB>>(&optimizer, &checkpoint_path) {
                progress.eprintln(&format!("Warning: Failed to save optimizer state: {e}"));
            }

            // Save observation normalizer if enabled
            if let Some(ref norm) = obs_normalizer {
                if let Err(e) = save_normalizer(norm, &checkpoint_path) {
                    progress.eprintln(&format!("Warning: Failed to save normalizer: {e}"));
                }
            }

            // Save return normalizer if enabled
            if let Some(ref norm) = return_normalizer {
                if let Err(e) = save_return_normalizer(norm, &checkpoint_path) {
                    progress.eprintln(&format!("Warning: Failed to save return normalizer: {e}"));
                }
            }

            // Save RNG state for reproducible continuation
            if let Err(e) = save_rng_state(&mut rng, &checkpoint_path) {
                progress.eprintln(&format!("Warning: Failed to save RNG state: {e}"));
            }

            // Add checkpoint to opponent pool
            if let Some(ref mut pool) = opponent_pool {
                pool.add_checkpoint(checkpoint_path.clone(), metadata.step);
                // Save qi scores periodically
                if let Err(e) = pool.save_qi_scores() {
                    progress.eprintln(&format!("Warning: Failed to save qi scores: {e}"));
                }
            }

            // Update rating history and compute ratings
            let checkpoint_name = checkpoint_path
                .file_name()
                .expect("checkpoint has filename")
                .to_string_lossy()
                .to_string();

            if let Some(ref mut history) = rating_history {
                // Register this checkpoint and set it as current
                history.on_checkpoint_saved(&checkpoint_name, global_step);

                // Compute ratings and log metrics (with timing)
                let elo_start = std::time::Instant::now();
                let result = history.compute_ratings();
                let elo_compute_ms = elo_start.elapsed().as_secs_f64() * 1000.0;

                logger.log_scalar(
                    "training/elo_compute_ms",
                    elo_compute_ms as f32,
                    global_step,
                )?;
                logger.log_scalar(
                    "training/current_elo",
                    result.current_elo as f32,
                    global_step,
                )?;
                logger.log_scalar("training/best_elo", result.best_elo as f32, global_step)?;
                logger.log_scalar("training/best_step", result.best_step as f32, global_step)?;
                logger.log_scalar(
                    "training/rating_games",
                    result.total_games as f32,
                    global_step,
                )?;

                // Print rating status
                progress.println(&format!(
                    "[Rating] Elo: {:.0} | Best: {:.0} @ step {} | Games: {}",
                    result.current_elo, result.best_elo, result.best_step, result.total_games
                ));

                // Update "best" symlink to currently highest-rated checkpoint
                let best_name = format!("step_{:08}", result.best_step);
                if let Err(e) = checkpoint_manager.set_best_checkpoint(&best_name) {
                    progress.eprintln(&format!("Warning: Failed to update best checkpoint: {e}"));
                }

                // Generate Elo graph in checkpoint directory
                let graph_path = checkpoint_path.join("elo_graph.png");
                if let Err(e) = history.generate_graph(&graph_path) {
                    progress.eprintln(&format!("Warning: Failed to generate Elo graph: {e}"));
                }

                // Update root symlink to point to latest checkpoint's graph
                let root_graph = run_dir.join("checkpoints/elo_graph.png");
                let relative_target = format!("{checkpoint_name}/elo_graph.png");
                // Remove existing symlink if present
                let _ = std::fs::remove_file(&root_graph);
                #[cfg(unix)]
                if let Err(e) = std::os::unix::fs::symlink(&relative_target, &root_graph) {
                    progress.eprintln(&format!("Warning: Failed to create Elo graph symlink: {e}"));
                }
            }

            // Log checkpoint save message
            let filename = checkpoint_path
                .file_name()
                .expect("checkpoint has filename")
                .to_string_lossy();
            let checkpoint_msg = if num_players > 1 {
                // Multiplayer: show Swiss points and training metrics
                let (avg_points, _draw_rate) =
                    compute_avg_points(&recent_outcomes, num_players_usize);
                let p0_points = avg_points.first().copied().unwrap_or(0.0);
                if let Some(ref m) = last_metrics {
                    format!(
                        "Saved checkpoint at step {global_step} (pts: {p0_points:.2} | loss: {:.3} | ent: {:.2} | ev: {:.2}) -> {filename}",
                        m.policy_loss, m.entropy, m.explained_variance
                    )
                } else {
                    format!(
                        "Saved checkpoint at step {global_step} (avg pts: {p0_points:.2}) -> {filename}"
                    )
                }
            } else {
                // Single-player: show avg_return and training metrics
                if let Some(ref m) = last_metrics {
                    format!(
                        "Saved checkpoint at step {global_step} (ret: {avg_return:.1} | loss: {:.3} | ent: {:.2} | ev: {:.2}) -> {filename}",
                        m.policy_loss, m.entropy, m.explained_variance
                    )
                } else {
                    format!(
                        "Saved checkpoint at step {global_step} (avg return: {avg_return:.1}) -> {filename}"
                    )
                }
            };
            progress.println(&checkpoint_msg);

            // Print qi histogram and log percentile metrics if debug_qi is enabled
            if config.debug_qi {
                if let Some(ref pool) = opponent_pool {
                    if pool.has_opponents() {
                        // Log percentile metrics
                        let percentiles = pool.compute_qi_percentiles();
                        for (i, &pos) in percentiles.iter().enumerate() {
                            let pct = (i + 1) * 10;
                            logger.log_scalar(
                                &format!("qi/percentile_p{pct}"),
                                pos as f32,
                                global_step,
                            )?;
                        }

                        // Print histogram to stderr
                        let histogram = pool.generate_qi_histogram();
                        for line in histogram.lines() {
                            progress.eprintln(line);
                        }

                        // Save probability graph to checkpoint directory
                        if let Err(e) = pool.save_qi_probability_graph(&checkpoint_path) {
                            progress.eprintln(&format!("Warning: Failed to save qi graph: {e}"));
                        }
                    }
                }
            }

            last_checkpoint_step = global_step;
            episodes_since_checkpoint.clear();

            // Track checkpoints saved for subprocess reload mode
            checkpoints_saved_this_run += 1;
            if max_checkpoints_this_run > 0
                && checkpoints_saved_this_run >= max_checkpoints_this_run
            {
                exited_for_reload = true;
                break;
            }
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
            network_type: config.network_type.clone(),
            num_conv_layers: config.num_conv_layers,
            conv_channels: config.conv_channels.clone(),
            kernel_size: config.kernel_size,
            cnn_fc_hidden_size: config.cnn_fc_hidden_size,
            cnn_num_fc_layers: config.cnn_num_fc_layers,
            obs_shape,
            env_name: E::NAME.to_string(),
        };

        // Determine whether to auto-update "best" symlink (same logic as regular checkpoints)
        let auto_update_best = use_avg_return_for_best;

        let checkpoint_path = checkpoint_manager.save(&model, &metadata, auto_update_best)?;

        // Save optimizer state alongside model
        if let Err(e) = save_optimizer::<TB, _, ActorCritic<TB>>(&optimizer, &checkpoint_path) {
            progress.eprintln(&format!("Warning: Failed to save optimizer state: {e}"));
        }

        // Save observation normalizer if enabled
        if let Some(ref norm) = obs_normalizer {
            if let Err(e) = save_normalizer(norm, &checkpoint_path) {
                progress.eprintln(&format!("Warning: Failed to save normalizer: {e}"));
            }
        }

        // Save return normalizer if enabled
        if let Some(ref norm) = return_normalizer {
            if let Err(e) = save_return_normalizer(norm, &checkpoint_path) {
                progress.eprintln(&format!("Warning: Failed to save return normalizer: {e}"));
            }
        }

        // Save RNG state for reproducible continuation
        if let Err(e) = save_rng_state(&mut rng, &checkpoint_path) {
            progress.eprintln(&format!("Warning: Failed to save RNG state: {e}"));
        }

        // Add checkpoint to opponent pool
        if let Some(ref mut pool) = opponent_pool {
            pool.add_checkpoint(checkpoint_path.clone(), metadata.step);
            // Save qi scores
            if let Err(e) = pool.save_qi_scores() {
                progress.eprintln(&format!("Warning: Failed to save qi scores: {e}"));
            }
            // Save qi probability graph (only in debug_qi mode)
            if config.debug_qi {
                if let Err(e) = pool.save_qi_probability_graph(&checkpoint_path) {
                    progress.eprintln(&format!("Warning: Failed to save qi graph: {e}"));
                }
            }
        }

        // Update rating history and compute ratings (final checkpoint)
        let checkpoint_name = checkpoint_path
            .file_name()
            .expect("checkpoint has filename")
            .to_string_lossy()
            .to_string();

        if let Some(ref mut history) = rating_history {
            // Register this checkpoint and set it as current
            history.on_checkpoint_saved(&checkpoint_name, global_step);

            // Compute ratings and log metrics (with timing)
            let elo_start = std::time::Instant::now();
            let result = history.compute_ratings();
            let elo_compute_ms = elo_start.elapsed().as_secs_f64() * 1000.0;

            logger.log_scalar(
                "training/elo_compute_ms",
                elo_compute_ms as f32,
                global_step,
            )?;
            logger.log_scalar(
                "training/current_elo",
                result.current_elo as f32,
                global_step,
            )?;
            logger.log_scalar("training/best_elo", result.best_elo as f32, global_step)?;
            logger.log_scalar("training/best_step", result.best_step as f32, global_step)?;
            logger.log_scalar(
                "training/rating_games",
                result.total_games as f32,
                global_step,
            )?;

            // Print rating status
            progress.println(&format!(
                "[Rating] Elo: {:.0} | Best: {:.0} @ step {} | Games: {}",
                result.current_elo, result.best_elo, result.best_step, result.total_games
            ));

            // Update "best" symlink to currently highest-rated checkpoint
            let best_name = format!("step_{:08}", result.best_step);
            if let Err(e) = checkpoint_manager.set_best_checkpoint(&best_name) {
                progress.eprintln(&format!("Warning: Failed to update best checkpoint: {e}"));
            }

            // Generate Elo graph in checkpoint directory
            let graph_path = checkpoint_path.join("elo_graph.png");
            if let Err(e) = history.generate_graph(&graph_path) {
                progress.eprintln(&format!("Warning: Failed to generate Elo graph: {e}"));
            }

            // Update root symlink to point to latest checkpoint's graph
            let root_graph = run_dir.join("checkpoints/elo_graph.png");
            let relative_target = format!("{checkpoint_name}/elo_graph.png");
            // Remove existing symlink if present
            let _ = std::fs::remove_file(&root_graph);
            #[cfg(unix)]
            if let Err(e) = std::os::unix::fs::symlink(&relative_target, &root_graph) {
                progress.eprintln(&format!("Warning: Failed to create Elo graph symlink: {e}"));
            }
        }

        // Log final checkpoint save message
        let filename = checkpoint_path
            .file_name()
            .expect("checkpoint has filename")
            .to_string_lossy();
        let checkpoint_msg = if num_players > 1 {
            // Multiplayer: show Swiss points and training metrics
            let (avg_points, _draw_rate) = compute_avg_points(&recent_outcomes, num_players_usize);
            let p0_points = avg_points.first().copied().unwrap_or(0.0);
            if let Some(ref m) = last_metrics {
                format!(
                    "Saved final checkpoint at step {global_step} (pts: {p0_points:.2} | loss: {:.3} | ent: {:.2} | ev: {:.2}) -> {filename}",
                    m.policy_loss, m.entropy, m.explained_variance
                )
            } else {
                format!(
                    "Saved final checkpoint at step {global_step} (avg pts: {p0_points:.2}) -> {filename}"
                )
            }
        } else {
            // Single-player: show avg_return and training metrics
            if let Some(ref m) = last_metrics {
                format!(
                    "Saved final checkpoint at step {global_step} (ret: {avg_return:.1} | loss: {:.3} | ent: {:.2} | ev: {:.2}) -> {filename}",
                    m.policy_loss, m.entropy, m.explained_variance
                )
            } else {
                format!(
                    "Saved final checkpoint at step {global_step} (avg return: {avg_return:.1}) -> {filename}"
                )
            }
        };
        progress.println(&checkpoint_msg);
    }

    // Finish progress bar appropriately based on exit reason
    let did_train = global_step > initial_global_step;
    if exited_for_reload {
        // Subprocess reload exit: finish quietly without completion messages
        progress.finish_quiet();
    } else if running.load(Ordering::SeqCst) && did_train {
        // Natural completion: show full completion output
        progress.finish();
        println!("---");
        println!("Training complete!");
        if !recent_returns.is_empty() {
            let avg_return: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            println!("Final average return (last 100 episodes): {avg_return:.1}");
        }
    } else if running.load(Ordering::SeqCst) {
        // Spawned at goal, didn't train - remove the bar completely to avoid stray lines
        progress.finish_and_clear();
    } else {
        // User interrupt: show interrupted status
        progress.finish_interrupted();
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
                eval::run_evaluation::<<TB as burn::tensor::backend::AutodiffBackend>::InnerBackend>(
                    &eval_args, &device,
                )
            })
        }
        Some(Command::Tournament(tournament_args)) => {
            let backend_name = tournament_args
                .backend
                .as_deref()
                .unwrap_or_else(|| backend::default_backend());
            backend::warn_if_better_backend_available(backend_name);
            dispatch_backend!(backend_name, device, {
                tournament::run_tournament::<
                    <TB as burn::tensor::backend::AutodiffBackend>::InnerBackend,
                >(&tournament_args, &device)
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

/// Run training in supervisor mode, managing subprocess lifecycle
///
/// This is used when `--reload-every-n-checkpoints` is set to combat memory leaks
/// by periodically restarting the training subprocess.
fn run_as_supervisor(args: &CliArgs) -> Result<()> {
    use crate::supervisor::TrainingSupervisor;

    // Set up graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc_handler(r);

    // Create supervisor based on mode (resume vs fresh)
    let mut supervisor = if let Some(ref resume_path) = args.resume {
        // Resume mode: use existing run directory
        let run_dir = resume_path.clone();

        // Validate checkpoint exists
        let checkpoint_dir = run_dir.join("checkpoints/latest");
        if !checkpoint_dir.exists() {
            bail!(
                "No checkpoint found at {}. Cannot resume.",
                checkpoint_dir.display()
            );
        }

        let config_path = run_dir.join("config.toml");
        let mut config = Config::load_from_path(&config_path)
            .with_context(|| format!("Failed to load config from {}", config_path.display()))?;
        config.apply_resume_overrides(args);

        println!("burn-ppo v{} (supervisor mode)", env!("CARGO_PKG_VERSION"));
        println!(
            "Run: {}",
            run_dir.file_name().unwrap_or_default().to_string_lossy()
        );
        println!(
            "Reloading subprocess every {} checkpoints",
            args.reload_every_n_checkpoints
        );

        TrainingSupervisor::new_resume(
            run_dir,
            args.reload_every_n_checkpoints,
            config.total_timesteps,
            args.total_timesteps,
            args.max_training_time.clone(),
            running,
            args.debug_qi,
            args.debug_opponents,
        )
    } else {
        // Fresh run: load config to get run name, but let child create the directory
        let config = Config::load(args, None)?;
        config.validate()?;

        let run_dir = config.run_path();
        let run_name = config
            .run_name
            .clone()
            .expect("run_name should be set during Config::load()");

        // Check if run directory already exists
        if !check_run_exists_and_prompt(&run_dir)? {
            return Ok(());
        }

        println!("burn-ppo v{} (supervisor mode)", env!("CARGO_PKG_VERSION"));
        println!("Run: {run_name}");
        println!(
            "Reloading subprocess every {} checkpoints",
            args.reload_every_n_checkpoints
        );

        TrainingSupervisor::new_fresh(
            run_dir,
            args.reload_every_n_checkpoints,
            config.total_timesteps,
            args.total_timesteps,
            args.max_training_time.clone(),
            running,
            args.config.clone(),
            run_name,
            args.seed,
            args.debug_qi,
            args.debug_opponents,
        )
    };

    supervisor.run()
}

fn run_training_cli(args: &CliArgs) -> Result<()> {
    // If reload is requested and this is NOT already a subprocess, become supervisor
    // Skip supervisor mode for --fork since it's a one-time operation
    if args.reload_every_n_checkpoints > 0
        && args.max_checkpoints_this_run == 0
        && args.fork.is_none()
    {
        return run_as_supervisor(args);
    }

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

    // Subprocess mode: suppress header output (supervisor already printed it)
    let is_subprocess = args.max_checkpoints_this_run > 0;

    if !is_subprocess {
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
    }

    // Set up graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc_handler(r);

    // Get backend from args or use default
    let backend_name = args
        .backend
        .as_deref()
        .unwrap_or_else(|| backend::default_backend());
    if !is_subprocess {
        println!(
            "Backend: {} ({})",
            backend::get_backend_display_name(backend_name),
            backend_name
        );
        backend::warn_if_better_backend_available(backend_name);

        // Print rating guide for understanding Openskill ratings
        print_rating_guide();
    }

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
    let elapsed_time_offset_ms = args.elapsed_time_offset_ms;
    let max_checkpoints_this_run = args.max_checkpoints_this_run;
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
                elapsed_time_offset_ms,
                max_checkpoints_this_run,
                move |i| CartPole::new(seed + i as u64),
            ),
            "connect_four" => run_training::<TB, ConnectFour, _>(
                &mode,
                &config,
                &run_dir,
                resumed_metadata.as_ref(),
                &device,
                &running,
                elapsed_time_offset_ms,
                max_checkpoints_this_run,
                move |i| ConnectFour::new(seed + i as u64),
            ),
            "liars_dice" => {
                let reward_shaping_coef = config.reward_shaping_coef;
                run_training::<TB, LiarsDice, _>(
                    &mode,
                    &config,
                    &run_dir,
                    resumed_metadata.as_ref(),
                    &device,
                    &running,
                    elapsed_time_offset_ms,
                    max_checkpoints_this_run,
                    move |i| LiarsDice::new_with_config(seed + i as u64, reward_shaping_coef),
                )
            }
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
