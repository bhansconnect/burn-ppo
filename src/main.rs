#![recursion_limit = "256"]

mod backend;
mod checkpoint;
mod config;
mod env;
mod envs;
mod metrics;
mod network;
mod ppo;
mod progress;
mod utils;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use clap::Parser;
use rand::SeedableRng;

use crate::backend::{init_device, backend_name, TrainingBackend};
use crate::checkpoint::{CheckpointManager, CheckpointMetadata, load_optimizer, save_optimizer};
use crate::config::{CliArgs, Config};
use crate::env::VecEnv;
use crate::envs::CartPole;
use crate::metrics::MetricsLogger;
use crate::network::ActorCritic;
use crate::ppo::{collect_rollouts, compute_gae, ppo_update, RolloutBuffer};
use crate::progress::TrainingProgress;

/// Extract run name from a checkpoint path
///
/// e.g., "runs/cartpole_001/checkpoints/best" -> "cartpole_001"
fn extract_run_name_from_checkpoint_path(checkpoint_path: &std::path::Path) -> Option<String> {
    // Navigate up from checkpoint: checkpoints/<name> -> checkpoints -> run_dir
    let run_dir = checkpoint_path.parent()?.parent()?;
    run_dir.file_name()?.to_str().map(String::from)
}

/// Check if run directory exists and prompt user for deletion confirmation
///
/// Returns Ok(true) if we should proceed (either didn't exist or user confirmed deletion)
/// Returns Ok(false) if user declined deletion
fn check_run_exists_and_prompt(run_dir: &std::path::Path) -> Result<bool> {
    if !run_dir.exists() {
        return Ok(true);
    }

    // Gather info about the existing run
    let checkpoints_dir = run_dir.join("checkpoints");
    let checkpoint_count = if checkpoints_dir.exists() {
        std::fs::read_dir(&checkpoints_dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter(|e| {
                        e.file_name()
                            .to_str()
                            .map(|n| n.starts_with("step_"))
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0)
    } else {
        0
    };

    eprintln!();
    eprintln!(
        "Warning: Run '{}' already exists at {:?}",
        run_dir.file_name().unwrap().to_string_lossy(),
        run_dir
    );
    eprintln!("This run contains {} checkpoint(s).", checkpoint_count);
    eprintln!();
    eprint!("Delete existing run and continue? [y/N]: ");

    // Flush stderr to ensure prompt is displayed
    use std::io::Write;
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
    Fork {
        checkpoint_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let args = CliArgs::parse();

    // Determine training mode
    let mode = if let Some(resume_path) = &args.resume {
        // Resume mode: continue existing run
        let run_dir = resume_path.clone();
        let checkpoint_dir = run_dir.join("checkpoints/latest");
        if !checkpoint_dir.exists() {
            bail!(
                "No checkpoint found at {:?}. Cannot resume.",
                checkpoint_dir
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
                "Checkpoint not found at {:?}. Cannot fork.",
                checkpoint_dir
            );
        }
        TrainingMode::Fork { checkpoint_dir }
    } else {
        TrainingMode::Fresh
    };

    // Load config based on mode
    let (config, run_dir, resumed_metadata) = match &mode {
        TrainingMode::Fresh => {
            let config = Config::load(&args, None)?;
            let run_dir = config.run_path();

            // Check if run directory already exists
            if !check_run_exists_and_prompt(&run_dir)? {
                return Ok(());
            }

            (config, run_dir, None)
        }
        TrainingMode::Resume { run_dir, checkpoint_dir } => {
            // Load config from the run directory
            let config_path = run_dir.join("config.toml");
            let mut config = Config::load_from_path(&config_path)
                .with_context(|| format!("Failed to load config from {:?}", config_path))?;

            // Apply limited resume overrides (only total_timesteps allowed)
            config.apply_resume_overrides(&args);

            // Load checkpoint metadata
            let metadata_path = checkpoint_dir.join("metadata.json");
            let metadata_json = std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("Failed to read {:?}", metadata_path))?;
            let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

            (config, run_dir.clone(), Some(metadata))
        }
        TrainingMode::Fork { checkpoint_dir } => {
            // Extract parent run name from checkpoint path
            let parent_run_name = extract_run_name_from_checkpoint_path(checkpoint_dir);

            // Use new config from CLI with forked_from set
            let config = Config::load(&args, parent_run_name)?;
            let run_dir = config.run_path();

            // Check if run directory already exists
            if !check_run_exists_and_prompt(&run_dir)? {
                return Ok(());
            }

            // Load checkpoint metadata
            let metadata_path = checkpoint_dir.join("metadata.json");
            let metadata_json = std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("Failed to read {:?}", metadata_path))?;
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
            println!("Mode: Forking from {:?}", checkpoint_dir)
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
    println!("Run: {}", config.run_name.as_ref().unwrap());

    // Set up graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc_handler(r);

    // Initialize device
    let device = init_device();
    println!("Backend: {}", backend_name());

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

    // Initialize RNG
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    // Create vectorized environment
    let num_envs = config.num_envs();
    let seed = config.seed;
    let mut vec_env = VecEnv::new(num_envs, |i| CartPole::new(seed + i as u64));

    let obs_dim = vec_env.observation_dim();
    let action_count = vec_env.action_count();
    println!(
        "Created {} environments (obs_dim={}, actions={})",
        num_envs, obs_dim, action_count
    );

    // Create optimizer with gradient clipping
    let optimizer_config = AdamConfig::new()
        .with_epsilon(config.adam_epsilon as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            config.max_grad_norm as f32,
        )));

    // Initialize model and optimizer based on mode
    let (mut model, mut optimizer, mut global_step, mut recent_returns, best_return) = match &mode {
        TrainingMode::Fresh => {
            let model: ActorCritic<TrainingBackend> =
                ActorCritic::new(obs_dim, action_count, &config, &device);
            let optimizer = optimizer_config.init();
            println!("Created ActorCritic network");
            (model, optimizer, 0, Vec::new(), f32::NEG_INFINITY)
        }
        TrainingMode::Resume { checkpoint_dir, .. } | TrainingMode::Fork { checkpoint_dir } => {
            let metadata = resumed_metadata.as_ref().unwrap();

            // Check for training completion
            if metadata.step >= config.total_timesteps {
                println!(
                    "Training already complete at step {}. Use --total-timesteps to extend.",
                    metadata.step
                );
                return Ok(());
            }

            // Load model from checkpoint
            let (model, _) =
                CheckpointManager::load::<TrainingBackend>(checkpoint_dir, &config, &device)
                    .with_context(|| format!("Failed to load checkpoint from {:?}", checkpoint_dir))?;

            // Create optimizer and try to load saved state
            let optimizer = optimizer_config.init();
            let optimizer = match load_optimizer::<TrainingBackend, _, ActorCritic<TrainingBackend>>(
                optimizer,
                checkpoint_dir,
                &device,
            ) {
                Ok(opt) => opt,
                Err(e) => {
                    eprintln!("Warning: Failed to load optimizer state: {}", e);
                    optimizer_config.init()
                }
            };

            let step = metadata.step;
            let recent_returns = metadata.recent_returns.clone();
            let best_return = metadata.best_avg_return;

            println!(
                "Loaded checkpoint from step {} (avg return: {:.1})",
                step, metadata.avg_return
            );

            (model, optimizer, step, recent_returns, best_return)
        }
    };

    // Create rollout buffer
    let mut buffer: RolloutBuffer<TrainingBackend> =
        RolloutBuffer::new(config.num_steps, num_envs, obs_dim, &device);

    // Create metrics logger
    let mut logger = MetricsLogger::new(&run_dir)?;
    if matches!(mode, TrainingMode::Fresh) {
        logger.log_hparams(&config)?;
    }
    logger.flush()?;

    // Create checkpoint manager
    let mut checkpoint_manager = CheckpointManager::new(&run_dir)?;
    checkpoint_manager.set_best_avg_return(best_return);
    let mut last_checkpoint_step = global_step;

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

    // Progress bar
    let progress = TrainingProgress::new(config.total_timesteps as u64);

    // Timing for SPS calculation
    let training_start = std::time::Instant::now();

    // Training loop
    for update in 0..num_updates {
        if !running.load(Ordering::SeqCst) {
            println!("\nInterrupted by user");
            break;
        }

        // Learning rate annealing (using total updates for proper continuation)
        let lr = if config.lr_anneal {
            let actual_update = update_offset + update;
            let progress = actual_update as f64 / total_updates as f64;
            config.learning_rate * (1.0 - progress)
        } else {
            config.learning_rate
        };

        // Collect rollouts
        let completed_episodes = collect_rollouts(
            &model,
            &mut vec_env,
            &mut buffer,
            config.num_steps,
            &device,
            &mut rng,
        );

        // Track episode returns
        for ep in &completed_episodes {
            recent_returns.push(ep.total_reward);
            if recent_returns.len() > 100 {
                recent_returns.remove(0);
            }
        }

        // Compute bootstrap value
        let obs_flat = vec_env.get_observations();
        let obs_tensor: Tensor<TrainingBackend, 2> =
            Tensor::<TrainingBackend, 1>::from_floats(obs_flat.as_slice(), &device)
                .reshape([num_envs, obs_dim]);
        let (_, last_values) = model.forward(obs_tensor);

        // Compute GAE
        compute_gae(
            &mut buffer,
            last_values,
            config.gamma as f32,
            config.gae_lambda as f32,
            &device,
        );

        // PPO update
        let (updated_model, metrics) = ppo_update(model, &buffer, &mut optimizer, &config, &mut rng);
        model = updated_model;

        global_step += steps_per_update;

        // Update progress bar
        let avg_return = if recent_returns.is_empty() {
            0.0
        } else {
            recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
        };
        progress.update(global_step as u64, avg_return);

        // Log metrics
        if global_step % config.log_freq == 0 || update == 0 {
            logger.log_scalar("train/policy_loss", metrics.policy_loss, global_step)?;
            logger.log_scalar("train/value_loss", metrics.value_loss, global_step)?;
            logger.log_scalar("train/entropy", metrics.entropy, global_step)?;
            logger.log_scalar("train/approx_kl", metrics.approx_kl, global_step)?;
            logger.log_scalar("train/clip_fraction", metrics.clip_fraction, global_step)?;
            logger.log_scalar("train/learning_rate", lr as f32, global_step)?;
            logger.log_scalar("train/explained_variance", metrics.explained_variance, global_step)?;
            logger.log_scalar("train/total_loss", metrics.total_loss, global_step)?;
            logger.log_scalar("train/value_mean", metrics.value_mean, global_step)?;
            logger.log_scalar("train/returns_mean", metrics.returns_mean, global_step)?;

            // Steps per second
            let elapsed = training_start.elapsed().as_secs_f32();
            let sps = if elapsed > 0.0 { global_step as f32 / elapsed } else { 0.0 };
            logger.log_scalar("charts/SPS", sps, global_step)?;

            if !recent_returns.is_empty() {
                let avg_return: f32 =
                    recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
                logger.log_scalar("episode/return", avg_return, global_step)?;
            }

            logger.flush()?;
        }

        // Log individual episode returns
        for ep in &completed_episodes {
            logger.log_scalar("episode/return_single", ep.total_reward, global_step)?;
            logger.log_scalar("episode/length", ep.length as f32, global_step)?;
        }

        // Checkpointing
        if global_step - last_checkpoint_step >= config.checkpoint_freq {
            let avg_return = if recent_returns.is_empty() {
                0.0
            } else {
                recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
            };

            let metadata = CheckpointMetadata {
                step: global_step,
                avg_return,
                rng_seed: config.seed,
                best_avg_return: checkpoint_manager.best_avg_return(),
                recent_returns: recent_returns.clone(),
                obs_dim,
                action_count,
                forked_from: config.forked_from.clone(),
            };

            let checkpoint_path = checkpoint_manager.save(&model, &metadata)?;

            // Save optimizer state alongside model
            if let Err(e) = save_optimizer::<TrainingBackend, _, ActorCritic<TrainingBackend>>(
                &optimizer,
                &checkpoint_path,
            ) {
                eprintln!("Warning: Failed to save optimizer state: {}", e);
            }

            println!(
                "Saved checkpoint at step {} (avg return: {:.1}) -> {:?}",
                global_step,
                avg_return,
                checkpoint_path.file_name().unwrap()
            );
            last_checkpoint_step = global_step;
        }
    }

    progress.finish();
    println!("---");
    println!("Training complete!");

    if !recent_returns.is_empty() {
        let avg_return: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
        println!("Final average return (last 100 episodes): {:.1}", avg_return);
    }

    Ok(())
}

/// Set up Ctrl+C handler for graceful shutdown
fn ctrlc_handler(running: Arc<AtomicBool>) {
    ctrlc::set_handler(move || {
        running.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");
}
