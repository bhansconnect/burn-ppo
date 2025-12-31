mod checkpoint;
mod config;
mod env;
mod envs;
mod metrics;
mod network;
mod ppo;
mod utils;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::*;
use clap::Parser;
use rand::SeedableRng;

use crate::checkpoint::{CheckpointManager, CheckpointMetadata};
use crate::config::{CliArgs, Config};
use crate::env::VecEnv;
use crate::envs::CartPole;
use crate::metrics::MetricsLogger;
use crate::network::ActorCritic;
use crate::ppo::{collect_rollouts, compute_gae, ppo_update, RolloutBuffer};

/// Backend type for training (WGPU with autodiff)
type TrainingBackend = Autodiff<Wgpu>;

fn main() -> Result<()> {
    let args = CliArgs::parse();
    let config = Config::load(&args)?;

    println!("burn-ppo v{}", env!("CARGO_PKG_VERSION"));
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
    let device = WgpuDevice::default();
    println!("Device: {:?}", device);

    // Create run directory
    let run_dir = config.run_path();
    std::fs::create_dir_all(&run_dir)?;
    std::fs::create_dir_all(run_dir.join("checkpoints"))?;

    // Save config snapshot for reproducibility
    let config_snapshot_path = run_dir.join("config.toml");
    std::fs::write(&config_snapshot_path, toml::to_string_pretty(&config)?)?;

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

    // Create model
    let mut model: ActorCritic<TrainingBackend> =
        ActorCritic::new(obs_dim, action_count, &config, &device);
    println!("Created ActorCritic network");

    // Create optimizer
    let optimizer_config = AdamConfig::new().with_epsilon(config.adam_epsilon as f32);
    let mut optimizer = optimizer_config.init();

    // Create rollout buffer
    let mut buffer: RolloutBuffer<TrainingBackend> =
        RolloutBuffer::new(config.num_steps, num_envs, obs_dim, &device);

    // Create metrics logger
    let mut logger = MetricsLogger::new(&run_dir)?;
    logger.log_hparams(&config)?;
    logger.flush()?;

    // Create checkpoint manager
    let mut checkpoint_manager = CheckpointManager::new(&run_dir)?;
    let mut last_checkpoint_step = 0;

    // Training state
    let mut global_step = 0;
    let num_updates = config.total_timesteps / (config.num_steps * num_envs);
    let steps_per_update = config.num_steps * num_envs;

    println!(
        "Training for {} timesteps ({} updates of {} steps each)",
        config.total_timesteps, num_updates, steps_per_update
    );
    println!("---");

    // Episode tracking
    let mut recent_returns: Vec<f32> = Vec::new();

    // Training loop
    for update in 0..num_updates {
        if !running.load(Ordering::SeqCst) {
            println!("\nInterrupted by user");
            break;
        }

        // Learning rate annealing
        let lr = if config.lr_anneal {
            let progress = update as f64 / num_updates as f64;
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

        // Log metrics
        if global_step % config.log_freq == 0 || update == 0 {
            logger.log_scalar("train/policy_loss", metrics.policy_loss, global_step)?;
            logger.log_scalar("train/value_loss", metrics.value_loss, global_step)?;
            logger.log_scalar("train/entropy", metrics.entropy, global_step)?;
            logger.log_scalar("train/approx_kl", metrics.approx_kl, global_step)?;
            logger.log_scalar("train/clip_fraction", metrics.clip_fraction, global_step)?;
            logger.log_scalar("train/learning_rate", lr as f32, global_step)?;

            if !recent_returns.is_empty() {
                let avg_return: f32 =
                    recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
                logger.log_scalar("episode/return", avg_return, global_step)?;
            }

            logger.flush()?;

            // Console output
            let avg_return = if recent_returns.is_empty() {
                0.0
            } else {
                recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
            };
            println!(
                "Step {:>7} | Return: {:>6.1} | Policy Loss: {:>7.4} | Entropy: {:>5.3} | KL: {:>6.4}",
                global_step, avg_return, metrics.policy_loss, metrics.entropy, metrics.approx_kl
            );
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
            };

            let checkpoint_path = checkpoint_manager.save(&model, &metadata)?;
            println!(
                "Saved checkpoint at step {} (avg return: {:.1}) -> {:?}",
                global_step,
                avg_return,
                checkpoint_path.file_name().unwrap()
            );
            last_checkpoint_step = global_step;
        }
    }

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
