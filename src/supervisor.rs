//! Subprocess supervisor for memory leak mitigation
//!
//! Runs training in a child process and restarts it every N checkpoints.
//! This works around memory leaks that cannot be fixed by periodically
//! restarting the training process while preserving all training state.

use std::path::PathBuf;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::checkpoint::load_metadata;
use crate::count_checkpoints;

/// Supervisor that manages training subprocess lifecycle
pub struct TrainingSupervisor {
    /// Directory containing the training run
    run_dir: PathBuf,
    /// Number of checkpoints to save before restarting subprocess
    reload_every_n: usize,
    /// Total timesteps target (for completion check)
    total_timesteps: usize,
    /// Optional override for `total_timesteps` to pass to child
    total_timesteps_override: Option<usize>,
    /// Optional max training time to pass to child
    max_training_time_override: Option<String>,
    /// Flag to signal graceful shutdown
    running: Arc<AtomicBool>,
    /// Whether this is a fresh run (first subprocess runs fresh, subsequent use --resume)
    is_fresh_run: bool,
    /// Config path for fresh runs
    config_path: Option<PathBuf>,
    /// Run name for fresh runs
    run_name: Option<String>,
    /// Seed override (pass through to subprocess)
    seed_override: Option<u64>,
}

impl TrainingSupervisor {
    /// Create a new training supervisor for resuming an existing run
    pub fn new_resume(
        run_dir: PathBuf,
        reload_every_n: usize,
        total_timesteps: usize,
        total_timesteps_override: Option<usize>,
        max_training_time_override: Option<String>,
        running: Arc<AtomicBool>,
    ) -> Self {
        Self {
            run_dir,
            reload_every_n,
            total_timesteps,
            total_timesteps_override,
            max_training_time_override,
            running,
            is_fresh_run: false,
            config_path: None,
            run_name: None,
            seed_override: None, // Seed not overridable for resume
        }
    }

    /// Create a new training supervisor for a fresh run
    pub fn new_fresh(
        run_dir: PathBuf,
        reload_every_n: usize,
        total_timesteps: usize,
        total_timesteps_override: Option<usize>,
        max_training_time_override: Option<String>,
        running: Arc<AtomicBool>,
        config_path: PathBuf,
        run_name: String,
        seed_override: Option<u64>,
    ) -> Self {
        Self {
            run_dir,
            reload_every_n,
            total_timesteps,
            total_timesteps_override,
            max_training_time_override,
            running,
            is_fresh_run: true,
            config_path: Some(config_path),
            run_name: Some(run_name),
            seed_override,
        }
    }

    /// Run the supervised training loop
    ///
    /// Spawns child processes that run training and restarts them every
    /// `reload_every_n` checkpoints until training is complete.
    pub fn run(&mut self) -> Result<()> {
        let mut total_elapsed = Duration::ZERO;
        let mut is_first_iteration = true;

        loop {
            if !self.running.load(Ordering::SeqCst) {
                eprintln!("\nSupervisor: interrupted by user");
                break;
            }

            // Check if training is complete before spawning (skip on first fresh run)
            if (!is_first_iteration || !self.is_fresh_run) && self.is_training_complete()? {
                eprintln!("Supervisor: training complete");
                break;
            }

            let checkpoint_baseline = count_checkpoints(&self.run_dir);

            // Spawn child process
            let start = Instant::now();
            let status = self.spawn_and_wait(total_elapsed, is_first_iteration)?;
            total_elapsed += start.elapsed();

            // After first iteration, always use resume mode
            is_first_iteration = false;

            if !status.success() {
                // Check if we were interrupted
                if !self.running.load(Ordering::SeqCst) {
                    eprintln!("\nSupervisor: subprocess interrupted");
                    break;
                }
                // Child failed unexpectedly
                anyhow::bail!(
                    "Training subprocess failed with exit code: {:?}",
                    status.code()
                );
            }

            // Check if training completed (child may have finished naturally)
            let new_checkpoint_count = count_checkpoints(&self.run_dir);
            let checkpoints_saved = new_checkpoint_count - checkpoint_baseline;

            // If subprocess saved 0 checkpoints and exited successfully, training is likely complete
            // (either it reached the end, or the step count is close enough that no more updates fit)
            if checkpoints_saved == 0 || self.is_training_complete()? {
                break;
            }
        }

        Ok(())
    }

    /// Spawn a child training process and wait for it to complete
    fn spawn_and_wait(
        &self,
        elapsed_offset: Duration,
        is_first_iteration: bool,
    ) -> Result<ExitStatus> {
        let exe = std::env::current_exe().context("Failed to get current executable path")?;

        let mut args = vec!["train".to_string()];

        // First iteration of a fresh run: use config file
        // All other cases: use --resume
        if is_first_iteration && self.is_fresh_run {
            // Fresh run: use config path and run name
            args.push("--config".to_string());
            args.push(
                self.config_path
                    .as_ref()
                    .expect("config_path set for fresh run")
                    .to_string_lossy()
                    .to_string(),
            );
            args.push("--run-name".to_string());
            args.push(
                self.run_name
                    .as_ref()
                    .expect("run_name set for fresh run")
                    .clone(),
            );
        } else {
            // Resume mode
            args.push("--resume".to_string());
            args.push(self.run_dir.to_string_lossy().to_string());
        }

        // Always pass elapsed offset and max checkpoints
        args.push("--elapsed-time-offset-ms".to_string());
        args.push(elapsed_offset.as_millis().to_string());
        args.push("--max-checkpoints-this-run".to_string());
        args.push(self.reload_every_n.to_string());

        // Pass through overridable args if set
        if let Some(ts) = self.total_timesteps_override {
            args.push("--total-timesteps".to_string());
            args.push(ts.to_string());
        }
        if let Some(ref time) = self.max_training_time_override {
            args.push("--max-training-time".to_string());
            args.push(time.clone());
        }
        if let Some(seed) = self.seed_override {
            args.push("--seed".to_string());
            args.push(seed.to_string());
        }

        let mut child = Command::new(exe)
            .args(&args)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn training subprocess")?;

        // Wait for child, checking for interrupts
        loop {
            if let Some(status) = child.try_wait()? {
                return Ok(status);
            }
            // Check for Ctrl+C
            if !self.running.load(Ordering::SeqCst) {
                // Kill the child process - it will handle cleanup via its own Ctrl+C handler
                // since stdin/stdout/stderr are inherited
                let _ = child.kill();
                return child
                    .wait()
                    .context("Failed to wait for child after interrupt");
            }
            // Sleep briefly before next poll
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Check if training has reached the target timesteps
    fn is_training_complete(&self) -> Result<bool> {
        let checkpoint_dir = self.run_dir.join("checkpoints").join("latest");
        if !checkpoint_dir.exists() {
            return Ok(false);
        }

        let metadata = load_metadata(&checkpoint_dir)?;
        Ok(metadata.step >= self.total_timesteps)
    }
}
