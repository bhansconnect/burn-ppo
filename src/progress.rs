/// Progress bars for training visualization
///
/// Uses indicatif for hierarchical progress display:
/// - Main bar: total training progress with ETA and SPS
/// - Epoch bar: PPO update epochs (shown during updates)

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Instant;

/// Training progress display with hierarchical bars
pub struct TrainingProgress {
    multi: MultiProgress,
    main_bar: ProgressBar,
    epoch_bar: ProgressBar,
    start_time: Instant,
    total_steps: u64,
}

impl TrainingProgress {
    /// Create new progress display
    pub fn new(total_steps: u64) -> Self {
        let multi = MultiProgress::new();

        // Main progress bar: total training progress
        let main_bar = multi.add(ProgressBar::new(total_steps));
        main_bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) | {msg}",
            )
            .expect("valid template")
            .progress_chars("##-"),
        );
        main_bar.set_message("Starting...");

        // Epoch bar: PPO update epochs (hidden initially)
        let epoch_bar = multi.add(ProgressBar::new(0));
        epoch_bar.set_style(
            ProgressStyle::with_template("  Epoch {pos}/{len} | {msg}")
                .expect("valid template"),
        );
        epoch_bar.set_length(0); // Hidden until update starts

        Self {
            multi,
            main_bar,
            epoch_bar,
            start_time: Instant::now(),
            total_steps,
        }
    }

    /// Update the main progress bar
    pub fn update(&self, step: u64, avg_return: f32) {
        self.main_bar.set_position(step);

        // Calculate SPS
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let sps = if elapsed > 0.0 {
            step as f32 / elapsed
        } else {
            0.0
        };

        self.main_bar.set_message(format!(
            "SPS: {:.0} | Return: {:.1}",
            sps, avg_return
        ));
    }

    /// Start a PPO update phase (shows epoch bar)
    pub fn start_ppo_update(&self, num_epochs: u64) {
        self.epoch_bar.set_length(num_epochs);
        self.epoch_bar.set_position(0);
        self.epoch_bar.set_message("Starting update...");
    }

    /// Update epoch progress during PPO update
    pub fn update_epoch(&self, epoch: u64, loss: f32) {
        self.epoch_bar.set_position(epoch + 1);
        self.epoch_bar.set_message(format!("Loss: {:.4}", loss));
    }

    /// Finish the PPO update phase (hides epoch bar)
    pub fn finish_ppo_update(&self) {
        self.epoch_bar.set_length(0);
    }

    /// Finish training and close progress bars
    pub fn finish(&self) {
        self.main_bar.finish_with_message("Training complete!");
        self.epoch_bar.finish_and_clear();
    }

    /// Get the MultiProgress for proper terminal handling
    pub fn multi(&self) -> &MultiProgress {
        &self.multi
    }
}
