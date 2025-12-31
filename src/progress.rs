/// Progress bar for training visualization
///
/// Uses indicatif for progress display with ETA and metrics.

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

/// Training progress display
pub struct TrainingProgress {
    main_bar: ProgressBar,
    start_time: Instant,
}

impl TrainingProgress {
    /// Create new progress display
    pub fn new(total_steps: u64) -> Self {
        let main_bar = ProgressBar::new(total_steps);
        main_bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}/{duration_precise}] [{bar:40.cyan/blue}] {pos}/{len} | {msg}",
            )
            .expect("valid template")
            .progress_chars("##-"),
        );
        main_bar.set_message("Starting...");

        Self {
            main_bar,
            start_time: Instant::now(),
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

    /// Finish training and close progress bar
    pub fn finish(&self) {
        self.main_bar.finish_with_message("Training complete!");
    }
}
