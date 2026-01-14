/// Progress bar for training visualization
///
/// Uses indicatif for progress display with ETA and metrics.
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

/// Training progress display
pub struct TrainingProgress {
    main_bar: ProgressBar,
    start_time: Instant,
    /// Elapsed time offset from previous subprocess runs (for reload mode)
    elapsed_offset: Duration,
}

impl TrainingProgress {
    /// Create progress display with player count for multiplayer games
    #[cfg(test)]
    pub fn new_with_players(total_steps: u64, num_players: usize) -> Self {
        Self::new_with_offset(total_steps, num_players, Duration::ZERO)
    }

    /// Create progress display with elapsed time offset from previous subprocess runs
    ///
    /// Used when training is restarted via `--reload-every-n-checkpoints` to maintain
    /// accurate SPS calculations across subprocess boundaries.
    pub fn new_with_offset(total_steps: u64, num_players: usize, elapsed_offset: Duration) -> Self {
        let main_bar = ProgressBar::new(total_steps);

        // Use slightly shorter bar for multiplayer to fit more info
        let template = if num_players > 1 {
            "{spinner:.green} [{elapsed_precise}/{duration_precise}] [{bar:30.cyan/blue}] {pos}/{len} | {msg}"
        } else {
            "{spinner:.green} [{elapsed_precise}/{duration_precise}] [{bar:40.cyan/blue}] {pos}/{len} | {msg}"
        };

        main_bar.set_style(
            ProgressStyle::with_template(template)
                .expect("valid template")
                .progress_chars("##-"),
        );
        main_bar.set_message("Starting...");

        Self {
            main_bar,
            start_time: Instant::now(),
            elapsed_offset,
        }
    }

    /// Total elapsed time including offset from previous subprocess runs
    fn total_elapsed(&self) -> Duration {
        self.elapsed_offset + self.start_time.elapsed()
    }

    /// Update the main progress bar (single-player format)
    pub fn update(&self, step: u64, avg_return: f32) {
        self.main_bar.set_position(step);

        // Calculate SPS using total elapsed time (including offset from previous runs)
        let elapsed = self.total_elapsed().as_secs_f32();
        let sps = if elapsed > 0.0 {
            step as f32 / elapsed
        } else {
            0.0
        };

        self.main_bar
            .set_message(format!("SPS: {sps:.0} | Return: {avg_return:.1}"));
    }

    /// Update with multiplayer statistics
    ///
    /// Shows per-player returns and average Swiss points.
    /// Format: `SPS: N | P0: ret (pts) | P1: ret (pts) | ... | D%`
    pub fn update_multiplayer(
        &self,
        step: u64,
        returns_per_player: &[f32],
        avg_points: &[f32],
        draw_rate: f32,
    ) {
        self.main_bar.set_position(step);

        // Calculate SPS using total elapsed time (including offset from previous runs)
        let elapsed = self.total_elapsed().as_secs_f32();
        let sps = if elapsed > 0.0 {
            step as f32 / elapsed
        } else {
            0.0
        };

        // Unified format for all N-player games
        let stats_str: String = returns_per_player
            .iter()
            .zip(avg_points.iter())
            .enumerate()
            .map(|(i, (r, pts))| format!("P{i}: {r:.2} ({pts:.2}pts)"))
            .collect::<Vec<_>>()
            .join(" | ");

        let msg = format!("SPS: {sps:.0} | {stats_str} | {:.0}%D", draw_rate * 100.0);
        self.main_bar.set_message(msg);
    }

    /// Print a message above the progress bar without breaking the display
    ///
    /// This properly clears the bar, prints the message, then redraws the bar.
    pub fn println(&self, msg: &str) {
        self.main_bar.suspend(|| println!("{msg}"));
    }

    /// Print an error message above the progress bar without breaking the display
    pub fn eprintln(&self, msg: &str) {
        self.main_bar.suspend(|| eprintln!("{msg}"));
    }

    /// Finish training and close progress bar
    pub fn finish(&self) {
        self.main_bar.finish_with_message("Training complete!");
    }

    /// Finish progress bar due to interruption (preserves current position)
    pub fn finish_interrupted(&self) {
        self.main_bar.abandon_with_message("Interrupted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_progress_creation_no_panic() {
        // Test that creation works for various player counts
        let _ = TrainingProgress::new_with_players(1000, 1);
        let _ = TrainingProgress::new_with_players(1000, 2);
        let _ = TrainingProgress::new_with_players(1000, 4);
    }

    #[test]
    fn test_training_progress_update_no_panic() {
        let progress = TrainingProgress::new_with_players(1000, 1);
        // Should not panic
        progress.update(500, 125.5);
        progress.finish();
    }

    #[test]
    fn test_training_progress_update_multiplayer_no_panic() {
        let progress = TrainingProgress::new_with_players(1000, 2);
        // Should not panic: returns=[0.5, 0.5], avg_points=[0.45, 0.45], draw_rate=0.1
        progress.update_multiplayer(500, &[0.5, 0.5], &[0.45, 0.45], 0.1);
        progress.finish();
    }

    #[test]
    fn test_training_progress_update_multiplayer_3player_no_panic() {
        let progress = TrainingProgress::new_with_players(1000, 3);
        // Should not panic: returns, avg_points for 3 players
        progress.update_multiplayer(500, &[0.33, 0.33, 0.34], &[1.0, 1.0, 1.0], 0.1);
        progress.finish();
    }
}
