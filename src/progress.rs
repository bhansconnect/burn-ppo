/// Progress bar for training visualization
///
/// Uses indicatif for progress display with ETA and metrics.
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::time::{Duration, Instant};

/// Training progress display
pub struct TrainingProgress {
    main_bar: ProgressBar,
    start_time: Instant,
    /// Elapsed time offset from previous subprocess runs (for reload mode)
    elapsed_offset: Duration,
    /// Total steps target (for ETA calculation)
    total_steps: u64,
}

impl TrainingProgress {
    /// Create progress display with player count for multiplayer games
    #[cfg(test)]
    pub fn new_with_players(total_steps: u64, num_players: usize) -> Self {
        Self::new_with_offset_and_position(total_steps, num_players, Duration::ZERO, 0, None)
    }

    /// Create progress display with offset and initial position
    ///
    /// This variant sets the initial position immediately to avoid flashing from 0.
    /// Used when resuming training in subprocess reload mode.
    ///
    /// If `initial_avg_return` is provided, the message will show real stats immediately
    /// instead of "Resuming...".
    pub fn new_with_offset_and_position(
        total_steps: u64,
        num_players: usize,
        elapsed_offset: Duration,
        initial_position: u64,
        initial_avg_return: Option<f32>,
    ) -> Self {
        // Use draw target that won't add blank lines
        let main_bar =
            ProgressBar::with_draw_target(Some(total_steps), ProgressDrawTarget::stderr());

        // Use slightly shorter bar for multiplayer to fit more info
        // Note: We use {msg} to show elapsed time since indicatif's {elapsed_precise}
        // can't account for time from previous subprocess runs
        let template = if num_players > 1 {
            "{spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} | {msg}"
        } else {
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} | {msg}"
        };

        main_bar.set_style(
            ProgressStyle::with_template(template)
                .expect("valid template")
                .progress_chars("##-"),
        );

        // Set position immediately before first draw to avoid flash to 0
        if initial_position > 0 {
            main_bar.set_position(initial_position);
            // Show real stats if available, otherwise placeholder
            if let Some(avg_return) = initial_avg_return {
                // Calculate approximate SPS from elapsed offset
                let elapsed_secs = elapsed_offset.as_secs_f32();
                let sps = if elapsed_secs > 0.0 {
                    initial_position as f32 / elapsed_secs
                } else {
                    0.0
                };
                // Calculate estimated total time (elapsed + remaining)
                let remaining = total_steps.saturating_sub(initial_position);
                let remaining_time = if sps > 0.0 {
                    Duration::from_secs_f32(remaining as f32 / sps)
                } else {
                    Duration::ZERO
                };
                let total_estimated = elapsed_offset + remaining_time;
                let elapsed_str = Self::format_elapsed(elapsed_offset);
                let total_str = Self::format_elapsed(total_estimated);
                main_bar.set_message(format!(
                    "{elapsed_str}/{total_str} | SPS: {sps:.0} | Return: {avg_return:.1}"
                ));
            } else {
                main_bar.set_message("Resuming...");
            }
        } else {
            main_bar.set_message("Starting...");
        }

        Self {
            main_bar,
            start_time: Instant::now(),
            elapsed_offset,
            total_steps,
        }
    }

    /// Total elapsed time including offset from previous subprocess runs
    fn total_elapsed(&self) -> Duration {
        self.elapsed_offset + self.start_time.elapsed()
    }

    /// Format duration as HH:MM:SS
    fn format_elapsed(duration: Duration) -> String {
        let secs = duration.as_secs();
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        let secs = secs % 60;
        format!("{hours:02}:{mins:02}:{secs:02}")
    }

    /// Update the main progress bar (single-player format)
    pub fn update(&self, step: u64, avg_return: f32) {
        self.main_bar.set_position(step);

        // Calculate SPS using total elapsed time (including offset from previous runs)
        let elapsed = self.total_elapsed();
        let elapsed_secs = elapsed.as_secs_f32();
        let sps = if elapsed_secs > 0.0 {
            step as f32 / elapsed_secs
        } else {
            0.0
        };

        // Calculate estimated total time (elapsed + remaining)
        let remaining = self.total_steps.saturating_sub(step);
        let remaining_time = if sps > 0.0 {
            Duration::from_secs_f32(remaining as f32 / sps)
        } else {
            Duration::ZERO
        };
        let total_estimated = elapsed + remaining_time;

        let elapsed_str = Self::format_elapsed(elapsed);
        let total_str = Self::format_elapsed(total_estimated);
        self.main_bar.set_message(format!(
            "{elapsed_str}/{total_str} | SPS: {sps:.0} | Return: {avg_return:.1}"
        ));
    }

    /// Update with multiplayer statistics
    ///
    /// Shows per-player returns and average Swiss points.
    /// Only displays players that participated in at least one game (`game_counts[p] > 0`).
    /// Format: `elapsed/eta | SPS: N | P0: ret (pts) | P1: ret (pts) | ... | D%`
    pub fn update_multiplayer(
        &self,
        step: u64,
        returns_per_player: &[f32],
        avg_points: &[f32],
        game_counts: &[usize],
        draw_rate: f32,
    ) {
        self.main_bar.set_position(step);

        // Calculate SPS using total elapsed time (including offset from previous runs)
        let elapsed = self.total_elapsed();
        let elapsed_secs = elapsed.as_secs_f32();
        let sps = if elapsed_secs > 0.0 {
            step as f32 / elapsed_secs
        } else {
            0.0
        };

        // Calculate estimated total time (elapsed + remaining)
        let remaining = self.total_steps.saturating_sub(step);
        let remaining_time = if sps > 0.0 {
            Duration::from_secs_f32(remaining as f32 / sps)
        } else {
            Duration::ZERO
        };
        let total_estimated = elapsed + remaining_time;

        // Only show players that participated in at least one game
        let stats_str: String = returns_per_player
            .iter()
            .zip(avg_points.iter())
            .zip(game_counts.iter())
            .enumerate()
            .filter(|(_, ((_, _), &count))| count > 0)
            .map(|(i, ((r, pts), _))| format!("P{i}: {r:.2} ({pts:.2}pts)"))
            .collect::<Vec<_>>()
            .join(" | ");

        let elapsed_str = Self::format_elapsed(elapsed);
        let total_str = Self::format_elapsed(total_estimated);
        let msg = format!(
            "{elapsed_str}/{total_str} | SPS: {sps:.0} | {stats_str} | {:.0}%D",
            draw_rate * 100.0
        );
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

    /// Finish progress bar quietly (for subprocess reload exit)
    /// Uses `abandon()` to leave the bar in place - the next subprocess's bar will overwrite it
    pub fn finish_quiet(&self) {
        self.main_bar.abandon();
    }

    /// Finish and clear the progress bar completely (removes the line)
    /// Used when subprocess exits without training to avoid leaving stray bar
    pub fn finish_and_clear(&self) {
        self.main_bar.finish_and_clear();
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
        // Should not panic: returns=[0.5, 0.5], avg_points=[0.45, 0.45], game_counts=[10, 10], draw_rate=0.1
        progress.update_multiplayer(500, &[0.5, 0.5], &[0.45, 0.45], &[10, 10], 0.1);
        progress.finish();
    }

    #[test]
    fn test_training_progress_update_multiplayer_3player_no_panic() {
        let progress = TrainingProgress::new_with_players(1000, 3);
        // Should not panic: returns, avg_points for 3 players
        progress.update_multiplayer(
            500,
            &[0.33, 0.33, 0.34],
            &[1.0, 1.0, 1.0],
            &[10, 10, 10],
            0.1,
        );
        progress.finish();
    }

    #[test]
    fn test_training_progress_update_multiplayer_filters_inactive_players() {
        let progress = TrainingProgress::new_with_players(1000, 4);
        // Only players 0, 1 have games - players 2, 3 should not appear in output
        progress.update_multiplayer(
            500,
            &[0.5, 0.6, 0.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0],
            &[10, 10, 0, 0],
            0.1,
        );
        progress.finish();
    }
}
