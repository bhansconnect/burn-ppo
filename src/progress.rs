/// Progress bar for training visualization
///
/// Uses indicatif for progress display with ETA and metrics.
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

/// Training progress display
pub struct TrainingProgress {
    main_bar: ProgressBar,
    start_time: Instant,
    num_players: usize,
}

impl TrainingProgress {
    /// Create progress display with player count for multiplayer games
    pub fn new_with_players(total_steps: u64, num_players: usize) -> Self {
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
            num_players,
        }
    }

    /// Update the main progress bar (single-player format)
    pub fn update(&self, step: u64, avg_return: f32) {
        self.main_bar.set_position(step);

        // Calculate SPS
        let elapsed = self.start_time.elapsed().as_secs_f32();
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
    /// Shows per-player returns and win rates for 2-player games,
    /// or just per-player returns for N>2 player games.
    pub fn update_multiplayer(
        &self,
        step: u64,
        returns_per_player: &[f32],
        win_rates: &[f32],
        draw_rate: f32,
    ) {
        self.main_bar.set_position(step);

        let elapsed = self.start_time.elapsed().as_secs_f32();
        let sps = if elapsed > 0.0 {
            step as f32 / elapsed
        } else {
            0.0
        };

        let msg = if self.num_players == 2 {
            // Compact 2-player format: SPS: 1250 | P0: 0.52 (48%W) | P1: 0.51 (47%W) | 5%D
            format!(
                "SPS: {:.0} | P0: {:.2} ({:.0}%W) | P1: {:.2} ({:.0}%W) | {:.0}%D",
                sps,
                returns_per_player.first().unwrap_or(&0.0),
                win_rates.first().unwrap_or(&0.0) * 100.0,
                returns_per_player.get(1).unwrap_or(&0.0),
                win_rates.get(1).unwrap_or(&0.0) * 100.0,
                draw_rate * 100.0
            )
        } else {
            // N-player format: just show returns
            let returns_str: String = returns_per_player
                .iter()
                .enumerate()
                .map(|(i, r)| format!("P{i}: {r:.2}"))
                .collect::<Vec<_>>()
                .join(" | ");
            format!("SPS: {sps:.0} | {returns_str}")
        };

        self.main_bar.set_message(msg);
    }

    /// Print a message above the progress bar without breaking the display
    ///
    /// This properly clears the bar, prints the message, then redraws the bar.
    pub fn println(&self, msg: &str) {
        self.main_bar.suspend(|| println!("{msg}"));
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
    fn test_training_progress_creation_single_player() {
        let progress = TrainingProgress::new_with_players(1000, 1);
        assert_eq!(progress.num_players, 1);
    }

    #[test]
    fn test_training_progress_creation_multiplayer() {
        let progress = TrainingProgress::new_with_players(1000, 2);
        assert_eq!(progress.num_players, 2);

        let progress3 = TrainingProgress::new_with_players(1000, 3);
        assert_eq!(progress3.num_players, 3);
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
        // Should not panic with 2-player format
        progress.update_multiplayer(500, &[0.5, 0.5], &[0.45, 0.45], 0.1);
        progress.finish();
    }

    #[test]
    fn test_training_progress_update_multiplayer_3player_no_panic() {
        let progress = TrainingProgress::new_with_players(1000, 3);
        // Should not panic with N-player format
        progress.update_multiplayer(500, &[0.33, 0.33, 0.34], &[0.3, 0.3, 0.3], 0.1);
        progress.finish();
    }
}
