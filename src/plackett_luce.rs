//! Plackett-Luce rating system for multi-player game rankings
//!
//! This module implements maximum likelihood estimation for the Plackett-Luce model,
//! which generalizes Bradley-Terry to multi-player rankings. Key features:
//!
//! - **Transitive inference**: A>B and B>C implies A>C even without direct matchups
//! - **Principled uncertainty**: Full Hessian inversion for accurate 95% confidence intervals
//! - **Tie handling**: Fractional win attribution for tied positions
//! - **Elo-scale output**: Anchored at 1000 for lowest-rated player
//!
//! # Mathematical Background
//!
//! For players with skill parameters γ₁, ..., γₙ (log-odds scale), the probability
//! of observing ranking r₁ > r₂ > ... > rₙ is:
//!
//! ```text
//! P(r₁ > r₂ > ... > rₙ) = ∏ᵢ₌₁ⁿ⁻¹ [exp(γᵣᵢ) / Σⱼ₌ᵢⁿ exp(γᵣⱼ)]
//! ```
//!
//! The log-likelihood is maximized using the MM (Minorization-Maximization) algorithm.

use std::collections::BTreeMap;

/// A single game result with player indices and their placements
#[derive(Debug, Clone)]
pub struct GameResult {
    /// Indices of players who participated in this game
    pub players: Vec<usize>,
    /// Placement for each player (1 = first place, can have ties)
    /// Must be same length as `players`
    pub placements: Vec<usize>,
}

impl GameResult {
    /// Create a new game result
    pub fn new(players: Vec<usize>, placements: Vec<usize>) -> Self {
        assert_eq!(
            players.len(),
            placements.len(),
            "players and placements must have same length"
        );
        Self {
            players,
            placements,
        }
    }
}

/// Rating result for a single player
#[derive(Debug, Clone, Copy)]
pub struct PlayerRating {
    /// Rating on Elo scale (lowest player anchored at 1000)
    pub rating: f64,
    /// Standard deviation of rating estimate (Elo scale)
    pub uncertainty: f64,
}

impl PlayerRating {
    /// Get 95% confidence interval bounds
    #[cfg(test)]
    pub fn confidence_interval(&self) -> (f64, f64) {
        (
            self.rating - 2.0 * self.uncertainty,
            self.rating + 2.0 * self.uncertainty,
        )
    }
}

impl Default for PlayerRating {
    fn default() -> Self {
        Self {
            rating: 1000.0,
            uncertainty: 350.0,
        }
    }
}

/// Statistics about the rating computation
#[derive(Debug, Clone, Copy)]
pub struct RatingStats {
    /// Whether the MM algorithm converged
    pub converged: bool,
    /// Number of iterations used
    pub iterations_used: usize,
    /// Maximum parameter change at termination
    pub final_delta: f64,
    /// Total computation time in milliseconds
    pub computation_time_ms: f64,
}

/// Result of rating computation including ratings and statistics
#[derive(Debug, Clone)]
pub struct RatingResult {
    /// Computed ratings for each player
    pub ratings: Vec<PlayerRating>,
    /// Statistics about the computation
    pub stats: RatingStats,
}

/// Configuration for the Plackett-Luce optimizer
#[derive(Debug, Clone)]
pub struct PlackettLuceConfig {
    /// Maximum iterations for MM algorithm
    pub max_iterations: usize,
    /// Convergence threshold (max change in any gamma)
    pub convergence_threshold: f64,
    /// Small constant for numerical stability
    pub epsilon: f64,
    /// Elo rating for the lowest-rated player (anchor point)
    pub anchor_elo: f64,
    /// Optional index of player to use as anchor (0 uncertainty, rating = `anchor_elo`)
    /// If None, uses lowest-rated player with games
    pub anchor_player_index: Option<usize>,
    /// Inflation factor for confidence intervals (default: 1.3)
    /// Fisher Information CIs are asymptotically correct but undercover at small sample sizes.
    /// This empirical correction widens CIs to achieve closer to 95% coverage.
    pub ci_inflation_factor: f64,
}

impl Default for PlackettLuceConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            epsilon: 1e-10,
            anchor_elo: 1000.0,
            anchor_player_index: None,
            ci_inflation_factor: 1.3,
        }
    }
}

/// Scale factor for converting between log-odds (gamma) and Elo
/// Elo = 1500 + 400 * gamma / ln(10)
const ELO_SCALE: f64 = 400.0 / std::f64::consts::LN_10; // ~173.72

/// Convert from internal log-odds scale to Elo scale
fn gamma_to_elo(gamma: f64) -> f64 {
    1500.0 + ELO_SCALE * gamma
}

/// Convert from Elo scale to internal log-odds scale
fn elo_to_gamma(elo: f64) -> f64 {
    (elo - 1500.0) / ELO_SCALE
}

/// Convert uncertainty from log-odds scale to Elo scale
fn uncertainty_to_elo(sigma: f64) -> f64 {
    ELO_SCALE * sigma
}

/// Numerically stable log-sum-exp computation
/// logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
#[cfg(test)]
fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    max_val
        + values
            .iter()
            .map(|&x| (x - max_val).exp())
            .sum::<f64>()
            .ln()
}

/// Numerically stable softmax computation
fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_shifted: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f64 = exp_shifted.iter().sum();
    if sum_exp == 0.0 {
        let n = logits.len();
        return vec![1.0 / n as f64; n];
    }
    exp_shifted.iter().map(|&x| x / sum_exp).collect()
}

/// A comparison extracted from a game for the MM algorithm
/// Represents "winner beat all losers simultaneously"
#[derive(Debug, Clone)]
struct Comparison {
    /// Index of the winning player
    winner: usize,
    /// Indices of all losing players in this comparison
    losers: Vec<usize>,
    /// Weight for fractional wins (1.0 for no ties, less for tied positions)
    weight: f64,
}

/// Expand games into weighted comparisons, handling ties via fractional attribution
fn expand_games_to_comparisons(games: &[GameResult]) -> Vec<Comparison> {
    let mut comparisons = Vec::new();

    for game in games {
        if game.players.len() <= 1 {
            continue;
        }

        // Group players by placement
        let mut placement_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (local_idx, &placement) in game.placements.iter().enumerate() {
            placement_groups
                .entry(placement)
                .or_default()
                .push(game.players[local_idx]);
        }

        // Get sorted placements (1st, 2nd, 3rd, etc.)
        let sorted_placements: Vec<usize> = placement_groups.keys().copied().collect();

        // For each position, the players there beat all players in worse positions
        for (pos_idx, &current_placement) in sorted_placements.iter().enumerate() {
            let current_players = &placement_groups[&current_placement];

            // Collect all players in strictly worse positions
            let mut lower_players: Vec<usize> = Vec::new();
            for &later_placement in &sorted_placements[pos_idx + 1..] {
                lower_players.extend(&placement_groups[&later_placement]);
            }

            if lower_players.is_empty() {
                continue; // No one to beat
            }

            let tie_size = current_players.len();

            if tie_size == 1 {
                // No tie - full weight for single winner
                comparisons.push(Comparison {
                    winner: current_players[0],
                    losers: lower_players,
                    weight: 1.0,
                });
            } else {
                // Tied players share wins fractionally
                // Each tied player gets 1/k of a win against lower players
                let weight = 1.0 / tie_size as f64;
                for &tied_player in current_players {
                    comparisons.push(Comparison {
                        winner: tied_player,
                        losers: lower_players.clone(),
                        weight,
                    });
                }
            }
        }
    }

    comparisons
}

/// Count how many games each player participated in
fn count_games_per_player(games: &[GameResult], num_players: usize) -> Vec<usize> {
    let mut counts = vec![0; num_players];
    for game in games {
        for &player in &game.players {
            if player < num_players {
                counts[player] += 1;
            }
        }
    }
    counts
}

/// MM algorithm update step
/// Returns new gammas (log-odds scale)
fn mm_update(
    comparisons: &[Comparison],
    gammas: &[f64],
    num_players: usize,
    epsilon: f64,
) -> Vec<f64> {
    // wins[i] = weighted count of wins for player i
    let mut wins = vec![0.0; num_players];
    // denom[i] = sum of (weight / sum_exp) over comparisons involving player i
    let mut denom = vec![0.0; num_players];

    for comp in comparisons {
        // Add weighted win for the winner
        wins[comp.winner] += comp.weight;

        // Compute sum of exp(gamma) for all players in this comparison
        let mut participants = vec![comp.winner];
        participants.extend(&comp.losers);

        let sum_exp: f64 = participants.iter().map(|&p| gammas[p].exp()).sum();

        if sum_exp > epsilon {
            // Add to denominator for all participants
            let contribution = comp.weight / sum_exp;
            for &p in &participants {
                denom[p] += contribution;
            }
        }
    }

    // MM update: alpha[i] = wins[i] / denom[i], then gamma[i] = ln(alpha[i])
    let mut new_gammas = vec![0.0; num_players];
    for i in 0..num_players {
        if wins[i] > epsilon && denom[i] > epsilon {
            new_gammas[i] = (wins[i] / denom[i]).ln();
        } else if denom[i] > epsilon {
            // Player participated but never won - assign low gamma
            new_gammas[i] = gammas[i] - 1.0;
        } else {
            // Player not in any comparison - keep current
            new_gammas[i] = gammas[i];
        }
    }

    new_gammas
}

/// Compute the Hessian matrix of the log-likelihood
/// Returns negative Hessian (Fisher information), which is positive semi-definite
fn compute_hessian(
    comparisons: &[Comparison],
    gammas: &[f64],
    num_players: usize,
) -> Vec<Vec<f64>> {
    let mut hessian = vec![vec![0.0; num_players]; num_players];

    for comp in comparisons {
        // Get all participants in this comparison
        let mut participants = vec![comp.winner];
        participants.extend(&comp.losers);

        // Compute softmax probabilities
        let participant_gammas: Vec<f64> = participants.iter().map(|&p| gammas[p]).collect();
        let probs = softmax(&participant_gammas);

        // Hessian contribution:
        // For negative Hessian (Fisher information):
        // H[i,i] += weight * p_i * (1 - p_i)
        // H[i,j] -= weight * p_i * p_j for i != j
        for (local_i, &player_i) in participants.iter().enumerate() {
            for (local_j, &player_j) in participants.iter().enumerate() {
                if local_i == local_j {
                    hessian[player_i][player_j] +=
                        comp.weight * probs[local_i] * (1.0 - probs[local_i]);
                } else {
                    hessian[player_i][player_j] -= comp.weight * probs[local_i] * probs[local_j];
                }
            }
        }
    }

    hessian
}

/// Invert a matrix using Gaussian elimination with partial pivoting
/// Returns the inverse, or a fallback high-uncertainty matrix if singular
#[expect(
    clippy::needless_range_loop,
    reason = "matrix operations require index-based iteration for clarity"
)]
fn invert_matrix(matrix: &[Vec<f64>], epsilon: f64) -> Vec<Vec<f64>> {
    let n = matrix.len();
    if n == 0 {
        return Vec::new();
    }

    // Create augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.extend(std::iter::repeat_n(0.0, n));
            new_row[n + i] = 1.0;
            new_row
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot (row with largest absolute value in this column)
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        // Swap rows
        aug.swap(col, max_row);

        // Check for singularity
        if aug[col][col].abs() < epsilon {
            // Near-singular - return high-uncertainty fallback
            return (0..n)
                .map(|i| {
                    let mut row = vec![0.0; n];
                    row[i] = 100.0; // High variance on diagonal
                    row
                })
                .collect();
        }

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate column in other rows
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse from right half
    aug.iter().map(|row| row[n..].to_vec()).collect()
}

/// Compute player ratings using Plackett-Luce maximum likelihood estimation
///
/// # Arguments
/// * `num_players` - Total number of players
/// * `games` - Slice of game results
/// * `config` - Optimizer configuration
///
/// # Returns
/// `RatingResult` containing ratings indexed by player ID and computation statistics
pub fn compute_ratings(
    num_players: usize,
    games: &[GameResult],
    config: &PlackettLuceConfig,
) -> RatingResult {
    let start_time = std::time::Instant::now();

    let default_stats = RatingStats {
        converged: true,
        iterations_used: 0,
        final_delta: 0.0,
        computation_time_ms: 0.0,
    };

    if num_players == 0 {
        return RatingResult {
            ratings: Vec::new(),
            stats: RatingStats {
                computation_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                ..default_stats
            },
        };
    }

    // Handle empty games - return default ratings
    if games.is_empty() {
        return RatingResult {
            ratings: (0..num_players)
                .map(|_| PlayerRating {
                    rating: config.anchor_elo,
                    uncertainty: 350.0,
                })
                .collect(),
            stats: RatingStats {
                computation_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                ..default_stats
            },
        };
    }

    // Track which players have game data
    let games_played = count_games_per_player(games, num_players);

    // Expand games into weighted comparisons
    let comparisons = expand_games_to_comparisons(games);

    if comparisons.is_empty() {
        return RatingResult {
            ratings: (0..num_players)
                .map(|_| PlayerRating {
                    rating: config.anchor_elo,
                    uncertainty: 350.0,
                })
                .collect(),
            stats: RatingStats {
                computation_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                ..default_stats
            },
        };
    }

    // Initialize gammas to 0 (all players equal)
    let mut gammas = vec![0.0; num_players];

    // MM algorithm iterations with statistics tracking
    let mut iterations_used = 0;
    let mut final_delta = f64::MAX;
    let mut converged = false;

    for iteration in 0..config.max_iterations {
        iterations_used = iteration + 1;
        let new_gammas = mm_update(&comparisons, &gammas, num_players, config.epsilon);

        // Center gammas (mean = 0) to prevent drift
        let mean = new_gammas.iter().sum::<f64>() / num_players as f64;
        let centered: Vec<f64> = new_gammas.iter().map(|&g| g - mean).collect();

        // Check convergence
        let max_change = gammas
            .iter()
            .zip(centered.iter())
            .map(|(&old, &new)| (old - new).abs())
            .fold(0.0, f64::max);

        final_delta = max_change;
        gammas = centered;

        if max_change < config.convergence_threshold {
            converged = true;
            break;
        }

        // Safety check for NaN/Inf
        if gammas.iter().any(|&g| !g.is_finite()) {
            gammas = vec![0.0; num_players];
            break;
        }
    }

    // Find anchor player: use configured anchor if valid, else lowest-rated with games
    let anchor_idx = if let Some(idx) = config.anchor_player_index {
        // Use specified anchor if valid (exists and has games)
        if idx < num_players && games_played[idx] > 0 {
            Some(idx)
        } else {
            // Fallback to lowest-rated player with games
            gammas
                .iter()
                .enumerate()
                .filter(|(i, _)| games_played[*i] > 0)
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
        }
    } else {
        // Default: lowest-rated player with games
        gammas
            .iter()
            .enumerate()
            .filter(|(i, _)| games_played[*i] > 0)
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    };

    // Compute Hessian for uncertainty estimation
    let hessian = compute_hessian(&comparisons, &gammas, num_players);

    // Build mapping from original indices to reduced indices (excluding anchor)
    let active_players: Vec<usize> = (0..num_players)
        .filter(|&i| games_played[i] > 0 && Some(i) != anchor_idx)
        .collect();

    // Build reduced Hessian (exclude anchor row/column) for constrained inversion
    let reduced_size = active_players.len();
    let mut reduced_hessian = vec![vec![0.0; reduced_size]; reduced_size];

    for (ri, &orig_i) in active_players.iter().enumerate() {
        for (rj, &orig_j) in active_players.iter().enumerate() {
            reduced_hessian[ri][rj] = hessian[orig_i][orig_j];
        }
        // Add small regularization for numerical stability only
        reduced_hessian[ri][ri] += 1e-6;
    }

    // Invert reduced matrix
    let reduced_cov = invert_matrix(&reduced_hessian, config.epsilon);

    // Extract uncertainties: anchor gets 0, others get sqrt(diagonal)
    let mut uncertainties = vec![2.0; num_players]; // Default high uncertainty

    // Anchor player gets 0 uncertainty (by definition - it's our reference)
    if let Some(anchor) = anchor_idx {
        uncertainties[anchor] = 0.0;
    }

    // Active players (non-anchor) get uncertainty from reduced covariance
    for (ri, &orig_i) in active_players.iter().enumerate() {
        if reduced_cov[ri][ri] > 0.0 {
            uncertainties[orig_i] = reduced_cov[ri][ri].sqrt();
        }
    }

    // Anchor lowest-rated player to anchor_elo
    let min_gamma = gammas
        .iter()
        .enumerate()
        .filter(|(i, _)| games_played[*i] > 0)
        .map(|(_, &g)| g)
        .fold(f64::INFINITY, f64::min);

    let anchor_gamma_target = elo_to_gamma(config.anchor_elo);

    // Shift all gammas so minimum maps to anchor
    let shift = if min_gamma.is_finite() {
        anchor_gamma_target - min_gamma
    } else {
        0.0
    };

    let computation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // Convert to Elo scale and return
    // Apply CI inflation factor to correct for Fisher Information underestimating uncertainty
    let inflation = config.ci_inflation_factor;
    let ratings = gammas
        .iter()
        .zip(uncertainties.iter())
        .enumerate()
        .map(|(i, (&gamma, &sigma))| {
            if games_played[i] > 0 {
                PlayerRating {
                    rating: gamma_to_elo(gamma + shift),
                    uncertainty: uncertainty_to_elo(sigma) * inflation,
                }
            } else {
                // Player without games gets default rating with high uncertainty
                PlayerRating {
                    rating: config.anchor_elo,
                    uncertainty: 350.0,
                }
            }
        })
        .collect();

    RatingResult {
        ratings,
        stats: RatingStats {
            converged,
            iterations_used,
            final_delta,
            computation_time_ms,
        },
    }
}

/// Convenience function to compute ratings with default config (returns just ratings for tests)
#[cfg(test)]
pub fn compute_ratings_default(num_players: usize, games: &[GameResult]) -> Vec<PlayerRating> {
    compute_ratings(num_players, games, &PlackettLuceConfig::default()).ratings
}

/// Print a guide to interpreting Plackett-Luce ratings (Elo scale)
pub fn print_rating_guide() {
    println!();
    println!("Rating Guide (Plackett-Luce / Elo scale):");
    println!("  Win probability: +100 pts -> 64% | +200 -> 76% | +400 -> 91% | +800 -> 99%");
    println!("  Uncertainty (sigma): high = few games, may shift. Low = stable rating.");
    println!(
        "  Comparing: if 95% CIs (rating +/- 2*sigma) overlap, difference may not be significant."
    );
    println!("  Lowest-rated player anchored at 1000.");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Numerical Utilities ====================

    #[test]
    fn test_logsumexp_basic() {
        let vals = vec![1.0, 2.0, 3.0];
        let result = logsumexp(&vals);
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_large_values() {
        // Test numerical stability with large values (should not overflow)
        let vals = vec![1000.0, 1001.0, 1002.0];
        let result = logsumexp(&vals);
        assert!(result.is_finite());
        assert!(result > 1000.0);
        // Expected: 1002 + ln(exp(-2) + exp(-1) + 1) ≈ 1002 + ln(1.503) ≈ 1002.41
        assert!((result - 1002.41).abs() < 0.1);
    }

    #[test]
    fn test_logsumexp_small_values() {
        // Test numerical stability with small values (should not underflow)
        let vals = vec![-1000.0, -1001.0, -1002.0];
        let result = logsumexp(&vals);
        assert!(result.is_finite());
        assert!(result < -999.0);
    }

    #[test]
    fn test_logsumexp_empty() {
        let vals: Vec<f64> = vec![];
        assert!(logsumexp(&vals).is_infinite() && logsumexp(&vals) < 0.0);
    }

    #[test]
    fn test_logsumexp_single() {
        let vals = vec![5.0];
        assert!((logsumexp(&vals) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 3);
        // Sum should be 1
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        // Higher logit = higher prob
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_equal() {
        let logits = vec![1.0, 1.0, 1.0];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_empty() {
        let logits: Vec<f64> = vec![];
        assert!(softmax(&logits).is_empty());
    }

    // ==================== Elo Conversion ====================

    #[test]
    fn test_gamma_to_elo_zero() {
        // gamma = 0 should give Elo = 1500
        assert!((gamma_to_elo(0.0) - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_to_elo_roundtrip() {
        for elo in [1000.0, 1500.0, 2000.0, 800.0, 2500.0] {
            let gamma = elo_to_gamma(elo);
            let back = gamma_to_elo(gamma);
            assert!((back - elo).abs() < 1e-10);
        }
    }

    #[test]
    fn test_elo_scale_400_difference() {
        // +400 Elo should correspond to ~10:1 odds, which is ln(10) in gamma
        let gamma_diff = std::f64::consts::LN_10;
        let elo_high = gamma_to_elo(gamma_diff / 2.0);
        let elo_low = gamma_to_elo(-gamma_diff / 2.0);
        assert!((elo_high - elo_low - 400.0).abs() < 1e-10);
    }

    // ==================== Two Player Games ====================

    #[test]
    fn test_simple_two_player_winner_higher() {
        // Player 0 always beats Player 1
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1], vec![1, 2]))
            .collect();

        let ratings = compute_ratings_default(2, &games);
        assert!(
            ratings[0].rating > ratings[1].rating,
            "Winner should be rated higher"
        );
    }

    #[test]
    fn test_two_player_rating_gap() {
        // Player 0 beats player 1 90% of the time (9 wins, 1 loss)
        let mut games: Vec<GameResult> = (0..9)
            .map(|_| GameResult::new(vec![0, 1], vec![1, 2]))
            .collect();
        games.push(GameResult::new(vec![0, 1], vec![2, 1]));

        let ratings = compute_ratings_default(2, &games);
        let gap = ratings[0].rating - ratings[1].rating;

        // 90% win rate ≈ +380 Elo (1/(1+10^(-380/400)) ≈ 0.90)
        assert!(
            gap > 250.0 && gap < 500.0,
            "90% win rate should give ~380 Elo gap, got {gap}"
        );
    }

    #[test]
    fn test_two_player_equal_wins() {
        // Each player wins half the games
        let mut games = Vec::new();
        for _ in 0..10 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2]));
            games.push(GameResult::new(vec![0, 1], vec![2, 1]));
        }

        let ratings = compute_ratings_default(2, &games);
        let gap = (ratings[0].rating - ratings[1].rating).abs();

        assert!(gap < 50.0, "Equal win rates should give similar ratings");
    }

    // ==================== Transitive Relationships ====================

    #[test]
    #[expect(
        clippy::similar_names,
        reason = "gap_ab/ac/bc are intentionally named for player pairs"
    )]
    fn test_three_player_transitive() {
        // A beats B, B beats C - system should infer A > B > C
        let mut games = Vec::new();
        // A beats B 5 times
        for _ in 0..5 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2]));
        }
        // B beats C 5 times
        for _ in 0..5 {
            games.push(GameResult::new(vec![1, 2], vec![1, 2]));
        }

        let ratings = compute_ratings_default(3, &games);

        assert!(
            ratings[0].rating > ratings[1].rating,
            "A should be rated higher than B"
        );
        assert!(
            ratings[1].rating > ratings[2].rating,
            "B should be rated higher than C"
        );
        assert!(
            ratings[0].rating > ratings[2].rating,
            "A should be rated higher than C (transitivity)"
        );

        // The gap A-C should be roughly A-B + B-C
        let gap_ac = ratings[0].rating - ratings[2].rating;
        let gap_ab = ratings[0].rating - ratings[1].rating;
        let gap_bc = ratings[1].rating - ratings[2].rating;
        assert!(
            (gap_ac - (gap_ab + gap_bc)).abs() < 100.0,
            "Transitive gap should be approximately additive"
        );
    }

    #[test]
    fn test_transitive_no_direct_matchup() {
        // A beats B, B beats C, but A never plays C
        // System should still rate A > C
        let mut games = Vec::new();
        for _ in 0..10 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2])); // A beats B
        }
        for _ in 0..10 {
            games.push(GameResult::new(vec![1, 2], vec![1, 2])); // B beats C
        }
        // Note: No A vs C games

        let ratings = compute_ratings_default(3, &games);

        assert!(
            ratings[0].rating > ratings[2].rating,
            "A should be rated higher than C even without direct matchup"
        );
    }

    // ==================== Multi-Player Games ====================

    #[test]
    fn test_four_player_game_ordering() {
        // 4-player game: rankings 1st, 2nd, 3rd, 4th consistently
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1, 2, 3], vec![1, 2, 3, 4]))
            .collect();

        let ratings = compute_ratings_default(4, &games);

        assert!(ratings[0].rating > ratings[1].rating);
        assert!(ratings[1].rating > ratings[2].rating);
        assert!(ratings[2].rating > ratings[3].rating);
    }

    #[test]
    fn test_three_player_game() {
        // 3-player games with consistent ordering
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1, 2], vec![1, 2, 3]))
            .collect();

        let ratings = compute_ratings_default(3, &games);

        assert!(ratings[0].rating > ratings[1].rating);
        assert!(ratings[1].rating > ratings[2].rating);
    }

    // ==================== Tie Handling ====================

    #[test]
    fn test_ties_two_way() {
        // Two players always tie
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1], vec![1, 1]))
            .collect();

        let ratings = compute_ratings_default(2, &games);
        let gap = (ratings[0].rating - ratings[1].rating).abs();

        assert!(gap < 50.0, "Tied players should have similar ratings");
    }

    #[test]
    fn test_ties_three_way() {
        // Three players always tie for 1st
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1, 2], vec![1, 1, 1]))
            .collect();

        let ratings = compute_ratings_default(3, &games);

        // All should have similar ratings
        let max_gap = ratings
            .iter()
            .flat_map(|r1| ratings.iter().map(|r2| (r1.rating - r2.rating).abs()))
            .fold(0.0, f64::max);

        assert!(
            max_gap < 50.0,
            "All tied players should have similar ratings"
        );
    }

    #[test]
    fn test_ties_partial() {
        // Players 0 and 1 tie for 1st, player 2 gets 3rd
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1, 2], vec![1, 1, 3]))
            .collect();

        let ratings = compute_ratings_default(3, &games);

        // Players 0 and 1 should have similar ratings
        assert!(
            (ratings[0].rating - ratings[1].rating).abs() < 50.0,
            "Tied players should have similar ratings"
        );
        // Both should be above player 2
        assert!(ratings[0].rating > ratings[2].rating);
        assert!(ratings[1].rating > ratings[2].rating);
    }

    #[test]
    fn test_ties_for_second() {
        // Player 0 wins, players 1 and 2 tie for 2nd
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1, 2], vec![1, 2, 2]))
            .collect();

        let ratings = compute_ratings_default(3, &games);

        // Player 0 should be highest
        assert!(ratings[0].rating > ratings[1].rating);
        assert!(ratings[0].rating > ratings[2].rating);
        // Players 1 and 2 should have similar ratings
        assert!(
            (ratings[1].rating - ratings[2].rating).abs() < 50.0,
            "Tied players should have similar ratings"
        );
    }

    // ==================== Uncertainty ====================

    #[test]
    fn test_uncertainty_decreases_with_more_games() {
        // Use varied outcomes to avoid extreme probability scenarios
        // that lead to near-singular Hessians
        let mut few_games = Vec::new();
        for _ in 0..5 {
            few_games.push(GameResult::new(vec![0, 1], vec![1, 2])); // 0 beats 1
            few_games.push(GameResult::new(vec![0, 1], vec![1, 2])); // 0 beats 1
            few_games.push(GameResult::new(vec![0, 1], vec![2, 1])); // 1 beats 0
        }

        let mut many_games = Vec::new();
        for _ in 0..50 {
            many_games.push(GameResult::new(vec![0, 1], vec![1, 2])); // 0 beats 1
            many_games.push(GameResult::new(vec![0, 1], vec![1, 2])); // 0 beats 1
            many_games.push(GameResult::new(vec![0, 1], vec![2, 1])); // 1 beats 0
        }

        let ratings_few = compute_ratings_default(2, &few_games);
        let ratings_many = compute_ratings_default(2, &many_games);

        // Both players should have reasonable uncertainty (with regularization)
        assert!(
            ratings_few[0].uncertainty > 0.0 && ratings_few[0].uncertainty < 2000.0,
            "Uncertainty should be reasonable: {}",
            ratings_few[0].uncertainty
        );

        assert!(
            ratings_many[0].uncertainty < ratings_few[0].uncertainty,
            "More games should lead to lower uncertainty: few={}, many={}",
            ratings_few[0].uncertainty,
            ratings_many[0].uncertainty
        );
    }

    #[test]
    fn test_confidence_interval() {
        let rating = PlayerRating {
            rating: 1600.0,
            uncertainty: 100.0,
        };

        let (low, high) = rating.confidence_interval();
        assert!((low - 1400.0).abs() < 1e-10);
        assert!((high - 1800.0).abs() < 1e-10);
    }

    #[test]
    #[expect(
        clippy::needless_range_loop,
        reason = "symmetry check requires comparing i,j vs j,i"
    )]
    fn test_hessian_symmetry() {
        let games = vec![
            GameResult::new(vec![0, 1, 2], vec![1, 2, 3]),
            GameResult::new(vec![0, 1, 2], vec![2, 1, 3]),
        ];

        let comparisons = expand_games_to_comparisons(&games);
        let gammas = vec![0.0, 0.0, 0.0];
        let hessian = compute_hessian(&comparisons, &gammas, 3);

        // Hessian should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (hessian[i][j] - hessian[j][i]).abs() < 1e-10,
                    "Hessian should be symmetric"
                );
            }
        }
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_empty_games() {
        let games: Vec<GameResult> = vec![];
        let ratings = compute_ratings_default(3, &games);

        assert_eq!(ratings.len(), 3);
        for r in &ratings {
            assert!((r.rating - 1000.0).abs() < 1.0); // Anchor Elo
        }
    }

    #[test]
    fn test_single_player() {
        let ratings = compute_ratings_default(1, &[]);
        assert_eq!(ratings.len(), 1);
        // Single player with no games should get anchor rating with high uncertainty
        assert!(
            (ratings[0].rating - 1000.0).abs() < 1.0,
            "Single player should get anchor rating (1000), got {}",
            ratings[0].rating
        );
        assert!(
            ratings[0].uncertainty > 200.0,
            "Single player should have high uncertainty, got {}",
            ratings[0].uncertainty
        );
    }

    #[test]
    fn test_zero_players() {
        let ratings = compute_ratings_default(0, &[]);
        assert!(ratings.is_empty());
    }

    #[test]
    fn test_player_without_games() {
        // Player 2 has no games
        let games = vec![
            GameResult::new(vec![0, 1], vec![1, 2]),
            GameResult::new(vec![0, 1], vec![1, 2]),
        ];

        let ratings = compute_ratings_default(3, &games);

        assert_eq!(ratings.len(), 3);
        // Player 2 should have default rating with high uncertainty
        assert!((ratings[2].rating - 1000.0).abs() < 1.0);
        assert!(ratings[2].uncertainty > 300.0);
    }

    #[test]
    fn test_player_never_wins() {
        // Player 1 never wins
        let games: Vec<GameResult> = (0..20)
            .map(|_| GameResult::new(vec![0, 1], vec![1, 2]))
            .collect();

        let ratings = compute_ratings_default(2, &games);

        // Should not produce NaN or Inf
        assert!(ratings[0].rating.is_finite());
        assert!(ratings[1].rating.is_finite());
        // Winner should still be rated higher
        assert!(ratings[0].rating > ratings[1].rating);
    }

    // ==================== Anchoring ====================

    #[test]
    fn test_anchor_at_1000() {
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1, 2], vec![1, 2, 3]))
            .collect();

        let ratings = compute_ratings_default(3, &games);

        // Lowest rated player should be at anchor (1000)
        let min_rating = ratings
            .iter()
            .map(|r| r.rating)
            .fold(f64::INFINITY, f64::min);
        assert!(
            (min_rating - 1000.0).abs() < 1.0,
            "Lowest player should be anchored at 1000"
        );
    }

    #[test]
    fn test_custom_anchor() {
        let games: Vec<GameResult> = (0..10)
            .map(|_| GameResult::new(vec![0, 1], vec![1, 2]))
            .collect();

        let config = PlackettLuceConfig {
            anchor_elo: 800.0,
            ..Default::default()
        };

        let result = compute_ratings(2, &games, &config);

        // Loser should be at anchor (800)
        assert!(
            (result.ratings[1].rating - 800.0).abs() < 1.0,
            "Lowest player should be anchored at custom value"
        );
    }

    // ==================== Config ====================

    #[test]
    fn test_config_default() {
        let config = PlackettLuceConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.anchor_elo, 1000.0);
    }

    // ==================== Comparison Expansion ====================

    #[test]
    fn test_expand_no_ties() {
        let game = GameResult::new(vec![0, 1, 2], vec![1, 2, 3]);
        let comparisons = expand_games_to_comparisons(&[game]);

        // Should produce 2 comparisons:
        // 1. Player 0 beats {1, 2}
        // 2. Player 1 beats {2}
        assert_eq!(comparisons.len(), 2);

        // First comparison: winner 0, losers [1, 2]
        assert_eq!(comparisons[0].winner, 0);
        assert_eq!(comparisons[0].losers.len(), 2);
        assert!((comparisons[0].weight - 1.0).abs() < 1e-10);

        // Second comparison: winner 1, losers [2]
        assert_eq!(comparisons[1].winner, 1);
        assert_eq!(comparisons[1].losers, vec![2]);
    }

    #[test]
    fn test_expand_with_ties() {
        let game = GameResult::new(vec![0, 1, 2], vec![1, 1, 3]);
        let comparisons = expand_games_to_comparisons(&[game]);

        // Players 0 and 1 tied for 1st, both beat player 2
        // Should produce 2 comparisons, each with weight 0.5
        assert_eq!(comparisons.len(), 2);

        for comp in &comparisons {
            assert!(comp.winner == 0 || comp.winner == 1);
            assert_eq!(comp.losers, vec![2]);
            assert!((comp.weight - 0.5).abs() < 1e-10);
        }
    }

    // ==================== Advanced Validation Tests ====================

    #[test]
    fn test_circular_preferences_rock_paper_scissors() {
        // Rock-paper-scissors scenario: A > B > C > A
        // All players should end up with similar ratings since there's no clear best
        let mut games = Vec::new();
        for _ in 0..10 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2])); // A beats B
            games.push(GameResult::new(vec![1, 2], vec![1, 2])); // B beats C
            games.push(GameResult::new(vec![2, 0], vec![1, 2])); // C beats A
        }

        let ratings = compute_ratings_default(3, &games);

        // All ratings should be close since no player dominates
        let max_gap = ratings
            .iter()
            .flat_map(|r1| ratings.iter().map(|r2| (r1.rating - r2.rating).abs()))
            .fold(0.0, f64::max);

        assert!(
            max_gap < 100.0,
            "Circular preferences should give similar ratings, max gap was {max_gap}"
        );
    }

    #[test]
    fn test_determinism_same_input_same_output() {
        // Same input should always produce same output
        let games: Vec<GameResult> = (0..20)
            .map(|i| {
                if i % 3 == 0 {
                    GameResult::new(vec![0, 1], vec![1, 2])
                } else if i % 3 == 1 {
                    GameResult::new(vec![0, 1], vec![2, 1])
                } else {
                    GameResult::new(vec![0, 1], vec![1, 1])
                }
            })
            .collect();

        let ratings1 = compute_ratings_default(2, &games);
        let ratings2 = compute_ratings_default(2, &games);

        assert!(
            (ratings1[0].rating - ratings2[0].rating).abs() < 1e-10,
            "Ratings should be deterministic"
        );
        assert!(
            (ratings1[1].rating - ratings2[1].rating).abs() < 1e-10,
            "Ratings should be deterministic"
        );
    }

    #[test]
    fn test_large_scale_many_players() {
        // Test with 20 players in a hierarchy
        let mut games = Vec::new();
        for i in 0..19 {
            // Player i beats player i+1 consistently
            for _ in 0..5 {
                games.push(GameResult::new(vec![i, i + 1], vec![1, 2]));
            }
        }

        let ratings = compute_ratings_default(20, &games);

        // Verify ordering: player 0 > player 1 > ... > player 19
        for i in 0..19 {
            assert!(
                ratings[i].rating > ratings[i + 1].rating,
                "Player {i} should be rated higher than player {}",
                i + 1
            );
        }

        // All ratings should be finite
        for (i, r) in ratings.iter().enumerate() {
            assert!(r.rating.is_finite(), "Player {i} rating should be finite");
            assert!(
                r.uncertainty.is_finite(),
                "Player {i} uncertainty should be finite"
            );
        }
    }

    #[test]
    fn test_large_scale_many_games() {
        // Test stability with many games
        let games: Vec<GameResult> = (0..500)
            .map(|i| {
                if i % 2 == 0 {
                    GameResult::new(vec![0, 1], vec![1, 2])
                } else {
                    GameResult::new(vec![0, 1], vec![2, 1])
                }
            })
            .collect();

        let ratings = compute_ratings_default(2, &games);

        // With 50-50 split, ratings should be similar
        let gap = (ratings[0].rating - ratings[1].rating).abs();
        assert!(
            gap < 50.0,
            "Equal win rates over many games should give similar ratings"
        );

        // Uncertainty should be reasonable (regularization prevents very low values)
        assert!(
            ratings[0].uncertainty < 1500.0,
            "Uncertainty should be reasonable with many games: {}",
            ratings[0].uncertainty
        );
    }

    #[test]
    fn test_unbalanced_game_counts() {
        // Player 0 plays many games, player 1 plays few
        let mut games = Vec::new();

        // Player 0 vs Player 1: 5 games (0 wins 3, 1 wins 2)
        for _ in 0..3 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2]));
        }
        for _ in 0..2 {
            games.push(GameResult::new(vec![0, 1], vec![2, 1]));
        }

        // Player 0 vs Player 2: 20 games (0 wins 12, 2 wins 8)
        for _ in 0..12 {
            games.push(GameResult::new(vec![0, 2], vec![1, 2]));
        }
        for _ in 0..8 {
            games.push(GameResult::new(vec![0, 2], vec![2, 1]));
        }

        let ratings = compute_ratings_default(3, &games);

        // All ratings should be finite and reasonable
        for r in &ratings {
            assert!(r.rating.is_finite());
            assert!(r.rating >= 500.0 && r.rating <= 2500.0);
        }

        // Player 0 has 25 games total, Player 1 has only 5 games, Player 2 has 20 games
        // Players with more games should generally have lower uncertainty
        // Note: Due to regularization the effect may be dampened, but ordering should hold
        assert!(
            ratings[0].uncertainty.is_finite() && ratings[0].uncertainty > 0.0,
            "Player 0 uncertainty should be positive and finite"
        );
        assert!(
            ratings[1].uncertainty.is_finite() && ratings[1].uncertainty > 0.0,
            "Player 1 uncertainty should be positive and finite"
        );
        // Player 1 (5 games) should have higher uncertainty than Player 0 (25 games)
        assert!(
            ratings[1].uncertainty > ratings[0].uncertainty,
            "Player with fewer games ({}) should have higher uncertainty than player with more games ({})",
            ratings[1].uncertainty, ratings[0].uncertainty
        );
    }

    #[test]
    fn test_win_probability_60_percent() {
        // 60% win rate should give approximately +70 Elo
        // P(win) = 1/(1+10^(-d/400)) => d = 400 * log10(p/(1-p))
        // For p=0.6: d = 400 * log10(0.6/0.4) = 400 * log10(1.5) ≈ 70
        let mut games = Vec::new();
        for _ in 0..60 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2]));
        }
        for _ in 0..40 {
            games.push(GameResult::new(vec![0, 1], vec![2, 1]));
        }

        let ratings = compute_ratings_default(2, &games);
        let gap = ratings[0].rating - ratings[1].rating;

        // 60% win rate should give ~70 Elo gap (tighter tolerance: 50-110)
        assert!(
            gap > 50.0 && gap < 110.0,
            "60% win rate should give ~70 Elo gap, got {gap}"
        );
        // Verify ordering: player 0 (60% winner) must be higher
        assert!(
            ratings[0].rating > ratings[1].rating,
            "Player with 60% win rate should be rated higher"
        );
    }

    #[test]
    fn test_win_probability_75_percent() {
        // 75% win rate should give approximately +190 Elo
        // d = 400 * log10(0.75/0.25) = 400 * log10(3) ≈ 191
        let mut games = Vec::new();
        for _ in 0..75 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2]));
        }
        for _ in 0..25 {
            games.push(GameResult::new(vec![0, 1], vec![2, 1]));
        }

        let ratings = compute_ratings_default(2, &games);
        let gap = ratings[0].rating - ratings[1].rating;

        // 75% win rate should give ~190 Elo gap (tighter tolerance: 150-250)
        assert!(
            gap > 150.0 && gap < 250.0,
            "75% win rate should give ~190 Elo gap, got {gap}"
        );
        // Verify ordering
        assert!(
            ratings[0].rating > ratings[1].rating,
            "Player with 75% win rate should be rated higher"
        );
    }

    #[test]
    fn test_win_probability_90_percent() {
        // 90% win rate should give approximately +382 Elo
        // d = 400 * log10(0.9/0.1) = 400 * log10(9) ≈ 382
        let mut games = Vec::new();
        for _ in 0..90 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2]));
        }
        for _ in 0..10 {
            games.push(GameResult::new(vec![0, 1], vec![2, 1]));
        }

        let ratings = compute_ratings_default(2, &games);
        let gap = ratings[0].rating - ratings[1].rating;

        // 90% win rate should give ~380 Elo gap (tolerance: 300-480)
        assert!(
            gap > 300.0 && gap < 480.0,
            "90% win rate should give ~380 Elo gap, got {gap}"
        );
        // Verify ordering
        assert!(
            ratings[0].rating > ratings[1].rating,
            "Player with 90% win rate should be rated higher"
        );
        // This gap should be larger than the 75% gap (~190)
        assert!(
            gap > 250.0,
            "90% win rate gap should be larger than 75% gap (~190), got {gap}"
        );
    }

    #[test]
    fn test_sparse_comparison_graph_long_chain() {
        // Chain: 0 > 1 > 2 > 3 > 4, each only plays adjacent opponent
        let mut games = Vec::new();
        for i in 0..4 {
            for _ in 0..10 {
                games.push(GameResult::new(vec![i, i + 1], vec![1, 2]));
            }
        }

        let ratings = compute_ratings_default(5, &games);

        // Should infer full ordering through transitivity
        for i in 0..4 {
            assert!(
                ratings[i].rating > ratings[i + 1].rating,
                "Player {i} should be rated higher than player {} (transitivity through chain)",
                i + 1
            );
        }

        // Player 0 should be much higher than player 4
        let total_gap = ratings[0].rating - ratings[4].rating;
        assert!(
            total_gap > 500.0,
            "Gap from best to worst in chain should be substantial"
        );
    }

    #[test]
    fn test_mixed_game_sizes() {
        // Mix of 2-player and 4-player games
        let mut games = Vec::new();

        // 2-player games
        for _ in 0..10 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2])); // 0 beats 1
        }

        // 4-player games
        for _ in 0..10 {
            games.push(GameResult::new(vec![0, 1, 2, 3], vec![1, 2, 3, 4]));
        }

        let ratings = compute_ratings_default(4, &games);

        // Should maintain ordering: 0 > 1 > 2 > 3
        assert!(ratings[0].rating > ratings[1].rating);
        assert!(ratings[1].rating > ratings[2].rating);
        assert!(ratings[2].rating > ratings[3].rating);
    }

    #[test]
    fn test_all_players_equal_round_robin() {
        // Complete round-robin where each player beats every other player once
        // Net effect: everyone is equal
        let mut games = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    games.push(GameResult::new(vec![i, j], vec![1, 2]));
                }
            }
        }

        let ratings = compute_ratings_default(4, &games);

        // All ratings should be similar
        let max_gap = ratings
            .iter()
            .flat_map(|r1| ratings.iter().map(|r2| (r1.rating - r2.rating).abs()))
            .fold(0.0, f64::max);

        assert!(
            max_gap < 100.0,
            "Complete round-robin should give similar ratings, max gap was {max_gap}"
        );
    }

    #[test]
    fn test_numerical_stability_no_nan_inf() {
        // Various edge cases that might produce NaN/Inf
        let test_cases: Vec<Vec<GameResult>> = vec![
            // Single game
            vec![GameResult::new(vec![0, 1], vec![1, 2])],
            // All ties
            (0..10)
                .map(|_| GameResult::new(vec![0, 1], vec![1, 1]))
                .collect(),
            // One player dominates completely
            (0..100)
                .map(|_| GameResult::new(vec![0, 1], vec![1, 2]))
                .collect(),
            // Complex multi-player
            (0..50)
                .map(|_| GameResult::new(vec![0, 1, 2, 3], vec![1, 2, 3, 4]))
                .collect(),
        ];

        for (i, games) in test_cases.iter().enumerate() {
            let ratings = compute_ratings_default(4, games);
            for (j, r) in ratings.iter().enumerate() {
                assert!(
                    r.rating.is_finite(),
                    "Case {i}, player {j}: rating should be finite, got {}",
                    r.rating
                );
                assert!(
                    r.uncertainty.is_finite(),
                    "Case {i}, player {j}: uncertainty should be finite, got {}",
                    r.uncertainty
                );
                assert!(
                    r.uncertainty >= 0.0,
                    "Case {i}, player {j}: uncertainty should be non-negative"
                );
            }
        }
    }

    #[test]
    #[expect(
        clippy::similar_names,
        reason = "gap_a_to_b/b_to_c/a_to_c are intentionally named for player pairs"
    )]
    fn test_rating_gaps_are_additive_in_chain() {
        // If A > B by X and B > C by Y, A > C should be approximately X + Y
        let mut games = Vec::new();

        // A beats B with 80% rate (20 games)
        for _ in 0..16 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2]));
        }
        for _ in 0..4 {
            games.push(GameResult::new(vec![0, 1], vec![2, 1]));
        }

        // B beats C with 80% rate (20 games)
        for _ in 0..16 {
            games.push(GameResult::new(vec![1, 2], vec![1, 2]));
        }
        for _ in 0..4 {
            games.push(GameResult::new(vec![1, 2], vec![2, 1]));
        }

        let ratings = compute_ratings_default(3, &games);

        let rating_gap_a_to_b = ratings[0].rating - ratings[1].rating;
        let rating_gap_b_to_c = ratings[1].rating - ratings[2].rating;
        let rating_gap_a_to_c = ratings[0].rating - ratings[2].rating;

        // Gap A-C should be approximately gap A-B + gap B-C
        let expected_total = rating_gap_a_to_b + rating_gap_b_to_c;
        let diff = (rating_gap_a_to_c - expected_total).abs();

        assert!(
            diff < 100.0,
            "Rating gaps should be approximately additive: actual={rating_gap_a_to_c}, expected={expected_total}, diff={diff}"
        );
    }

    #[test]
    fn test_tournament_simulation_realistic() {
        // Simulate a realistic tournament with varying skill levels
        // Players 0-2 are strong (beat lower players 70%)
        // Players 3-5 are medium
        // Players 6-9 are weak
        let mut games = Vec::new();

        // Strong vs Medium matchups
        for strong in 0..3 {
            for medium in 3..6 {
                // Strong wins 70% of 10 games = 7 wins
                for _ in 0..7 {
                    games.push(GameResult::new(vec![strong, medium], vec![1, 2]));
                }
                for _ in 0..3 {
                    games.push(GameResult::new(vec![strong, medium], vec![2, 1]));
                }
            }
        }

        // Medium vs Weak matchups
        for medium in 3..6 {
            for weak in 6..10 {
                // Medium wins 65% of 10 games
                for _ in 0..6 {
                    games.push(GameResult::new(vec![medium, weak], vec![1, 2]));
                }
                for _ in 0..4 {
                    games.push(GameResult::new(vec![medium, weak], vec![2, 1]));
                }
            }
        }

        // Strong vs Weak matchups (strong wins 85%)
        for strong in 0..3 {
            for weak in 6..10 {
                for _ in 0..8 {
                    games.push(GameResult::new(vec![strong, weak], vec![1, 2]));
                }
                for _ in 0..2 {
                    games.push(GameResult::new(vec![strong, weak], vec![2, 1]));
                }
            }
        }

        let ratings = compute_ratings_default(10, &games);

        // Verify tier ordering
        let strong_avg: f64 = ratings[0..3].iter().map(|r| r.rating).sum::<f64>() / 3.0;
        let medium_avg: f64 = ratings[3..6].iter().map(|r| r.rating).sum::<f64>() / 3.0;
        let weak_avg: f64 = ratings[6..10].iter().map(|r| r.rating).sum::<f64>() / 4.0;

        assert!(
            strong_avg > medium_avg,
            "Strong players should be rated higher than medium: strong={strong_avg:.1}, medium={medium_avg:.1}"
        );
        assert!(
            medium_avg > weak_avg,
            "Medium players should be rated higher than weak: medium={medium_avg:.1}, weak={weak_avg:.1}"
        );

        // Gap between strong and weak should be substantial (at least 200 Elo)
        let total_gap = strong_avg - weak_avg;
        assert!(
            total_gap > 200.0,
            "Gap between strong and weak tiers should be substantial, got {total_gap:.1}"
        );

        // Strong-Medium gap should be meaningful (70% win rate ≈ 85 Elo)
        let strong_medium_gap = strong_avg - medium_avg;
        assert!(
            strong_medium_gap > 50.0,
            "Strong-Medium gap should be meaningful (>50 Elo), got {strong_medium_gap:.1}"
        );

        // Medium-Weak gap should also be meaningful (65% win rate ≈ 55 Elo)
        let medium_weak_gap = medium_avg - weak_avg;
        assert!(
            medium_weak_gap > 30.0,
            "Medium-Weak gap should be meaningful (>30 Elo), got {medium_weak_gap:.1}"
        );
    }

    #[test]
    fn test_five_player_ffa_consistent_rankings() {
        // 5-player free-for-all games with consistent performance
        let games: Vec<GameResult> = (0..30)
            .map(|_| GameResult::new(vec![0, 1, 2, 3, 4], vec![1, 2, 3, 4, 5]))
            .collect();

        let ratings = compute_ratings_default(5, &games);

        // Verify strict ordering
        for i in 0..4 {
            assert!(
                ratings[i].rating > ratings[i + 1].rating,
                "Player {} should be rated higher than player {}",
                i,
                i + 1
            );
        }

        // Check meaningful rating spread
        // With extreme outcomes (player 0 always 1st, player 4 always 5th),
        // the spread can be very large - this is mathematically correct since
        // deterministic outcomes imply infinite skill gaps (capped by regularization)
        let spread = ratings[0].rating - ratings[4].rating;
        assert!(
            spread > 200.0,
            "Rating spread should be meaningful in 5-player games, got {spread}"
        );
    }

    #[test]
    fn test_mm_algorithm_convergence() {
        // Test that the algorithm converges - more iterations should produce same ordering
        // Using clearer data where one player wins 75% (converges faster than 50/50)
        let mut games = Vec::new();
        for _ in 0..30 {
            games.push(GameResult::new(vec![0, 1, 2], vec![1, 2, 3])); // 0 wins
        }
        for _ in 0..10 {
            games.push(GameResult::new(vec![0, 1, 2], vec![2, 1, 3])); // 1 wins
        }

        let config_few = PlackettLuceConfig {
            max_iterations: 100,
            ..Default::default()
        };

        let config_many = PlackettLuceConfig {
            max_iterations: 200,
            ..Default::default()
        };

        let result_few = compute_ratings(3, &games, &config_few);
        let result_many = compute_ratings(3, &games, &config_many);

        // Both should produce the same relative ordering: 0 > 1 > 2
        assert!(
            result_few.ratings[0].rating > result_few.ratings[1].rating,
            "Few iterations: Player 0 should beat Player 1"
        );
        assert!(
            result_few.ratings[1].rating > result_few.ratings[2].rating,
            "Few iterations: Player 1 should beat Player 2"
        );
        assert!(
            result_many.ratings[0].rating > result_many.ratings[1].rating,
            "Many iterations: Player 0 should beat Player 1"
        );
        assert!(
            result_many.ratings[1].rating > result_many.ratings[2].rating,
            "Many iterations: Player 1 should beat Player 2"
        );

        // The relative gaps should be similar (within 30% of each other)
        let gap_few = result_few.ratings[0].rating - result_few.ratings[2].rating;
        let gap_many = result_many.ratings[0].rating - result_many.ratings[2].rating;
        let ratio = gap_few / gap_many;
        assert!(
            (0.7..=1.3).contains(&ratio),
            "Rating gaps should be similar between few and many iterations, ratio={ratio}"
        );
    }

    #[test]
    fn test_expansion_complex_tie_scenario() {
        // Complex tie: [1, 2, 2, 4] - 1st place, two tied for 2nd, 4th place
        let game = GameResult::new(vec![0, 1, 2, 3], vec![1, 2, 2, 4]);
        let comparisons = expand_games_to_comparisons(&[game]);

        // Player 0 (1st) beats all others
        let p0_comps: Vec<_> = comparisons.iter().filter(|c| c.winner == 0).collect();
        assert!(
            !p0_comps.is_empty(),
            "Player 0 should have winning comparisons"
        );

        // Players 1 and 2 (tied 2nd) should each beat player 3, with fractional weight
        let p1_p2_beat_p3: Vec<_> = comparisons
            .iter()
            .filter(|c| (c.winner == 1 || c.winner == 2) && c.losers.contains(&3))
            .collect();

        // Should have 2 comparisons (one for each tied player beating P3)
        assert_eq!(
            p1_p2_beat_p3.len(),
            2,
            "Both tied players should beat player 3"
        );

        // Each should have weight 0.5
        for comp in &p1_p2_beat_p3 {
            assert!(
                (comp.weight - 0.5).abs() < 1e-10,
                "Tied players should split weight"
            );
        }
    }

    #[test]
    fn test_ratings_robust_to_game_order() {
        // Results shouldn't depend significantly on the order games are listed
        let mut games1 = Vec::new();
        let mut games2 = Vec::new();

        // Same games, different order
        for _ in 0..20 {
            games1.push(GameResult::new(vec![0, 1], vec![1, 2]));
        }
        for _ in 0..10 {
            games1.push(GameResult::new(vec![0, 1], vec![2, 1]));
        }

        // Interleaved order
        for _ in 0..10 {
            games2.push(GameResult::new(vec![0, 1], vec![1, 2]));
            games2.push(GameResult::new(vec![0, 1], vec![2, 1]));
            games2.push(GameResult::new(vec![0, 1], vec![1, 2]));
        }

        let ratings1 = compute_ratings_default(2, &games1);
        let ratings2 = compute_ratings_default(2, &games2);

        // Ratings should be identical (same data, just different order)
        for i in 0..2 {
            let diff = (ratings1[i].rating - ratings2[i].rating).abs();
            assert!(
                diff < 1.0,
                "Player {i}: ratings should be order-independent, diff={diff}"
            );
        }
    }

    // ==================== Constrained Hessian and Statistics Tests ====================

    #[test]
    fn test_anchor_has_zero_uncertainty() {
        // Anchor player (lowest rated with games) should have ~0 uncertainty
        // since they define the reference point
        let games: Vec<GameResult> = (0..20)
            .map(|_| GameResult::new(vec![0, 1, 2], vec![1, 2, 3]))
            .collect();
        let result = compute_ratings(3, &games, &PlackettLuceConfig::default());

        // Player 2 is always last → anchor (lowest gamma)
        assert!(
            result.ratings[2].uncertainty < 5.0,
            "Anchor should have ~0 uncertainty, got {}",
            result.ratings[2].uncertainty
        );
        // Other players should have positive uncertainty
        assert!(
            result.ratings[0].uncertainty > 10.0,
            "Non-anchor should have positive uncertainty"
        );
        assert!(
            result.ratings[1].uncertainty > 10.0,
            "Non-anchor should have positive uncertainty"
        );
    }

    #[test]
    fn test_uncertainty_sqrt_scaling() {
        // σ should scale as 1/√N (fundamental MLE property)
        // With 10x more games, uncertainty should be ~3.16x lower
        let games_10: Vec<GameResult> = (0..10)
            .map(|i| {
                if i % 3 == 0 {
                    GameResult::new(vec![0, 1], vec![1, 2])
                } else if i % 3 == 1 {
                    GameResult::new(vec![0, 1], vec![2, 1])
                } else {
                    GameResult::new(vec![0, 1], vec![1, 1])
                }
            })
            .collect();

        let games_100: Vec<GameResult> = (0..100)
            .map(|i| {
                if i % 3 == 0 {
                    GameResult::new(vec![0, 1], vec![1, 2])
                } else if i % 3 == 1 {
                    GameResult::new(vec![0, 1], vec![2, 1])
                } else {
                    GameResult::new(vec![0, 1], vec![1, 1])
                }
            })
            .collect();

        let r10 = compute_ratings(2, &games_10, &PlackettLuceConfig::default());
        let r100 = compute_ratings(2, &games_100, &PlackettLuceConfig::default());

        // Non-anchor player uncertainty (player 0 is higher-rated in both due to more wins)
        // Find the non-anchor player (higher uncertainty)
        let sigma_10 = r10.ratings[0].uncertainty.max(r10.ratings[1].uncertainty);
        let sigma_100 = r100.ratings[0].uncertainty.max(r100.ratings[1].uncertainty);

        // 10x more games → √10 ≈ 3.16x lower uncertainty
        let ratio = sigma_10 / sigma_100;
        assert!(
            ratio > 2.0 && ratio < 5.0,
            "10x games should give ~3x lower σ, got ratio {ratio} (σ10={sigma_10}, σ100={sigma_100})"
        );
    }

    #[test]
    fn test_stats_all_fields_valid() {
        let games: Vec<GameResult> = (0..50)
            .map(|_| GameResult::new(vec![0, 1], vec![1, 2]))
            .collect();
        let result = compute_ratings(2, &games, &PlackettLuceConfig::default());

        assert!(result.stats.iterations_used > 0);
        assert!(result.stats.iterations_used <= 100);
        assert!(result.stats.computation_time_ms >= 0.0);
        assert!(result.stats.final_delta.is_finite());
        // Should converge with clear winner
        assert!(result.stats.converged);
        assert!(result.stats.final_delta < 1e-6);
    }

    #[test]
    fn test_stats_non_convergence_reported() {
        // Use a scenario that requires more iterations to converge:
        // A beats B, B beats C, we need to propagate ratings through transitivity
        let mut games = Vec::new();
        for _ in 0..20 {
            games.push(GameResult::new(vec![0, 1], vec![1, 2])); // A beats B
            games.push(GameResult::new(vec![1, 2], vec![1, 2])); // B beats C
        }

        let config = PlackettLuceConfig {
            max_iterations: 1, // Force very early stop
            ..Default::default()
        };
        let result = compute_ratings(3, &games, &config);

        assert_eq!(result.stats.iterations_used, 1);
        assert!(!result.stats.converged);
        // Final delta should be larger than threshold since we didn't converge
        assert!(result.stats.final_delta > config.convergence_threshold);
    }

    #[test]
    fn test_uncertainty_independent_of_inactive_player_count() {
        // Same game data, different number of "bystander" players
        // Uncertainty should be similar for active players
        let games_2p: Vec<GameResult> = (0..20)
            .map(|_| GameResult::new(vec![0, 1], vec![1, 2]))
            .collect();

        let games_10p: Vec<GameResult> = (0..20)
            .map(|_| GameResult::new(vec![0, 1], vec![1, 2]))
            .collect();

        let r2 = compute_ratings(2, &games_2p, &PlackettLuceConfig::default());
        let r10 = compute_ratings(10, &games_10p, &PlackettLuceConfig::default());

        // Non-anchor player (player 0, winner) should have similar uncertainty
        // regardless of how many inactive players exist
        let diff = (r2.ratings[0].uncertainty - r10.ratings[0].uncertainty).abs();
        assert!(
            diff < 50.0,
            "Uncertainty shouldn't depend heavily on inactive players: diff={diff}"
        );
    }

    #[test]
    fn test_reduced_hessian_gives_varied_uncertainties() {
        // Verify we get meaningful uncertainty estimates (not all the same constant)
        let games: Vec<GameResult> = (0..100)
            .map(|i| {
                GameResult::new(
                    vec![0, 1, 2],
                    vec![(i % 3) + 1, ((i + 1) % 3) + 1, ((i + 2) % 3) + 1],
                )
            })
            .collect();

        let result = compute_ratings(3, &games, &PlackettLuceConfig::default());

        // Uncertainties should vary: anchor has ~0, others have positive
        let uncertainties: Vec<f64> = result.ratings.iter().map(|r| r.uncertainty).collect();

        // Check that not all are the same (which would indicate fallback behavior)
        let min_u = uncertainties.iter().copied().fold(f64::INFINITY, f64::min);
        let max_u = uncertainties
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            max_u - min_u > 10.0,
            "Uncertainties should vary: min={min_u}, max={max_u}"
        );

        // Anchor (lowest rated) should have smallest uncertainty
        let anchor_idx = result
            .ratings
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.rating.partial_cmp(&b.1.rating).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            result.ratings[anchor_idx].uncertainty < 5.0,
            "Anchor (idx {anchor_idx}) should have near-zero uncertainty, got {}",
            result.ratings[anchor_idx].uncertainty
        );
    }
}
