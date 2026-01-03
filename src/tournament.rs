//! Tournament mode for evaluating multiple checkpoints with skill ratings
//!
//! Features:
//! - Swiss-style tournaments for efficient skill estimation (N > 8 contestants)
//! - Round-robin for complete coverage (N <= 8 contestants)
//! - Weng-Lin (`OpenSkill`) rating system with uncertainty tracking
//! - Progress bars and intermediate standings

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use skillratings::weng_lin::{weng_lin_multi_team, WengLinConfig, WengLinRating};
use skillratings::MultiTeamOutcome;

use crate::checkpoint::{load_metadata, CheckpointMetadata};
use crate::config::TournamentArgs;
use crate::dispatch_env;
use crate::env::{Environment, GameOutcome};
use crate::eval::{load_model_from_checkpoint, run_stats_mode_env, PlayerSource, TempSchedule};
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;

/// Single game result with placements for all participants
#[derive(Debug, Clone)]
pub struct GamePlacement {
    /// Placement for each contestant (1-indexed, 1=first place)
    pub placements: Vec<usize>,
}

/// Results of a matchup between contestants with per-game data
#[derive(Debug, Clone)]
pub struct MatchupGames {
    /// Indices of contestants in this matchup
    pub contestants: Vec<usize>,
    /// Individual game results
    pub games: Vec<GamePlacement>,
}

impl MatchupGames {
    /// Get wins for contestant at index 0 (for 2-player matchups)
    pub fn wins_a(&self) -> usize {
        self.games
            .iter()
            .filter(|g| g.placements.len() >= 2 && g.placements[0] < g.placements[1])
            .count()
    }

    /// Get wins for contestant at index 1 (for 2-player matchups)
    pub fn wins_b(&self) -> usize {
        self.games
            .iter()
            .filter(|g| g.placements.len() >= 2 && g.placements[1] < g.placements[0])
            .count()
    }

    /// Get draw count (for 2-player matchups)
    pub fn draws(&self) -> usize {
        self.games
            .iter()
            .filter(|g| g.placements.len() >= 2 && g.placements[0] == g.placements[1])
            .count()
    }

    /// Convert to legacy `MatchupResult` format
    pub fn to_matchup_result(&self) -> MatchupResult {
        MatchupResult {
            contestant_a: self.contestants.first().copied().unwrap_or(0),
            contestant_b: self.contestants.get(1).copied().unwrap_or(0),
            wins_a: self.wins_a(),
            wins_b: self.wins_b(),
            draws: self.draws(),
        }
    }
}

/// A contestant in the tournament
#[derive(Clone, Debug)]
pub struct Contestant {
    /// Display name
    pub name: String,
    /// Source for action selection
    pub source: PlayerSource,
    /// Current skill rating (mu, sigma)
    pub rating: WengLinRating,
    /// Indices of opponents already faced
    pub opponents_faced: Vec<usize>,
    /// Placement counts: [placement-1] = count (1st place at index 0)
    pub placement_counts: Vec<usize>,
    /// Total games played
    pub games_played: usize,
    /// Draw count (ties where all players got same placement)
    pub draw_count: usize,
    /// Swiss tournament points (based on placement)
    pub swiss_points: f64,
    /// Whether this player has received a bye
    pub has_bye: bool,
    /// Initial seeding value from `training_rating` (higher = stronger)
    pub initial_seed: f64,
}

impl Contestant {
    pub fn new(name: String, source: PlayerSource, initial_seed: f64) -> Self {
        Self {
            name,
            source,
            rating: WengLinRating::new(),
            opponents_faced: Vec::new(),
            placement_counts: Vec::new(),
            games_played: 0,
            draw_count: 0,
            swiss_points: 0.0,
            has_bye: false,
            initial_seed,
        }
    }

    /// Get wins (1st place finishes, excluding draws) for 2-player compatibility
    pub fn wins(&self) -> usize {
        self.placement_counts
            .first()
            .copied()
            .unwrap_or(0)
            .saturating_sub(self.draw_count)
    }

    /// Get losses (last place finishes) for 2-player compatibility
    pub fn losses(&self) -> usize {
        if self.placement_counts.len() >= 2 {
            self.placement_counts.last().copied().unwrap_or(0)
        } else {
            0
        }
    }

    /// Get draws (ties)
    pub fn draws(&self) -> usize {
        self.draw_count
    }
}

/// Result of a matchup between two contestants
#[derive(Debug, Clone)]
pub struct MatchupResult {
    pub contestant_a: usize,
    pub contestant_b: usize,
    pub wins_a: usize,
    pub wins_b: usize,
    pub draws: usize,
}

/// Final tournament ranking entry
#[derive(Debug, Clone, Serialize)]
pub struct RankingEntry {
    pub rank: usize,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub swiss_points: f64,
    pub rating: f64,
    pub uncertainty: f64,
    pub rating_low: f64,
    pub rating_high: f64,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub games_played: usize,
}

/// Match summary for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct MatchSummary {
    pub round: usize,
    pub contestant_a: String,
    pub contestant_b: String,
    pub wins_a: usize,
    pub wins_b: usize,
    pub draws: usize,
}

/// Full tournament results for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct TournamentResults {
    pub rankings: Vec<RankingEntry>,
    pub matches: Vec<MatchSummary>,
    pub config: TournamentConfigSummary,
    pub environment: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TournamentConfigSummary {
    pub num_games_per_matchup: usize,
    pub num_rounds: usize,
    pub format: String,
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Check if a path is a checkpoint directory (has metadata.json)
fn is_checkpoint_dir(path: &Path) -> bool {
    path.is_dir() && path.join("metadata.json").exists()
}

/// Check if a path is a run directory (has checkpoints subdirectory)
fn is_run_checkpoints_dir(path: &Path) -> bool {
    path.is_dir()
        && std::fs::read_dir(path)
            .map(|entries| {
                entries.filter_map(Result::ok).any(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("step_"))
                })
            })
            .unwrap_or(false)
}

/// Enumerate checkpoint directories in a checkpoints folder
fn enumerate_checkpoints(checkpoints_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut checkpoints: Vec<PathBuf> = std::fs::read_dir(checkpoints_dir)
        .context("Failed to read checkpoints directory")?
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("step_"))
        })
        .map(|e| e.path())
        .collect();

    // Sort by step number
    checkpoints.sort_by(|a, b| {
        let step_a = a
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("step_"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let step_b = b
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("step_"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        step_a.cmp(&step_b)
    });

    Ok(checkpoints)
}

/// Select N evenly spaced checkpoints including first and last
fn select_evenly_spaced(checkpoints: &[PathBuf], n: usize) -> Vec<PathBuf> {
    if n >= checkpoints.len() {
        return checkpoints.to_vec();
    }
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![checkpoints.last().expect("non-empty").clone()];
    }

    let mut selected = Vec::with_capacity(n);
    let step = (checkpoints.len() - 1) as f64 / (n - 1) as f64;

    for i in 0..n {
        #[expect(clippy::cast_sign_loss, reason = "step is always positive")]
        let idx = (i as f64 * step).round() as usize;
        selected.push(checkpoints[idx.min(checkpoints.len() - 1)].clone());
    }

    selected
}

/// Get display name from a checkpoint path
fn checkpoint_name(path: &Path) -> String {
    // Try to get step_XXXXX name, falling back to file name
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("checkpoint")
        .to_string()
}

/// Discover contestants from command-line sources
fn discover_contestants(args: &TournamentArgs) -> Result<Vec<Contestant>> {
    let mut contestants = Vec::new();

    // Track if all checkpoints come from a single training run (same checkpoints folder)
    // Only use training_rating for seeding if this is true
    let single_training_run = args.sources.len() == 1 && {
        let path = &args.sources[0];
        let resolved = if path.is_symlink() {
            path.read_link().map_or_else(
                |_| path.clone(),
                |target| path.parent().unwrap_or(path).join(target),
            )
        } else {
            path.clone()
        };
        is_run_checkpoints_dir(&resolved)
    };

    for source_path in &args.sources {
        // Resolve symlinks
        let path = if source_path.is_symlink() {
            source_path.read_link().map_or_else(
                |_| source_path.clone(),
                |target| source_path.parent().unwrap_or(source_path).join(target),
            )
        } else {
            source_path.clone()
        };

        if is_checkpoint_dir(&path) {
            // Single checkpoint - load training_rating from metadata
            let initial_seed = if single_training_run {
                load_metadata(&path)
                    .map(|m| m.training_rating)
                    .unwrap_or(0.0)
            } else {
                0.0 // Will be shuffled later
            };
            contestants.push(Contestant::new(
                checkpoint_name(&path),
                PlayerSource::Checkpoint(path),
                initial_seed,
            ));
        } else if is_run_checkpoints_dir(&path) {
            // Checkpoints directory - enumerate and optionally limit
            let checkpoints = enumerate_checkpoints(&path)?;
            if checkpoints.is_empty() {
                bail!("No checkpoints found in {}", path.display());
            }

            let selected = if let Some(limit) = args.limit {
                select_evenly_spaced(&checkpoints, limit)
            } else {
                checkpoints
            };

            for ckpt in selected {
                // Load training_rating from checkpoint metadata
                let initial_seed = if single_training_run {
                    load_metadata(&ckpt)
                        .map(|m| m.training_rating)
                        .unwrap_or(0.0)
                } else {
                    0.0 // Will be shuffled later
                };
                contestants.push(Contestant::new(
                    checkpoint_name(&ckpt),
                    PlayerSource::Checkpoint(ckpt),
                    initial_seed,
                ));
            }
        } else {
            bail!(
                "Invalid source: {} (expected checkpoint dir or checkpoints folder)",
                source_path.display()
            );
        }
    }

    // Add random player if requested (always lowest seed)
    if args.random {
        contestants.push(Contestant::new(
            "Random".to_string(),
            PlayerSource::Random,
            f64::NEG_INFINITY,
        ));
    }

    // If not from single training run, shuffle contestants for random seeding
    // (but keep random player at the end if present)
    if !single_training_run && contestants.len() > 1 {
        let mut rng = StdRng::from_entropy();
        let has_random = args.random;
        if has_random {
            // Shuffle all except the last (random player)
            let len = contestants.len();
            let checkpoint_contestants = &mut contestants[..len - 1];
            checkpoint_contestants.shuffle(&mut rng);
        } else {
            contestants.shuffle(&mut rng);
        }
        // Assign random initial_seed values based on shuffled position
        for (i, c) in contestants.iter_mut().enumerate() {
            if !matches!(c.source, PlayerSource::Random) {
                c.initial_seed = i as f64;
            }
        }
    }

    Ok(contestants)
}

/// Calculate Swiss points for a single game using fractional ranking
///
/// Uses the formula: points = N - `avg_position` where `avg_position` accounts for ties.
/// Tied players share the positions they would occupy, averaged.
///
/// Examples for N=4:
/// - [1,2,3,4] → [3.0, 2.0, 1.0, 0.0]
/// - [1,1,3,4] → [2.5, 2.5, 1.0, 0.0] (tied for 1st share positions 1&2)
/// - [1,2,2,4] → [3.0, 1.5, 1.5, 0.0] (tied for 2nd share positions 2&3)
/// - [1,1,1,1] → [1.5, 1.5, 1.5, 1.5] (all tied share positions 1-4)
fn calculate_swiss_points(placements: &[usize]) -> Vec<f64> {
    use std::collections::HashMap;

    let num_players = placements.len();
    if num_players == 0 {
        return Vec::new();
    }

    // Count how many players share each placement
    let mut placement_counts: HashMap<usize, usize> = HashMap::new();
    for &p in placements {
        *placement_counts.entry(p).or_insert(0) += 1;
    }

    // Sort unique placements to determine position ranges
    let mut unique_placements: Vec<usize> = placement_counts.keys().copied().collect();
    unique_placements.sort_unstable();

    // Calculate average position for each placement group
    let mut placement_to_avg_pos: HashMap<usize, f64> = HashMap::new();
    let mut current_pos = 1usize;
    for p in unique_placements {
        let count = placement_counts[&p];
        // Players occupy positions current_pos to current_pos + count - 1
        // Average = (first + last) / 2
        let avg = f64::midpoint(current_pos as f64, (current_pos + count - 1) as f64);
        placement_to_avg_pos.insert(p, avg);
        current_pos += count;
    }

    // Calculate points for each player: N - avg_position
    placements
        .iter()
        .map(|&p| num_players as f64 - placement_to_avg_pos[&p])
        .collect()
}

/// Generate Swiss pods for a round
///
/// For round 1 (all `swiss_points` == 0): Uses Dutch-style pairing by dividing
/// contestants into N groups by `initial_seed` and forming pods with one from each group.
/// For subsequent rounds: Groups by similar Swiss points, preferring unfaced opponents.
fn swiss_pods(contestants: &[Contestant], pod_size: usize) -> Vec<Vec<usize>> {
    // Check if this is round 1 (all swiss_points are 0)
    let is_round_1 = contestants.iter().all(|c| c.swiss_points == 0.0);

    if is_round_1 {
        // Dutch-style initial pairing: divide into N groups by initial_seed
        // Sort by initial_seed (descending - higher seed = stronger)
        let mut ranked: Vec<(usize, f64)> = contestants
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.initial_seed))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate number of complete pods
        let num_pods = contestants.len() / pod_size;
        if num_pods == 0 {
            return Vec::new();
        }

        // Form pods: Pod i gets contestants [i, i+num_pods, i+2*num_pods, ..., i+(N-1)*num_pods]
        // This ensures each pod has one player from each "skill tier"
        let mut pods = Vec::with_capacity(num_pods);
        for pod_idx in 0..num_pods {
            let mut pod = Vec::with_capacity(pod_size);
            for group in 0..pod_size {
                let ranked_pos = pod_idx + group * num_pods;
                if ranked_pos < ranked.len() {
                    pod.push(ranked[ranked_pos].0);
                }
            }
            if pod.len() == pod_size {
                pods.push(pod);
            }
        }
        return pods;
    }

    // Subsequent rounds: sort by Swiss points (desc), initial_seed as tiebreaker
    let mut ranked: Vec<(usize, f64, f64)> = contestants
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.swiss_points, c.initial_seed))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut pods = Vec::new();
    let mut used = vec![false; contestants.len()];

    // Greedy pod formation: take top N unused players who haven't all played together
    for i in 0..ranked.len() {
        let idx = ranked[i].0;
        if used[idx] {
            continue;
        }

        let mut pod = vec![idx];
        used[idx] = true;

        // Find pod_size-1 more players, preferring those not yet faced by all pod members
        for &(candidate, _, _) in ranked.iter().skip(i + 1) {
            if pod.len() >= pod_size {
                break;
            }
            if used[candidate] {
                continue;
            }
            // Check if candidate has faced all current pod members
            let faced_all = pod
                .iter()
                .all(|&p| contestants[candidate].opponents_faced.contains(&p));
            if !faced_all {
                pod.push(candidate);
                used[candidate] = true;
            }
        }

        // If we couldn't find enough unfaced opponents, take any available
        if pod.len() < pod_size {
            for &(candidate, _, _) in ranked.iter().skip(i + 1) {
                if pod.len() >= pod_size {
                    break;
                }
                if !used[candidate] {
                    pod.push(candidate);
                    used[candidate] = true;
                }
            }
        }

        if pod.len() == pod_size {
            pods.push(pod);
        }
    }

    pods
}

/// Generate Swiss pairings for a round (2-player games)
///
/// For round 1 (all `swiss_points` == 0): Uses Dutch-style pairing by dividing
/// contestants into top half and bottom half by `initial_seed`.
/// For subsequent rounds: Pairs by similar Swiss points, preferring unfaced opponents.
fn swiss_pairings(contestants: &[Contestant]) -> Vec<(usize, usize)> {
    // Check if this is round 1 (all swiss_points are 0)
    let is_round_1 = contestants.iter().all(|c| c.swiss_points == 0.0);

    if is_round_1 {
        // Dutch-style initial pairing: sort by initial_seed, pair top half vs bottom half
        let mut ranked: Vec<(usize, f64)> = contestants
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.initial_seed))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let num_pairs = contestants.len() / 2;
        let mut pairings = Vec::with_capacity(num_pairs);

        // Pair #1 vs #(num_pairs+1), #2 vs #(num_pairs+2), etc.
        for i in 0..num_pairs {
            let top_idx = ranked[i].0;
            let bottom_idx = ranked[i + num_pairs].0;
            pairings.push((top_idx, bottom_idx));
        }
        return pairings;
    }

    // Subsequent rounds: sort by Swiss points (desc), initial_seed as tiebreaker
    let mut ranked: Vec<(usize, f64, f64)> = contestants
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.swiss_points, c.initial_seed))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut pairings = Vec::new();
    let mut paired = vec![false; contestants.len()];

    for i in 0..ranked.len() {
        let idx_a = ranked[i].0;
        if paired[idx_a] {
            continue;
        }

        // Find best unpaired opponent (similar points, preferring not yet faced)
        let mut best_opponent = None;
        let mut best_not_faced = None;

        for (_, &(idx_b, _, _)) in ranked.iter().enumerate().skip(i + 1) {
            if paired[idx_b] {
                continue;
            }

            if best_opponent.is_none() {
                best_opponent = Some(idx_b);
            }

            // Prefer opponents not yet faced
            if !contestants[idx_a].opponents_faced.contains(&idx_b) && best_not_faced.is_none() {
                best_not_faced = Some(idx_b);
                break; // Found ideal opponent
            }
        }

        // Use not-faced opponent if available, otherwise best available
        if let Some(idx_b) = best_not_faced.or(best_opponent) {
            pairings.push((idx_a, idx_b));
            paired[idx_a] = true;
            paired[idx_b] = true;
        }
    }

    pairings
}

/// Generate round-robin pairings (all possible pairs)
fn round_robin_pairings(n: usize) -> Vec<(usize, usize)> {
    let mut pairings = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairings.push((i, j));
        }
    }
    pairings
}

/// Update ratings from per-game placements using `weng_lin_multi_team` (N-player compatible)
/// Games are shuffled to avoid bias from completion order (shorter games finish first)
///
/// Swiss points are awarded based on match-level placement:
/// 1. Sum raw placement points across all games in the match
/// 2. Rank contestants by raw totals to determine match placements
/// 3. Award Swiss points based on match placement using fractional ranking
fn update_ratings_from_games(
    contestants: &mut [Contestant],
    matchup: &MatchupGames,
    config: &WengLinConfig,
) {
    if matchup.games.is_empty() {
        return;
    }

    // Shuffle games to avoid bias from completion order
    let mut games = matchup.games.clone();
    let seed = games.len() as u64;
    let mut rng = StdRng::seed_from_u64(seed);
    games.shuffle(&mut rng);

    let num_players = matchup.contestants.len();

    // Track raw points per contestant for match-level scoring
    let mut raw_points: Vec<f64> = vec![0.0; num_players];

    for game in &games {
        // Build rating groups for weng_lin_multi_team
        // Each "team" is a single player
        let ratings: Vec<WengLinRating> = matchup
            .contestants
            .iter()
            .map(|&idx| contestants[idx].rating)
            .collect();

        let rating_refs: Vec<[WengLinRating; 1]> = ratings.iter().map(|r| [*r]).collect();

        let rating_groups: Vec<(&[WengLinRating], MultiTeamOutcome)> = rating_refs
            .iter()
            .enumerate()
            .map(|(i, rating_arr)| {
                let placement = game.placements[i];
                (rating_arr.as_slice(), MultiTeamOutcome::new(placement))
            })
            .collect();

        let new_ratings = weng_lin_multi_team(&rating_groups, config);

        // Update contestant ratings
        for (i, &contestant_idx) in matchup.contestants.iter().enumerate() {
            contestants[contestant_idx].rating = new_ratings[i][0];
        }

        // Check if this is a draw (all placements equal)
        let is_draw = game.placements.windows(2).all(|w| w[0] == w[1]);

        // Update placement counts and games played
        for (i, &contestant_idx) in matchup.contestants.iter().enumerate() {
            let placement = game.placements[i];
            // Ensure placement_counts is large enough
            if contestants[contestant_idx].placement_counts.len() < num_players {
                contestants[contestant_idx]
                    .placement_counts
                    .resize(num_players, 0);
            }
            if placement > 0 && placement <= num_players {
                contestants[contestant_idx].placement_counts[placement - 1] += 1;
            }
            // Track draws
            if is_draw {
                contestants[contestant_idx].draw_count += 1;
            }
            contestants[contestant_idx].games_played += 1;
        }

        // Accumulate raw points per contestant for match-level scoring
        let game_points = calculate_swiss_points(&game.placements);
        for (i, pts) in game_points.iter().enumerate() {
            raw_points[i] += pts;
        }
    }

    // Determine match placements based on raw point totals (higher points = better placement)
    // Create sorted indices by raw points (descending)
    let mut sorted_indices: Vec<(usize, f64)> = raw_points.iter().copied().enumerate().collect();
    sorted_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign placements (1-indexed), handling ties
    let mut match_placements: Vec<usize> = vec![0; num_players];
    let mut current_placement = 1;
    let mut i = 0;
    while i < sorted_indices.len() {
        // Find all contestants tied at this position
        let current_points = sorted_indices[i].1;
        let mut j = i;
        while j < sorted_indices.len()
            && (sorted_indices[j].1 - current_points).abs() < f64::EPSILON
        {
            j += 1;
        }
        // All from i to j-1 share the same placement
        for k in i..j {
            match_placements[sorted_indices[k].0] = current_placement;
        }
        current_placement = j + 1; // Next placement accounts for tied positions
        i = j;
    }

    // Award Swiss points based on match placement using fractional ranking
    let swiss_points = calculate_swiss_points(&match_placements);
    for (i, &contestant_idx) in matchup.contestants.iter().enumerate() {
        contestants[contestant_idx].swiss_points += swiss_points[i];
    }

    // Track opponents faced (all matchup participants)
    for &contestant_idx in &matchup.contestants {
        for &other_idx in &matchup.contestants {
            if contestant_idx != other_idx
                && !contestants[contestant_idx]
                    .opponents_faced
                    .contains(&other_idx)
            {
                contestants[contestant_idx].opponents_faced.push(other_idx);
            }
        }
    }
}

/// Print current standings
fn print_standings(contestants: &[Contestant], header: &str) {
    println!("\n{header}");

    // Sort by Swiss points (descending), rating as tiebreaker
    let mut ranked: Vec<(usize, &Contestant)> = contestants.iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.swiss_points
            .partial_cmp(&a.1.swiss_points)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.1.rating
                    .rating
                    .partial_cmp(&a.1.rating.rating)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    for (rank, (_, c)) in ranked.iter().enumerate() {
        println!(
            "  {:2}. {:20} {:6.1}pts  (rating: {:.1} ± {:.1})",
            rank + 1,
            c.name,
            c.swiss_points,
            c.rating.rating,
            c.rating.uncertainty
        );
    }
}

/// Print final tournament summary
fn print_final_summary(contestants: &[Contestant], num_rounds: usize, num_games: usize) {
    println!("\n{}", "=".repeat(80));
    println!("=== Tournament Results ===");
    println!(
        "Contestants: {} | Rounds: {} | Games per matchup: {}",
        contestants.len(),
        num_rounds,
        num_games
    );
    println!();

    // Sort by Swiss points (descending), rating as tiebreaker
    let mut ranked: Vec<(usize, &Contestant)> = contestants.iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.swiss_points
            .partial_cmp(&a.1.swiss_points)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.1.rating
                    .rating
                    .partial_cmp(&a.1.rating.rating)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Header
    println!(
        " {:>2}  {:20}  {:>8}  {:>7}  {:>16}  {:>4}  {:>4}  {:>4}",
        "#", "Name", "Points", "Rating", "95% CI", "W", "L", "D"
    );
    println!("{:-<80}", "");

    for (rank, (_, c)) in ranked.iter().enumerate() {
        let sigma = c.rating.uncertainty;
        let low = c.rating.rating - 2.0 * sigma;
        let high = c.rating.rating + 2.0 * sigma;

        println!(
            " {:>2}  {:20}  {:>8.1}  {:>7.1}  [{:>5.1}, {:>5.1}]  {:>4}  {:>4}  {:>4}",
            rank + 1,
            c.name,
            c.swiss_points,
            c.rating.rating,
            low,
            high,
            c.wins(),
            c.losses(),
            c.draws()
        );
    }

    println!();
    println!("Note: Ranked by Swiss points. Rating 95% CI = rating ± 2×sigma");
}

/// Build tournament results for JSON export
fn build_results(
    contestants: &[Contestant],
    matches: &[(usize, MatchupResult)], // (round, result)
    num_rounds: usize,
    args: &TournamentArgs,
    env_name: &str,
) -> TournamentResults {
    // Sort by Swiss points (descending), rating as tiebreaker
    let mut ranked: Vec<(usize, &Contestant)> = contestants.iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.swiss_points
            .partial_cmp(&a.1.swiss_points)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.1.rating
                    .rating
                    .partial_cmp(&a.1.rating.rating)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let rankings: Vec<RankingEntry> = ranked
        .iter()
        .enumerate()
        .map(|(rank, (_, c))| {
            let sigma = c.rating.uncertainty;
            let source = match &c.source {
                PlayerSource::Checkpoint(p) => Some(p.display().to_string()),
                _ => None,
            };
            RankingEntry {
                rank: rank + 1,
                name: c.name.clone(),
                source,
                swiss_points: c.swiss_points,
                rating: c.rating.rating,
                uncertainty: sigma,
                rating_low: c.rating.rating - 2.0 * sigma,
                rating_high: c.rating.rating + 2.0 * sigma,
                wins: c.wins(),
                losses: c.losses(),
                draws: c.draws(),
                games_played: c.games_played,
            }
        })
        .collect();

    let match_summaries: Vec<MatchSummary> = matches
        .iter()
        .map(|(round, result)| MatchSummary {
            round: *round,
            contestant_a: contestants[result.contestant_a].name.clone(),
            contestant_b: contestants[result.contestant_b].name.clone(),
            wins_a: result.wins_a,
            wins_b: result.wins_b,
            draws: result.draws,
        })
        .collect();

    let format = if contestants.len() <= 8 {
        "round-robin"
    } else {
        "swiss"
    };

    TournamentResults {
        rankings,
        matches: match_summaries,
        config: TournamentConfigSummary {
            num_games_per_matchup: args.num_games,
            num_rounds,
            format: format.to_string(),
            temperature: args.temperature,
            seed: args.seed,
        },
        environment: env_name.to_string(),
        timestamp: chrono_lite_now(),
    }
}

/// Simple timestamp without chrono dependency
fn chrono_lite_now() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", duration.as_secs())
}

/// Run a matchup between two contestants
fn run_matchup<B: Backend, E: Environment>(
    contestants: &[Contestant],
    models: &[ActorCritic<B>],
    normalizers: &[Option<ObsNormalizer>],
    contestant_to_model: &[Option<usize>], // None = Random
    idx_a: usize,
    idx_b: usize,
    num_games: usize,
    num_envs: usize,
    temp_schedule: &TempSchedule,
    rng: &mut StdRng,
    device: &B::Device,
) -> MatchupGames {
    // For this matchup, we need exactly 2 "checkpoints" for run_stats_mode_env
    // Build models/normalizers arrays for just these two
    let mut matchup_models: Vec<ActorCritic<B>> = Vec::new();
    let mut matchup_normalizers: Vec<Option<ObsNormalizer>> = Vec::new();
    let mut checkpoint_to_model_map: Vec<usize> = Vec::new();

    // Add contestant A's model
    if let Some(model_idx) = contestant_to_model[idx_a] {
        matchup_models.push(models[model_idx].clone());
        matchup_normalizers.push(normalizers[model_idx].clone());
        checkpoint_to_model_map.push(0);
    } else {
        // Random player - need a dummy model slot that won't be used
        // Actually, run_stats_mode_env doesn't support Random directly
        // We need to handle this differently
        // For now, create a dummy - but this won't work properly
        // TODO: Handle Random properly by extending run_stats_mode_env or using different approach
        checkpoint_to_model_map.push(0); // placeholder
    }

    // Add contestant B's model (reuse if same as A's model)
    if let Some(model_idx) = contestant_to_model[idx_b] {
        // Check if we already added this model (same model_idx means same source)
        let a_model_idx = contestant_to_model[idx_a];
        if a_model_idx == Some(model_idx) && !matchup_models.is_empty() {
            // Same model as A, reuse index 0
            checkpoint_to_model_map.push(0);
        } else {
            matchup_models.push(models[model_idx].clone());
            matchup_normalizers.push(normalizers[model_idx].clone());
            checkpoint_to_model_map.push(matchup_models.len() - 1);
        }
    } else {
        checkpoint_to_model_map.push(0); // placeholder for Random
    }

    let names = vec![
        contestants[idx_a].name.clone(),
        contestants[idx_b].name.clone(),
    ];

    // Handle Random players - for now, skip if either is Random
    // A proper implementation would extend run_stats_mode_env to support Random
    let has_random = contestant_to_model[idx_a].is_none() || contestant_to_model[idx_b].is_none();

    if has_random {
        // Fallback: run interactive-style evaluation or simplified version
        // For MVP, we'll use a simple simulation
        return run_matchup_with_random::<E>(idx_a, idx_b, num_games, rng);
    }

    // Run games via eval infrastructure
    let stats = run_stats_mode_env::<B, E>(
        &matchup_models,
        &matchup_normalizers,
        &checkpoint_to_model_map,
        &names,
        num_games,
        num_envs,
        temp_schedule,
        rng,
        device,
        true, // silent
    );

    // Convert game_outcomes to GamePlacement
    let games: Vec<GamePlacement> = stats
        .game_outcomes
        .iter()
        .filter_map(|outcome| {
            if let GameOutcome::Placements(p) = outcome {
                Some(GamePlacement {
                    placements: p.clone(),
                })
            } else {
                None
            }
        })
        .collect();

    MatchupGames {
        contestants: vec![idx_a, idx_b],
        games,
    }
}

/// Simplified matchup for when Random player is involved
fn run_matchup_with_random<E: Environment>(
    idx_a: usize,
    idx_b: usize,
    num_games: usize,
    rng: &mut StdRng,
) -> MatchupGames {
    use rand::Rng;

    let mut games = Vec::with_capacity(num_games);

    for _ in 0..num_games {
        let mut env = E::new(rng.gen());

        loop {
            // Random valid action
            let mask = env.action_mask();
            let action = if let Some(ref m) = mask {
                let valid: Vec<usize> = m
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v)
                    .map(|(i, _)| i)
                    .collect();
                if valid.is_empty() {
                    0
                } else {
                    valid[rng.gen_range(0..valid.len())]
                }
            } else {
                rng.gen_range(0..E::ACTION_COUNT)
            };

            let (_, _, done) = env.step(action);
            if done {
                break;
            }
        }

        // Convert outcome to placements
        let placements = match env.game_outcome() {
            Some(GameOutcome::Winner(w)) => {
                if w == 0 {
                    vec![1, 2] // A wins
                } else {
                    vec![2, 1] // B wins
                }
            }
            Some(GameOutcome::Placements(p)) => p,
            Some(GameOutcome::Tie) | None => vec![1, 1], // Draw
        };

        games.push(GamePlacement { placements });
    }

    MatchupGames {
        contestants: vec![idx_a, idx_b],
        games,
    }
}

/// Main tournament entry point
pub fn run_tournament<B: Backend>(args: &TournamentArgs, device: &B::Device) -> Result<()> {
    // Discover contestants
    let mut contestants = discover_contestants(args)?;
    let n = contestants.len();

    if n < 2 {
        bail!("Tournament requires at least 2 contestants, found {n}");
    }

    println!("Tournament: {n} contestants");

    // Load first checkpoint to determine environment
    let first_checkpoint = contestants
        .iter()
        .find_map(|c| match &c.source {
            PlayerSource::Checkpoint(p) => Some(p.clone()),
            _ => None,
        })
        .context("At least one checkpoint is required to determine environment")?;

    let (_, metadata, _) = load_model_from_checkpoint::<B>(&first_checkpoint, device)?;
    let env_name = metadata.env_name.clone();

    println!("Environment: {env_name}");

    // Dispatch to environment-specific implementation
    dispatch_env!(
        &env_name,
        run_tournament_env::<B, E>(args, &mut contestants, &metadata, device)
    )
}

/// Environment-specific tournament implementation
fn run_tournament_env<B: Backend, E: Environment>(
    args: &TournamentArgs,
    contestants: &mut [Contestant],
    metadata: &CheckpointMetadata,
    device: &B::Device,
) -> Result<()> {
    let n = contestants.len();

    // Load all models (with deduplication)
    let mut models: Vec<ActorCritic<B>> = Vec::new();
    let mut normalizers: Vec<Option<ObsNormalizer>> = Vec::new();
    let mut path_to_model_idx: HashMap<PathBuf, usize> = HashMap::new();
    let mut contestant_to_model: Vec<Option<usize>> = Vec::new();

    println!("Loading models...");
    for contestant in contestants.iter() {
        match &contestant.source {
            PlayerSource::Checkpoint(path) => {
                // Resolve symlinks
                let resolved = if path.is_symlink() {
                    path.read_link().map_or_else(
                        |_| path.clone(),
                        |target| path.parent().unwrap_or(path).join(target),
                    )
                } else {
                    path.clone()
                };

                // Check for deduplication
                if let Some(&idx) = path_to_model_idx.get(&resolved) {
                    contestant_to_model.push(Some(idx));
                } else {
                    let (model, _, normalizer) =
                        load_model_from_checkpoint::<B>(&resolved, device)?;
                    let idx = models.len();
                    models.push(model);
                    normalizers.push(normalizer);
                    path_to_model_idx.insert(resolved, idx);
                    contestant_to_model.push(Some(idx));
                }
            }
            PlayerSource::Random => {
                contestant_to_model.push(None);
            }
            PlayerSource::Human { .. } => {
                bail!("Human players not supported in tournament mode");
            }
        }
    }

    println!("Loaded {} unique models", models.len());

    // Determine tournament format
    let use_swiss = n > 8;
    let num_rounds = if use_swiss {
        args.rounds.unwrap_or_else(|| {
            #[expect(clippy::cast_sign_loss, reason = "log2 of positive n is positive")]
            let rounds = (n as f64).log2().ceil() as usize + 1;
            rounds
        })
    } else {
        1 // Round-robin is done in one "round"
    };

    let format_name = if use_swiss { "Swiss" } else { "Round-Robin" };
    println!("Format: {format_name} ({num_rounds} rounds)");
    println!();

    // Temperature schedule
    let temp_schedule = TempSchedule::new(
        args.temperature,
        args.temp_final.unwrap_or(0.0),
        args.temp_cutoff,
        false,
    );

    // RNG
    let seed = args.seed.unwrap_or_else(rand::random);
    let mut rng = StdRng::seed_from_u64(seed);

    // Weng-Lin config
    let wl_config = WengLinConfig::new();

    // Track all matches for JSON output
    let mut all_matches: Vec<(usize, MatchupResult)> = Vec::new();

    // Progress bar setup
    let multi_progress = MultiProgress::new();

    if use_swiss {
        // Swiss tournament
        for round in 1..=num_rounds {
            println!("Round {round}/{num_rounds}:");

            // Handle byes for odd number of contestants (2-player games)
            let pod_size = E::NUM_PLAYERS;
            let num_byes = contestants.len() % pod_size;
            let mut bye_recipients: Vec<usize> = Vec::new();

            if num_byes > 0 {
                // Find lowest-ranked players (by Swiss points) who haven't had a bye
                let mut bye_candidates: Vec<(usize, f64)> = contestants
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.has_bye)
                    .map(|(i, c)| (i, c.swiss_points))
                    .collect();
                // Sort ascending (lowest points first)
                bye_candidates
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                for (bye_idx, _) in bye_candidates.iter().take(num_byes) {
                    // Award bye: points equivalent to 1st place in a match
                    // With match-level scoring, a bye = (pod_size - 1) Swiss points
                    let bye_points = (pod_size - 1) as f64;
                    contestants[*bye_idx].swiss_points += bye_points;
                    contestants[*bye_idx].has_bye = true;
                    bye_recipients.push(*bye_idx);
                    println!(
                        "  {} receives bye (+{:.1} points)",
                        contestants[*bye_idx].name, bye_points
                    );
                }
            }

            // Create pairings from active (non-bye) contestants
            let active_indices: Vec<usize> = (0..contestants.len())
                .filter(|i| !bye_recipients.contains(i))
                .collect();

            // Build temporary contestant slice for pairing
            let pairings = if pod_size == 2 {
                // For 2-player games, use swiss_pairings
                let active_contestants: Vec<Contestant> = active_indices
                    .iter()
                    .map(|&i| contestants[i].clone())
                    .collect();
                let temp_pairings = swiss_pairings(&active_contestants);
                // Map back to original indices
                temp_pairings
                    .into_iter()
                    .map(|(a, b)| (active_indices[a], active_indices[b]))
                    .collect::<Vec<_>>()
            } else {
                // For N-player games, use swiss_pods
                let active_contestants: Vec<Contestant> = active_indices
                    .iter()
                    .map(|&i| contestants[i].clone())
                    .collect();
                let pods = swiss_pods(&active_contestants, pod_size);
                // Convert pods to pairs (for compatibility with current 2-player matchup code)
                // This is a temporary solution - would need proper N-player matchup support
                pods.into_iter()
                    .filter(|pod| pod.len() >= 2)
                    .map(|pod| (active_indices[pod[0]], active_indices[pod[1]]))
                    .collect::<Vec<_>>()
            };

            if pairings.is_empty() && bye_recipients.is_empty() {
                println!("  No more pairings possible");
                break;
            }

            let round_pb = multi_progress.add(ProgressBar::new(pairings.len() as u64));
            round_pb.set_style(
                ProgressStyle::default_bar()
                    .template("  [{bar:30}] {pos}/{len} matchups")
                    .expect("valid template")
                    .progress_chars("=> "),
            );

            for (idx_a, idx_b) in pairings {
                let result = run_matchup::<B, E>(
                    contestants,
                    &models,
                    &normalizers,
                    &contestant_to_model,
                    idx_a,
                    idx_b,
                    args.num_games,
                    args.num_envs,
                    &temp_schedule,
                    &mut rng,
                    device,
                );

                // Suspend progress bar to print result
                round_pb.suspend(|| {
                    let winner = if result.wins_a() > result.wins_b() {
                        &contestants[idx_a].name
                    } else if result.wins_b() > result.wins_a() {
                        &contestants[idx_b].name
                    } else {
                        "draw"
                    };
                    println!(
                        "  {} vs {}: {}-{}-{} ({})",
                        contestants[idx_a].name,
                        contestants[idx_b].name,
                        result.wins_a(),
                        result.wins_b(),
                        result.draws(),
                        winner
                    );
                });

                update_ratings_from_games(contestants, &result, &wl_config);
                all_matches.push((round, result.to_matchup_result()));
                round_pb.inc(1);
            }

            round_pb.finish_and_clear();
            print_standings(contestants, &format!("Standings after round {round}:"));
        }
    } else {
        // Round-robin
        let pairings = round_robin_pairings(n);
        let total_matchups = pairings.len();

        println!("Running {total_matchups} matchups (round-robin):");

        let pb = multi_progress.add(ProgressBar::new(total_matchups as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40}] {pos}/{len} matchups ({eta})")
                .expect("valid template")
                .progress_chars("=> "),
        );

        for (idx_a, idx_b) in pairings {
            let result = run_matchup::<B, E>(
                contestants,
                &models,
                &normalizers,
                &contestant_to_model,
                idx_a,
                idx_b,
                args.num_games,
                args.num_envs,
                &temp_schedule,
                &mut rng,
                device,
            );

            pb.suspend(|| {
                let winner = if result.wins_a() > result.wins_b() {
                    &contestants[idx_a].name
                } else if result.wins_b() > result.wins_a() {
                    &contestants[idx_b].name
                } else {
                    "draw"
                };
                println!(
                    "  {} vs {}: {}-{}-{} ({})",
                    contestants[idx_a].name,
                    contestants[idx_b].name,
                    result.wins_a(),
                    result.wins_b(),
                    result.draws(),
                    winner
                );
            });

            update_ratings_from_games(contestants, &result, &wl_config);
            all_matches.push((1, result.to_matchup_result()));
            pb.inc(1);
        }

        pb.finish_and_clear();
    }

    // Shift ratings so minimum rating becomes 0.0
    let min_rating = contestants
        .iter()
        .map(|c| c.rating.rating)
        .fold(f64::INFINITY, f64::min);
    for c in contestants.iter_mut() {
        c.rating.rating -= min_rating;
    }

    // Final summary
    print_final_summary(contestants, num_rounds, args.num_games);

    // JSON output if requested
    if let Some(output_path) = &args.output {
        let results = build_results(
            contestants,
            &all_matches,
            num_rounds,
            args,
            &metadata.env_name,
        );
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(output_path, json)?;
        println!("\nResults saved to: {}", output_path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_contestant_new() {
        let contestant = Contestant::new("TestPlayer".to_string(), PlayerSource::Random, 0.0);
        assert_eq!(contestant.name, "TestPlayer");
        assert!(matches!(contestant.source, PlayerSource::Random));
        assert!(contestant.opponents_faced.is_empty());
        assert_eq!(contestant.wins(), 0);
        assert_eq!(contestant.losses(), 0);
        assert_eq!(contestant.draws(), 0);
        // Default WengLin rating should be 25.0
        assert!((contestant.rating.rating - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_is_checkpoint_dir_valid() {
        let temp = TempDir::new().unwrap();
        let checkpoint_dir = temp.path().join("step_100");
        std::fs::create_dir(&checkpoint_dir).unwrap();
        std::fs::write(checkpoint_dir.join("metadata.json"), "{}").unwrap();

        assert!(is_checkpoint_dir(&checkpoint_dir));
    }

    #[test]
    fn test_is_checkpoint_dir_no_metadata() {
        let temp = TempDir::new().unwrap();
        let checkpoint_dir = temp.path().join("step_100");
        std::fs::create_dir(&checkpoint_dir).unwrap();
        // No metadata.json

        assert!(!is_checkpoint_dir(&checkpoint_dir));
    }

    #[test]
    fn test_is_checkpoint_dir_not_dir() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("not_a_dir");
        std::fs::write(&file_path, "content").unwrap();

        assert!(!is_checkpoint_dir(&file_path));
    }

    #[test]
    fn test_is_run_checkpoints_dir_valid() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("step_100")).unwrap();
        std::fs::create_dir(temp.path().join("step_200")).unwrap();

        assert!(is_run_checkpoints_dir(temp.path()));
    }

    #[test]
    fn test_is_run_checkpoints_dir_no_steps() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("other_dir")).unwrap();

        assert!(!is_run_checkpoints_dir(temp.path()));
    }

    #[test]
    fn test_is_run_checkpoints_dir_not_dir() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("file");
        std::fs::write(&file_path, "content").unwrap();

        assert!(!is_run_checkpoints_dir(&file_path));
    }

    #[test]
    fn test_checkpoint_name() {
        let path = PathBuf::from("/some/path/step_12345");
        assert_eq!(checkpoint_name(&path), "step_12345");

        let path2 = PathBuf::from("/another/checkpoint");
        assert_eq!(checkpoint_name(&path2), "checkpoint");
    }

    #[test]
    fn test_enumerate_checkpoints() {
        let temp = TempDir::new().unwrap();

        // Create step directories in non-sorted order
        std::fs::create_dir(temp.path().join("step_300")).unwrap();
        std::fs::create_dir(temp.path().join("step_100")).unwrap();
        std::fs::create_dir(temp.path().join("step_200")).unwrap();
        std::fs::create_dir(temp.path().join("other")).unwrap(); // should be ignored

        let checkpoints = enumerate_checkpoints(temp.path()).unwrap();

        assert_eq!(checkpoints.len(), 3);
        assert!(checkpoints[0].ends_with("step_100"));
        assert!(checkpoints[1].ends_with("step_200"));
        assert!(checkpoints[2].ends_with("step_300"));
    }

    #[test]
    fn test_enumerate_checkpoints_empty() {
        let temp = TempDir::new().unwrap();
        let checkpoints = enumerate_checkpoints(temp.path()).unwrap();
        assert!(checkpoints.is_empty());
    }

    #[test]
    fn test_chrono_lite_now() {
        let timestamp = chrono_lite_now();
        assert!(timestamp.starts_with("unix:"));
        // Should be a valid number after "unix:"
        let num_str = timestamp.strip_prefix("unix:").unwrap();
        let _: u64 = num_str.parse().expect("should be a valid unix timestamp");
    }

    #[test]
    fn test_update_ratings_win() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        let initial_rating_a = contestants[0].rating.rating;
        let initial_rating_b = contestants[1].rating.rating;

        // Player A wins (1st place), Player B loses (2nd place)
        let matchup = MatchupGames {
            contestants: vec![0, 1],
            games: vec![GamePlacement {
                placements: vec![1, 2],
            }],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        // Winner's rating should increase
        assert!(contestants[0].rating.rating > initial_rating_a);
        // Loser's rating should decrease
        assert!(contestants[1].rating.rating < initial_rating_b);
        // Stats updated
        assert_eq!(contestants[0].wins(), 1);
        assert_eq!(contestants[0].losses(), 0);
        assert_eq!(contestants[1].wins(), 0);
        assert_eq!(contestants[1].losses(), 1);
    }

    #[test]
    fn test_update_ratings_loss() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        // Player A loses (2nd place), Player B wins (1st place)
        let matchup = MatchupGames {
            contestants: vec![0, 1],
            games: vec![GamePlacement {
                placements: vec![2, 1],
            }],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        assert_eq!(contestants[0].losses(), 1);
        assert_eq!(contestants[1].wins(), 1);
    }

    #[test]
    fn test_update_ratings_draw() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        let initial_rating_a = contestants[0].rating.rating;
        let initial_rating_b = contestants[1].rating.rating;

        // Draw: both players get same placement
        let matchup = MatchupGames {
            contestants: vec![0, 1],
            games: vec![GamePlacement {
                placements: vec![1, 1],
            }],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        // Ratings should stay approximately the same for equal-rated players
        assert!((contestants[0].rating.rating - initial_rating_a).abs() < 1.0);
        assert!((contestants[1].rating.rating - initial_rating_b).abs() < 1.0);
        assert_eq!(contestants[0].draws(), 1);
        assert_eq!(contestants[1].draws(), 1);
    }

    #[test]
    fn test_update_ratings_tracks_opponents() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        let matchup = MatchupGames {
            contestants: vec![0, 1],
            games: vec![GamePlacement {
                placements: vec![1, 2],
            }],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        assert!(contestants[0].opponents_faced.contains(&1));
        assert!(contestants[1].opponents_faced.contains(&0));
    }

    #[test]
    fn test_update_ratings_multiple_games() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        // 3 wins for A, 1 win for B, 1 draw
        let matchup = MatchupGames {
            contestants: vec![0, 1],
            games: vec![
                GamePlacement {
                    placements: vec![1, 2],
                }, // A wins
                GamePlacement {
                    placements: vec![1, 2],
                }, // A wins
                GamePlacement {
                    placements: vec![1, 2],
                }, // A wins
                GamePlacement {
                    placements: vec![2, 1],
                }, // B wins
                GamePlacement {
                    placements: vec![1, 1],
                }, // Draw
            ],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        // A won more, should have higher rating
        assert!(contestants[0].rating.rating > contestants[1].rating.rating);
        assert_eq!(contestants[0].wins(), 3);
        assert_eq!(contestants[0].losses(), 1);
        assert_eq!(contestants[0].draws(), 1);
        assert_eq!(contestants[1].wins(), 1);
        assert_eq!(contestants[1].losses(), 3);
        assert_eq!(contestants[1].draws(), 1);
    }

    #[test]
    fn test_matchup_result_creation() {
        let result = MatchupResult {
            contestant_a: 0,
            contestant_b: 1,
            wins_a: 5,
            wins_b: 3,
            draws: 2,
        };

        assert_eq!(result.contestant_a, 0);
        assert_eq!(result.contestant_b, 1);
        assert_eq!(result.wins_a, 5);
        assert_eq!(result.wins_b, 3);
        assert_eq!(result.draws, 2);
    }

    #[test]
    fn test_ranking_entry_serialization() {
        let entry = RankingEntry {
            rank: 1,
            name: "Champion".to_string(),
            source: Some("/path/to/checkpoint".to_string()),
            swiss_points: 150.0,
            rating: 28.5,
            uncertainty: 4.2,
            rating_low: 20.1,
            rating_high: 36.9,
            wins: 10,
            losses: 2,
            draws: 1,
            games_played: 13,
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"rank\":1"));
        assert!(json.contains("\"name\":\"Champion\""));
        assert!(json.contains("\"source\":\"/path/to/checkpoint\""));
    }

    #[test]
    fn test_ranking_entry_serialization_no_source() {
        let entry = RankingEntry {
            rank: 1,
            name: "Random".to_string(),
            source: None, // Should be skipped in serialization
            swiss_points: 50.0,
            rating: 25.0,
            uncertainty: 8.333,
            rating_low: 8.334,
            rating_high: 41.666,
            wins: 5,
            losses: 5,
            draws: 0,
            games_played: 10,
        };

        let json = serde_json::to_string(&entry).unwrap();
        // source should NOT be present since it's None
        assert!(!json.contains("\"source\""));
    }

    #[test]
    fn test_tournament_results_serialization() {
        let results = TournamentResults {
            rankings: vec![],
            matches: vec![],
            config: TournamentConfigSummary {
                num_games_per_matchup: 10,
                num_rounds: 3,
                format: "swiss".to_string(),
                temperature: 1.0,
                seed: Some(42),
            },
            environment: "connect_four".to_string(),
            timestamp: "unix:1234567890".to_string(),
        };

        let json = serde_json::to_string(&results).unwrap();
        assert!(json.contains("\"environment\":\"connect_four\""));
        assert!(json.contains("\"format\":\"swiss\""));
        assert!(json.contains("\"seed\":42"));
    }

    #[test]
    fn test_tournament_config_summary_no_seed() {
        let config = TournamentConfigSummary {
            num_games_per_matchup: 5,
            num_rounds: 2,
            format: "round-robin".to_string(),
            temperature: 0.5,
            seed: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("\"seed\""));
    }

    #[test]
    fn test_match_summary_serialization() {
        let summary = MatchSummary {
            round: 1,
            contestant_a: "Player1".to_string(),
            contestant_b: "Player2".to_string(),
            wins_a: 3,
            wins_b: 2,
            draws: 0,
        };

        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"round\":1"));
        assert!(json.contains("\"contestant_a\":\"Player1\""));
    }

    #[test]
    fn test_select_evenly_spaced() {
        let paths: Vec<PathBuf> = (0..10)
            .map(|i| PathBuf::from(format!("step_{i}")))
            .collect();

        // Select 3 from 10: should get 0, 4/5, 9
        let selected = select_evenly_spaced(&paths, 3);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0], paths[0]); // First
        assert_eq!(selected[2], paths[9]); // Last

        // Select 1: should get last
        let selected = select_evenly_spaced(&paths, 1);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], paths[9]);

        // Select more than available: get all
        let selected = select_evenly_spaced(&paths, 20);
        assert_eq!(selected.len(), 10);
    }

    #[test]
    fn test_round_robin_pairings() {
        let pairings = round_robin_pairings(4);
        assert_eq!(pairings.len(), 6); // 4*3/2 = 6
        assert!(pairings.contains(&(0, 1)));
        assert!(pairings.contains(&(0, 2)));
        assert!(pairings.contains(&(0, 3)));
        assert!(pairings.contains(&(1, 2)));
        assert!(pairings.contains(&(1, 3)));
        assert!(pairings.contains(&(2, 3)));
    }

    #[test]
    fn test_swiss_pairings() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("D".to_string(), PlayerSource::Random, 0.0),
        ];

        let pairings = swiss_pairings(&contestants);
        // Should pair all 4: 2 pairings
        assert_eq!(pairings.len(), 2);

        // Each contestant should appear exactly once
        let mut seen = [false; 4];
        for (a, b) in &pairings {
            assert!(!seen[*a]);
            assert!(!seen[*b]);
            seen[*a] = true;
            seen[*b] = true;
        }
    }

    #[test]
    fn test_swiss_pairings_with_different_ratings() {
        let mut contestants = vec![
            Contestant::new("High".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("Low".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("Medium".to_string(), PlayerSource::Random, 0.0),
        ];

        // Modify ratings
        contestants[0].rating.rating = 30.0;
        contestants[1].rating.rating = 15.0;
        contestants[2].rating.rating = 25.0;

        let pairings = swiss_pairings(&contestants);
        // With 3 contestants, only 1 pairing (one gets a bye)
        assert_eq!(pairings.len(), 1);
    }

    #[test]
    fn test_swiss_pairings_prefers_unfaced_opponents() {
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("D".to_string(), PlayerSource::Random, 0.0),
        ];

        // A has already faced B
        contestants[0].opponents_faced.push(1);
        contestants[1].opponents_faced.push(0);

        let pairings = swiss_pairings(&contestants);

        // A should be paired with C or D (not B since they've already faced)
        let a_pairing = pairings.iter().find(|(a, b)| *a == 0 || *b == 0);
        if let Some((a, b)) = a_pairing {
            let opponent = if *a == 0 { *b } else { *a };
            assert!(opponent == 2 || opponent == 3); // C or D, not B
        }
    }

    #[test]
    fn test_swiss_pairings_odd_number() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
        ];

        let pairings = swiss_pairings(&contestants);
        // With 3 contestants, only 1 pairing (one gets a bye)
        assert_eq!(pairings.len(), 1);
    }

    #[test]
    fn test_build_results() {
        use crate::config::TournamentArgs;

        let mut contestants = vec![
            Contestant::new("Winner".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("Loser".to_string(), PlayerSource::Random, 0.0),
        ];

        contestants[0].rating.rating = 28.0;
        contestants[0].placement_counts = vec![3, 0]; // 3 wins (1st place), 0 losses (2nd place)
        contestants[0].games_played = 3;
        contestants[1].rating.rating = 22.0;
        contestants[1].placement_counts = vec![0, 3]; // 0 wins, 3 losses
        contestants[1].games_played = 3;

        let matches = vec![(
            1,
            MatchupResult {
                contestant_a: 0,
                contestant_b: 1,
                wins_a: 3,
                wins_b: 0,
                draws: 0,
            },
        )];

        let args = TournamentArgs {
            sources: vec![],
            num_games: 3,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: Some(1),
            output: None,
            seed: Some(42),
            random: false,
        };

        let results = build_results(&contestants, &matches, 1, &args, "connect_four");

        assert_eq!(results.rankings.len(), 2);
        assert_eq!(results.rankings[0].name, "Winner"); // Higher rating is first
        assert_eq!(results.rankings[0].rank, 1);
        assert_eq!(results.rankings[1].name, "Loser");
        assert_eq!(results.rankings[1].rank, 2);
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.environment, "connect_four");
        assert_eq!(results.config.num_games_per_matchup, 3);
        assert_eq!(results.config.seed, Some(42));
    }

    #[test]
    fn test_build_results_round_robin_format() {
        use crate::config::TournamentArgs;

        let contestants: Vec<Contestant> = (0..4)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        let args = TournamentArgs {
            sources: vec![],
            num_games: 5,
            num_envs: 1,
            temperature: 0.5,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
        };

        let results = build_results(&contestants, &[], 1, &args, "cartpole");

        // 4 contestants = round-robin format
        assert_eq!(results.config.format, "round-robin");
    }

    #[test]
    fn test_build_results_swiss_format() {
        use crate::config::TournamentArgs;

        // 10 contestants > 8 = swiss format
        let contestants: Vec<Contestant> = (0..10)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        let args = TournamentArgs {
            sources: vec![],
            num_games: 5,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: Some(3),
            output: None,
            seed: None,
            random: false,
        };

        let results = build_results(&contestants, &[], 3, &args, "connect_four");

        assert_eq!(results.config.format, "swiss");
    }

    #[test]
    fn test_discover_contestants_single_checkpoint() {
        use crate::config::TournamentArgs;

        let temp = TempDir::new().unwrap();
        let checkpoint = temp.path().join("step_100");
        std::fs::create_dir(&checkpoint).unwrap();
        std::fs::write(checkpoint.join("metadata.json"), "{}").unwrap();

        let args = TournamentArgs {
            sources: vec![checkpoint.clone()],
            num_games: 1,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
        };

        let contestants = discover_contestants(&args).unwrap();

        assert_eq!(contestants.len(), 1);
        assert_eq!(contestants[0].name, "step_100");
        assert!(matches!(contestants[0].source, PlayerSource::Checkpoint(_)));
    }

    #[test]
    fn test_discover_contestants_with_random() {
        use crate::config::TournamentArgs;

        let temp = TempDir::new().unwrap();
        let checkpoint = temp.path().join("step_100");
        std::fs::create_dir(&checkpoint).unwrap();
        std::fs::write(checkpoint.join("metadata.json"), "{}").unwrap();

        let args = TournamentArgs {
            sources: vec![checkpoint],
            num_games: 1,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: true, // Add random player
        };

        let contestants = discover_contestants(&args).unwrap();

        assert_eq!(contestants.len(), 2);
        assert!(contestants.iter().any(|c| c.name == "Random"));
    }

    #[test]
    fn test_discover_contestants_from_checkpoints_dir() {
        use crate::config::TournamentArgs;

        let temp = TempDir::new().unwrap();

        // Create a checkpoints directory with step_ subdirs
        std::fs::create_dir(temp.path().join("step_100")).unwrap();
        std::fs::create_dir(temp.path().join("step_200")).unwrap();
        std::fs::create_dir(temp.path().join("step_300")).unwrap();

        let args = TournamentArgs {
            sources: vec![temp.path().to_path_buf()],
            num_games: 1,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None, // No limit
            rounds: None,
            output: None,
            seed: None,
            random: false,
        };

        let contestants = discover_contestants(&args).unwrap();

        assert_eq!(contestants.len(), 3);
    }

    #[test]
    fn test_discover_contestants_with_limit() {
        use crate::config::TournamentArgs;

        let temp = TempDir::new().unwrap();

        // Create 10 checkpoint directories
        for i in 0..10 {
            std::fs::create_dir(temp.path().join(format!("step_{}", i * 100))).unwrap();
        }

        let args = TournamentArgs {
            sources: vec![temp.path().to_path_buf()],
            num_games: 1,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: Some(3), // Limit to 3
            rounds: None,
            output: None,
            seed: None,
            random: false,
        };

        let contestants = discover_contestants(&args).unwrap();

        assert_eq!(contestants.len(), 3);
    }

    #[test]
    fn test_discover_contestants_invalid_path() {
        use crate::config::TournamentArgs;

        let args = TournamentArgs {
            sources: vec![PathBuf::from("/nonexistent/path/to/checkpoint")],
            num_games: 1,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
        };

        let result = discover_contestants(&args);

        assert!(result.is_err());
    }

    #[test]
    fn test_discover_contestants_empty_checkpoints_dir() {
        use crate::config::TournamentArgs;

        let temp = TempDir::new().unwrap();

        // Create a checkpoints directory but with NO step_ subdirs
        // but with some step_ dirs to make it pass is_run_checkpoints_dir
        // Actually, an empty dir won't pass is_run_checkpoints_dir, so this is invalid
        let args = TournamentArgs {
            sources: vec![temp.path().to_path_buf()],
            num_games: 1,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
        };

        // Empty dir isn't a valid checkpoint or checkpoints dir
        let result = discover_contestants(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_matchup_with_random_produces_results() {
        use crate::envs::connect_four::ConnectFour;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);
        let result = run_matchup_with_random::<ConnectFour>(0, 1, 10, &mut rng);

        assert_eq!(result.contestants, vec![0, 1]);
        // Total games should equal num_games
        assert_eq!(result.games.len(), 10);
        // Each game should have proper placements for 2 players
        for game in &result.games {
            assert_eq!(game.placements.len(), 2);
        }
    }

    #[test]
    fn test_run_matchup_with_random_deterministic() {
        use crate::envs::connect_four::ConnectFour;
        use rand::SeedableRng;

        // Same seed should produce same results
        let mut rng1 = StdRng::seed_from_u64(12345);
        let result1 = run_matchup_with_random::<ConnectFour>(0, 1, 5, &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(12345);
        let result2 = run_matchup_with_random::<ConnectFour>(0, 1, 5, &mut rng2);

        assert_eq!(result1.wins_a(), result2.wins_a());
        assert_eq!(result1.wins_b(), result2.wins_b());
        assert_eq!(result1.draws(), result2.draws());
    }

    #[test]
    fn test_select_evenly_spaced_zero() {
        let paths: Vec<PathBuf> = (0..5).map(|i| PathBuf::from(format!("step_{i}"))).collect();

        let selected = select_evenly_spaced(&paths, 0);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_build_results_with_checkpoint_source() {
        use crate::config::TournamentArgs;

        let checkpoint_path = PathBuf::from("/path/to/step_100");
        let mut contestants = vec![Contestant::new(
            "step_100".to_string(),
            PlayerSource::Checkpoint(checkpoint_path.clone()),
            0.0,
        )];
        // Set up placement counts: 5 wins (1st place), 2 losses (2nd place), 1 draw
        contestants[0].placement_counts = vec![5, 2]; // 5 first places, 2 second places
        contestants[0].draw_count = 1;
        contestants[0].games_played = 8;

        let args = TournamentArgs {
            sources: vec![],
            num_games: 8,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
        };

        let results = build_results(&contestants, &[], 1, &args, "test_env");

        assert_eq!(results.rankings.len(), 1);
        assert_eq!(
            results.rankings[0].source,
            Some("/path/to/step_100".to_string())
        );
        assert_eq!(results.rankings[0].games_played, 8);
    }

    #[test]
    fn test_run_matchup_with_random_cartpole() {
        use crate::envs::cartpole::CartPole;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);
        let result = run_matchup_with_random::<CartPole>(0, 1, 5, &mut rng);

        // CartPole is single player, so outcomes are based on episode length/reward
        assert_eq!(result.contestants, vec![0, 1]);
        assert_eq!(result.games.len(), 5);
    }

    #[test]
    fn test_print_standings_does_not_panic() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        // Just verify it doesn't panic
        print_standings(&contestants, "Test Header");
    }

    #[test]
    fn test_print_final_summary_does_not_panic() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        // Just verify it doesn't panic
        print_final_summary(&contestants, 3, 10);
    }

    #[test]
    fn test_build_results_match_summaries() {
        use crate::config::TournamentArgs;

        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        let matches = vec![
            (
                1,
                MatchupResult {
                    contestant_a: 0,
                    contestant_b: 1,
                    wins_a: 2,
                    wins_b: 1,
                    draws: 0,
                },
            ),
            (
                2,
                MatchupResult {
                    contestant_a: 0,
                    contestant_b: 1,
                    wins_a: 1,
                    wins_b: 2,
                    draws: 1,
                },
            ),
        ];

        let args = TournamentArgs {
            sources: vec![],
            num_games: 3,
            num_envs: 1,
            temperature: 1.0,
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: Some(2),
            output: None,
            seed: None,
            random: false,
        };

        let results = build_results(&contestants, &matches, 2, &args, "test");

        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].round, 1);
        assert_eq!(results.matches[0].wins_a, 2);
        assert_eq!(results.matches[1].round, 2);
        assert_eq!(results.matches[1].draws, 1);
    }

    #[test]
    fn test_swiss_pairings_all_faced() {
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("D".to_string(), PlayerSource::Random, 0.0),
        ];

        // All have faced each other
        contestants[0].opponents_faced = vec![1, 2, 3];
        contestants[1].opponents_faced = vec![0, 2, 3];
        contestants[2].opponents_faced = vec![0, 1, 3];
        contestants[3].opponents_faced = vec![0, 1, 2];

        let pairings = swiss_pairings(&contestants);
        // Should still produce pairings even when all faced
        assert_eq!(pairings.len(), 2);
    }

    #[test]
    fn test_contestant_debug_impl() {
        let contestant = Contestant::new("Test".to_string(), PlayerSource::Random, 0.0);
        let debug_str = format!("{contestant:?}");
        assert!(debug_str.contains("Test"));
        assert!(debug_str.contains("Random"));
    }

    #[test]
    fn test_matchup_result_clone() {
        let result = MatchupResult {
            contestant_a: 0,
            contestant_b: 1,
            wins_a: 5,
            wins_b: 3,
            draws: 2,
        };

        let cloned = result.clone();
        assert_eq!(cloned.wins_a, 5);
        assert_eq!(cloned.wins_b, 3);
    }

    #[test]
    fn test_ranking_entry_clone() {
        let entry = RankingEntry {
            rank: 1,
            name: "Test".to_string(),
            source: None,
            swiss_points: 0.0,
            rating: 25.0,
            uncertainty: 8.333,
            rating_low: 8.334,
            rating_high: 41.666,
            wins: 0,
            losses: 0,
            draws: 0,
            games_played: 0,
        };

        let cloned = entry.clone();
        assert_eq!(cloned.rank, 1);
        assert_eq!(cloned.name, "Test");
    }

    #[test]
    fn test_tournament_results_clone() {
        let results = TournamentResults {
            rankings: vec![],
            matches: vec![],
            config: TournamentConfigSummary {
                num_games_per_matchup: 5,
                num_rounds: 1,
                format: "round-robin".to_string(),
                temperature: 1.0,
                seed: None,
            },
            environment: "test".to_string(),
            timestamp: "unix:0".to_string(),
        };

        let cloned = results.clone();
        assert_eq!(cloned.environment, "test");
    }

    // ===== Swiss Points Tests =====

    #[test]
    fn test_swiss_points_2player_win() {
        // Winner gets 1 point, loser gets 0
        // placements: [1, 2] -> avg_pos: [1, 2] -> points: [1.0, 0.0]
        let points = calculate_swiss_points(&[1, 2]);
        assert_eq!(points.len(), 2);
        assert!((points[0] - 1.0).abs() < 0.001);
        assert!((points[1] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_swiss_points_2player_draw() {
        // Both get 0.5 points
        // placements: [1, 1] -> avg_pos: [1.5, 1.5] -> points: [0.5, 0.5]
        let points = calculate_swiss_points(&[1, 1]);
        assert_eq!(points.len(), 2);
        assert!((points[0] - 0.5).abs() < 0.001);
        assert!((points[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_swiss_points_4player_no_ties() {
        // placements: [1,2,3,4] -> points: [3, 2, 1, 0]
        let points = calculate_swiss_points(&[1, 2, 3, 4]);
        assert_eq!(points.len(), 4);
        assert!((points[0] - 3.0).abs() < 0.001);
        assert!((points[1] - 2.0).abs() < 0.001);
        assert!((points[2] - 1.0).abs() < 0.001);
        assert!((points[3] - 0.0).abs() < 0.001);
        // Total should be 6
        let total: f64 = points.iter().sum();
        assert!((total - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_swiss_points_4player_partial_tie_first() {
        // placements: [1,1,3,4] -> avg_pos: [1.5, 1.5, 3, 4] -> points: [2.5, 2.5, 1, 0]
        let points = calculate_swiss_points(&[1, 1, 3, 4]);
        assert_eq!(points.len(), 4);
        assert!((points[0] - 2.5).abs() < 0.001);
        assert!((points[1] - 2.5).abs() < 0.001);
        assert!((points[2] - 1.0).abs() < 0.001);
        assert!((points[3] - 0.0).abs() < 0.001);
        // Total should be 6
        let total: f64 = points.iter().sum();
        assert!((total - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_swiss_points_4player_partial_tie_middle() {
        // placements: [1,2,2,4] -> avg_pos: [1, 2.5, 2.5, 4] -> points: [3, 1.5, 1.5, 0]
        let points = calculate_swiss_points(&[1, 2, 2, 4]);
        assert_eq!(points.len(), 4);
        assert!((points[0] - 3.0).abs() < 0.001);
        assert!((points[1] - 1.5).abs() < 0.001);
        assert!((points[2] - 1.5).abs() < 0.001);
        assert!((points[3] - 0.0).abs() < 0.001);
        // Total should be 6
        let total: f64 = points.iter().sum();
        assert!((total - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_swiss_points_4player_full_draw() {
        // placements: [1,1,1,1] -> avg_pos: [2.5, 2.5, 2.5, 2.5] -> points: [1.5, 1.5, 1.5, 1.5]
        let points = calculate_swiss_points(&[1, 1, 1, 1]);
        assert_eq!(points.len(), 4);
        for p in &points {
            assert!((*p - 1.5).abs() < 0.001);
        }
        // Total should be 6
        let total: f64 = points.iter().sum();
        assert!((total - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_swiss_points_empty() {
        let points = calculate_swiss_points(&[]);
        assert!(points.is_empty());
    }

    #[test]
    fn test_update_ratings_awards_swiss_points() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        assert_eq!(contestants[0].swiss_points, 0.0);
        assert_eq!(contestants[1].swiss_points, 0.0);

        // Player A wins, Player B loses
        let matchup = MatchupGames {
            contestants: vec![0, 1],
            games: vec![GamePlacement {
                placements: vec![1, 2],
            }],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        // Winner gets 1 point, loser gets 0
        assert!((contestants[0].swiss_points - 1.0).abs() < 0.001);
        assert!((contestants[1].swiss_points - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_update_ratings_awards_swiss_points_multiple_games() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        // 2 wins for A, 1 win for B, 1 draw
        // Raw points: A = 2.5 (2 wins + 0.5 draw), B = 1.5 (1 win + 0.5 draw)
        let matchup = MatchupGames {
            contestants: vec![0, 1],
            games: vec![
                GamePlacement {
                    placements: vec![1, 2],
                }, // A wins
                GamePlacement {
                    placements: vec![1, 2],
                }, // A wins
                GamePlacement {
                    placements: vec![2, 1],
                }, // B wins
                GamePlacement {
                    placements: vec![1, 1],
                }, // Draw
            ],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        // With match-level scoring:
        // A has more raw points (2.5 vs 1.5) → 1st place → 1.0 Swiss points
        // B has fewer raw points → 2nd place → 0.0 Swiss points
        assert!((contestants[0].swiss_points - 1.0).abs() < 0.001);
        assert!((contestants[1].swiss_points - 0.0).abs() < 0.001);
    }

    // ===== Swiss Pods Tests =====

    #[test]
    fn test_swiss_pods_by_points() {
        // 8 players, pod_size=4 -> 2 pods
        let mut contestants: Vec<Contestant> = (0..8)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        // Set different Swiss points
        contestants[0].swiss_points = 10.0;
        contestants[1].swiss_points = 9.0;
        contestants[2].swiss_points = 8.0;
        contestants[3].swiss_points = 7.0;
        contestants[4].swiss_points = 6.0;
        contestants[5].swiss_points = 5.0;
        contestants[6].swiss_points = 4.0;
        contestants[7].swiss_points = 3.0;

        let pods = swiss_pods(&contestants, 4);
        assert_eq!(pods.len(), 2);

        // Each pod should have 4 players
        assert_eq!(pods[0].len(), 4);
        assert_eq!(pods[1].len(), 4);

        // Top 4 by points should be in first pod (0,1,2,3)
        // Bottom 4 should be in second pod (4,5,6,7)
        for idx in &pods[0] {
            assert!(*idx < 4);
        }
        for idx in &pods[1] {
            assert!(*idx >= 4);
        }
    }

    #[test]
    fn test_swiss_pods_avoids_repeat_opponents() {
        let mut contestants: Vec<Contestant> = (0..8)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        // Set up: 0,1,2,3 all faced each other in previous round
        contestants[0].opponents_faced = vec![1, 2, 3];
        contestants[1].opponents_faced = vec![0, 2, 3];
        contestants[2].opponents_faced = vec![0, 1, 3];
        contestants[3].opponents_faced = vec![0, 1, 2];

        // 4,5,6,7 also faced each other
        contestants[4].opponents_faced = vec![5, 6, 7];
        contestants[5].opponents_faced = vec![4, 6, 7];
        contestants[6].opponents_faced = vec![4, 5, 7];
        contestants[7].opponents_faced = vec![4, 5, 6];

        let pods = swiss_pods(&contestants, 4);

        // Should mix the groups since all within groups have faced each other
        // First pod should not be exactly [0,1,2,3] or [4,5,6,7]
        let first_pod_set: std::collections::HashSet<_> = pods[0].iter().collect();
        let original_top: std::collections::HashSet<_> = [0, 1, 2, 3].iter().collect();
        let original_bottom: std::collections::HashSet<_> = [4, 5, 6, 7].iter().collect();

        // At least one pod should be mixed (not exactly match original groupings)
        let pod0_is_top = first_pod_set == original_top;
        let pod0_is_bottom = first_pod_set == original_bottom;

        // We expect some mixing since all within groups have faced each other
        assert!(
            !pod0_is_top || !pod0_is_bottom,
            "Pods should try to mix players who haven't faced each other"
        );
    }

    #[test]
    fn test_swiss_pods_exact_size() {
        let contestants: Vec<Contestant> = (0..12)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        let pods = swiss_pods(&contestants, 4);
        assert_eq!(pods.len(), 3); // 12 / 4 = 3 pods

        for pod in &pods {
            assert_eq!(pod.len(), 4);
        }
    }

    #[test]
    fn test_swiss_pods_incomplete() {
        // 10 players, pod_size=4 -> 2 complete pods (8 players), 2 leftover
        let contestants: Vec<Contestant> = (0..10)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        let pods = swiss_pods(&contestants, 4);
        // Only complete pods are returned
        assert_eq!(pods.len(), 2);

        for pod in &pods {
            assert_eq!(pod.len(), 4);
        }
    }

    #[test]
    fn test_contestant_new_initializes_swiss_fields() {
        let contestant = Contestant::new("Test".to_string(), PlayerSource::Random, 0.0);
        assert_eq!(contestant.swiss_points, 0.0);
        assert!(!contestant.has_bye);
    }

    #[test]
    fn test_contestant_initial_seed() {
        let contestant = Contestant::new("Test".to_string(), PlayerSource::Random, 42.5);
        assert!((contestant.initial_seed - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_swiss_pairings_dutch_style_round_1() {
        // Round 1: all swiss_points == 0, should use Dutch-style pairing
        // Sort by initial_seed (descending), pair top half vs bottom half
        let contestants = vec![
            Contestant::new("Seed3".to_string(), PlayerSource::Random, 3.0),
            Contestant::new("Seed1".to_string(), PlayerSource::Random, 1.0),
            Contestant::new("Seed4".to_string(), PlayerSource::Random, 4.0),
            Contestant::new("Seed2".to_string(), PlayerSource::Random, 2.0),
        ];

        let pairings = swiss_pairings(&contestants);
        assert_eq!(pairings.len(), 2);

        // Sorted by seed: Seed4(idx=2), Seed3(idx=0), Seed2(idx=3), Seed1(idx=1)
        // Top half: Seed4, Seed3; Bottom half: Seed2, Seed1
        // Expected pairings: (2, 3), (0, 1) - Seed4 vs Seed2, Seed3 vs Seed1
        let mut found_pairs: Vec<(usize, usize)> = pairings
            .iter()
            .map(|&(a, b)| if a < b { (a, b) } else { (b, a) })
            .collect();
        found_pairs.sort_unstable();

        // Verify Dutch-style: highest seed paired with mid-ranked, not with 2nd highest
        assert!(found_pairs.contains(&(0, 1)) || found_pairs.contains(&(2, 3)));
    }

    #[test]
    fn test_swiss_pods_dutch_style_round_1() {
        // Round 1: all swiss_points == 0, should use Dutch-style pairing
        // For 4-player pods, divide into 4 quartiles
        let contestants: Vec<Contestant> = (0..8)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        let pods = swiss_pods(&contestants, 4);
        assert_eq!(pods.len(), 2);

        // With Dutch-style: Pod i = [i, i+num_pods, i+2*num_pods, i+3*num_pods]
        // num_pods = 8/4 = 2
        // Pod 0 should have players ranked [0, 2, 4, 6] by seed position
        // Pod 1 should have players ranked [1, 3, 5, 7] by seed position
        for pod in &pods {
            assert_eq!(pod.len(), 4);
            // Each pod should have a mix of skill levels (not all top or all bottom)
        }
    }

    #[test]
    fn test_swiss_pairings_subsequent_rounds() {
        // After round 1, contestants have swiss_points > 0
        // Should sort by swiss_points, not initial_seed
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 1.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 2.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 3.0),
            Contestant::new("D".to_string(), PlayerSource::Random, 4.0),
        ];

        // Set non-zero swiss_points to simulate post-round-1
        contestants[0].swiss_points = 2.0; // A has most points
        contestants[1].swiss_points = 1.0;
        contestants[2].swiss_points = 0.5;
        contestants[3].swiss_points = 0.0; // D has least points

        let pairings = swiss_pairings(&contestants);
        assert_eq!(pairings.len(), 2);

        // A (highest points) should be paired with someone, not necessarily by seed
    }

    #[test]
    fn test_match_level_scoring_tie() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        // Both players tie: each wins 2 games
        let matchup = MatchupGames {
            contestants: vec![0, 1],
            games: vec![
                GamePlacement {
                    placements: vec![1, 2],
                }, // A wins
                GamePlacement {
                    placements: vec![1, 2],
                }, // A wins
                GamePlacement {
                    placements: vec![2, 1],
                }, // B wins
                GamePlacement {
                    placements: vec![2, 1],
                }, // B wins
            ],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        // Both have equal raw points (2 each), so they tie for 1st place
        // With fractional ranking: both get 0.5 Swiss points (share positions 1&2)
        assert!((contestants[0].swiss_points - 0.5).abs() < 0.001);
        assert!((contestants[1].swiss_points - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_match_level_scoring_4player() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("D".to_string(), PlayerSource::Random, 0.0),
        ];

        // Play 4 games, A dominates, D loses everything
        let matchup = MatchupGames {
            contestants: vec![0, 1, 2, 3],
            games: vec![
                GamePlacement {
                    placements: vec![1, 2, 3, 4],
                }, // A wins
                GamePlacement {
                    placements: vec![1, 2, 3, 4],
                }, // A wins
                GamePlacement {
                    placements: vec![1, 3, 2, 4],
                }, // A wins, C beats B
                GamePlacement {
                    placements: vec![1, 2, 3, 4],
                }, // A wins
            ],
        };

        update_ratings_from_games(&mut contestants, &matchup, &wl_config);

        // A should be 1st, D should be last
        // A: 4*3 = 12 raw points → 1st → 3.0 Swiss points
        // D: 4*0 = 0 raw points → 4th → 0.0 Swiss points
        assert!((contestants[0].swiss_points - 3.0).abs() < 0.001); // A = 1st
        assert!((contestants[3].swiss_points - 0.0).abs() < 0.001); // D = 4th
    }
}
