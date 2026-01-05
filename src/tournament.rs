//! Tournament mode for evaluating multiple checkpoints with skill ratings
//!
//! Features:
//! - Swiss-style tournaments for efficient skill estimation (N > 8 contestants)
//! - Round-robin for complete coverage (N <= 8 contestants)
//! - Weng-Lin (`OpenSkill`) rating system with uncertainty tracking
//! - Progress bars and intermediate standings

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::Itertools;
use plotters::backend::BitMapBackend;
use plotters::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use skillratings::weng_lin::{weng_lin_multi_team, WengLinConfig, WengLinRating};
use skillratings::MultiTeamOutcome;

use crate::checkpoint::{load_metadata, CheckpointMetadata};
use crate::config::TournamentArgs;
use crate::dispatch_env;
use crate::env::Environment;
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
    /// Get total Swiss points for contestant at given index
    fn total_points(&self, idx: usize) -> f64 {
        self.games
            .iter()
            .map(|g| {
                let points = calculate_swiss_points(&g.placements);
                points.get(idx).copied().unwrap_or(0.0)
            })
            .sum()
    }

    /// Get average Swiss points per game for contestant at given index
    pub fn avg_points(&self, idx: usize) -> f64 {
        if self.games.is_empty() {
            return 0.0;
        }
        self.total_points(idx) / self.games.len() as f64
    }

    /// Get draw count (all contestants tied for 1st)
    pub fn full_draws(&self) -> usize {
        self.games
            .iter()
            .filter(|g| !g.placements.is_empty() && g.placements.iter().all(|&p| p == 1))
            .count()
    }

    /// Format result summary for display
    pub fn summary(&self, names: &[&str]) -> String {
        let num_contestants = self.contestants.len();
        let points: Vec<f64> = (0..num_contestants).map(|i| self.avg_points(i)).collect();

        // Find winner (highest avg points)
        let (best_idx, _) = points
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in points comparison"))
            .unwrap_or((0, &0.0));

        let names_str = names.join(" vs ");
        let points_str: String = points
            .iter()
            .map(|p| format!("{p:.2}"))
            .collect::<Vec<_>>()
            .join("-");

        let winner = if points.iter().all(|&p| (p - points[0]).abs() < 0.001) {
            "draw".to_string()
        } else {
            names.get(best_idx).unwrap_or(&"?").to_string()
        };

        format!("{names_str}: {points_str} ({winner})")
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

    /// Get wins (1st place finishes, excluding draws) - for test assertions
    #[cfg(test)]
    pub fn wins(&self) -> usize {
        self.placement_counts
            .first()
            .copied()
            .unwrap_or(0)
            .saturating_sub(self.draw_count)
    }

    /// Get losses (last place finishes) - for test assertions
    #[cfg(test)]
    pub fn losses(&self) -> usize {
        if self.placement_counts.len() >= 2 {
            self.placement_counts.last().copied().unwrap_or(0)
        } else {
            0
        }
    }

    /// Get draws (ties) - for test assertions
    #[cfg(test)]
    pub fn draws(&self) -> usize {
        self.draw_count
    }
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
    /// Placement counts: [1st place finishes, 2nd place finishes, ...]
    pub placement_counts: Vec<usize>,
    /// Number of full draws (all players tied for 1st)
    pub draw_count: usize,
    pub games_played: usize,
}

/// Pod match summary for JSON output (N-player generic)
#[derive(Debug, Clone, Serialize)]
pub struct PodSummary {
    pub round: usize,
    /// Names of contestants in this pod
    pub contestants: Vec<String>,
    /// Average Swiss points per game for each contestant
    pub avg_points: Vec<f64>,
    /// Number of games played
    pub games: usize,
    /// Number of full draws (all tied for 1st)
    pub draws: usize,
}

/// Full tournament results for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct TournamentResults {
    pub rankings: Vec<RankingEntry>,
    pub pods: Vec<PodSummary>,
    pub config: TournamentConfigSummary,
    pub environment: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TournamentConfigSummary {
    pub num_games_per_matchup: usize,
    pub num_rounds: usize,
    pub format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Check if a path is a checkpoint directory (has metadata.json)
fn is_checkpoint_dir(path: &Path) -> bool {
    path.is_dir() && path.join("metadata.json").exists()
}

/// Check if a path is a checkpoints directory (contains step_* subdirectories)
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

/// Check if a path is a run directory (has checkpoints subdirectory)
fn is_run_dir(path: &Path) -> bool {
    path.is_dir() && path.join("checkpoints").is_dir()
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

/// Get the "best" checkpoint from a checkpoints directory.
/// Priority: best symlink > highest `training_rating` (with warning)
fn get_best_checkpoint(checkpoints_dir: &Path) -> Option<PathBuf> {
    let best_symlink = checkpoints_dir.join("best");

    // If "best" symlink exists, use it
    if best_symlink.exists() {
        let resolved = if best_symlink.is_symlink() {
            best_symlink.read_link().ok().map(|target| {
                if target.is_absolute() {
                    target
                } else {
                    checkpoints_dir.join(target)
                }
            })
        } else {
            Some(best_symlink)
        };

        if let Some(path) = resolved {
            if is_checkpoint_dir(&path) {
                return Some(path);
            }
        }
    }

    // Fallback: scan for highest training_rating (with warning)
    eprintln!(
        "Warning: No 'best' symlink in {}, falling back to highest training_rating",
        checkpoints_dir.display()
    );

    let checkpoints = enumerate_checkpoints(checkpoints_dir).ok()?;
    checkpoints.into_iter().max_by(|a, b| {
        let rating_a = load_metadata(a).map(|m| m.training_rating).unwrap_or(0.0);
        let rating_b = load_metadata(b).map(|m| m.training_rating).unwrap_or(0.0);
        rating_a
            .partial_cmp(&rating_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Select checkpoints with priority: best, latest, then evenly distributed
/// - limit 1: best only
/// - limit 2: best + latest
/// - limit 3+: best + latest + (n-2) evenly distributed from remaining
fn select_checkpoints_with_priority(
    checkpoints_dir: &Path,
    checkpoints: &[PathBuf],
    limit: usize,
) -> Vec<PathBuf> {
    if limit == 0 || checkpoints.is_empty() {
        return Vec::new();
    }

    let best = get_best_checkpoint(checkpoints_dir);
    let latest = checkpoints.last().cloned();

    match limit {
        1 => {
            // Just best (or latest as final fallback)
            best.or(latest).into_iter().collect()
        }
        2 => {
            // Best + latest (deduplicated if same)
            let mut result = Vec::new();
            if let Some(b) = &best {
                result.push(b.clone());
            }
            if let Some(l) = &latest {
                if best.as_ref() != Some(l) {
                    result.push(l.clone());
                }
            }
            result
        }
        _ => {
            // Best + latest + (n-2) evenly distributed from remaining
            let mut result = Vec::new();
            let mut excluded: HashSet<PathBuf> = HashSet::new();

            if let Some(b) = &best {
                result.push(b.clone());
                excluded.insert(b.clone());
            }
            if let Some(l) = &latest {
                if !excluded.contains(l) {
                    result.push(l.clone());
                    excluded.insert(l.clone());
                }
            }

            // Get remaining checkpoints (excluding best and latest)
            let remaining: Vec<PathBuf> = checkpoints
                .iter()
                .filter(|c| !excluded.contains(*c))
                .cloned()
                .collect();

            // Select (limit - result.len()) evenly from remaining
            let extra_needed = limit.saturating_sub(result.len());
            let extra = select_evenly_spaced(&remaining, extra_needed);
            result.extend(extra);

            result
        }
    }
}

/// Compute display names for a list of paths.
///
/// Algorithm:
/// 1. Strip the longest common prefix (full folders only) from all paths
/// 2. Collapse common middle runs (folders identical across ALL paths at the same
///    position from the end) with "..."
///
/// For a single path, just returns the filename.
pub fn compute_display_names(paths: &[PathBuf]) -> Vec<String> {
    if paths.is_empty() {
        return Vec::new();
    }

    // Single path: just return the filename
    if paths.len() == 1 {
        return vec![paths[0]
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string()];
    }

    // Build a list of path components for each path (in forward order)
    let components: Vec<Vec<&str>> = paths
        .iter()
        .map(|p| p.iter().filter_map(|c| c.to_str()).collect())
        .collect();

    // Find the longest common prefix
    let prefix_len = common_prefix_len(&components);

    // Strip the common prefix from all paths
    let stripped: Vec<Vec<&str>> = components
        .iter()
        .map(|comps| comps[prefix_len..].to_vec())
        .collect();

    // Find common middle runs (aligned from end)
    let common_offsets = find_common_middle_offsets(&stripped);

    // Reconstruct display names, collapsing common middles
    stripped
        .iter()
        .map(|comps| collapse_common_middles(comps, &common_offsets))
        .collect()
}

/// Find the length of the longest common prefix (in path components).
/// Never strips the last component (filename) - always leaves at least 1.
fn common_prefix_len(components: &[Vec<&str>]) -> usize {
    if components.is_empty() {
        return 0;
    }

    let min_len = components.iter().map(Vec::len).min().unwrap_or(0);
    // Never strip the last component (filename)
    let max_prefix = min_len.saturating_sub(1);
    let first = &components[0];

    for i in 0..max_prefix {
        if !components.iter().all(|c| c[i] == first[i]) {
            return i;
        }
    }
    max_prefix
}

/// Find offsets from end where all paths have the same component.
/// Returns a set of offsets (e.g., 2 means second-to-last component).
/// Offset 1 (last component) is excluded to keep filenames distinct.
fn find_common_middle_offsets(components: &[Vec<&str>]) -> std::collections::HashSet<usize> {
    use std::collections::HashSet;

    let mut common = HashSet::new();

    if components.is_empty() {
        return common;
    }

    // Find minimum length across all paths
    let min_len = components.iter().map(Vec::len).min().unwrap_or(0);

    // Check each position from the end (excluding the last component - keep filenames distinct)
    // offset_from_end: 2 = second-to-last, 3 = third-to-last, etc.
    for offset_from_end in 2..=min_len {
        let first_val = components[0][components[0].len() - offset_from_end];
        let all_match = components
            .iter()
            .all(|c| c[c.len() - offset_from_end] == first_val);
        if all_match {
            common.insert(offset_from_end);
        }
    }

    common
}

/// Collapse consecutive common middle components into "..."
fn collapse_common_middles(
    comps: &[&str],
    common_offsets: &std::collections::HashSet<usize>,
) -> String {
    if comps.is_empty() {
        return String::new();
    }

    let len = comps.len();
    let mut result = Vec::new();
    let mut in_common_run = false;

    for (i, &comp) in comps.iter().enumerate() {
        let offset_from_end = len - i;
        if common_offsets.contains(&offset_from_end) {
            if !in_common_run {
                result.push("...");
                in_common_run = true;
            }
            // Skip this component (already represented by "...")
        } else {
            result.push(comp);
            in_common_run = false;
        }
    }

    result.join("/")
}

/// Discover contestants from command-line sources
fn discover_contestants(args: &TournamentArgs) -> Result<Vec<Contestant>> {
    // Track if all checkpoints come from a single training run (same checkpoints folder)
    // Only use training_rating for seeding if this is true
    let single_training_run = args.sources.len() == 1 && {
        let path = &args.sources[0];
        let mut resolved = if path.is_symlink() {
            path.read_link().map_or_else(
                |_| path.clone(),
                |target| path.parent().unwrap_or(path).join(target),
            )
        } else {
            path.clone()
        };
        // Auto-append /checkpoints for run directories
        if is_run_dir(&resolved) {
            resolved = resolved.join("checkpoints");
        }
        is_run_checkpoints_dir(&resolved)
    };

    // Resolve all source paths and identify run directories
    let resolved_sources: Vec<PathBuf> = args
        .sources
        .iter()
        .map(|source_path| {
            let mut resolved = if source_path.is_symlink() {
                source_path.read_link().map_or_else(
                    |_| source_path.clone(),
                    |target| source_path.parent().unwrap_or(source_path).join(target),
                )
            } else {
                source_path.clone()
            };
            // Auto-append /checkpoints for run directories
            if is_run_dir(&resolved) {
                resolved = resolved.join("checkpoints");
            }
            resolved
        })
        .collect();

    // Count run directories for limit splitting
    let num_run_dirs = resolved_sources
        .iter()
        .filter(|p| is_run_checkpoints_dir(p))
        .count();

    // Calculate per-folder limits if limit is set and there are multiple run dirs
    let per_folder_limits: Vec<usize> = if let Some(total_limit) = args.limit {
        if num_run_dirs > 0 {
            let base = total_limit / num_run_dirs;
            let remainder = total_limit % num_run_dirs;
            // Distribute remainder to earlier folders
            (0..num_run_dirs)
                .map(|i| base + usize::from(i < remainder))
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let mut run_dir_idx = 0;

    // First pass: collect all checkpoint paths with their initial seeds
    let mut checkpoint_data: Vec<(PathBuf, f64)> = Vec::new();

    for path in resolved_sources {
        if is_checkpoint_dir(&path) {
            // Single checkpoint - load training_rating from metadata
            let initial_seed = if single_training_run {
                load_metadata(&path)
                    .map(|m| m.training_rating)
                    .unwrap_or(25.0)
            } else {
                0.0 // Will be shuffled later
            };
            checkpoint_data.push((path, initial_seed));
        } else if is_run_checkpoints_dir(&path) {
            // Checkpoints directory - enumerate and optionally limit
            let checkpoints = enumerate_checkpoints(&path)?;
            if checkpoints.is_empty() {
                bail!("No checkpoints found in {}", path.display());
            }

            // Get per-folder limit (if any)
            let folder_limit = per_folder_limits.get(run_dir_idx).copied();
            run_dir_idx += 1;

            let selected = match folder_limit {
                Some(limit) => select_checkpoints_with_priority(&path, &checkpoints, limit),
                None => checkpoints,
            };

            for ckpt in selected {
                // Load training_rating from checkpoint metadata
                let initial_seed = if single_training_run {
                    load_metadata(&ckpt)
                        .map(|m| m.training_rating)
                        .unwrap_or(25.0)
                } else {
                    0.0 // Will be shuffled later
                };
                checkpoint_data.push((ckpt, initial_seed));
            }
        } else {
            bail!(
                "Invalid source: {} (expected checkpoint dir or checkpoints folder)",
                path.display()
            );
        }
    }

    // Deduplicate by resolved path - keep first occurrence
    let mut seen_paths: HashSet<PathBuf> = HashSet::new();
    let original_count = checkpoint_data.len();
    checkpoint_data.retain(|(path, _)| seen_paths.insert(path.clone()));

    if checkpoint_data.len() < original_count {
        let removed = original_count - checkpoint_data.len();
        println!("Warning: Removed {removed} duplicate checkpoint(s) (same resolved path)");
    }

    // Compute unique display names for all checkpoints
    let paths: Vec<PathBuf> = checkpoint_data.iter().map(|(p, _)| p.clone()).collect();
    let display_names = compute_display_names(&paths);

    // Create contestants with computed display names
    let mut contestants: Vec<Contestant> = checkpoint_data
        .into_iter()
        .zip(display_names)
        .map(|((path, initial_seed), name)| {
            Contestant::new(name, PlayerSource::Checkpoint(path), initial_seed)
        })
        .collect();

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
///
/// Public so training and eval can use the same metric as tournaments.
pub fn calculate_swiss_points(placements: &[usize]) -> Vec<f64> {
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

/// Check if any pair in a pod has faced each other before
fn has_repeat_opponents(pod: &[usize], contestants: &[Contestant]) -> bool {
    for i in 0..pod.len() {
        for j in (i + 1)..pod.len() {
            if contestants[pod[i]].opponents_faced.contains(&pod[j]) {
                return true;
            }
        }
    }
    false
}

/// Generate Swiss pods for a round (generalized for any `pod_size` including 2-player)
///
/// Uses Dutch-style pairing:
/// - Round 1: Divide by `initial_seed` into N groups, form pods with one from each group
/// - Subsequent rounds: Group by score brackets, apply same Dutch pairing within each bracket
/// - Floaters (odd players) carry down to the next bracket
/// - Greedy swap in last group to avoid repeat opponents
fn swiss_pods(contestants: &[Contestant], pod_size: usize) -> Vec<Vec<usize>> {
    if contestants.len() < pod_size {
        return Vec::new();
    }

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

        let ranked_indices: Vec<usize> = ranked.iter().map(|(idx, _)| *idx).collect();
        return form_dutch_pods(&ranked_indices, pod_size, contestants);
    }

    // Subsequent rounds: sort by Swiss points (desc), rating as tiebreaker
    let mut ranked: Vec<(usize, f64, f64)> = contestants
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.swiss_points, c.rating.rating))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Group into score brackets
    let mut brackets: Vec<Vec<usize>> = Vec::new();
    let mut current_score = f64::MAX;
    for &(idx, points, _) in &ranked {
        if (points - current_score).abs() > 0.001 {
            brackets.push(Vec::new());
            current_score = points;
        }
        brackets
            .last_mut()
            .expect("brackets should have at least one element")
            .push(idx);
    }

    let mut all_pods = Vec::new();
    let mut floaters: Vec<usize> = Vec::new();

    for bracket in brackets {
        // Floaters from higher bracket join at the top (FIDE: must pair first)
        let pool: Vec<usize> = floaters.drain(..).chain(bracket).collect();

        // Form Dutch-style pods within this bracket
        let (pods, new_floaters) = form_dutch_pods_with_floaters(&pool, pod_size, contestants);
        all_pods.extend(pods);
        floaters = new_floaters;
    }

    // Any remaining floaters can't form a complete pod
    all_pods
}

/// Form Dutch-style pods: divide into N groups, take one from each group per pod
/// Returns the formed pods (does not handle floaters)
fn form_dutch_pods(
    ranked_indices: &[usize],
    pod_size: usize,
    contestants: &[Contestant],
) -> Vec<Vec<usize>> {
    let (pods, _) = form_dutch_pods_with_floaters(ranked_indices, pod_size, contestants);
    pods
}

/// Form Dutch-style pods with floater handling
/// Returns (pods, floaters) where floaters are players who couldn't form a complete pod
fn form_dutch_pods_with_floaters(
    ranked_indices: &[usize],
    pod_size: usize,
    contestants: &[Contestant],
) -> (Vec<Vec<usize>>, Vec<usize>) {
    if ranked_indices.len() < pod_size {
        return (Vec::new(), ranked_indices.to_vec());
    }

    // Calculate number of complete pods we can form
    let num_pods = ranked_indices.len() / pod_size;
    if num_pods == 0 {
        return (Vec::new(), ranked_indices.to_vec());
    }

    // Create mutable copy of indices for potential swaps
    let mut indices = ranked_indices.to_vec();

    // Form pods: Pod i gets contestants [i, i+num_pods, i+2*num_pods, ..., i+(N-1)*num_pods]
    // This ensures each pod has one player from each "skill tier" (Dutch pairing)
    let mut pods = Vec::with_capacity(num_pods);
    for pod_idx in 0..num_pods {
        let mut pod = Vec::with_capacity(pod_size);
        for group in 0..pod_size {
            let ranked_pos = pod_idx + group * num_pods;
            if ranked_pos < indices.len() {
                pod.push(indices[ranked_pos]);
            }
        }

        // Check for repeat opponents and try greedy swap in last group
        if pod.len() == pod_size && has_repeat_opponents(&pod, contestants) {
            // Try swapping the last group's player with later players in the same group
            let last_group_start = (pod_size - 1) * num_pods;
            let current_last_pos = pod_idx + last_group_start;

            for swap_offset in 1..(num_pods - pod_idx) {
                let swap_pos = current_last_pos + swap_offset;
                if swap_pos < indices.len() {
                    // Try this swap
                    let mut test_pod = pod[..pod_size - 1].to_vec();
                    test_pod.push(indices[swap_pos]);

                    if !has_repeat_opponents(&test_pod, contestants) {
                        // Swap is good - apply it
                        indices.swap(current_last_pos, swap_pos);
                        pod = test_pod;
                        break;
                    }
                }
            }
        }

        if pod.len() == pod_size {
            pods.push(pod);
        }
    }

    // Floaters are the remaining players who couldn't form a complete pod
    let floaters: Vec<usize> = indices[num_pods * pod_size..].to_vec();

    (pods, floaters)
}

/// Generate round-robin pods for N-player games
///
/// Returns all unique combinations of `pod_size` contestants.
/// For `pod_size`=2, this is equivalent to all pairs.
/// For larger `pod_size`, uses itertools combinations.
fn round_robin_pods(n: usize, pod_size: usize) -> Vec<Vec<usize>> {
    (0..n).combinations(pod_size).collect()
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

/// Print a quick guide to interpreting Openskill ratings
pub fn print_rating_guide() {
    println!();
    println!("Rating Guide (Openskill):");
    println!("  Win probability: +4 pts → 67% | +8 → 82% | +12 → 90% | +16 → 95%");
    println!("  Uncertainty (σ): high = few games, may shift. Low = stable rating.");
    println!("  Comparing: if 95% CIs (±2σ) overlap, difference may not be significant.");
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

    // Determine number of players from first contestant's placement_counts
    let num_players = contestants.first().map_or(2, |c| c.placement_counts.len());

    // Build header dynamically based on num_players
    let placement_headers: Vec<String> = (1..=num_players).map(ordinal).collect();
    let placement_header_str: String =
        placement_headers.iter().fold(String::new(), |mut acc, h| {
            use std::fmt::Write;
            let _ = write!(acc, "{h:>4}");
            acc
        });

    // First, print results sorted by name (reverse order: newest networks first)
    let mut by_name: Vec<&Contestant> = contestants.iter().collect();
    by_name.sort_by(|a, b| b.name.cmp(&a.name));

    println!("=== Results by Name (newest first) ===");
    println!(
        "     {:20}  {:>8}  {:>7}  {:>16}  {}",
        "Name", "Points", "Rating", "95% CI", placement_header_str
    );
    let width = 60 + placement_headers.len() * 4;
    println!("{:-<width$}", "");

    for c in &by_name {
        let sigma = c.rating.uncertainty;
        let low = c.rating.rating - 2.0 * sigma;
        let high = c.rating.rating + 2.0 * sigma;

        let placement_str: String =
            c.placement_counts
                .iter()
                .fold(String::new(), |mut acc, &count| {
                    use std::fmt::Write;
                    let _ = write!(acc, "{count:>4}");
                    acc
                });

        println!(
            "     {:20}  {:>8.1}  {:>7.1}  [{:>5.1}, {:>5.1}]  {}",
            c.name, c.swiss_points, c.rating.rating, low, high, placement_str
        );
    }
    println!();

    // Sort by Swiss points (descending), rating as tiebreaker
    println!("=== Results by Swiss Points ===");
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

    println!(
        " {:>2}  {:20}  {:>8}  {:>7}  {:>16}  {}",
        "#", "Name", "Points", "Rating", "95% CI", placement_header_str
    );
    let width = 60 + placement_headers.len() * 4;
    println!("{:-<width$}", "");

    for (rank, (_, c)) in ranked.iter().enumerate() {
        let sigma = c.rating.uncertainty;
        let low = c.rating.rating - 2.0 * sigma;
        let high = c.rating.rating + 2.0 * sigma;

        // Build placement counts string
        let placement_str: String =
            c.placement_counts
                .iter()
                .fold(String::new(), |mut acc, &count| {
                    use std::fmt::Write;
                    let _ = write!(acc, "{count:>4}");
                    acc
                });

        println!(
            " {:>2}  {:20}  {:>8.1}  {:>7.1}  [{:>5.1}, {:>5.1}]  {}",
            rank + 1,
            c.name,
            c.swiss_points,
            c.rating.rating,
            low,
            high,
            placement_str
        );
    }

    println!();
    println!("Note: Ranked by Swiss points. Rating 95% CI = rating ± 2×sigma");
}

/// Get ordinal suffix (1st, 2nd, 3rd, 4th, ...)
fn ordinal(n: usize) -> String {
    let suffix = match n % 10 {
        1 if n % 100 != 11 => "st",
        2 if n % 100 != 12 => "nd",
        3 if n % 100 != 13 => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

/// Extract training step from contestant name
fn extract_step(name: &str) -> Option<usize> {
    let step_part = name.rsplit('/').next().unwrap_or(name);
    step_part.strip_prefix("step_").and_then(|s| {
        let trimmed = s.trim_start_matches('0');
        if trimmed.is_empty() {
            Some(0)
        } else {
            trimmed.parse().ok()
        }
    })
}

/// Extract run identifier from display name
fn extract_run_id(name: &str) -> &str {
    // "A/.../step_001" -> "A", "step_001" -> ""
    if let Some(idx) = name.find("/.../") {
        &name[..idx]
    } else if name.starts_with("step_") {
        ""
    } else if let Some(idx) = name.rfind('/') {
        &name[..idx]
    } else {
        ""
    }
}

/// Format step number for axis labels (e.g., 10240 -> "10k")
fn format_step(step: usize) -> String {
    if step >= 1_000_000 {
        format!("{}M", step / 1_000_000)
    } else if step >= 1_000 {
        format!("{}k", step / 1_000)
    } else {
        step.to_string()
    }
}

/// Generate and display a rating graph for tournament contestants
fn generate_rating_graph(contestants: &[Contestant]) -> Result<()> {
    // Helper for x-axis label formatting (must be defined before use)
    #[expect(
        clippy::cast_sign_loss,
        reason = "x-axis values are non-negative training steps"
    )]
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "plotters API requires &f64 for label formatters"
    )]
    fn format_x_label(x: &f64) -> String {
        format_step(x.max(0.0) as usize)
    }

    // 1. Separate checkpoints from random player
    let (checkpoints, random): (Vec<_>, Vec<_>) = contestants
        .iter()
        .partition(|c| matches!(c.source, PlayerSource::Checkpoint(_)));

    if checkpoints.is_empty() {
        println!("No checkpoints to graph");
        return Ok(());
    }

    // 2. Extract data: (step, rating, lower, upper, run_id)
    let mut data: Vec<(usize, f64, f64, f64, String)> = checkpoints
        .iter()
        .filter_map(|c| {
            let step = extract_step(&c.name)?;
            let sigma = c.rating.uncertainty;
            let lower = c.rating.rating - 2.0 * sigma;
            let upper = c.rating.rating + 2.0 * sigma;
            let run = extract_run_id(&c.name).to_string();
            Some((step, c.rating.rating, lower, upper, run))
        })
        .collect();

    if data.is_empty() {
        println!("No checkpoints with parseable step numbers to graph");
        return Ok(());
    }

    // 3. Sort by step
    data.sort_by_key(|(step, ..)| *step);

    // 4. Group by run
    let mut runs: HashMap<String, Vec<(usize, f64, f64, f64)>> = HashMap::new();
    for (step, rating, lower, upper, run) in data {
        runs.entry(run)
            .or_default()
            .push((step, rating, lower, upper));
    }

    // 5. Calculate axis ranges
    let all_points: Vec<_> = runs.values().flatten().collect();
    let x_min = all_points.iter().map(|(s, ..)| *s).min().unwrap_or(0);
    let x_max = all_points.iter().map(|(s, ..)| *s).max().unwrap_or(1);
    let y_min = all_points
        .iter()
        .map(|(_, _, l, _)| *l)
        .fold(f64::MAX, f64::min);
    let y_max = all_points
        .iter()
        .map(|(_, _, _, u)| *u)
        .fold(f64::MIN, f64::max);

    // Include random baseline in y range
    let random_rating = random.first().map(|r| r.rating.rating);
    let y_min = random_rating.map_or(y_min, |r| y_min.min(r - 1.0));
    let y_max = random_rating.map_or(y_max, |r| y_max.max(r + 1.0));

    // Add padding to y range
    let y_range = y_max - y_min;
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;

    // 6. Create temp file
    let temp_path = std::env::temp_dir().join(format!(
        "tournament_rating_{}.png",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));

    // 7. Create chart
    let root = BitMapBackend::new(&temp_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Rating by Training Step", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min as f64..x_max as f64, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_label_formatter(&format_x_label)
        .y_label_formatter(&|y| format!("{y:.1}"))
        .x_desc("Training Step")
        .y_desc("Rating")
        .draw()?;

    // 8. Define colors for runs
    let colors = [BLUE, RED, GREEN, MAGENTA, CYAN];

    for (i, (run_name, points)) in runs.iter().enumerate() {
        let color = colors[i % colors.len()];

        // Sort points by step for this run
        let mut points = points.clone();
        points.sort_by_key(|(s, ..)| *s);

        // Draw confidence band (filled area)
        let upper_bound: Vec<_> = points.iter().map(|(s, _, _, u)| (*s as f64, *u)).collect();
        let lower_bound: Vec<_> = points.iter().map(|(s, _, l, _)| (*s as f64, *l)).collect();

        // Create polygon for filled area
        let mut polygon: Vec<(f64, f64)> = upper_bound.clone();
        polygon.extend(lower_bound.iter().rev());

        chart.draw_series(std::iter::once(Polygon::new(
            polygon,
            color.mix(0.2).filled(),
        )))?;

        // Draw rating line
        let rating_line: Vec<_> = points.iter().map(|(s, r, _, _)| (*s as f64, *r)).collect();
        let label = if run_name.is_empty() {
            "rating"
        } else {
            run_name
        };

        chart
            .draw_series(LineSeries::new(rating_line, color.stroke_width(2)))?
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
    }

    // 9. Draw random baseline if present
    if let Some(rr) = random_rating {
        chart
            .draw_series(LineSeries::new(
                vec![(x_min as f64, rr), (x_max as f64, rr)],
                BLACK.stroke_width(2),
            ))?
            .label("random")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(2)));
    }

    // 10. Draw legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;

    // 11. Print path and open
    println!("\nGraph saved to: {}", temp_path.display());

    #[cfg(target_os = "macos")]
    let _ = Command::new("open").arg(&temp_path).spawn();

    #[cfg(target_os = "linux")]
    let _ = Command::new("xdg-open").arg(&temp_path).spawn();

    #[cfg(target_os = "windows")]
    let _ = Command::new("cmd")
        .args(["/C", "start", ""])
        .arg(&temp_path)
        .spawn();

    Ok(())
}

/// Build tournament results for JSON export
fn build_results(
    contestants: &[Contestant],
    pods: &[(usize, MatchupGames)], // (round, result)
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
                placement_counts: c.placement_counts.clone(),
                draw_count: c.draw_count,
                games_played: c.games_played,
            }
        })
        .collect();

    let pod_summaries: Vec<PodSummary> = pods
        .iter()
        .map(|(round, result)| {
            let num_contestants = result.contestants.len();
            PodSummary {
                round: *round,
                contestants: result
                    .contestants
                    .iter()
                    .map(|&idx| contestants[idx].name.clone())
                    .collect(),
                avg_points: (0..num_contestants).map(|i| result.avg_points(i)).collect(),
                games: result.games.len(),
                draws: result.full_draws(),
            }
        })
        .collect();

    let format = if contestants.len() <= 8 {
        "round-robin"
    } else {
        "swiss"
    };

    TournamentResults {
        rankings,
        pods: pod_summaries,
        config: TournamentConfigSummary {
            num_games_per_matchup: args.num_games,
            num_rounds,
            format: format.to_string(),
            temp: args.temp,
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

/// Run games for a pod of N contestants
fn run_pod<B: Backend, E: Environment>(
    contestants: &[Contestant],
    models: &[ActorCritic<B>],
    normalizers: &[Option<ObsNormalizer>],
    contestant_to_model: &[Option<usize>], // None = Random
    pod: &[usize],
    num_games: usize,
    num_envs: usize,
    temp_schedule: &TempSchedule,
    rng: &mut StdRng,
    device: &B::Device,
) -> MatchupGames {
    use std::collections::HashMap;

    let num_players = E::NUM_PLAYERS;
    assert_eq!(
        pod.len(),
        num_players,
        "Pod size {} must match NUM_PLAYERS {}",
        pod.len(),
        num_players
    );

    // Check for Random players
    let has_random = pod.iter().any(|&idx| contestant_to_model[idx].is_none());

    if has_random {
        return run_pod_with_random::<E>(pod, num_games, rng);
    }

    // Build models array, deduplicating when same model plays multiple positions
    let mut pod_models: Vec<Option<ActorCritic<B>>> = Vec::new();
    let mut pod_normalizers: Vec<Option<ObsNormalizer>> = Vec::new();
    let mut checkpoint_to_model_map: Vec<usize> = Vec::new();
    let mut model_cache: HashMap<usize, usize> = HashMap::new();

    for &contestant_idx in pod {
        let model_idx = contestant_to_model[contestant_idx]
            .expect("contestant should have model (Random players handled separately)");
        if let Some(&cached_idx) = model_cache.get(&model_idx) {
            // Reuse existing model
            checkpoint_to_model_map.push(cached_idx);
        } else {
            // Add new model
            let new_idx = pod_models.len();
            pod_models.push(Some(models[model_idx].clone()));
            pod_normalizers.push(normalizers[model_idx].clone());
            model_cache.insert(model_idx, new_idx);
            checkpoint_to_model_map.push(new_idx);
        }
    }

    let names: Vec<String> = pod.iter().map(|&i| contestants[i].name.clone()).collect();

    // Run games via eval infrastructure
    let stats = run_stats_mode_env::<B, E>(
        &pod_models,
        &pod_normalizers,
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
        .map(|outcome| GamePlacement {
            placements: outcome.0.clone(),
        })
        .collect();

    MatchupGames {
        contestants: pod.to_vec(),
        games,
    }
}

/// Simplified pod games for when Random players are involved
fn run_pod_with_random<E: Environment>(
    pod: &[usize],
    num_games: usize,
    rng: &mut StdRng,
) -> MatchupGames {
    use rand::Rng;

    let num_players = E::NUM_PLAYERS;
    let mut games = Vec::with_capacity(num_games);

    for _ in 0..num_games {
        let mut env = E::new(rng.gen());

        loop {
            // Random valid action for all players
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

        // Get placements from game outcome (or default to all tied for 1st)
        let placements = env
            .game_outcome()
            .map_or_else(|| vec![1; num_players], |outcome| outcome.0);

        games.push(GamePlacement { placements });
    }

    MatchupGames {
        contestants: pod.to_vec(),
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

    // Temperature schedule - use environment default if not specified
    let temp_schedule = TempSchedule::new(
        args.temp.unwrap_or(E::DEFAULT_TEMP),
        args.temp_final.unwrap_or(0.0),
        args.temp_cutoff,
        false,
    );

    // RNG
    let seed = args.seed.unwrap_or_else(rand::random);
    let mut rng = StdRng::seed_from_u64(seed);

    // Weng-Lin config
    let wl_config = WengLinConfig::new();

    // Track all pod results for JSON output
    let mut all_pods: Vec<(usize, MatchupGames)> = Vec::new();

    // Progress bar setup
    let multi_progress = MultiProgress::new();

    let pod_size = E::NUM_PLAYERS;

    if use_swiss {
        // Swiss tournament
        let rounds_pb = multi_progress.add(ProgressBar::new(num_rounds as u64));
        rounds_pb.set_style(
            ProgressStyle::default_bar()
                .template("Round [{bar:20}] {pos}/{len} | {elapsed}/{duration} (ETA: {eta})")
                .expect("valid template")
                .progress_chars("=> "),
        );

        for round in 1..=num_rounds {
            // Handle byes for contestants that can't form complete pods
            let num_byes = contestants.len() % pod_size;
            let mut bye_recipients: Vec<usize> = Vec::new();

            if num_byes > 0 {
                // Find lowest-ranked players (by Swiss points, then rating) who haven't had a bye
                let mut bye_candidates: Vec<(usize, f64, f64)> = contestants
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.has_bye)
                    .map(|(i, c)| (i, c.swiss_points, c.rating.rating))
                    .collect();
                // Sort ascending by points, then ascending by rating (lowest gets bye)
                bye_candidates.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                });

                for (bye_idx, _, _) in bye_candidates.iter().take(num_byes) {
                    // Award bye: points equivalent to 1st place in a match
                    let bye_points = (pod_size - 1) as f64;
                    contestants[*bye_idx].swiss_points += bye_points;
                    contestants[*bye_idx].has_bye = true;
                    bye_recipients.push(*bye_idx);
                    multi_progress.suspend(|| {
                        println!(
                            "  {} receives bye (+{:.1} points)",
                            contestants[*bye_idx].name, bye_points
                        );
                    });
                }
            }

            // Create pods from active (non-bye) contestants
            let active_indices: Vec<usize> = (0..contestants.len())
                .filter(|i| !bye_recipients.contains(i))
                .collect();

            let active_contestants: Vec<Contestant> = active_indices
                .iter()
                .map(|&i| contestants[i].clone())
                .collect();
            let temp_pods = swiss_pods(&active_contestants, pod_size);
            // Map pod indices back to original contestant indices
            let pods: Vec<Vec<usize>> = temp_pods
                .into_iter()
                .map(|pod| pod.into_iter().map(|i| active_indices[i]).collect())
                .collect();

            if pods.is_empty() && bye_recipients.is_empty() {
                println!("  No pods possible");
                break;
            }

            let matchup_pb = multi_progress.add(ProgressBar::new(pods.len() as u64));
            matchup_pb.set_style(
                ProgressStyle::default_bar()
                    .template("  [{bar:30}] {pos}/{len} matchups")
                    .expect("valid template")
                    .progress_chars("=> "),
            );

            for pod in pods {
                let result = run_pod::<B, E>(
                    contestants,
                    &models,
                    &normalizers,
                    &contestant_to_model,
                    &pod,
                    args.num_games,
                    args.num_envs,
                    &temp_schedule,
                    &mut rng,
                    device,
                );

                // Suspend progress bars to print result
                multi_progress.suspend(|| {
                    let names: Vec<&str> =
                        pod.iter().map(|&i| contestants[i].name.as_str()).collect();
                    println!("  {}", result.summary(&names));
                });

                update_ratings_from_games(contestants, &result, &wl_config);
                all_pods.push((round, result));
                matchup_pb.inc(1);
            }

            matchup_pb.finish_and_clear();
            rounds_pb.inc(1);
            multi_progress.suspend(|| {
                print_standings(contestants, &format!("Standings after round {round}:"));
            });
        }
        rounds_pb.finish_and_clear();
    } else {
        // Round-robin
        let pods = round_robin_pods(n, pod_size);
        let total_pods = pods.len();

        println!("Running {total_pods} pod games (round-robin):");

        let pb = multi_progress.add(ProgressBar::new(total_pods as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40}] {pos}/{len} pods ({eta})")
                .expect("valid template")
                .progress_chars("=> "),
        );

        for pod in pods {
            let result = run_pod::<B, E>(
                contestants,
                &models,
                &normalizers,
                &contestant_to_model,
                &pod,
                args.num_games,
                args.num_envs,
                &temp_schedule,
                &mut rng,
                device,
            );

            // Suspend progress bar to print result
            multi_progress.suspend(|| {
                let names: Vec<&str> = pod.iter().map(|&i| contestants[i].name.as_str()).collect();
                println!("  {}", result.summary(&names));
            });

            update_ratings_from_games(contestants, &result, &wl_config);
            all_pods.push((1, result)); // Round 1 for all round-robin games
            pb.inc(1);
        }

        pb.finish_and_clear();
    }

    // Final summary
    print_rating_guide();
    print_final_summary(contestants, num_rounds, args.num_games);

    // Graph output if requested
    if args.graph {
        if let Err(e) = generate_rating_graph(contestants) {
            eprintln!("Failed to generate graph: {e}");
        }
    }

    // JSON output if requested
    if let Some(output_path) = &args.output {
        let results = build_results(contestants, &all_pods, num_rounds, args, &metadata.env_name);
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
    fn test_compute_display_names_unique() {
        // All names unique - common prefix "/" stripped, no common middle
        let paths = vec![
            PathBuf::from("/a/step_001"),
            PathBuf::from("/b/step_002"),
            PathBuf::from("/c/step_003"),
        ];
        let names = compute_display_names(&paths);
        // Common prefix "/" stripped, different parents so full relative paths shown
        assert_eq!(names, vec!["a/step_001", "b/step_002", "c/step_003"]);
    }

    #[test]
    fn test_compute_display_names_duplicate() {
        // Same checkpoint name, common "checkpoints" folder collapsed to "..."
        let paths = vec![
            PathBuf::from("/runs/Q/checkpoints/step_001"),
            PathBuf::from("/runs/Z/checkpoints/step_001"),
        ];
        let names = compute_display_names(&paths);
        // Common prefix "/runs" stripped, common middle "checkpoints" collapsed
        assert_eq!(names, vec!["Q/.../step_001", "Z/.../step_001"]);
    }

    #[test]
    fn test_compute_display_names_partial_duplicates() {
        // Different path lengths - no common middle when aligned from end
        let paths = vec![
            PathBuf::from("/a/step_001"),
            PathBuf::from("/b/Q/step_002"),
            PathBuf::from("/b/Z/step_002"),
        ];
        let names = compute_display_names(&paths);
        // Common prefix "/" stripped, no common middle (different at -2 position)
        assert_eq!(names, vec!["a/step_001", "b/Q/step_002", "b/Z/step_002"]);
    }

    #[test]
    fn test_compute_display_names_deep() {
        // Common "Q/checkpoints" middle collapsed to "..."
        let paths = vec![
            PathBuf::from("/runs/A/Q/checkpoints/step_001"),
            PathBuf::from("/runs/B/Q/checkpoints/step_001"),
        ];
        let names = compute_display_names(&paths);
        // Common prefix "/runs" stripped, common middle "Q/checkpoints" collapsed
        assert_eq!(names, vec!["A/.../step_001", "B/.../step_001"]);
    }

    #[test]
    fn test_compute_display_names_single() {
        let paths = vec![PathBuf::from("/a/b/step_001")];
        let names = compute_display_names(&paths);
        assert_eq!(names, vec!["step_001"]);
    }

    #[test]
    fn test_compute_display_names_empty() {
        let paths: Vec<PathBuf> = vec![];
        let names = compute_display_names(&paths);
        assert!(names.is_empty());
    }

    #[test]
    fn test_compute_display_names_common_prefix_stripped() {
        // All paths share "runs/" prefix
        let paths = vec![
            PathBuf::from("runs/a/step_001"),
            PathBuf::from("runs/b/step_002"),
        ];
        let names = compute_display_names(&paths);
        assert_eq!(names, vec!["a/step_001", "b/step_002"]);
    }

    #[test]
    fn test_compute_display_names_common_middle_collapsed() {
        // Common "checkpoints" folder in middle
        let paths = vec![
            PathBuf::from("phase1/checkpoints/step_001"),
            PathBuf::from("phase2/checkpoints/step_002"),
        ];
        let names = compute_display_names(&paths);
        assert_eq!(names, vec!["phase1/.../step_001", "phase2/.../step_002"]);
    }

    #[test]
    fn test_compute_display_names_prefix_and_middle_combined() {
        let paths = vec![
            PathBuf::from("runs/phase1/checkpoints/step_001"),
            PathBuf::from("runs/phase2/checkpoints/step_002"),
        ];
        let names = compute_display_names(&paths);
        assert_eq!(names, vec!["phase1/.../step_001", "phase2/.../step_002"]);
    }

    #[test]
    fn test_compute_display_names_multiple_middle_runs() {
        // Multiple non-adjacent common sections
        let paths = vec![PathBuf::from("a/X/b/Y/c"), PathBuf::from("a/X/d/Y/e")];
        // Common prefix "a/X/" stripped first
        // Then Y is common middle -> collapsed to "..."
        let names = compute_display_names(&paths);
        assert_eq!(names, vec!["b/.../c", "d/.../e"]);
    }

    #[test]
    fn test_compute_display_names_no_common_parts() {
        let paths = vec![PathBuf::from("a/b/c"), PathBuf::from("x/y/z")];
        let names = compute_display_names(&paths);
        assert_eq!(names, vec!["a/b/c", "x/y/z"]);
    }

    #[test]
    fn test_compute_display_names_different_length_paths() {
        // Align from end for common middle detection
        let paths = vec![
            PathBuf::from("runs/phase1/checkpoints/step_001"),
            PathBuf::from("phase2/checkpoints/step_002"),
        ];
        let names = compute_display_names(&paths);
        // "checkpoints" is common at -2 from end
        assert_eq!(
            names,
            vec!["runs/phase1/.../step_001", "phase2/.../step_002"]
        );
    }

    #[test]
    fn test_compute_display_names_identical_paths() {
        // All paths identical - should still show filename, not empty string
        let paths = vec![
            PathBuf::from("/runs/checkpoints/step_001"),
            PathBuf::from("/runs/checkpoints/step_001"),
            PathBuf::from("/runs/checkpoints/step_001"),
        ];
        let names = compute_display_names(&paths);
        assert_eq!(names, vec!["step_001", "step_001", "step_001"]);
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
    fn test_matchup_games_creation() {
        let result = MatchupGames {
            contestants: vec![0, 1],
            games: vec![
                GamePlacement {
                    placements: vec![1, 2],
                }, // P0 wins
                GamePlacement {
                    placements: vec![2, 1],
                }, // P1 wins
                GamePlacement {
                    placements: vec![1, 1],
                }, // Draw
            ],
        };

        assert_eq!(result.contestants, vec![0, 1]);
        assert_eq!(result.games.len(), 3);
        // Check avg points
        // P0: win (1pt) + loss (0pt) + draw (0.5pt) = 1.5 / 3 = 0.5
        assert!((result.avg_points(0) - 0.5).abs() < 0.001);
        assert!((result.avg_points(1) - 0.5).abs() < 0.001);
        assert_eq!(result.full_draws(), 1);
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
            placement_counts: vec![10, 2, 1], // 10 1st, 2 2nd, 1 3rd
            draw_count: 1,
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
            placement_counts: vec![5, 5], // 5 wins, 5 losses
            draw_count: 0,
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
            pods: vec![],
            config: TournamentConfigSummary {
                num_games_per_matchup: 10,
                num_rounds: 3,
                format: "swiss".to_string(),
                temp: Some(1.0),
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
            temp: Some(0.5),
            seed: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("\"seed\""));
    }

    #[test]
    fn test_pod_summary_serialization() {
        let summary = PodSummary {
            round: 1,
            contestants: vec!["Player1".to_string(), "Player2".to_string()],
            avg_points: vec![2.0, 1.0],
            games: 5,
            draws: 0,
        };

        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"round\":1"));
        assert!(json.contains("\"contestants\""));
        assert!(json.contains("Player1"));
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
    fn test_round_robin_pods_2player() {
        let pods = round_robin_pods(4, 2);
        assert_eq!(pods.len(), 6); // C(4,2) = 6
        assert!(pods.contains(&vec![0, 1]));
        assert!(pods.contains(&vec![0, 2]));
        assert!(pods.contains(&vec![0, 3]));
        assert!(pods.contains(&vec![1, 2]));
        assert!(pods.contains(&vec![1, 3]));
        assert!(pods.contains(&vec![2, 3]));
    }

    #[test]
    fn test_round_robin_pods_4player() {
        let pods = round_robin_pods(5, 4);
        assert_eq!(pods.len(), 5); // C(5,4) = 5
                                   // Each pod should have 4 unique contestants
        for pod in &pods {
            assert_eq!(pod.len(), 4);
        }
    }

    #[test]
    fn test_swiss_pods_2player() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("D".to_string(), PlayerSource::Random, 0.0),
        ];

        let pods = swiss_pods(&contestants, 2);
        // Should pair all 4: 2 pods
        assert_eq!(pods.len(), 2);

        // Each contestant should appear exactly once
        let mut seen = [false; 4];
        for pod in &pods {
            for &idx in pod {
                assert!(!seen[idx]);
                seen[idx] = true;
            }
        }
    }

    #[test]
    fn test_swiss_pods_4player() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("D".to_string(), PlayerSource::Random, 0.0),
        ];

        let pods = swiss_pods(&contestants, 4);
        // Should make 1 pod with all 4
        assert_eq!(pods.len(), 1);
        assert_eq!(pods[0].len(), 4);
    }

    #[test]
    fn test_swiss_pods_with_different_ratings() {
        let mut contestants = vec![
            Contestant::new("High".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("Low".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("Medium".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("MidHigh".to_string(), PlayerSource::Random, 0.0),
        ];

        // Modify ratings
        contestants[0].rating.rating = 30.0;
        contestants[1].rating.rating = 15.0;
        contestants[2].rating.rating = 25.0;
        contestants[3].rating.rating = 28.0;

        let pods = swiss_pods(&contestants, 2);
        // With 4 contestants, should get 2 pods
        assert_eq!(pods.len(), 2);
    }

    #[test]
    fn test_swiss_pods_prefers_unfaced_opponents() {
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("D".to_string(), PlayerSource::Random, 0.0),
        ];

        // A has already faced B
        contestants[0].opponents_faced.push(1);
        contestants[1].opponents_faced.push(0);

        let pods = swiss_pods(&contestants, 2);

        // A should be paired with C or D (not B since they've already faced)
        let a_pod = pods.iter().find(|pod| pod.contains(&0));
        if let Some(pod) = a_pod {
            let opponent = if pod[0] == 0 { pod[1] } else { pod[0] };
            assert!(opponent == 2 || opponent == 3); // C or D, not B
        }
    }

    #[test]
    fn test_swiss_pods_odd_number() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("C".to_string(), PlayerSource::Random, 0.0),
        ];

        let pods = swiss_pods(&contestants, 2);
        // With 3 contestants, only 1 pod (one gets a bye)
        assert_eq!(pods.len(), 1);
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

        // Create MatchupGames with 3 games where Winner (idx 0) wins all
        let matchup_games = MatchupGames {
            contestants: vec![0, 1],
            games: vec![
                GamePlacement {
                    placements: vec![1, 2],
                },
                GamePlacement {
                    placements: vec![1, 2],
                },
                GamePlacement {
                    placements: vec![1, 2],
                },
            ],
        };
        let pods = vec![(1, matchup_games)];

        let args = TournamentArgs {
            sources: vec![],
            backend: None,
            num_games: 3,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: Some(1),
            output: None,
            seed: Some(42),
            random: false,
            graph: false,
        };

        let results = build_results(&contestants, &pods, 1, &args, "connect_four");

        assert_eq!(results.rankings.len(), 2);
        assert_eq!(results.rankings[0].name, "Winner"); // Higher rating is first
        assert_eq!(results.rankings[0].rank, 1);
        assert_eq!(results.rankings[1].name, "Loser");
        assert_eq!(results.rankings[1].rank, 2);
        assert_eq!(results.pods.len(), 1);
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
            backend: None,
            num_games: 5,
            num_envs: 1,
            temp: Some(0.5),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
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
            backend: None,
            num_games: 5,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: Some(3),
            output: None,
            seed: None,
            random: false,
            graph: false,
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
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
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
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: true, // Add random player
            graph: false,
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
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None, // No limit
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
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
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: Some(3), // Limit to 3
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
        };

        let contestants = discover_contestants(&args).unwrap();

        assert_eq!(contestants.len(), 3);
    }

    #[test]
    fn test_discover_contestants_invalid_path() {
        use crate::config::TournamentArgs;

        let args = TournamentArgs {
            sources: vec![PathBuf::from("/nonexistent/path/to/checkpoint")],
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
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
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
        };

        // Empty dir isn't a valid checkpoint or checkpoints dir
        let result = discover_contestants(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_best_checkpoint_with_symlink() {
        let temp = TempDir::new().unwrap();

        // Create checkpoints with metadata files
        for step in [100, 200, 300] {
            let ckpt_dir = temp.path().join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            let metadata = format!(
                r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":10.0,"training_uncertainty":8.333}}"#
            );
            std::fs::write(ckpt_dir.join("metadata.json"), metadata).unwrap();
        }

        // Create "best" symlink pointing to step_200
        std::os::unix::fs::symlink("step_200", temp.path().join("best")).unwrap();

        let best = get_best_checkpoint(temp.path()).unwrap();
        // Should use the "best" symlink (step_200)
        assert!(best.ends_with("step_200"));
    }

    #[test]
    fn test_get_best_checkpoint_fallback_to_rating() {
        let temp = TempDir::new().unwrap();

        // Create checkpoints with metadata files containing different training_ratings
        // No "best" symlink - should fall back to highest rating
        for (step, rating) in [(100, 10.0), (200, 30.0), (300, 20.0)] {
            let ckpt_dir = temp.path().join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            let metadata = format!(
                r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":{rating},"training_uncertainty":8.333}}"#
            );
            std::fs::write(ckpt_dir.join("metadata.json"), metadata).unwrap();
        }

        let best = get_best_checkpoint(temp.path()).unwrap();
        // Should fall back to highest training_rating (step_200 with rating 30.0)
        assert!(best.ends_with("step_200"));
    }

    #[test]
    fn test_get_best_checkpoint_no_metadata() {
        let temp = TempDir::new().unwrap();

        // Create checkpoints without metadata files and no "best" symlink
        std::fs::create_dir(temp.path().join("step_100")).unwrap();
        std::fs::create_dir(temp.path().join("step_200")).unwrap();

        // Should fall back to highest training_rating (which defaults to 0.0)
        // With equal ratings, max_by picks last evaluated, which is step_200
        let best = get_best_checkpoint(temp.path()).unwrap();
        assert!(best.ends_with("step_200"));
    }

    #[test]
    fn test_discover_contestants_limit_split_between_folders() {
        use crate::config::TournamentArgs;

        let temp1 = TempDir::new().unwrap();
        let temp2 = TempDir::new().unwrap();

        // Create 5 checkpoints in each folder with metadata
        for i in 0..5 {
            let ckpt1 = temp1.path().join(format!("step_{}", i * 100));
            let ckpt2 = temp2.path().join(format!("step_{}", i * 100));
            std::fs::create_dir(&ckpt1).unwrap();
            std::fs::create_dir(&ckpt2).unwrap();
            let step = i * 100;
            let metadata = format!(
                r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":10.0,"training_uncertainty":8.333}}"#
            );
            std::fs::write(ckpt1.join("metadata.json"), &metadata).unwrap();
            std::fs::write(ckpt2.join("metadata.json"), &metadata).unwrap();
        }

        // Create "best" symlinks pointing to step_200 (not latest)
        std::os::unix::fs::symlink("step_200", temp1.path().join("best")).unwrap();
        std::os::unix::fs::symlink("step_200", temp2.path().join("best")).unwrap();

        let args = TournamentArgs {
            sources: vec![temp1.path().to_path_buf(), temp2.path().to_path_buf()],
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: Some(4), // 4 total = 2 per folder (best + latest from each)
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
        };

        let contestants = discover_contestants(&args).unwrap();

        // Should get 4 total: best (step_200) + latest (step_400) from each folder
        assert_eq!(contestants.len(), 4);
    }

    #[test]
    fn test_discover_contestants_limit_1_per_folder_picks_best() {
        use crate::config::TournamentArgs;

        let temp1 = TempDir::new().unwrap();
        let temp2 = TempDir::new().unwrap();

        // Create checkpoints with metadata - different best in each folder
        for (step, rating) in [(100, 10.0), (200, 30.0), (300, 20.0)] {
            let ckpt_dir = temp1.path().join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            let metadata = format!(
                r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":{rating},"training_uncertainty":8.333}}"#
            );
            std::fs::write(ckpt_dir.join("metadata.json"), metadata).unwrap();
        }

        for (step, rating) in [(100, 5.0), (200, 15.0), (300, 25.0)] {
            let ckpt_dir = temp2.path().join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            let metadata = format!(
                r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":{rating},"training_uncertainty":8.333}}"#
            );
            std::fs::write(ckpt_dir.join("metadata.json"), metadata).unwrap();
        }

        // Verify get_best_checkpoint works for each folder independently
        // (no "best" symlink, so falls back to training_rating)
        let best1 = get_best_checkpoint(temp1.path()).unwrap();
        let best2 = get_best_checkpoint(temp2.path()).unwrap();
        assert!(
            best1.ends_with("step_200"),
            "Expected best1 to be step_200, got: {}",
            best1.display()
        );
        assert!(
            best2.ends_with("step_300"),
            "Expected best2 to be step_300, got: {}",
            best2.display()
        );

        let args = TournamentArgs {
            sources: vec![temp1.path().to_path_buf(), temp2.path().to_path_buf()],
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: Some(2), // 2 total = 1 per folder, should pick best-rated
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
        };

        let contestants = discover_contestants(&args).unwrap();

        // Should get 2 total (1 best from each folder)
        assert_eq!(contestants.len(), 2);

        // First should be step_200 from temp1 (rating 30.0)
        // Second should be step_300 from temp2 (rating 25.0)
        let paths: Vec<_> = contestants
            .iter()
            .filter_map(|c| match &c.source {
                PlayerSource::Checkpoint(p) => Some(p.clone()),
                _ => None,
            })
            .collect();

        // Contestants are shuffled when not single_training_run, so check unordered
        let path_strs: Vec<String> = paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        assert!(
            path_strs.iter().any(|p| p.contains("step_200")),
            "Expected one path to contain step_200, got: {path_strs:?}"
        );
        assert!(
            path_strs.iter().any(|p| p.contains("step_300")),
            "Expected one path to contain step_300, got: {path_strs:?}"
        );
    }

    #[test]
    fn test_run_pod_with_random_produces_results() {
        use crate::envs::connect_four::ConnectFour;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);
        let result = run_pod_with_random::<ConnectFour>(&[0, 1], 10, &mut rng);

        assert_eq!(result.contestants, vec![0, 1]);
        // Total games should equal num_games
        assert_eq!(result.games.len(), 10);
        // Each game should have proper placements for 2 players
        for game in &result.games {
            assert_eq!(game.placements.len(), 2);
        }
    }

    #[test]
    fn test_run_pod_with_random_deterministic() {
        use crate::envs::connect_four::ConnectFour;
        use rand::SeedableRng;

        // Same seed should produce same results
        let mut rng1 = StdRng::seed_from_u64(12345);
        let result1 = run_pod_with_random::<ConnectFour>(&[0, 1], 5, &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(12345);
        let result2 = run_pod_with_random::<ConnectFour>(&[0, 1], 5, &mut rng2);

        // Check avg_points are equal (deterministic results)
        assert!((result1.avg_points(0) - result2.avg_points(0)).abs() < 0.001);
        assert!((result1.avg_points(1) - result2.avg_points(1)).abs() < 0.001);
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
            backend: None,
            num_games: 8,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
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
    fn test_run_pod_with_random_cartpole() {
        use crate::envs::cartpole::CartPole;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);
        // CartPole is single player, so pod size is 1
        let result = run_pod_with_random::<CartPole>(&[0], 5, &mut rng);

        // CartPole is single player, so outcomes are based on episode length/reward
        assert_eq!(result.contestants, vec![0]);
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
    fn test_build_results_pod_summaries() {
        use crate::config::TournamentArgs;

        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random, 0.0),
            Contestant::new("B".to_string(), PlayerSource::Random, 0.0),
        ];

        let pods = vec![
            (
                1,
                MatchupGames {
                    contestants: vec![0, 1],
                    games: vec![
                        GamePlacement {
                            placements: vec![1, 2],
                        },
                        GamePlacement {
                            placements: vec![1, 2],
                        },
                        GamePlacement {
                            placements: vec![2, 1],
                        },
                    ],
                },
            ),
            (
                2,
                MatchupGames {
                    contestants: vec![0, 1],
                    games: vec![
                        GamePlacement {
                            placements: vec![2, 1],
                        },
                        GamePlacement {
                            placements: vec![2, 1],
                        },
                        GamePlacement {
                            placements: vec![1, 1],
                        }, // Draw
                        GamePlacement {
                            placements: vec![1, 2],
                        },
                    ],
                },
            ),
        ];

        let args = TournamentArgs {
            sources: vec![],
            backend: None,
            num_games: 3,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: Some(2),
            output: None,
            seed: None,
            random: false,
            graph: false,
        };

        let results = build_results(&contestants, &pods, 2, &args, "test");

        assert_eq!(results.pods.len(), 2);
        assert_eq!(results.pods[0].round, 1);
        assert_eq!(results.pods[1].round, 2);
        assert_eq!(results.pods[1].draws, 1);
    }

    #[test]
    fn test_swiss_pods_all_faced() {
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

        let pods = swiss_pods(&contestants, 2);
        // Should still produce pods even when all faced
        assert_eq!(pods.len(), 2);
    }

    #[test]
    fn test_contestant_debug_impl() {
        let contestant = Contestant::new("Test".to_string(), PlayerSource::Random, 0.0);
        let debug_str = format!("{contestant:?}");
        assert!(debug_str.contains("Test"));
        assert!(debug_str.contains("Random"));
    }

    #[test]
    fn test_matchup_games_clone() {
        let result = MatchupGames {
            contestants: vec![0, 1],
            games: vec![
                GamePlacement {
                    placements: vec![1, 2],
                },
                GamePlacement {
                    placements: vec![2, 1],
                },
            ],
        };

        let cloned = result.clone();
        assert_eq!(cloned.contestants, vec![0, 1]);
        assert_eq!(cloned.games.len(), 2);
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
            placement_counts: vec![0, 0],
            draw_count: 0,
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
            pods: vec![],
            config: TournamentConfigSummary {
                num_games_per_matchup: 5,
                num_rounds: 1,
                format: "round-robin".to_string(),
                temp: Some(1.0),
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
        // 8 players all in same score bracket, pod_size=4 -> 2 pods with Dutch-style mixing
        let mut contestants: Vec<Contestant> = (0..8)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        // All players in same score bracket (same swiss_points)
        // This should use Dutch-style pairing within the bracket
        for c in &mut contestants {
            c.swiss_points = 5.0; // Same points for all
        }

        let pods = swiss_pods(&contestants, 4);
        assert_eq!(pods.len(), 2);

        // Each pod should have 4 players
        assert_eq!(pods[0].len(), 4);
        assert_eq!(pods[1].len(), 4);

        // Dutch-style: divide into 4 groups (quarters), one from each group per pod
        // With 8 players sorted by rating: [7,6,5,4,3,2,1,0] (highest rating first)
        // Groups: G0=[7,6], G1=[5,4], G2=[3,2], G3=[1,0]
        // Pod 0: [7,5,3,1], Pod 1: [6,4,2,0]
        // All pods should be complete
        let all_players: std::collections::HashSet<_> = pods.iter().flatten().copied().collect();
        assert_eq!(all_players.len(), 8);
    }

    #[test]
    fn test_swiss_pods_floaters_across_brackets() {
        // Test that floaters from higher score brackets carry down to lower brackets
        let mut contestants: Vec<Contestant> = (0..6)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        // Set up score brackets:
        // - Bracket 1 (3 pts): players 0, 1, 2 (3 players - can't form pod of 4)
        // - Bracket 2 (0 pts): players 3, 4, 5 (3 players - can't form pod of 4)
        // Combined: 6 players, should form 1 pod with 2 floaters
        contestants[0].swiss_points = 3.0;
        contestants[1].swiss_points = 3.0;
        contestants[2].swiss_points = 3.0;
        contestants[3].swiss_points = 0.0;
        contestants[4].swiss_points = 0.0;
        contestants[5].swiss_points = 0.0;

        let pods = swiss_pods(&contestants, 4);

        // 6 players / 4 = 1 complete pod, 2 floaters
        assert_eq!(pods.len(), 1);
        assert_eq!(pods[0].len(), 4);

        // The pod should contain players from both brackets (floaters + residents)
        let pod_set: std::collections::HashSet<_> = pods[0].iter().copied().collect();
        // Higher-bracket players should have priority (floaters pair first)
        assert!(pod_set.contains(&0) || pod_set.contains(&1) || pod_set.contains(&2));
    }

    #[test]
    fn test_has_repeat_opponents() {
        let mut contestants: Vec<Contestant> = (0..4)
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random, f64::from(i)))
            .collect();

        // No opponents faced yet
        assert!(!has_repeat_opponents(&[0, 1, 2, 3], &contestants));

        // Player 0 has faced player 1
        contestants[0].opponents_faced.push(1);
        contestants[1].opponents_faced.push(0);
        assert!(has_repeat_opponents(&[0, 1, 2, 3], &contestants));

        // Pod without the repeat
        assert!(!has_repeat_opponents(&[0, 2, 3], &contestants));
        assert!(!has_repeat_opponents(&[2, 3], &contestants));
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
    fn test_swiss_pods_dutch_style_round_1_2player() {
        // Round 1: all swiss_points == 0, should use Dutch-style pairing
        // Sort by initial_seed (descending), pair top half vs bottom half
        let contestants = vec![
            Contestant::new("Seed3".to_string(), PlayerSource::Random, 3.0),
            Contestant::new("Seed1".to_string(), PlayerSource::Random, 1.0),
            Contestant::new("Seed4".to_string(), PlayerSource::Random, 4.0),
            Contestant::new("Seed2".to_string(), PlayerSource::Random, 2.0),
        ];

        let pods = swiss_pods(&contestants, 2);
        assert_eq!(pods.len(), 2);

        // Each pod should have 2 contestants
        for pod in &pods {
            assert_eq!(pod.len(), 2);
        }
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
    fn test_swiss_pods_subsequent_rounds() {
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

        let pods = swiss_pods(&contestants, 2);
        assert_eq!(pods.len(), 2);

        // A (highest points) should be in a pod
        let a_pod = pods.iter().any(|pod| pod.contains(&0));
        assert!(a_pod);
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

    #[test]
    fn test_checkpoint_deduplication() {
        // Test that duplicate paths are removed from checkpoint_data
        let mut checkpoint_data: Vec<(PathBuf, f64)> = vec![
            (PathBuf::from("/runs/a/checkpoints/step_001"), 25.0),
            (PathBuf::from("/runs/b/checkpoints/step_001"), 26.0),
            (PathBuf::from("/runs/a/checkpoints/step_001"), 27.0), // duplicate of first
            (PathBuf::from("/runs/c/checkpoints/step_001"), 28.0),
            (PathBuf::from("/runs/b/checkpoints/step_001"), 29.0), // duplicate of second
        ];

        // Apply deduplication logic (same as in discover_contestants)
        let mut seen_paths: HashSet<PathBuf> = HashSet::new();
        let original_count = checkpoint_data.len();
        checkpoint_data.retain(|(path, _)| seen_paths.insert(path.clone()));

        // Should remove 2 duplicates
        assert_eq!(checkpoint_data.len(), 3);
        assert_eq!(original_count - checkpoint_data.len(), 2);

        // Verify remaining paths are unique
        let paths: Vec<&PathBuf> = checkpoint_data.iter().map(|(p, _)| p).collect();
        assert_eq!(
            paths,
            vec![
                &PathBuf::from("/runs/a/checkpoints/step_001"),
                &PathBuf::from("/runs/b/checkpoints/step_001"),
                &PathBuf::from("/runs/c/checkpoints/step_001"),
            ]
        );

        // Verify first occurrence's seed is kept
        assert_eq!(checkpoint_data[0].1, 25.0); // first a
        assert_eq!(checkpoint_data[1].1, 26.0); // first b
        assert_eq!(checkpoint_data[2].1, 28.0); // c
    }

    #[test]
    fn test_checkpoint_deduplication_no_duplicates() {
        // Test that no paths are removed when there are no duplicates
        let mut checkpoint_data: Vec<(PathBuf, f64)> = vec![
            (PathBuf::from("/runs/a/step_001"), 25.0),
            (PathBuf::from("/runs/b/step_001"), 26.0),
            (PathBuf::from("/runs/c/step_001"), 27.0),
        ];

        let mut seen_paths: HashSet<PathBuf> = HashSet::new();
        let original_count = checkpoint_data.len();
        checkpoint_data.retain(|(path, _)| seen_paths.insert(path.clone()));

        assert_eq!(checkpoint_data.len(), original_count);
        assert_eq!(checkpoint_data.len(), 3);
    }

    #[test]
    fn test_checkpoint_deduplication_all_same() {
        // Test that all duplicates except first are removed
        let mut checkpoint_data: Vec<(PathBuf, f64)> = vec![
            (PathBuf::from("/runs/a/step_001"), 25.0),
            (PathBuf::from("/runs/a/step_001"), 26.0),
            (PathBuf::from("/runs/a/step_001"), 27.0),
        ];

        let mut seen_paths: HashSet<PathBuf> = HashSet::new();
        checkpoint_data.retain(|(path, _)| seen_paths.insert(path.clone()));

        assert_eq!(checkpoint_data.len(), 1);
        assert_eq!(checkpoint_data[0].1, 25.0); // First occurrence kept
    }

    #[test]
    fn test_is_run_dir() {
        let temp = TempDir::new().unwrap();

        // Not a run dir (no checkpoints subdirectory)
        assert!(!is_run_dir(temp.path()));

        // Create checkpoints subdirectory
        std::fs::create_dir(temp.path().join("checkpoints")).unwrap();
        assert!(is_run_dir(temp.path()));
    }

    #[test]
    fn test_select_checkpoints_with_priority_limit_1() {
        let temp = TempDir::new().unwrap();

        // Create checkpoints with metadata
        for step in [100, 200, 300] {
            let ckpt_dir = temp.path().join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            std::fs::write(
                ckpt_dir.join("metadata.json"),
                format!(r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":10.0,"training_uncertainty":8.333}}"#),
            ).unwrap();
        }

        // Create "best" symlink pointing to step_200
        std::os::unix::fs::symlink("step_200", temp.path().join("best")).unwrap();

        let checkpoints = enumerate_checkpoints(temp.path()).unwrap();
        let selected = select_checkpoints_with_priority(temp.path(), &checkpoints, 1);

        assert_eq!(selected.len(), 1);
        assert!(selected[0].ends_with("step_200")); // best
    }

    #[test]
    fn test_select_checkpoints_with_priority_limit_2() {
        let temp = TempDir::new().unwrap();

        // Create checkpoints with metadata
        for step in [100, 200, 300] {
            let ckpt_dir = temp.path().join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            std::fs::write(
                ckpt_dir.join("metadata.json"),
                format!(r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":10.0,"training_uncertainty":8.333}}"#),
            ).unwrap();
        }

        // Create "best" symlink pointing to step_200
        std::os::unix::fs::symlink("step_200", temp.path().join("best")).unwrap();

        let checkpoints = enumerate_checkpoints(temp.path()).unwrap();
        let selected = select_checkpoints_with_priority(temp.path(), &checkpoints, 2);

        assert_eq!(selected.len(), 2);
        // First should be best (step_200), second should be latest (step_300)
        assert!(selected[0].ends_with("step_200")); // best
        assert!(selected[1].ends_with("step_300")); // latest
    }

    #[test]
    fn test_select_checkpoints_with_priority_limit_2_best_is_latest() {
        let temp = TempDir::new().unwrap();

        // Create checkpoints with metadata
        for step in [100, 200, 300] {
            let ckpt_dir = temp.path().join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            std::fs::write(
                ckpt_dir.join("metadata.json"),
                format!(r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":10.0,"training_uncertainty":8.333}}"#),
            ).unwrap();
        }

        // Create "best" symlink pointing to step_300 (same as latest)
        std::os::unix::fs::symlink("step_300", temp.path().join("best")).unwrap();

        let checkpoints = enumerate_checkpoints(temp.path()).unwrap();
        let selected = select_checkpoints_with_priority(temp.path(), &checkpoints, 2);

        // When best == latest, should only get 1 checkpoint (deduplicated)
        assert_eq!(selected.len(), 1);
        assert!(selected[0].ends_with("step_300"));
    }

    #[test]
    fn test_select_checkpoints_with_priority_limit_3plus() {
        let temp = TempDir::new().unwrap();

        // Create 5 checkpoints with metadata
        for step in [100, 200, 300, 400, 500] {
            let ckpt_dir = temp.path().join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            std::fs::write(
                ckpt_dir.join("metadata.json"),
                format!(r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":10.0,"training_uncertainty":8.333}}"#),
            ).unwrap();
        }

        // Create "best" symlink pointing to step_200
        std::os::unix::fs::symlink("step_200", temp.path().join("best")).unwrap();

        let checkpoints = enumerate_checkpoints(temp.path()).unwrap();
        let selected = select_checkpoints_with_priority(temp.path(), &checkpoints, 4);

        assert_eq!(selected.len(), 4);
        // First should be best (step_200), second should be latest (step_500)
        assert!(selected[0].ends_with("step_200")); // best
        assert!(selected[1].ends_with("step_500")); // latest
                                                    // Remaining 2 should be evenly distributed from [100, 300, 400]
    }

    #[test]
    fn test_run_dir_path_resolution() {
        use crate::config::TournamentArgs;

        let temp = TempDir::new().unwrap();

        // Create run structure: temp/checkpoints/step_*
        let checkpoints_dir = temp.path().join("checkpoints");
        std::fs::create_dir(&checkpoints_dir).unwrap();

        for step in [100, 200] {
            let ckpt_dir = checkpoints_dir.join(format!("step_{step}"));
            std::fs::create_dir(&ckpt_dir).unwrap();
            std::fs::write(
                ckpt_dir.join("metadata.json"),
                format!(r#"{{"step":{step},"avg_return":0.0,"rng_seed":42,"obs_dim":4,"action_count":2,"num_players":1,"hidden_size":64,"num_hidden":2,"split_networks":false,"activation":"tanh","env_name":"test","training_rating":10.0,"training_uncertainty":8.333}}"#),
            ).unwrap();
        }

        // Pass the run directory (not checkpoints directory)
        let args = TournamentArgs {
            sources: vec![temp.path().to_path_buf()], // Note: NOT temp/checkpoints
            backend: None,
            num_games: 1,
            num_envs: 1,
            temp: Some(1.0),
            temp_final: None,
            temp_cutoff: None,
            limit: None,
            rounds: None,
            output: None,
            seed: None,
            random: false,
            graph: false,
        };

        let contestants = discover_contestants(&args).unwrap();
        // Should auto-detect run directory and find checkpoints
        assert_eq!(contestants.len(), 2);
    }
}
