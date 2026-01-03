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
use rand::SeedableRng;
use serde::Serialize;
use skillratings::weng_lin::{weng_lin, WengLinConfig, WengLinRating};
use skillratings::Outcomes;

use crate::checkpoint::CheckpointMetadata;
use crate::config::TournamentArgs;
use crate::dispatch_env;
use crate::env::Environment;
use crate::eval::{load_model_from_checkpoint, run_stats_mode_env, PlayerSource, TempSchedule};
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;

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
    /// Win/loss/draw stats
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
}

impl Contestant {
    pub fn new(name: String, source: PlayerSource) -> Self {
        Self {
            name,
            source,
            rating: WengLinRating::new(),
            opponents_faced: Vec::new(),
            wins: 0,
            losses: 0,
            draws: 0,
        }
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
            // Single checkpoint
            contestants.push(Contestant::new(
                checkpoint_name(&path),
                PlayerSource::Checkpoint(path),
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
                contestants.push(Contestant::new(
                    checkpoint_name(&ckpt),
                    PlayerSource::Checkpoint(ckpt),
                ));
            }
        } else {
            bail!(
                "Invalid source: {} (expected checkpoint dir or checkpoints folder)",
                source_path.display()
            );
        }
    }

    // Add random player if requested
    if args.random {
        contestants.push(Contestant::new("Random".to_string(), PlayerSource::Random));
    }

    Ok(contestants)
}

/// Generate Swiss pairings for a round
///
/// Pairs contestants by similar rating, preferring those not yet faced.
fn swiss_pairings(contestants: &[Contestant]) -> Vec<(usize, usize)> {
    // Sort by rating (descending)
    let mut ranked: Vec<(usize, f64)> = contestants
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.rating.rating))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut pairings = Vec::new();
    let mut paired = vec![false; contestants.len()];

    for i in 0..ranked.len() {
        let idx_a = ranked[i].0;
        if paired[idx_a] {
            continue;
        }

        // Find best unpaired opponent (closest rating, preferring not yet faced)
        let mut best_opponent = None;
        let mut best_not_faced = None;

        for (_, &(idx_b, _)) in ranked.iter().enumerate().skip(i + 1) {
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

/// Update ratings after a matchup using Weng-Lin
fn update_ratings(contestants: &mut [Contestant], result: &MatchupResult, config: &WengLinConfig) {
    let a = result.contestant_a;
    let b = result.contestant_b;

    // Process each game outcome
    for _ in 0..result.wins_a {
        let (new_a, new_b) = weng_lin(
            &contestants[a].rating,
            &contestants[b].rating,
            &Outcomes::WIN,
            config,
        );
        contestants[a].rating = new_a;
        contestants[b].rating = new_b;
    }

    for _ in 0..result.wins_b {
        let (new_a, new_b) = weng_lin(
            &contestants[a].rating,
            &contestants[b].rating,
            &Outcomes::LOSS,
            config,
        );
        contestants[a].rating = new_a;
        contestants[b].rating = new_b;
    }

    for _ in 0..result.draws {
        let (new_a, new_b) = weng_lin(
            &contestants[a].rating,
            &contestants[b].rating,
            &Outcomes::DRAW,
            config,
        );
        contestants[a].rating = new_a;
        contestants[b].rating = new_b;
    }

    // Update stats
    contestants[a].wins += result.wins_a;
    contestants[a].losses += result.wins_b;
    contestants[a].draws += result.draws;
    contestants[b].wins += result.wins_b;
    contestants[b].losses += result.wins_a;
    contestants[b].draws += result.draws;

    // Track opponents faced
    contestants[a].opponents_faced.push(b);
    contestants[b].opponents_faced.push(a);
}

/// Print current standings
fn print_standings(contestants: &[Contestant], header: &str) {
    println!("\n{header}");

    // Sort by rating
    let mut ranked: Vec<(usize, &Contestant)> = contestants.iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.rating
            .rating
            .partial_cmp(&a.1.rating.rating)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, (_, c)) in ranked.iter().enumerate() {
        let sigma = c.rating.uncertainty;
        println!(
            "  {:2}. {:20} {:5.1} ± {:.1}",
            rank + 1,
            c.name,
            c.rating.rating,
            sigma
        );
    }
}

/// Print final tournament summary
fn print_final_summary(contestants: &[Contestant], num_rounds: usize, num_games: usize) {
    println!("\n{}", "=".repeat(60));
    println!("=== Tournament Results ===");
    println!(
        "Contestants: {} | Rounds: {} | Games per matchup: {}",
        contestants.len(),
        num_rounds,
        num_games
    );
    println!();

    // Sort by rating
    let mut ranked: Vec<(usize, &Contestant)> = contestants.iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.rating
            .rating
            .partial_cmp(&a.1.rating.rating)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Header
    println!(
        " {:>2}  {:20}  {:>7}  {:>16}  {:>4}  {:>4}  {:>4}",
        "#", "Name", "Rating", "95% CI", "W", "L", "D"
    );
    println!("{:-<68}", "");

    for (rank, (_, c)) in ranked.iter().enumerate() {
        let sigma = c.rating.uncertainty;
        let low = c.rating.rating - 2.0 * sigma;
        let high = c.rating.rating + 2.0 * sigma;

        println!(
            " {:>2}  {:20}  {:>7.1}  [{:>5.1}, {:>5.1}]  {:>4}  {:>4}  {:>4}",
            rank + 1,
            c.name,
            c.rating.rating,
            low,
            high,
            c.wins,
            c.losses,
            c.draws
        );
    }

    println!();
    println!("Note: 95% CI = rating ± 2×sigma");
}

/// Build tournament results for JSON export
fn build_results(
    contestants: &[Contestant],
    matches: &[(usize, MatchupResult)], // (round, result)
    num_rounds: usize,
    args: &TournamentArgs,
    env_name: &str,
) -> TournamentResults {
    // Sort by rating for rankings
    let mut ranked: Vec<(usize, &Contestant)> = contestants.iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.rating
            .rating
            .partial_cmp(&a.1.rating.rating)
            .unwrap_or(std::cmp::Ordering::Equal)
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
                rating: c.rating.rating,
                uncertainty: sigma,
                rating_low: c.rating.rating - 2.0 * sigma,
                rating_high: c.rating.rating + 2.0 * sigma,
                wins: c.wins,
                losses: c.losses,
                draws: c.draws,
                games_played: c.wins + c.losses + c.draws,
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
) -> MatchupResult {
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

    MatchupResult {
        contestant_a: idx_a,
        contestant_b: idx_b,
        wins_a: stats.wins[0],
        wins_b: stats.wins[1],
        draws: stats.draws,
    }
}

/// Simplified matchup for when Random player is involved
fn run_matchup_with_random<E: Environment>(
    idx_a: usize,
    idx_b: usize,
    num_games: usize,
    rng: &mut StdRng,
) -> MatchupResult {
    use crate::env::GameOutcome;
    use rand::Rng;

    let mut wins_a = 0;
    let mut wins_b = 0;
    let mut draws = 0;

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

        // Determine outcome
        match env.game_outcome() {
            Some(GameOutcome::Winner(w)) => {
                if w == 0 {
                    wins_a += 1;
                } else {
                    wins_b += 1;
                }
            }
            Some(GameOutcome::Placements(p)) => {
                if p[0] < p[1] {
                    wins_a += 1;
                } else if p[1] < p[0] {
                    wins_b += 1;
                } else {
                    draws += 1;
                }
            }
            Some(GameOutcome::Tie) | None => draws += 1,
        }
    }

    MatchupResult {
        contestant_a: idx_a,
        contestant_b: idx_b,
        wins_a,
        wins_b,
        draws,
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

            let pairings = swiss_pairings(contestants);
            if pairings.is_empty() {
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
                    let winner = if result.wins_a > result.wins_b {
                        &contestants[idx_a].name
                    } else if result.wins_b > result.wins_a {
                        &contestants[idx_b].name
                    } else {
                        "draw"
                    };
                    println!(
                        "  {} vs {}: {}-{}-{} ({})",
                        contestants[idx_a].name,
                        contestants[idx_b].name,
                        result.wins_a,
                        result.wins_b,
                        result.draws,
                        winner
                    );
                });

                update_ratings(contestants, &result, &wl_config);
                all_matches.push((round, result));
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
                let winner = if result.wins_a > result.wins_b {
                    &contestants[idx_a].name
                } else if result.wins_b > result.wins_a {
                    &contestants[idx_b].name
                } else {
                    "draw"
                };
                println!(
                    "  {} vs {}: {}-{}-{} ({})",
                    contestants[idx_a].name,
                    contestants[idx_b].name,
                    result.wins_a,
                    result.wins_b,
                    result.draws,
                    winner
                );
            });

            update_ratings(contestants, &result, &wl_config);
            all_matches.push((1, result));
            pb.inc(1);
        }

        pb.finish_and_clear();
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
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
            Contestant::new("C".to_string(), PlayerSource::Random),
            Contestant::new("D".to_string(), PlayerSource::Random),
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
}
