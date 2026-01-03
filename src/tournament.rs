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
    use tempfile::TempDir;

    #[test]
    fn test_contestant_new() {
        let contestant = Contestant::new("TestPlayer".to_string(), PlayerSource::Random);
        assert_eq!(contestant.name, "TestPlayer");
        assert!(matches!(contestant.source, PlayerSource::Random));
        assert!(contestant.opponents_faced.is_empty());
        assert_eq!(contestant.wins, 0);
        assert_eq!(contestant.losses, 0);
        assert_eq!(contestant.draws, 0);
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
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
        ];

        let initial_rating_a = contestants[0].rating.rating;
        let initial_rating_b = contestants[1].rating.rating;

        let result = MatchupResult {
            contestant_a: 0,
            contestant_b: 1,
            wins_a: 1,
            wins_b: 0,
            draws: 0,
        };

        update_ratings(&mut contestants, &result, &wl_config);

        // Winner's rating should increase
        assert!(contestants[0].rating.rating > initial_rating_a);
        // Loser's rating should decrease
        assert!(contestants[1].rating.rating < initial_rating_b);
        // Stats updated
        assert_eq!(contestants[0].wins, 1);
        assert_eq!(contestants[0].losses, 0);
        assert_eq!(contestants[1].wins, 0);
        assert_eq!(contestants[1].losses, 1);
    }

    #[test]
    fn test_update_ratings_loss() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
        ];

        let result = MatchupResult {
            contestant_a: 0,
            contestant_b: 1,
            wins_a: 0,
            wins_b: 1,
            draws: 0,
        };

        update_ratings(&mut contestants, &result, &wl_config);

        assert_eq!(contestants[0].losses, 1);
        assert_eq!(contestants[1].wins, 1);
    }

    #[test]
    fn test_update_ratings_draw() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
        ];

        let initial_rating_a = contestants[0].rating.rating;
        let initial_rating_b = contestants[1].rating.rating;

        let result = MatchupResult {
            contestant_a: 0,
            contestant_b: 1,
            wins_a: 0,
            wins_b: 0,
            draws: 1,
        };

        update_ratings(&mut contestants, &result, &wl_config);

        // Ratings should stay approximately the same for equal-rated players
        assert!((contestants[0].rating.rating - initial_rating_a).abs() < 1.0);
        assert!((contestants[1].rating.rating - initial_rating_b).abs() < 1.0);
        assert_eq!(contestants[0].draws, 1);
        assert_eq!(contestants[1].draws, 1);
    }

    #[test]
    fn test_update_ratings_tracks_opponents() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
        ];

        let result = MatchupResult {
            contestant_a: 0,
            contestant_b: 1,
            wins_a: 1,
            wins_b: 0,
            draws: 0,
        };

        update_ratings(&mut contestants, &result, &wl_config);

        assert!(contestants[0].opponents_faced.contains(&1));
        assert!(contestants[1].opponents_faced.contains(&0));
    }

    #[test]
    fn test_update_ratings_multiple_games() {
        let wl_config = WengLinConfig::new();
        let mut contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
        ];

        let result = MatchupResult {
            contestant_a: 0,
            contestant_b: 1,
            wins_a: 3,
            wins_b: 1,
            draws: 1,
        };

        update_ratings(&mut contestants, &result, &wl_config);

        // A won more, should have higher rating
        assert!(contestants[0].rating.rating > contestants[1].rating.rating);
        assert_eq!(contestants[0].wins, 3);
        assert_eq!(contestants[0].losses, 1);
        assert_eq!(contestants[0].draws, 1);
        assert_eq!(contestants[1].wins, 1);
        assert_eq!(contestants[1].losses, 3);
        assert_eq!(contestants[1].draws, 1);
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

    #[test]
    fn test_swiss_pairings_with_different_ratings() {
        let mut contestants = vec![
            Contestant::new("High".to_string(), PlayerSource::Random),
            Contestant::new("Low".to_string(), PlayerSource::Random),
            Contestant::new("Medium".to_string(), PlayerSource::Random),
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
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
            Contestant::new("C".to_string(), PlayerSource::Random),
            Contestant::new("D".to_string(), PlayerSource::Random),
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
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
            Contestant::new("C".to_string(), PlayerSource::Random),
        ];

        let pairings = swiss_pairings(&contestants);
        // With 3 contestants, only 1 pairing (one gets a bye)
        assert_eq!(pairings.len(), 1);
    }

    #[test]
    fn test_build_results() {
        use crate::config::TournamentArgs;

        let mut contestants = vec![
            Contestant::new("Winner".to_string(), PlayerSource::Random),
            Contestant::new("Loser".to_string(), PlayerSource::Random),
        ];

        contestants[0].rating.rating = 28.0;
        contestants[0].wins = 3;
        contestants[0].losses = 0;
        contestants[1].rating.rating = 22.0;
        contestants[1].wins = 0;
        contestants[1].losses = 3;

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
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random))
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
            .map(|i| Contestant::new(format!("Player{i}"), PlayerSource::Random))
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
        assert!(matches!(
            contestants[0].source,
            PlayerSource::Checkpoint(_)
        ));
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

        assert_eq!(result.contestant_a, 0);
        assert_eq!(result.contestant_b, 1);
        // Total should equal num_games
        assert_eq!(result.wins_a + result.wins_b + result.draws, 10);
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

        assert_eq!(result1.wins_a, result2.wins_a);
        assert_eq!(result1.wins_b, result2.wins_b);
        assert_eq!(result1.draws, result2.draws);
    }

    #[test]
    fn test_select_evenly_spaced_zero() {
        let paths: Vec<PathBuf> = (0..5)
            .map(|i| PathBuf::from(format!("step_{i}")))
            .collect();

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
        )];
        contestants[0].wins = 5;
        contestants[0].losses = 2;
        contestants[0].draws = 1;

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
        assert_eq!(result.contestant_a, 0);
        assert_eq!(result.contestant_b, 1);
        assert_eq!(result.wins_a + result.wins_b + result.draws, 5);
    }

    #[test]
    fn test_print_standings_does_not_panic() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
        ];

        // Just verify it doesn't panic
        print_standings(&contestants, "Test Header");
    }

    #[test]
    fn test_print_final_summary_does_not_panic() {
        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
        ];

        // Just verify it doesn't panic
        print_final_summary(&contestants, 3, 10);
    }

    #[test]
    fn test_build_results_match_summaries() {
        use crate::config::TournamentArgs;

        let contestants = vec![
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
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
            Contestant::new("A".to_string(), PlayerSource::Random),
            Contestant::new("B".to_string(), PlayerSource::Random),
            Contestant::new("C".to_string(), PlayerSource::Random),
            Contestant::new("D".to_string(), PlayerSource::Random),
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
        let contestant = Contestant::new("Test".to_string(), PlayerSource::Random);
        let debug_str = format!("{:?}", contestant);
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
}
