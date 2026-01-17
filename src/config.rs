use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::schedule::Schedule;

/// Parse a duration string with unit suffix (e.g., "30s", "5m", "2h")
/// Supports: s (seconds), m (minutes), h (hours)
fn parse_duration(s: &str) -> Result<std::time::Duration> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("empty duration string");
    }

    let (num_str, unit) = if let Some(num) = s.strip_suffix('s') {
        (num, 's')
    } else if let Some(num) = s.strip_suffix('m') {
        (num, 'm')
    } else if let Some(num) = s.strip_suffix('h') {
        (num, 'h')
    } else {
        // Default to seconds if no unit
        (s, 's')
    };

    let value: u64 = num_str
        .parse()
        .with_context(|| format!("invalid duration number: {num_str}"))?;

    let secs = match unit {
        's' => value,
        'm' => value * 60,
        'h' => value * 3600,
        _ => unreachable!(),
    };

    Ok(std::time::Duration::from_secs(secs))
}

/// PPO training and evaluation for discrete games
#[derive(Parser, Debug)]
#[command(name = "burn-ppo", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Train a model (default if no subcommand given)
    Train(Box<TrainArgs>),
    /// Evaluate trained models
    Eval(EvalArgs),
    /// Run a tournament between checkpoints with skill ratings
    Tournament(TournamentArgs),
}

/// Arguments for training
#[derive(Parser, Debug)]
pub struct TrainArgs {
    /// Path to TOML config file
    #[arg(short, long, default_value = "configs/cartpole.toml")]
    pub config: PathBuf,

    /// Resume from existing run directory (same config, continue training)
    #[arg(long, conflicts_with = "fork")]
    pub resume: Option<PathBuf>,

    /// Fork from a checkpoint path (new run, allows config changes)
    #[arg(long, conflicts_with = "resume")]
    pub fork: Option<PathBuf>,

    #[arg(long, help = "Compute backend (default: ndarray)")]
    pub backend: Option<String>,

    // --- Overrides ---
    #[arg(long, help = "Environment name (default: from config file)")]
    pub env: Option<String>,

    #[arg(
        long,
        help = "Number of parallel environments (default: auto/CPU cores)"
    )]
    pub num_envs: Option<usize>,

    #[arg(
        long,
        help = "Learning rate - static value or schedule (e.g., '0.0003' or '0.0003@0,0.00003@30M')"
    )]
    pub learning_rate: Option<String>,

    #[arg(long, help = "Total training steps (default: 1000000)")]
    pub total_steps: Option<usize>,

    /// Maximum training time with unit suffix (e.g., "30s", "5m", "2h")
    /// Training stops early if this time is reached before `total_steps`
    #[arg(long)]
    pub max_training_time: Option<String>,

    /// Reload subprocess every N checkpoint saves to combat memory leaks.
    /// When set >0, training runs in a subprocess that restarts every N checkpoints.
    #[arg(long, default_value = "10")]
    pub reload_every_n_checkpoints: usize,

    /// [Internal] Elapsed time offset in milliseconds from parent process.
    /// Used for accurate progress bar ETA when running as reloaded subprocess.
    #[arg(long, hide = true, default_value = "0")]
    pub elapsed_time_offset_ms: u64,

    /// [Internal] Max checkpoints to save before exiting (for reload mode).
    /// When >0, training exits after saving this many checkpoints.
    #[arg(long, hide = true, default_value = "0")]
    pub max_checkpoints_this_run: usize,

    #[arg(long, help = "Random seed (default: 42)")]
    pub seed: Option<u64>,

    #[arg(long)]
    pub run_name: Option<String>,

    #[arg(long, help = "Activation function (default: tanh)")]
    pub activation: Option<String>,

    // --- Network ---
    /// Network architecture type: "mlp" or "cnn"
    #[arg(long)]
    pub network_type: Option<String>,

    /// Use separate actor and critic networks instead of shared backbone
    #[arg(long, action = clap::ArgAction::Set, help = "Use separate actor/critic networks (default: false)")]
    pub split_networks: Option<bool>,

    #[arg(long, help = "Hidden layer size (default: 64)")]
    pub hidden_size: Option<usize>,

    #[arg(long, help = "Number of hidden layers (default: 2)")]
    pub num_hidden: Option<usize>,

    // --- CNN Network Parameters ---
    /// Number of convolutional layers
    #[arg(long)]
    pub num_conv_layers: Option<usize>,

    /// Kernel size for all conv layers
    #[arg(long)]
    pub kernel_size: Option<usize>,

    /// FC hidden layer size after conv
    #[arg(long)]
    pub cnn_fc_hidden_size: Option<usize>,

    /// Number of FC layers after conv
    #[arg(long)]
    pub cnn_num_fc_layers: Option<usize>,

    // --- PPO Hyperparameters ---
    #[arg(long, help = "Steps per rollout (default: 128)")]
    pub num_steps: Option<usize>,

    /// Reward shaping coefficient for dense rewards (default: 0.0)
    #[arg(long)]
    pub reward_shaping_coef: Option<f32>,

    #[arg(long, help = "Discount factor (default: 0.99)")]
    pub gamma: Option<f64>,

    #[arg(long, help = "GAE lambda (default: 0.95)")]
    pub gae_lambda: Option<f64>,

    #[arg(long, help = "PPO clipping epsilon (default: 0.2)")]
    pub clip_epsilon: Option<f64>,

    #[arg(long, help = "Enable value clipping (default: false)")]
    pub clip_value: Option<bool>,

    #[arg(
        long,
        help = "Entropy coefficient - static value or schedule (e.g., '0.01' or '0.02@0,0.005@30M')"
    )]
    pub entropy_coef: Option<String>,

    #[arg(
        long,
        help = "Adaptive entropy target - static or schedule (e.g., '0.5' or '0.7@0,0.7@3M,0.2@30M'), or 'none' to disable. Ratio of max entropy."
    )]
    pub adaptive_entropy: Option<String>,

    #[arg(long, help = "Minimum entropy coefficient (default: 0.001)")]
    pub adaptive_entropy_min_coef: Option<f64>,

    #[arg(long, help = "Maximum entropy coefficient (default: 0.1)")]
    pub adaptive_entropy_max_coef: Option<f64>,

    #[arg(
        long,
        help = "Adjustment step size for adaptive entropy (default: 0.001)"
    )]
    pub adaptive_entropy_delta: Option<f64>,

    #[arg(long, help = "Value loss coefficient (default: 0.5)")]
    pub value_coef: Option<f64>,

    #[arg(long, help = "Max gradient norm for clipping (default: 0.5)")]
    pub max_grad_norm: Option<f64>,

    #[arg(
        long,
        help = "KL divergence threshold for early stopping (default: disabled)"
    )]
    pub target_kl: Option<f64>,

    #[arg(long, action = clap::ArgAction::Set, help = "Enable observation normalization (default: false)")]
    pub normalize_obs: Option<bool>,

    #[arg(long, action = clap::ArgAction::Set, help = "Enable return normalization (default: true)")]
    pub normalize_returns: Option<bool>,

    // --- Training ---
    #[arg(long, help = "PPO epochs per update (default: 4)")]
    pub num_epochs: Option<usize>,

    #[arg(long, help = "Number of minibatches per epoch (default: 4)")]
    pub num_minibatches: Option<usize>,

    #[arg(long, help = "Adam optimizer epsilon (default: 0.00001)")]
    pub adam_epsilon: Option<f64>,

    // --- Checkpointing/Logging ---
    #[arg(long, help = "Directory for run outputs (default: runs)")]
    pub run_dir: Option<PathBuf>,

    #[arg(long, help = "Checkpoint save frequency in steps (default: 10000)")]
    pub checkpoint_freq: Option<usize>,

    #[arg(long, help = "Logging frequency in steps (default: 1000)")]
    pub log_freq: Option<usize>,

    // --- Opponent Pool Training ---
    #[arg(
        long,
        help = "Fraction of envs for opponent games (0.0 = disabled, default: 0.25)"
    )]
    pub opponent_pool_fraction: Option<f32>,

    #[arg(
        long,
        help = "qi score learning rate for opponent sampling (default: 0.01)"
    )]
    pub qi_eta: Option<f64>,

    #[arg(long, help = "Print selected opponents during training and evaluation")]
    pub debug_opponents: bool,
}

impl TrainArgs {
    /// Generate CLI args for passthrough to subprocess
    ///
    /// Returns all override arguments that should be passed to a subprocess.
    /// Excludes args already handled specially by supervisor (`total_steps`,
    /// `max_training_time`, `seed`, `debug_opponents`) and meta args (config, resume, etc.)
    pub fn to_passthrough_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // Helper macro to reduce boilerplate
        macro_rules! push_opt {
            ($field:expr, $flag:literal) => {
                if let Some(v) = &$field {
                    args.push($flag.to_string());
                    args.push(v.to_string());
                }
            };
        }

        // Environment
        push_opt!(self.env, "--env");
        push_opt!(self.num_envs, "--num-envs");
        push_opt!(self.reward_shaping_coef, "--reward-shaping-coef");

        // Network
        push_opt!(self.network_type, "--network-type");
        push_opt!(self.split_networks, "--split-networks");
        push_opt!(self.hidden_size, "--hidden-size");
        push_opt!(self.num_hidden, "--num-hidden");
        push_opt!(self.activation, "--activation");
        push_opt!(self.num_conv_layers, "--num-conv-layers");
        push_opt!(self.kernel_size, "--kernel-size");
        push_opt!(self.cnn_fc_hidden_size, "--cnn-fc-hidden-size");
        push_opt!(self.cnn_num_fc_layers, "--cnn-num-fc-layers");

        // PPO hyperparameters
        // Schedule types are passed as strings (e.g., "0.0003" or "0.0003@0,0.00003@30M")
        push_opt!(self.learning_rate, "--learning-rate");
        push_opt!(self.gamma, "--gamma");
        push_opt!(self.gae_lambda, "--gae-lambda");
        push_opt!(self.clip_epsilon, "--clip-epsilon");
        push_opt!(self.clip_value, "--clip-value");
        push_opt!(self.entropy_coef, "--entropy-coef");
        push_opt!(self.value_coef, "--value-coef");
        push_opt!(self.max_grad_norm, "--max-grad-norm");
        push_opt!(self.target_kl, "--target-kl");
        push_opt!(self.num_steps, "--num-steps");

        // Adaptive entropy
        if let Some(ref schedule) = self.adaptive_entropy {
            args.push("--adaptive-entropy".to_string());
            args.push(schedule.clone());
        }
        push_opt!(
            self.adaptive_entropy_min_coef,
            "--adaptive-entropy-min-coef"
        );
        push_opt!(
            self.adaptive_entropy_max_coef,
            "--adaptive-entropy-max-coef"
        );
        push_opt!(self.adaptive_entropy_delta, "--adaptive-entropy-delta");

        // Normalization
        push_opt!(self.normalize_obs, "--normalize-obs");
        push_opt!(self.normalize_returns, "--normalize-returns");

        // Training
        push_opt!(self.num_epochs, "--num-epochs");
        push_opt!(self.num_minibatches, "--num-minibatches");
        push_opt!(self.adam_epsilon, "--adam-epsilon");

        // Checkpointing/Logging
        if let Some(v) = &self.run_dir {
            args.push("--run-dir".to_string());
            args.push(v.to_string_lossy().to_string());
        }
        push_opt!(self.checkpoint_freq, "--checkpoint-freq");
        push_opt!(self.log_freq, "--log-freq");

        // Opponent pool training
        push_opt!(self.opponent_pool_fraction, "--opponent-pool-fraction");
        push_opt!(self.qi_eta, "--qi-eta");

        // Note: These are handled specially by supervisor and excluded:
        // - total_steps, max_training_time, seed, debug_opponents
        // - config, resume, fork, run_name, backend
        // - reload_every_n_checkpoints, elapsed_time_offset_ms, max_checkpoints_this_run

        args
    }
}

/// Arguments for evaluation
#[derive(Parser, Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "CLI args naturally use bool flags"
)]
pub struct EvalArgs {
    /// Checkpoint paths (one per player, use with --human and --random for mixed games)
    #[arg(long = "checkpoint", short = 'c')]
    pub checkpoints: Vec<PathBuf>,

    #[arg(long, help = "Compute backend (default: ndarray)")]
    pub backend: Option<String>,

    /// Human players (specify name for each human player)
    /// Example: --human Alice --human Bob
    #[arg(long = "human")]
    pub humans: Vec<String>,

    #[arg(
        long = "random",
        help = "Add a random player as baseline (default: false)"
    )]
    pub random: bool,

    /// Environment name (required if no checkpoint provided)
    #[arg(long = "env", short = 'e')]
    pub env_name: Option<String>,

    /// Number of games to play
    #[arg(short = 'n', long, default_value = "100")]
    pub num_games: usize,

    /// Number of parallel environments for stats mode
    #[arg(long, default_value = "64")]
    pub num_envs: usize,

    #[arg(long, help = "Show per-step output, watch mode (default: false)")]
    pub watch: bool,

    #[arg(long, help = "Step mode: press Enter to advance (default: false)")]
    pub step: bool,

    #[arg(long, help = "Enable smooth animation (default: false)")]
    pub animate: bool,

    /// Frames per second for animation
    #[arg(long, default_value = "10")]
    pub fps: u32,

    #[arg(long, help = "Random seed for reproducibility (default: random)")]
    pub seed: Option<u64>,

    #[arg(long, help = "Initial softmax temperature (default: env default)")]
    pub temp: Option<f32>,

    #[arg(long, help = "Temperature after cutoff (default: 0.0)")]
    pub temp_final: Option<f32>,

    #[arg(long, help = "Move number to switch temperature (default: disabled)")]
    pub temp_cutoff: Option<usize>,

    #[arg(long, help = "Disable temperature cutoff (default: false)")]
    pub no_temp_cutoff: bool,

    #[arg(
        long,
        help = "Gradually decay temperature over cutoff (default: false)"
    )]
    pub temp_decay: bool,
}

/// Arguments for tournament mode
#[derive(Parser, Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "CLI flags are naturally bools"
)]
pub struct TournamentArgs {
    /// Checkpoint paths or run directories to include
    /// For run directories, all checkpoints are discovered automatically
    #[arg(required = true)]
    pub sources: Vec<PathBuf>,

    #[arg(long, help = "Compute backend (default: ndarray)")]
    pub backend: Option<String>,

    /// Number of games per matchup between contestants
    #[arg(short = 'n', long, default_value = "100")]
    pub num_games: usize,

    /// Number of parallel environments
    #[arg(long, default_value = "64")]
    pub num_envs: usize,

    #[arg(long, help = "Swiss rounds (default: auto = ceil(log2(n)) + 1)")]
    pub rounds: Option<usize>,

    #[arg(long, help = "Max checkpoints per run directory (default: unlimited)")]
    pub limit_per_run: Option<usize>,

    #[arg(long, help = "Include a random agent as baseline (default: false)")]
    pub random: bool,

    #[arg(long, help = "Initial softmax temperature (default: env default)")]
    pub temp: Option<f32>,

    #[arg(long, help = "Temperature after cutoff (default: 0.0)")]
    pub temp_final: Option<f32>,

    #[arg(long, help = "Move number to switch temperature (default: disabled)")]
    pub temp_cutoff: Option<usize>,

    #[arg(long, help = "Disable temperature cutoff (default: false)")]
    pub no_temp_cutoff: bool,

    #[arg(long, help = "Random seed for reproducibility (default: random)")]
    pub seed: Option<u64>,

    #[arg(short = 'o', long, help = "Save results to JSON file (default: none)")]
    pub output: Option<PathBuf>,

    #[arg(
        long,
        help = "Generate rating graph over training steps (default: false)"
    )]
    pub graph: bool,

    #[arg(
        long = "round-robin",
        help = "Force round-robin format (default: auto-select based on matchup count)"
    )]
    pub round_robin: bool,
}

/// Legacy alias for backward compatibility
pub type CliArgs = TrainArgs;

/// Number of parallel environments - either auto-detected or explicit
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NumEnvs {
    Auto(String), // "auto"
    Explicit(usize),
}

impl Default for NumEnvs {
    fn default() -> Self {
        Self::Auto("auto".to_string())
    }
}

impl NumEnvs {
    pub fn resolve(&self) -> usize {
        match self {
            // 1x CPU cores (not 2x) - no async rollout/training overlap
            Self::Auto(_) => num_cpus::get(),
            Self::Explicit(n) => *n,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "config flags naturally use bools"
)]
pub struct Config {
    // Environment (required - must be specified in TOML config)
    pub env: String,
    #[serde(default)]
    pub num_envs: NumEnvs,
    #[serde(default = "default_num_steps")]
    pub num_steps: usize,
    /// Reward shaping coefficient for dense rewards (default 0.0).
    /// Used differently by each environment:
    /// - Liar's Dice: per-round survival bonus.
    ///
    /// Set to 0.0 for pure zero-sum/sparse rewards.
    #[serde(default = "default_reward_shaping_coef")]
    pub reward_shaping_coef: f32,

    // PPO hyperparameters
    /// Learning rate - can be static value or schedule with milestones.
    /// Examples: 0.0003 (static), [[0.0003, 0], [0.00003, 30000000]] (schedule)
    #[serde(default = "default_learning_rate")]
    pub learning_rate: Schedule,
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_gae_lambda")]
    pub gae_lambda: f64,
    #[serde(default = "default_clip_epsilon")]
    pub clip_epsilon: f64,
    #[serde(default)]
    pub clip_value: bool,
    /// Entropy coefficient - can be static value or schedule with milestones.
    /// Examples: 0.01 (static), [[0.02, 0], [0.005, 30000000]] (schedule)
    #[serde(default = "default_entropy_coef")]
    pub entropy_coef: Schedule,

    // Adaptive entropy control (PID-inspired target tracking)
    /// Adaptive entropy target schedule (ratio of max entropy).
    /// When set, enables adaptive control which adjusts coefficient to hit target.
    /// Examples: 0.5 (static 50%), [[0.7, 0], [0.7, 3000000], [0.2, 30000000]] (schedule)
    /// Overrides `entropy_coef` when set.
    #[serde(default)]
    pub adaptive_entropy: Option<Schedule>,
    /// Minimum entropy coefficient. Default: 0.001
    #[serde(default = "default_adaptive_entropy_min_coef")]
    pub adaptive_entropy_min_coef: f64,
    /// Maximum entropy coefficient. Default: 0.1
    #[serde(default = "default_adaptive_entropy_max_coef")]
    pub adaptive_entropy_max_coef: f64,
    /// Adjustment step size for bang-bang control. Default: 0.001
    #[serde(default = "default_adaptive_entropy_delta")]
    pub adaptive_entropy_delta: f64,

    #[serde(default = "default_value_coef")]
    pub value_coef: f64,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,
    /// KL divergence threshold for early stopping (None = disabled)
    /// If `approx_kl` exceeds this during an epoch, stop the epoch early.
    /// Typical values: 0.01-0.03, recommended 0.015-0.02
    #[serde(default)]
    pub target_kl: Option<f64>,
    /// Whether to normalize observations using running mean/std
    /// Helps with environments that have varying observation scales
    #[serde(default)]
    pub normalize_obs: bool,
    /// Whether to normalize rewards using running statistics of discounted returns.
    /// Divides rewards by `sqrt(running_return_variance)` for stable training.
    /// **Default behavior**: ON for single-player, OFF for multiplayer.
    /// Set explicitly to `true` or `false` to override.
    #[serde(default)]
    pub normalize_returns: Option<bool>,
    /// Clipping range for normalized rewards (default: 10.0)
    /// Normalized rewards are clamped to [-clip, +clip]
    #[serde(default = "default_return_clip")]
    pub return_clip: f32,

    // Training
    #[serde(default = "default_total_steps")]
    pub total_steps: usize,
    #[serde(default = "default_num_epochs")]
    pub num_epochs: usize,
    #[serde(default = "default_num_minibatches")]
    pub num_minibatches: usize,
    #[serde(default = "default_adam_epsilon")]
    pub adam_epsilon: f64,
    /// Maximum training wall-clock time (e.g., "30s", "5m", "2h")
    /// Training stops when this time is reached OR `total_steps`, whichever first
    #[serde(default)]
    pub max_training_time: Option<String>,

    // Network
    /// Network architecture type: "mlp" (default) or "cnn"
    #[serde(default = "default_network_type")]
    pub network_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden")]
    pub num_hidden: usize,
    #[serde(default = "default_activation")]
    pub activation: String,
    /// Use separate actor and critic networks instead of shared backbone
    #[serde(default)]
    pub split_networks: bool,

    // CNN-specific parameters (ignored when network_type = "mlp")
    /// Number of convolutional layers (default: 2)
    #[serde(default = "default_num_conv_layers")]
    pub num_conv_layers: usize,
    /// Channels per conv layer (default: [8, 8])
    #[serde(default = "default_conv_channels")]
    pub conv_channels: Vec<usize>,
    /// Kernel size for all conv layers (default: 3)
    #[serde(default = "default_kernel_size")]
    pub kernel_size: usize,
    /// FC hidden layer size after conv (default: 32)
    #[serde(default = "default_cnn_fc_hidden_size")]
    pub cnn_fc_hidden_size: usize,
    /// Number of FC layers after conv (default: 1)
    #[serde(default = "default_cnn_num_fc_layers")]
    pub cnn_num_fc_layers: usize,

    // Checkpointing
    #[serde(default = "default_run_dir")]
    pub run_dir: PathBuf,
    #[serde(default = "default_checkpoint_freq")]
    pub checkpoint_freq: usize,

    // Logging
    #[serde(default = "default_log_freq")]
    pub log_freq: usize,

    // Opponent Pool Training (OpenAI Five-style historical opponent training)
    /// Fraction of environments dedicated to opponent games (0.0 = disabled, 0.0-1.0)
    /// When > 0, that fraction of training games are played against historical checkpoints
    #[serde(default = "default_opponent_pool_fraction")]
    pub opponent_pool_fraction: f32,
    /// qi score learning rate for opponent sampling
    #[serde(default = "default_qi_eta")]
    pub qi_eta: f64,
    /// Print selected opponents during training and evaluation
    #[serde(default)]
    pub debug_opponents: bool,

    // Experiment
    #[serde(default = "default_seed")]
    pub seed: u64,
    pub run_name: Option<String>,
    /// Parent run name if this run was forked from another
    #[serde(default)]
    pub forked_from: Option<String>,
}

// Default value functions
const fn default_num_steps() -> usize {
    128
}
const fn default_reward_shaping_coef() -> f32 {
    0.0 // Pure zero-sum by default
}
fn default_learning_rate() -> Schedule {
    Schedule::constant(2.5e-4)
}
const fn default_gamma() -> f64 {
    0.99
}
const fn default_gae_lambda() -> f64 {
    0.95
}
const fn default_clip_epsilon() -> f64 {
    0.2
}
fn default_entropy_coef() -> Schedule {
    Schedule::constant(0.01)
}

// Adaptive entropy defaults
const fn default_adaptive_entropy_min_coef() -> f64 {
    0.001
}
const fn default_adaptive_entropy_max_coef() -> f64 {
    0.1
}
const fn default_adaptive_entropy_delta() -> f64 {
    0.001 // Adjustment step size
}

const fn default_value_coef() -> f64 {
    0.5
}
const fn default_max_grad_norm() -> f64 {
    0.5
}
const fn default_return_clip() -> f32 {
    10.0 // Standard VecNormalize default
}
const fn default_total_steps() -> usize {
    1_000_000
}
const fn default_num_epochs() -> usize {
    4
}
const fn default_num_minibatches() -> usize {
    4
}
const fn default_adam_epsilon() -> f64 {
    1e-5
}
const fn default_hidden_size() -> usize {
    64
}
const fn default_num_hidden() -> usize {
    2
}
fn default_activation() -> String {
    "tanh".to_string()
}
fn default_network_type() -> String {
    "mlp".to_string()
}
const fn default_num_conv_layers() -> usize {
    2
}
fn default_conv_channels() -> Vec<usize> {
    vec![8, 8]
}
const fn default_kernel_size() -> usize {
    3
}
const fn default_cnn_fc_hidden_size() -> usize {
    32
}
const fn default_cnn_num_fc_layers() -> usize {
    1
}
fn default_run_dir() -> PathBuf {
    PathBuf::from("runs")
}
const fn default_checkpoint_freq() -> usize {
    10_000
}
const fn default_log_freq() -> usize {
    1_000
}
const fn default_seed() -> u64 {
    42
}

// Opponent pool defaults
const fn default_opponent_pool_fraction() -> f32 {
    0.25 // 25% of envs play against opponents (0.0 = disabled)
}
const fn default_qi_eta() -> f64 {
    0.01 // OpenAI Five default
}

impl Default for Config {
    /// Default config for testing purposes.
    /// Note: `env` must be specified in TOML configs for actual training.
    fn default() -> Self {
        Self {
            env: "cartpole".to_string(), // For tests only
            num_envs: NumEnvs::default(),
            num_steps: default_num_steps(),
            reward_shaping_coef: default_reward_shaping_coef(),
            learning_rate: default_learning_rate(),
            gamma: default_gamma(),
            gae_lambda: default_gae_lambda(),
            clip_epsilon: default_clip_epsilon(),
            clip_value: false,
            entropy_coef: default_entropy_coef(),
            adaptive_entropy: None,
            adaptive_entropy_min_coef: default_adaptive_entropy_min_coef(),
            adaptive_entropy_max_coef: default_adaptive_entropy_max_coef(),
            adaptive_entropy_delta: default_adaptive_entropy_delta(),
            value_coef: default_value_coef(),
            max_grad_norm: default_max_grad_norm(),
            target_kl: None,
            normalize_obs: false,
            normalize_returns: None, // Smart default: on for single-player, off for multiplayer
            return_clip: default_return_clip(),
            total_steps: default_total_steps(),
            num_epochs: default_num_epochs(),
            num_minibatches: default_num_minibatches(),
            adam_epsilon: default_adam_epsilon(),
            max_training_time: None,
            network_type: default_network_type(),
            hidden_size: default_hidden_size(),
            num_hidden: default_num_hidden(),
            activation: default_activation(),
            split_networks: false,
            num_conv_layers: default_num_conv_layers(),
            conv_channels: default_conv_channels(),
            kernel_size: default_kernel_size(),
            cnn_fc_hidden_size: default_cnn_fc_hidden_size(),
            cnn_num_fc_layers: default_cnn_num_fc_layers(),
            run_dir: default_run_dir(),
            checkpoint_freq: default_checkpoint_freq(),
            log_freq: default_log_freq(),
            opponent_pool_fraction: default_opponent_pool_fraction(),
            qi_eta: default_qi_eta(),
            debug_opponents: false,
            seed: default_seed(),
            run_name: None,
            forked_from: None,
        }
    }
}

impl Config {
    /// Load config from TOML file, apply CLI overrides
    ///
    /// The `forked_from` parameter is set when forking from another run
    pub fn load(args: &CliArgs, forked_from: Option<&str>) -> Result<Self> {
        // Load base config - fail if file doesn't exist
        let config_path = args.config.display();
        let content = fs::read_to_string(&args.config)
            .with_context(|| format!("Config file not found: {config_path}"))?;
        let mut config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config: {config_path}"))?;

        // Apply CLI overrides
        config.apply_cli_overrides(args);

        // Store forked_from relationship
        config.forked_from = forked_from.map(String::from);

        // For fork mode, always generate child name (ignore --run-name if specified)
        if forked_from.is_some() {
            if config.run_name.is_some() {
                eprintln!(
                    "Warning: --run-name is ignored when forking; using auto-generated child name"
                );
            }
            config.run_name = Some(generate_run_name(&config, forked_from));
        } else if config.run_name.is_none() {
            // Generate run name if not specified (fresh training)
            config.run_name = Some(generate_run_name(&config, None));
        }

        Ok(config)
    }

    /// Load config from a specific TOML file path
    pub fn load_from_path(path: &std::path::Path) -> Result<Self> {
        let path_display = path.display();
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {path_display}"))?;
        toml::from_str(&content).with_context(|| format!("Failed to parse config: {path_display}"))
    }

    /// Apply all CLI overrides to this config
    fn apply_cli_overrides(&mut self, args: &CliArgs) {
        // Environment
        if let Some(env) = &args.env {
            self.env.clone_from(env);
        }
        if let Some(n) = args.num_envs {
            self.num_envs = NumEnvs::Explicit(n);
        }
        if let Some(v) = args.reward_shaping_coef {
            self.reward_shaping_coef = v;
        }

        // PPO hyperparameters
        if let Some(ref lr) = args.learning_rate {
            match Schedule::parse_cli(lr) {
                Ok(s) => self.learning_rate = s,
                Err(e) => eprintln!("Warning: invalid --learning-rate '{lr}': {e}"),
            }
        }
        if let Some(v) = args.gamma {
            self.gamma = v;
        }
        if let Some(v) = args.gae_lambda {
            self.gae_lambda = v;
        }
        if let Some(v) = args.clip_epsilon {
            self.clip_epsilon = v;
        }
        if let Some(v) = args.clip_value {
            self.clip_value = v;
        }
        if let Some(ref v) = args.entropy_coef {
            match Schedule::parse_cli(v) {
                Ok(s) => self.entropy_coef = s,
                Err(e) => eprintln!("Warning: invalid --entropy-coef '{v}': {e}"),
            }
        }
        if let Some(ref v) = args.adaptive_entropy {
            let lower = v.trim().to_lowercase();
            if lower == "none" || lower == "off" || lower == "disabled" {
                self.adaptive_entropy = None;
            } else {
                match Schedule::parse_cli(v) {
                    Ok(s) => self.adaptive_entropy = Some(s),
                    Err(e) => eprintln!("Warning: invalid --adaptive-entropy '{v}': {e}"),
                }
            }
        }
        if let Some(v) = args.adaptive_entropy_min_coef {
            self.adaptive_entropy_min_coef = v;
        }
        if let Some(v) = args.adaptive_entropy_max_coef {
            self.adaptive_entropy_max_coef = v;
        }
        if let Some(v) = args.adaptive_entropy_delta {
            self.adaptive_entropy_delta = v;
        }
        if let Some(v) = args.value_coef {
            self.value_coef = v;
        }
        if let Some(v) = args.max_grad_norm {
            self.max_grad_norm = v;
        }
        if let Some(v) = args.target_kl {
            self.target_kl = Some(v);
        }
        if let Some(v) = args.normalize_obs {
            self.normalize_obs = v;
        }
        if let Some(v) = args.normalize_returns {
            self.normalize_returns = Some(v);
        }
        if let Some(v) = args.num_steps {
            self.num_steps = v;
        }

        // Training
        if let Some(ts) = args.total_steps {
            self.total_steps = ts;
        }
        if let Some(ref t) = args.max_training_time {
            self.max_training_time = Some(t.clone());
        }
        if let Some(v) = args.num_epochs {
            self.num_epochs = v;
        }
        if let Some(v) = args.num_minibatches {
            self.num_minibatches = v;
        }
        if let Some(v) = args.adam_epsilon {
            self.adam_epsilon = v;
        }

        // Network
        if let Some(network_type) = &args.network_type {
            self.network_type.clone_from(network_type);
        }
        if let Some(v) = args.hidden_size {
            self.hidden_size = v;
        }
        if let Some(v) = args.num_hidden {
            self.num_hidden = v;
        }
        if let Some(activation) = &args.activation {
            self.activation.clone_from(activation);
        }
        if let Some(v) = args.split_networks {
            self.split_networks = v;
        }
        // CNN parameters
        if let Some(v) = args.num_conv_layers {
            self.num_conv_layers = v;
        }
        if let Some(v) = args.kernel_size {
            self.kernel_size = v;
        }
        if let Some(v) = args.cnn_fc_hidden_size {
            self.cnn_fc_hidden_size = v;
        }
        if let Some(v) = args.cnn_num_fc_layers {
            self.cnn_num_fc_layers = v;
        }

        // Checkpointing/Logging
        if let Some(v) = &args.run_dir {
            self.run_dir.clone_from(v);
        }
        if let Some(v) = args.checkpoint_freq {
            self.checkpoint_freq = v;
        }
        if let Some(v) = args.log_freq {
            self.log_freq = v;
        }

        // Opponent pool training
        if let Some(v) = args.opponent_pool_fraction {
            self.opponent_pool_fraction = v;
        }
        if let Some(v) = args.qi_eta {
            self.qi_eta = v;
        }
        if args.debug_opponents {
            self.debug_opponents = true;
        }

        // Experiment
        if let Some(s) = args.seed {
            self.seed = s;
        }
        if let Some(name) = &args.run_name {
            self.run_name = Some(name.clone());
        }
    }

    /// Apply limited CLI overrides for resume mode
    ///
    /// When resuming, we only allow extending `total_steps` and setting `max_training_time`.
    /// Other parameters are locked to the original run config.
    pub fn apply_resume_overrides(&mut self, args: &CliArgs) {
        // Only allow extending training duration and setting time limit
        if let Some(ts) = args.total_steps {
            self.total_steps = ts;
        }
        if let Some(ref t) = args.max_training_time {
            self.max_training_time = Some(t.clone());
        }

        // Collect warnings for ignored overrides
        let mut ignored = Vec::new();

        // Environment
        if args.env.is_some() {
            ignored.push("--env");
        }
        if args.num_envs.is_some() {
            ignored.push("--num-envs");
        }

        // PPO hyperparameters
        if args.learning_rate.is_some() {
            ignored.push("--learning-rate");
        }
        if args.gamma.is_some() {
            ignored.push("--gamma");
        }
        if args.gae_lambda.is_some() {
            ignored.push("--gae-lambda");
        }
        if args.clip_epsilon.is_some() {
            ignored.push("--clip-epsilon");
        }
        if args.clip_value.is_some() {
            ignored.push("--clip-value");
        }
        if args.entropy_coef.is_some() {
            ignored.push("--entropy-coef");
        }
        if args.adaptive_entropy.is_some() {
            ignored.push("--adaptive-entropy");
        }
        if args.adaptive_entropy_min_coef.is_some() {
            ignored.push("--adaptive-entropy-min-coef");
        }
        if args.adaptive_entropy_max_coef.is_some() {
            ignored.push("--adaptive-entropy-max-coef");
        }
        if args.adaptive_entropy_delta.is_some() {
            ignored.push("--adaptive-entropy-delta");
        }
        if args.value_coef.is_some() {
            ignored.push("--value-coef");
        }
        if args.max_grad_norm.is_some() {
            ignored.push("--max-grad-norm");
        }
        if args.target_kl.is_some() {
            ignored.push("--target-kl");
        }
        if args.normalize_obs.is_some() {
            ignored.push("--normalize-obs");
        }
        if args.normalize_returns.is_some() {
            ignored.push("--normalize-returns");
        }
        if args.num_steps.is_some() {
            ignored.push("--num-steps");
        }

        // Training
        if args.num_epochs.is_some() {
            ignored.push("--num-epochs");
        }
        if args.num_minibatches.is_some() {
            ignored.push("--num-minibatches");
        }
        if args.adam_epsilon.is_some() {
            ignored.push("--adam-epsilon");
        }

        // Network
        if args.hidden_size.is_some() {
            ignored.push("--hidden-size");
        }
        if args.num_hidden.is_some() {
            ignored.push("--num-hidden");
        }
        if args.activation.is_some() {
            ignored.push("--activation");
        }
        if args.split_networks.is_some() {
            ignored.push("--split-networks");
        }

        // Checkpointing/Logging
        if args.run_dir.is_some() {
            ignored.push("--run-dir");
        }
        if args.checkpoint_freq.is_some() {
            ignored.push("--checkpoint-freq");
        }
        if args.log_freq.is_some() {
            ignored.push("--log-freq");
        }

        // Opponent pool training
        if args.opponent_pool_fraction.is_some() {
            ignored.push("--opponent-pool-fraction");
        }
        if args.qi_eta.is_some() {
            ignored.push("--qi-eta");
        }

        // Experiment
        if args.seed.is_some() {
            ignored.push("--seed");
        }
        if args.run_name.is_some() {
            ignored.push("--run-name");
        }

        // Print single warning if any flags were ignored
        if !ignored.is_empty() {
            eprintln!(
                "Warning: {} ignored when resuming (use --fork to change config)",
                ignored.join(", ")
            );
        }
    }

    /// Get max training time as a Duration (if set)
    pub fn max_training_duration(&self) -> Result<Option<std::time::Duration>> {
        self.max_training_time
            .as_ref()
            .map(|s| parse_duration(s))
            .transpose()
    }

    /// Get the full path to the run directory
    ///
    /// # Panics
    /// Panics if `run_name` is None (should be set during `load()`)
    pub fn run_path(&self) -> PathBuf {
        self.run_dir.join(
            self.run_name
                .as_ref()
                .expect("run_name should be set during Config::load()"),
        )
    }

    /// Get resolved number of environments
    pub fn num_envs(&self) -> usize {
        self.num_envs.resolve()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        use anyhow::bail;

        // Validate environment name
        if !["cartpole", "connect_four", "liars_dice"].contains(&self.env.as_str()) {
            bail!(
                "Unknown environment '{}'. Supported: cartpole, connect_four, liars_dice",
                self.env
            );
        }

        // Validate learning_rate schedule has positive initial value
        if self.learning_rate.initial_value() <= 0.0 {
            bail!(
                "learning_rate must be > 0 (initial value: {})",
                self.learning_rate.initial_value()
            );
        }
        if self.gamma < 0.0 || self.gamma > 1.0 {
            bail!("gamma must be in [0, 1]");
        }
        if self.clip_epsilon <= 0.0 {
            bail!("clip_epsilon must be > 0");
        }
        // Validate entropy_coef schedule has non-negative initial value
        if self.entropy_coef.initial_value() < 0.0 {
            bail!(
                "entropy_coef must be >= 0 (initial value: {})",
                self.entropy_coef.initial_value()
            );
        }

        // Validate adaptive entropy config
        if self.adaptive_entropy.is_some() {
            if self.adaptive_entropy_min_coef < 0.0 {
                bail!("adaptive_entropy_min_coef must be >= 0");
            }
            if self.adaptive_entropy_max_coef <= self.adaptive_entropy_min_coef {
                bail!("adaptive_entropy_max_coef must be > adaptive_entropy_min_coef");
            }
            if self.adaptive_entropy_delta <= 0.0 {
                bail!("adaptive_entropy_delta must be > 0");
            }
        }

        if self.num_epochs == 0 {
            bail!("num_epochs must be > 0");
        }
        if self.num_minibatches == 0 {
            bail!("num_minibatches must be > 0");
        }

        // Check minibatch size is reasonable
        let batch_size = self.num_steps * self.num_envs();
        let minibatch_size = batch_size / self.num_minibatches;
        if minibatch_size < 4 {
            bail!("minibatch_size {minibatch_size} too small, increase num_steps or num_envs");
        }

        // Validate activation function
        if !["tanh", "relu"].contains(&self.activation.as_str()) {
            bail!(
                "Unknown activation '{}'. Supported: tanh, relu",
                self.activation
            );
        }

        // Validate network type
        if !["mlp", "cnn"].contains(&self.network_type.as_str()) {
            bail!(
                "Unknown network_type '{}'. Supported: mlp, cnn",
                self.network_type
            );
        }

        // Validate CNN parameters
        if self.num_conv_layers == 0 {
            bail!("num_conv_layers must be > 0");
        }
        if self.kernel_size == 0 {
            bail!("kernel_size must be > 0");
        }
        if self.conv_channels.is_empty() {
            bail!("conv_channels must not be empty");
        }

        // Validate opponent pool config
        if self.opponent_pool_fraction < 0.0 || self.opponent_pool_fraction > 1.0 {
            bail!("opponent_pool_fraction must be in [0.0, 1.0]");
        }
        if self.opponent_pool_fraction > 0.0 && self.qi_eta <= 0.0 {
            bail!("qi_eta must be > 0 when opponent pool is enabled");
        }

        Ok(())
    }
}

/// Extract the global counter from a run name
///
/// Handles both standard names like "`cartpole_001`" and
/// child names like "`cartpole_003_child_001`"
fn extract_run_counter(name: &str) -> Option<u32> {
    let parts: Vec<&str> = name.split('_').collect();

    // Look for "_child_" pattern (child run)
    if let Some(child_idx) = parts.iter().position(|&p| p == "child") {
        // For child runs like "cartpole_003_child_001", return the parent counter (003)
        // The counter is just before "child"
        if child_idx >= 1 {
            return parts[child_idx - 1].parse().ok();
        }
        return None;
    }

    // Standard name: counter is the last part (e.g., "cartpole_001")
    parts.last()?.parse().ok()
}

/// Find the next available global counter by scanning the runs directory
fn find_next_global_counter(run_dir: &std::path::Path, env: &str) -> u32 {
    let mut max_counter: u32 = 0;

    if let Ok(entries) = std::fs::read_dir(run_dir) {
        for entry in entries.filter_map(Result::ok) {
            if let Some(name) = entry.file_name().to_str() {
                // Only consider runs for this environment
                if !name.starts_with(&format!("{env}_")) {
                    continue;
                }
                // Skip child runs when finding global counter
                if name.contains("_child_") {
                    continue;
                }
                if let Some(counter) = extract_run_counter(name) {
                    max_counter = max_counter.max(counter);
                }
            }
        }
    }

    max_counter + 1
}

/// Find the next available child counter for a specific parent run
fn find_next_child_counter(run_dir: &std::path::Path, parent_name: &str) -> u32 {
    let mut max_counter: u32 = 0;
    let prefix = format!("{parent_name}_child_");

    if let Ok(entries) = std::fs::read_dir(run_dir) {
        for entry in entries.filter_map(Result::ok) {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&prefix) {
                    // Extract child counter from end of name
                    let suffix = &name[prefix.len()..];
                    if let Ok(counter) = suffix.parse::<u32>() {
                        max_counter = max_counter.max(counter);
                    }
                }
            }
        }
    }

    max_counter + 1
}

/// Generate a unique run name with incrementing counter
///
/// Fresh runs: `{env}_{counter:03}` (e.g., `cartpole_001`)
/// Child runs: `{parent_name}_child_{counter:03}` (e.g., `cartpole_003_child_001`)
fn generate_run_name(config: &Config, forked_from: Option<&str>) -> String {
    if let Some(parent_name) = forked_from {
        let child_counter = find_next_child_counter(&config.run_dir, parent_name);
        format!("{parent_name}_child_{child_counter:03}")
    } else {
        let counter = find_next_global_counter(&config.run_dir, &config.env);
        format!("{}_{:03}", config.env, counter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.env, "cartpole");
        assert_eq!(config.learning_rate.initial_value(), 2.5e-4);
        assert_eq!(config.adam_epsilon, 1e-5);
    }

    #[test]
    fn test_num_envs_auto() {
        let num_envs = NumEnvs::Auto("auto".to_string());
        assert!(num_envs.resolve() >= 2);
    }

    #[test]
    fn test_num_envs_explicit() {
        let num_envs = NumEnvs::Explicit(64);
        assert_eq!(num_envs.resolve(), 64);
    }

    #[test]
    fn test_config_validate_success() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_lr() {
        let config = Config {
            learning_rate: Schedule::constant(-0.1),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_gamma() {
        let config = Config {
            gamma: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_extract_run_counter_standard() {
        assert_eq!(extract_run_counter("cartpole_001"), Some(1));
        assert_eq!(extract_run_counter("cartpole_042"), Some(42));
        assert_eq!(extract_run_counter("connect4_123"), Some(123));
    }

    #[test]
    fn test_extract_run_counter_child() {
        // Child runs should return the parent counter
        assert_eq!(extract_run_counter("cartpole_003_child_001"), Some(3));
        assert_eq!(extract_run_counter("cartpole_003_child_002"), Some(3));
    }

    #[test]
    fn test_extract_run_counter_invalid() {
        assert_eq!(extract_run_counter("my_experiment"), None);
        assert_eq!(extract_run_counter(""), None);
        assert_eq!(extract_run_counter("cartpole_abc"), None);
    }

    #[test]
    fn test_find_next_global_counter_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(find_next_global_counter(dir.path(), "cartpole"), 1);
    }

    #[test]
    fn test_find_next_global_counter_with_runs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("cartpole_001")).unwrap();
        std::fs::create_dir(dir.path().join("cartpole_005")).unwrap();
        assert_eq!(find_next_global_counter(dir.path(), "cartpole"), 6);
    }

    #[test]
    fn test_find_next_global_counter_ignores_children() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("cartpole_003")).unwrap();
        std::fs::create_dir(dir.path().join("cartpole_003_child_001")).unwrap();
        std::fs::create_dir(dir.path().join("cartpole_003_child_002")).unwrap();
        // Should return 4, ignoring child runs
        assert_eq!(find_next_global_counter(dir.path(), "cartpole"), 4);
    }

    #[test]
    fn test_find_next_global_counter_different_envs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("cartpole_005")).unwrap();
        std::fs::create_dir(dir.path().join("connect4_010")).unwrap();
        // Should only count cartpole runs
        assert_eq!(find_next_global_counter(dir.path(), "cartpole"), 6);
        assert_eq!(find_next_global_counter(dir.path(), "connect4"), 11);
    }

    #[test]
    fn test_find_next_child_counter_no_children() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("cartpole_003")).unwrap();
        assert_eq!(find_next_child_counter(dir.path(), "cartpole_003"), 1);
    }

    #[test]
    fn test_find_next_child_counter_with_children() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("cartpole_003")).unwrap();
        std::fs::create_dir(dir.path().join("cartpole_003_child_001")).unwrap();
        std::fs::create_dir(dir.path().join("cartpole_003_child_003")).unwrap();
        assert_eq!(find_next_child_counter(dir.path(), "cartpole_003"), 4);
    }

    #[test]
    fn test_generate_run_name_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config {
            env: "cartpole".to_string(),
            run_dir: dir.path().to_path_buf(),
            ..Config::default()
        };
        let name = generate_run_name(&config, None);
        assert_eq!(name, "cartpole_001");
    }

    #[test]
    fn test_generate_run_name_child() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("cartpole_003")).unwrap();
        let config = Config {
            env: "cartpole".to_string(),
            run_dir: dir.path().to_path_buf(),
            ..Config::default()
        };
        let name = generate_run_name(&config, Some("cartpole_003"));
        assert_eq!(name, "cartpole_003_child_001");
    }
}
