use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

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

    /// Compute backend (defaults to best available: cuda > libtorch > wgpu > ndarray)
    #[arg(long)]
    pub backend: Option<String>,

    // --- Overrides ---
    #[arg(long)]
    pub env: Option<String>,

    #[arg(long)]
    pub num_envs: Option<usize>,

    #[arg(long)]
    pub learning_rate: Option<f64>,

    #[arg(long)]
    pub total_timesteps: Option<usize>,

    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long)]
    pub run_name: Option<String>,

    #[arg(long)]
    pub activation: Option<String>,

    // --- Network ---
    /// Use separate actor and critic networks instead of shared backbone
    #[arg(long, action = clap::ArgAction::Set)]
    pub split_networks: Option<bool>,

    #[arg(long)]
    pub hidden_size: Option<usize>,

    #[arg(long)]
    pub num_hidden: Option<usize>,

    // --- PPO Hyperparameters ---
    #[arg(long)]
    pub num_steps: Option<usize>,

    /// Enable/disable learning rate annealing (use --lr-anneal=false to disable)
    #[arg(long)]
    pub lr_anneal: Option<bool>,

    #[arg(long)]
    pub gamma: Option<f64>,

    #[arg(long)]
    pub gae_lambda: Option<f64>,

    #[arg(long)]
    pub clip_epsilon: Option<f64>,

    /// Enable/disable value clipping (use --clip-value=false to disable)
    #[arg(long)]
    pub clip_value: Option<bool>,

    #[arg(long)]
    pub entropy_coef: Option<f64>,

    /// Enable entropy coefficient annealing
    #[arg(long, action = clap::ArgAction::Set)]
    pub entropy_anneal: Option<bool>,

    #[arg(long)]
    pub value_coef: Option<f64>,

    #[arg(long)]
    pub max_grad_norm: Option<f64>,

    /// KL divergence threshold for early stopping
    #[arg(long)]
    pub target_kl: Option<f64>,

    /// Enable observation normalization
    #[arg(long, action = clap::ArgAction::Set)]
    pub normalize_obs: Option<bool>,

    // --- Training ---
    #[arg(long)]
    pub num_epochs: Option<usize>,

    #[arg(long)]
    pub num_minibatches: Option<usize>,

    #[arg(long)]
    pub adam_epsilon: Option<f64>,

    // --- Checkpointing/Logging ---
    #[arg(long)]
    pub run_dir: Option<PathBuf>,

    #[arg(long)]
    pub checkpoint_freq: Option<usize>,

    #[arg(long)]
    pub log_freq: Option<usize>,

    // --- Challenger Evaluation ---
    /// Enable challenger-style evaluation for multiplayer games
    #[arg(long, action = clap::ArgAction::Set)]
    pub challenger_eval: Option<bool>,

    #[arg(long)]
    pub challenger_games: Option<usize>,

    #[arg(long)]
    pub challenger_threshold: Option<f64>,

    #[arg(long)]
    pub challenger_temp: Option<f32>,

    #[arg(long)]
    pub challenger_temp_final: Option<f32>,

    #[arg(long)]
    pub challenger_temp_cutoff: Option<usize>,

    /// Enable temperature decay instead of hard cutoff
    #[arg(long, action = clap::ArgAction::Set)]
    pub challenger_temp_decay: Option<bool>,
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

    /// Compute backend (defaults to best available: cuda > libtorch > wgpu > ndarray)
    #[arg(long)]
    pub backend: Option<String>,

    /// Human players (specify name for each human player)
    /// Example: --human Alice --human Bob
    #[arg(long = "human")]
    pub humans: Vec<String>,

    /// Add a random player (useful for baseline comparisons)
    #[arg(long = "random")]
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

    /// Show per-step output (watch mode)
    #[arg(long)]
    pub watch: bool,

    /// Step mode: press Enter to advance each move (implies --watch)
    #[arg(long)]
    pub step: bool,

    /// Enable smooth animation (in-place frame updates)
    #[arg(long)]
    pub animate: bool,

    /// Frames per second for animation (default: 10)
    #[arg(long, default_value = "10")]
    pub fps: u32,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Initial softmax temperature (uses environment default if not specified)
    #[arg(long)]
    pub temp: Option<f32>,

    /// Temperature after cutoff (requires --temp-cutoff)
    #[arg(long)]
    pub temp_final: Option<f32>,

    /// Move number to switch from initial to final temperature
    #[arg(long)]
    pub temp_cutoff: Option<usize>,

    /// Gradually decay temperature over cutoff moves (requires --temp-cutoff)
    #[arg(long)]
    pub temp_decay: bool,
}

/// Arguments for tournament mode
#[derive(Parser, Debug)]
pub struct TournamentArgs {
    /// Checkpoint paths or run directories to include
    /// For run directories, all checkpoints are discovered automatically
    #[arg(required = true)]
    pub sources: Vec<PathBuf>,

    /// Compute backend (defaults to best available: cuda > libtorch > wgpu > ndarray)
    #[arg(long)]
    pub backend: Option<String>,

    /// Number of games per matchup between contestants
    #[arg(short = 'n', long, default_value = "100")]
    pub num_games: usize,

    /// Number of parallel environments
    #[arg(long, default_value = "64")]
    pub num_envs: usize,

    /// Number of Swiss rounds (default: auto = ceil(log2(n)) + 1)
    /// Ignored for round-robin (N <= 8 contestants)
    #[arg(long)]
    pub rounds: Option<usize>,

    /// Maximum checkpoints to select from each run directory
    /// Selects evenly spaced checkpoints including first and last
    #[arg(long)]
    pub limit: Option<usize>,

    /// Include a random agent as baseline
    #[arg(long)]
    pub random: bool,

    /// Initial softmax temperature (uses environment default if not specified)
    #[arg(long)]
    pub temp: Option<f32>,

    /// Temperature after cutoff (requires --temp-cutoff)
    #[arg(long)]
    pub temp_final: Option<f32>,

    /// Move number to switch from initial to final temperature
    #[arg(long)]
    pub temp_cutoff: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Save results to JSON file
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Generate and display a graph of ratings over training steps
    #[arg(long)]
    pub graph: bool,
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

    // PPO hyperparameters
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_true")]
    pub lr_anneal: bool,
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_gae_lambda")]
    pub gae_lambda: f64,
    #[serde(default = "default_clip_epsilon")]
    pub clip_epsilon: f64,
    #[serde(default = "default_true")]
    pub clip_value: bool,
    #[serde(default = "default_entropy_coef")]
    pub entropy_coef: f64,
    /// Whether to anneal entropy coefficient over training
    /// When true, decays from `entropy_coef` to 10% of initial value
    #[serde(default)]
    pub entropy_anneal: bool,
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

    // Training
    #[serde(default = "default_total_timesteps")]
    pub total_timesteps: usize,
    #[serde(default = "default_num_epochs")]
    pub num_epochs: usize,
    #[serde(default = "default_num_minibatches")]
    pub num_minibatches: usize,
    #[serde(default = "default_adam_epsilon")]
    pub adam_epsilon: f64,

    // Network
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden")]
    pub num_hidden: usize,
    #[serde(default = "default_activation")]
    pub activation: String,
    /// Use separate actor and critic networks instead of shared backbone
    #[serde(default)]
    pub split_networks: bool,

    // Checkpointing
    #[serde(default = "default_run_dir")]
    pub run_dir: PathBuf,
    #[serde(default = "default_checkpoint_freq")]
    pub checkpoint_freq: usize,

    // Logging
    #[serde(default = "default_log_freq")]
    pub log_freq: usize,

    // Challenger evaluation (multiplayer best checkpoint selection)
    /// Enable challenger-style evaluation for multiplayer games
    /// When enabled, new checkpoints must beat the current best to become "best"
    #[serde(default)]
    pub challenger_eval: bool,
    /// Number of games for challenger evaluation
    #[serde(default = "default_challenger_games")]
    pub challenger_games: usize,
    /// Win rate threshold to become new best (0.55 = 55%)
    #[serde(default = "default_challenger_threshold")]
    pub challenger_threshold: f64,
    /// Temperature for challenger evaluation action sampling (default: 0.3)
    #[serde(default = "default_challenger_temp")]
    pub challenger_temp: f32,
    /// Final temperature after cutoff (default: 0.0)
    #[serde(default)]
    pub challenger_temp_final: Option<f32>,
    /// Move number to switch/decay temperature (default: None = constant temp)
    #[serde(default)]
    pub challenger_temp_cutoff: Option<usize>,
    /// Use linear decay instead of hard cutoff (default: false)
    #[serde(default)]
    pub challenger_temp_decay: bool,

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
const fn default_learning_rate() -> f64 {
    2.5e-4
}
const fn default_true() -> bool {
    true
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
const fn default_entropy_coef() -> f64 {
    0.01
}
const fn default_value_coef() -> f64 {
    0.5
}
const fn default_max_grad_norm() -> f64 {
    0.5
}
const fn default_total_timesteps() -> usize {
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
const fn default_challenger_games() -> usize {
    64
}
const fn default_challenger_threshold() -> f64 {
    0.55
}
const fn default_challenger_temp() -> f32 {
    0.3
}

impl Default for Config {
    /// Default config for testing purposes.
    /// Note: `env` must be specified in TOML configs for actual training.
    fn default() -> Self {
        Self {
            env: "cartpole".to_string(), // For tests only
            num_envs: NumEnvs::default(),
            num_steps: default_num_steps(),
            learning_rate: default_learning_rate(),
            lr_anneal: default_true(),
            gamma: default_gamma(),
            gae_lambda: default_gae_lambda(),
            clip_epsilon: default_clip_epsilon(),
            clip_value: default_true(),
            entropy_coef: default_entropy_coef(),
            entropy_anneal: false,
            value_coef: default_value_coef(),
            max_grad_norm: default_max_grad_norm(),
            target_kl: None,
            normalize_obs: false,
            total_timesteps: default_total_timesteps(),
            num_epochs: default_num_epochs(),
            num_minibatches: default_num_minibatches(),
            adam_epsilon: default_adam_epsilon(),
            hidden_size: default_hidden_size(),
            num_hidden: default_num_hidden(),
            activation: default_activation(),
            split_networks: false,
            run_dir: default_run_dir(),
            checkpoint_freq: default_checkpoint_freq(),
            log_freq: default_log_freq(),
            challenger_eval: false,
            challenger_games: default_challenger_games(),
            challenger_threshold: default_challenger_threshold(),
            challenger_temp: default_challenger_temp(),
            challenger_temp_final: None,
            challenger_temp_cutoff: None,
            challenger_temp_decay: false,
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

        // PPO hyperparameters
        if let Some(lr) = args.learning_rate {
            self.learning_rate = lr;
        }
        if let Some(v) = args.lr_anneal {
            self.lr_anneal = v;
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
        if let Some(v) = args.entropy_coef {
            self.entropy_coef = v;
        }
        if let Some(v) = args.entropy_anneal {
            self.entropy_anneal = v;
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
        if let Some(v) = args.num_steps {
            self.num_steps = v;
        }

        // Training
        if let Some(ts) = args.total_timesteps {
            self.total_timesteps = ts;
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

        // Challenger evaluation
        if let Some(v) = args.challenger_eval {
            self.challenger_eval = v;
        }
        if let Some(v) = args.challenger_games {
            self.challenger_games = v;
        }
        if let Some(v) = args.challenger_threshold {
            self.challenger_threshold = v;
        }
        if let Some(v) = args.challenger_temp {
            self.challenger_temp = v;
        }
        if let Some(v) = args.challenger_temp_final {
            self.challenger_temp_final = Some(v);
        }
        if let Some(v) = args.challenger_temp_cutoff {
            self.challenger_temp_cutoff = Some(v);
        }
        if let Some(v) = args.challenger_temp_decay {
            self.challenger_temp_decay = v;
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
    /// When resuming, we only allow extending `total_timesteps`.
    /// Other parameters are locked to the original run config.
    pub fn apply_resume_overrides(&mut self, args: &CliArgs) {
        // Only allow extending training duration
        if let Some(ts) = args.total_timesteps {
            self.total_timesteps = ts;
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
        if args.lr_anneal.is_some() {
            ignored.push("--lr-anneal");
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
        if args.entropy_anneal.is_some() {
            ignored.push("--entropy-anneal");
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

        // Challenger evaluation
        if args.challenger_eval.is_some() {
            ignored.push("--challenger-eval");
        }
        if args.challenger_games.is_some() {
            ignored.push("--challenger-games");
        }
        if args.challenger_threshold.is_some() {
            ignored.push("--challenger-threshold");
        }
        if args.challenger_temp.is_some() {
            ignored.push("--challenger-temp");
        }
        if args.challenger_temp_final.is_some() {
            ignored.push("--challenger-temp-final");
        }
        if args.challenger_temp_cutoff.is_some() {
            ignored.push("--challenger-temp-cutoff");
        }
        if args.challenger_temp_decay.is_some() {
            ignored.push("--challenger-temp-decay");
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

        if self.learning_rate <= 0.0 {
            bail!("learning_rate must be > 0");
        }
        if self.gamma < 0.0 || self.gamma > 1.0 {
            bail!("gamma must be in [0, 1]");
        }
        if self.clip_epsilon <= 0.0 {
            bail!("clip_epsilon must be > 0");
        }
        if self.entropy_coef < 0.0 {
            bail!("entropy_coef must be >= 0");
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
        assert_eq!(config.learning_rate, 2.5e-4);
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
            learning_rate: -0.1,
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
