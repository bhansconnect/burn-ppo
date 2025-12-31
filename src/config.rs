use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// PPO training for discrete games
#[derive(Parser, Debug)]
#[command(name = "burn-ppo", version, about)]
pub struct CliArgs {
    /// Path to TOML config file
    #[arg(short, long, default_value = "configs/default.toml")]
    pub config: PathBuf,

    /// Resume from existing run directory
    #[arg(long)]
    pub resume: Option<PathBuf>,

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
}

/// Number of parallel environments - either auto-detected or explicit
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NumEnvs {
    Auto(String), // "auto"
    Explicit(usize),
}

impl Default for NumEnvs {
    fn default() -> Self {
        NumEnvs::Auto("auto".to_string())
    }
}

impl NumEnvs {
    pub fn resolve(&self) -> usize {
        match self {
            NumEnvs::Auto(_) => num_cpus::get() * 2,
            NumEnvs::Explicit(n) => *n,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Environment
    #[serde(default = "default_env")]
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
    #[serde(default = "default_value_coef")]
    pub value_coef: f64,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,

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

    // Checkpointing
    #[serde(default = "default_run_dir")]
    pub run_dir: PathBuf,
    #[serde(default = "default_checkpoint_freq")]
    pub checkpoint_freq: usize,

    // Logging
    #[serde(default = "default_log_freq")]
    pub log_freq: usize,

    // Experiment
    #[serde(default = "default_seed")]
    pub seed: u64,
    pub run_name: Option<String>,
}

// Default value functions
fn default_env() -> String {
    "cartpole".to_string()
}
fn default_num_steps() -> usize {
    128
}
fn default_learning_rate() -> f64 {
    2.5e-4
}
fn default_true() -> bool {
    true
}
fn default_gamma() -> f64 {
    0.99
}
fn default_gae_lambda() -> f64 {
    0.95
}
fn default_clip_epsilon() -> f64 {
    0.2
}
fn default_entropy_coef() -> f64 {
    0.01
}
fn default_value_coef() -> f64 {
    0.5
}
fn default_max_grad_norm() -> f64 {
    0.5
}
fn default_total_timesteps() -> usize {
    1_000_000
}
fn default_num_epochs() -> usize {
    4
}
fn default_num_minibatches() -> usize {
    4
}
fn default_adam_epsilon() -> f64 {
    1e-5
}
fn default_hidden_size() -> usize {
    64
}
fn default_num_hidden() -> usize {
    2
}
fn default_run_dir() -> PathBuf {
    PathBuf::from("runs")
}
fn default_checkpoint_freq() -> usize {
    10_000
}
fn default_log_freq() -> usize {
    1_000
}
fn default_seed() -> u64 {
    42
}

impl Default for Config {
    fn default() -> Self {
        Self {
            env: default_env(),
            num_envs: NumEnvs::default(),
            num_steps: default_num_steps(),
            learning_rate: default_learning_rate(),
            lr_anneal: default_true(),
            gamma: default_gamma(),
            gae_lambda: default_gae_lambda(),
            clip_epsilon: default_clip_epsilon(),
            clip_value: default_true(),
            entropy_coef: default_entropy_coef(),
            value_coef: default_value_coef(),
            max_grad_norm: default_max_grad_norm(),
            total_timesteps: default_total_timesteps(),
            num_epochs: default_num_epochs(),
            num_minibatches: default_num_minibatches(),
            adam_epsilon: default_adam_epsilon(),
            hidden_size: default_hidden_size(),
            num_hidden: default_num_hidden(),
            run_dir: default_run_dir(),
            checkpoint_freq: default_checkpoint_freq(),
            log_freq: default_log_freq(),
            seed: default_seed(),
            run_name: None,
        }
    }
}

impl Config {
    /// Load config from TOML file, apply CLI overrides
    pub fn load(args: &CliArgs) -> Result<Self> {
        // Load base config
        let mut config: Config = if args.config.exists() {
            let content = fs::read_to_string(&args.config)
                .with_context(|| format!("Failed to read config: {:?}", args.config))?;
            toml::from_str(&content)
                .with_context(|| format!("Failed to parse config: {:?}", args.config))?
        } else {
            Config::default()
        };

        // Apply CLI overrides
        if let Some(env) = &args.env {
            config.env = env.clone();
        }
        if let Some(n) = args.num_envs {
            config.num_envs = NumEnvs::Explicit(n);
        }
        if let Some(lr) = args.learning_rate {
            config.learning_rate = lr;
        }
        if let Some(ts) = args.total_timesteps {
            config.total_timesteps = ts;
        }
        if let Some(s) = args.seed {
            config.seed = s;
        }
        if let Some(name) = &args.run_name {
            config.run_name = Some(name.clone());
        }

        // Generate run name if not specified
        if config.run_name.is_none() {
            config.run_name = Some(generate_run_name(&config));
        }

        Ok(config)
    }

    /// Get the full path to the run directory
    pub fn run_path(&self) -> PathBuf {
        self.run_dir.join(self.run_name.as_ref().unwrap())
    }

    /// Get resolved number of environments
    pub fn num_envs(&self) -> usize {
        self.num_envs.resolve()
    }
}

fn generate_run_name(config: &Config) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}_{}_{}", config.env, config.seed, timestamp)
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
}
