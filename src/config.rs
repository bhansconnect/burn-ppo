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

    /// Resume from existing run directory (same config, continue training)
    #[arg(long, conflicts_with = "fork")]
    pub resume: Option<PathBuf>,

    /// Fork from a checkpoint path (new run, allows config changes)
    #[arg(long, conflicts_with = "resume")]
    pub fork: Option<PathBuf>,

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
            // 1x CPU cores (not 2x) - no async rollout/training overlap
            NumEnvs::Auto(_) => num_cpus::get(),
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
    /// Parent run name if this run was forked from another
    #[serde(default)]
    pub forked_from: Option<String>,
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
            forked_from: None,
        }
    }
}

impl Config {
    /// Load config from TOML file, apply CLI overrides
    ///
    /// The `forked_from` parameter is set when forking from another run
    pub fn load(args: &CliArgs, forked_from: Option<String>) -> Result<Self> {
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
        config.apply_cli_overrides(args);

        // Store forked_from relationship
        config.forked_from = forked_from.clone();

        // Generate run name if not specified
        if config.run_name.is_none() {
            config.run_name = Some(generate_run_name(&config, forked_from.as_deref()));
        }

        Ok(config)
    }

    /// Load config from a specific TOML file path
    pub fn load_from_path(path: &std::path::Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {:?}", path))?;
        toml::from_str(&content).with_context(|| format!("Failed to parse config: {:?}", path))
    }

    /// Apply all CLI overrides to this config
    fn apply_cli_overrides(&mut self, args: &CliArgs) {
        if let Some(env) = &args.env {
            self.env = env.clone();
        }
        if let Some(n) = args.num_envs {
            self.num_envs = NumEnvs::Explicit(n);
        }
        if let Some(lr) = args.learning_rate {
            self.learning_rate = lr;
        }
        if let Some(ts) = args.total_timesteps {
            self.total_timesteps = ts;
        }
        if let Some(s) = args.seed {
            self.seed = s;
        }
        if let Some(name) = &args.run_name {
            self.run_name = Some(name.clone());
        }
    }

    /// Apply limited CLI overrides for resume mode
    ///
    /// When resuming, we only allow extending total_timesteps.
    /// Other parameters are locked to the original run config.
    pub fn apply_resume_overrides(&mut self, args: &CliArgs) {
        // Only allow extending training duration
        if let Some(ts) = args.total_timesteps {
            self.total_timesteps = ts;
        }

        // Warn about ignored overrides
        if args.env.is_some() {
            eprintln!("Warning: --env is ignored when resuming");
        }
        if args.num_envs.is_some() {
            eprintln!("Warning: --num-envs is ignored when resuming");
        }
        if args.learning_rate.is_some() {
            eprintln!("Warning: --learning-rate is ignored when resuming");
        }
        if args.seed.is_some() {
            eprintln!("Warning: --seed is ignored when resuming");
        }
        if args.run_name.is_some() {
            eprintln!("Warning: --run-name is ignored when resuming");
        }
    }

    /// Get the full path to the run directory
    pub fn run_path(&self) -> PathBuf {
        self.run_dir.join(self.run_name.as_ref().unwrap())
    }

    /// Get resolved number of environments
    pub fn num_envs(&self) -> usize {
        self.num_envs.resolve()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        use anyhow::bail;

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
            bail!(
                "minibatch_size {} too small, increase num_steps or num_envs",
                minibatch_size
            );
        }

        Ok(())
    }
}

/// Extract the global counter from a run name
///
/// Handles both standard names like "cartpole_001" and
/// child names like "cartpole_003_child_001"
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
                if !name.starts_with(&format!("{}_", env)) {
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
    let prefix = format!("{}_child_", parent_name);

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
        format!("{}_child_{:03}", parent_name, child_counter)
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
        let mut config = Config::default();
        config.learning_rate = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_gamma() {
        let mut config = Config::default();
        config.gamma = 1.5;
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
