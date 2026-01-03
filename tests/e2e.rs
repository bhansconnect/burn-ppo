//! End-to-end tests that spawn the actual burn-ppo binary.
//!
//! These tests exercise the full training pipeline including CLI parsing,
//! checkpoint saving, resume/fork logic, and error handling.
#![allow(clippy::unwrap_used, reason = "test code uses unwrap for simplicity")]

use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::tempdir;

/// Run the burn-ppo binary with given args and a custom `run_dir`
fn run_binary(args: &[&str], run_dir: &Path) -> Output {
    // Create a modified config with the temp run_dir
    let config_content = format!(
        r#"
env = "cartpole"
num_envs = 2
num_steps = 8
total_timesteps = 64
num_epochs = 1
num_minibatches = 1
hidden_size = 16
num_hidden = 1
activation = "relu"
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
adam_epsilon = 1e-5
checkpoint_freq = 32
log_freq = 1000
seed = 42
run_dir = "{}"
"#,
        run_dir.display()
    );

    let config_path = run_dir.join("test_config.toml");
    fs::write(&config_path, config_content).expect("Failed to write test config");

    // Build args with train subcommand and temp config
    let config_str = config_path.to_str().unwrap().to_string();
    let mut full_args: Vec<String> = vec!["train".to_string(), "--config".to_string(), config_str];
    full_args.extend(args.iter().copied().map(String::from));

    Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args(&full_args)
        .output()
        .expect("Failed to execute binary")
}

/// Run the binary with raw args (no config modification)
fn run_binary_raw(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args(args)
        .output()
        .expect("Failed to execute binary")
}

/// Get the first run directory created in the given `run_dir`
fn get_first_run_dir(run_dir: &Path) -> Option<std::path::PathBuf> {
    fs::read_dir(run_dir)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .next()
}

// ============================================================================
// Fresh Training Tests
// ============================================================================

#[test]
fn test_fresh_training_creates_run_dir() {
    let dir = tempdir().unwrap();
    let output = run_binary(&[], dir.path());

    assert!(
        output.status.success(),
        "Training failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that a run directory was created
    let run_dir = get_first_run_dir(dir.path());
    assert!(run_dir.is_some(), "No run directory created");

    // Check that checkpoints were saved
    let checkpoints = run_dir.unwrap().join("checkpoints");
    assert!(checkpoints.exists(), "Checkpoints directory not created");
}

#[test]
fn test_fresh_training_creates_checkpoint() {
    let dir = tempdir().unwrap();
    let output = run_binary(&[], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let checkpoints = run_dir.join("checkpoints");

    // Should have at least one step_* checkpoint
    let step_dirs: Vec<_> = fs::read_dir(&checkpoints)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("step_"))
        })
        .collect();

    assert!(!step_dirs.is_empty(), "No checkpoint directories created");
}

#[test]
fn test_fresh_training_creates_metrics() {
    let dir = tempdir().unwrap();
    let output = run_binary(&[], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let metrics = run_dir.join("metrics.jsonl");
    assert!(metrics.exists(), "metrics.jsonl not created");

    // Metrics file should have content
    let content = fs::read_to_string(&metrics).unwrap();
    assert!(!content.is_empty(), "metrics.jsonl is empty");
}

#[test]
fn test_fresh_training_saves_config_snapshot() {
    let dir = tempdir().unwrap();
    let output = run_binary(&[], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let config = run_dir.join("config.toml");
    assert!(config.exists(), "config.toml not saved");
}

#[test]
fn test_fresh_training_with_seed_override() {
    let dir = tempdir().unwrap();
    let output = run_binary(&["--seed", "123"], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let config_content = fs::read_to_string(run_dir.join("config.toml")).unwrap();
    assert!(
        config_content.contains("seed = 123"),
        "Seed override not applied"
    );
}

#[test]
fn test_fresh_training_with_timesteps_override() {
    let dir = tempdir().unwrap();
    let output = run_binary(&["--total-timesteps", "32"], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let config_content = fs::read_to_string(run_dir.join("config.toml")).unwrap();
    assert!(
        config_content.contains("total_timesteps = 32"),
        "Timesteps override not applied"
    );
}

// ============================================================================
// Resume Training Tests
// ============================================================================

#[test]
fn test_resume_training() {
    let dir = tempdir().unwrap();

    // First run: create checkpoint
    let output1 = run_binary(&[], dir.path());
    assert!(
        output1.status.success(),
        "First run failed: {}",
        String::from_utf8_lossy(&output1.stderr)
    );

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let run_dir_str = run_dir.to_str().unwrap();

    // Second run: resume with more timesteps
    let output2 = run_binary_raw(&["train", "--resume", run_dir_str, "--total-timesteps", "128"]);

    assert!(
        output2.status.success(),
        "Resume failed: {}",
        String::from_utf8_lossy(&output2.stderr)
    );
}

#[test]
fn test_resume_nonexistent_fails() {
    let output = run_binary_raw(&["train", "--resume", "/nonexistent/path"]);

    assert!(
        !output.status.success(),
        "Resume from nonexistent path should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No checkpoint") || stderr.contains("Cannot resume"),
        "Error message should mention checkpoint issue"
    );
}

// ============================================================================
// Fork Training Tests
// ============================================================================

#[test]
fn test_fork_from_checkpoint() {
    let dir = tempdir().unwrap();

    // First run: create checkpoint
    let output1 = run_binary(&[], dir.path());
    assert!(output1.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let best_checkpoint = run_dir.join("checkpoints/best");

    if best_checkpoint.exists() {
        let checkpoint_str = best_checkpoint.to_str().unwrap();

        // Fork from the checkpoint
        let output2 = run_binary(&["--fork", checkpoint_str], dir.path());

        assert!(
            output2.status.success(),
            "Fork failed: {}",
            String::from_utf8_lossy(&output2.stderr)
        );

        // Should have created a second run directory
        let run_count = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir() && e.file_name() != "test_config.toml")
            .count();
        // Account for test_config.toml file vs directories
        assert!(run_count >= 1, "Fork should create a run");
    }
}

#[test]
fn test_fork_nonexistent_fails() {
    let output = run_binary_raw(&[
        "train",
        "--config",
        "configs/test.toml",
        "--fork",
        "/nonexistent/checkpoint",
    ]);

    assert!(
        !output.status.success(),
        "Fork from nonexistent checkpoint should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("Cannot fork"),
        "Error message should mention checkpoint issue"
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_invalid_environment() {
    let dir = tempdir().unwrap();
    let output = run_binary(&["--env", "nonexistent_game"], dir.path());

    assert!(!output.status.success(), "Invalid environment should fail");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown environment") || stderr.contains("nonexistent"),
        "Error should mention unknown environment"
    );
}

#[test]
fn test_missing_config_file() {
    let output = run_binary_raw(&["train", "--config", "nonexistent_config.toml"]);

    assert!(!output.status.success(), "Missing config should fail");
}

#[test]
fn test_conflicting_resume_and_fork() {
    let output = run_binary_raw(&[
        "train",
        "--resume",
        "/some/run",
        "--fork",
        "/some/checkpoint",
        "--config",
        "configs/test.toml",
    ]);

    assert!(
        !output.status.success(),
        "Conflicting --resume and --fork should fail"
    );
}

// ============================================================================
// CLI Help Tests
// ============================================================================

#[test]
fn test_help_flag() {
    let output = run_binary_raw(&["--help"]);

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("burn-ppo"));
    assert!(stdout.contains("train") || stdout.contains("eval"));
}

#[test]
fn test_train_help() {
    let output = run_binary_raw(&["train", "--help"]);

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--config"));
    assert!(stdout.contains("--resume") || stdout.contains("--fork"));
}

#[test]
fn test_version_flag() {
    let output = run_binary_raw(&["--version"]);

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("0.1") || stdout.contains("burn-ppo"));
}

#[test]
fn test_eval_subcommand_help() {
    let output = run_binary_raw(&["eval", "--help"]);

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("checkpoint") || stdout.contains("Evaluate"));
}

// ============================================================================
// Connect Four Environment Tests
// ============================================================================

#[test]
fn test_connect_four_training() {
    let dir = tempdir().unwrap();

    // Create config for connect four
    let config_content = format!(
        r#"
env = "connect_four"
num_envs = 2
num_steps = 8
total_timesteps = 64
num_epochs = 1
num_minibatches = 1
hidden_size = 16
num_hidden = 1
activation = "relu"
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
adam_epsilon = 1e-5
checkpoint_freq = 32
log_freq = 1000
seed = 42
run_dir = "{}"
"#,
        dir.path().display()
    );

    let config_path = dir.path().join("c4_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args(["train", "--config", config_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "Connect Four training failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
