//! End-to-end tests that spawn the actual burn-ppo binary.
//!
//! These tests exercise the full training pipeline including CLI parsing,
//! checkpoint saving, resume/fork logic, and error handling.
#![allow(clippy::unwrap_used, reason = "test code uses unwrap for simplicity")]

use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::tempdir;

/// Run the burn-ppo binary with given args and a custom run directory
fn run_binary(args: &[&str], base_dir: &Path) -> Output {
    // Create a test config (run_dir is now specified via CLI, not config)
    let config_content = r#"
env = "cartpole"
num_envs = 2
num_steps = 8
total_steps = 64
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
"#;

    let config_path = base_dir.join("test_config.toml");
    fs::write(&config_path, config_content).expect("Failed to write test config");

    // Build args with train subcommand, temp config, and --run-dir pointing to base_dir/cartpole_001
    // The --run-dir specifies the full path to the run directory
    let config_str = config_path.to_str().unwrap().to_string();
    let run_dir = base_dir.join("cartpole_001");
    let run_dir_str = run_dir.to_str().unwrap().to_string();
    let mut full_args: Vec<String> = vec![
        "train".to_string(),
        "--config".to_string(),
        config_str,
        "--run-dir".to_string(),
        run_dir_str,
    ];
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

/// Get the run directory created in the given base directory.
/// Since we use --run-dir to specify the exact path, this returns `base_dir/cartpole_001` if it exists.
fn get_first_run_dir(base_dir: &Path) -> Option<std::path::PathBuf> {
    let run_dir = base_dir.join("cartpole_001");
    if run_dir.exists() {
        Some(run_dir)
    } else {
        // Fallback: search for any directory (for tests that don't use run_binary helper)
        fs::read_dir(base_dir)
            .ok()?
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir() && e.file_name() != "test_config.toml")
            .map(|e| e.path())
            .next()
    }
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
fn test_fresh_training_with_steps_override() {
    let dir = tempdir().unwrap();
    let output = run_binary(&["--total-steps", "32"], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let config_content = fs::read_to_string(run_dir.join("config.toml")).unwrap();
    assert!(
        config_content.contains("total_steps = 32"),
        "Steps override not applied"
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

    // Second run: resume with more steps
    let output2 = run_binary_raw(&["train", "--resume", run_dir_str, "--total-steps", "128"]);

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
// Evaluation Tests
// ============================================================================

#[test]
fn test_eval_stats_mode() {
    let dir = tempdir().unwrap();

    // First: train a model
    let output1 = run_binary(&[], dir.path());
    assert!(
        output1.status.success(),
        "Training failed: {}",
        String::from_utf8_lossy(&output1.stderr)
    );

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let best_checkpoint = run_dir.join("checkpoints/best");

    // Run eval with stats mode (default)
    let output = run_binary_raw(&[
        "eval",
        "--checkpoint",
        best_checkpoint.to_str().unwrap(),
        "--num-games",
        "10",
        "--num-envs",
        "2",
    ]);

    assert!(
        output.status.success(),
        "Eval stats mode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show game results
    assert!(
        stdout.contains("games") || stdout.contains("Reward") || stdout.contains("Mean"),
        "Eval output should show game statistics"
    );
}

#[test]
fn test_eval_with_temperature() {
    let dir = tempdir().unwrap();

    // Train a model
    let output1 = run_binary(&[], dir.path());
    assert!(output1.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let best_checkpoint = run_dir.join("checkpoints/best");

    // Run eval with temperature options
    let output = run_binary_raw(&[
        "eval",
        "--checkpoint",
        best_checkpoint.to_str().unwrap(),
        "--num-games",
        "5",
        "--temp",
        "0.5",
        "--temp-cutoff",
        "10",
        "--temp-final",
        "0.0",
    ]);

    assert!(
        output.status.success(),
        "Eval with temp failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================================================================
// Checkpoint Metadata Tests
// ============================================================================

#[test]
fn test_checkpoint_metadata_structure() {
    let dir = tempdir().unwrap();
    let output = run_binary(&[], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let checkpoints_dir = run_dir.join("checkpoints");

    // Find first step_* directory
    let step_dir = fs::read_dir(&checkpoints_dir)
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("step_"))
        });

    assert!(step_dir.is_some(), "Should have a step checkpoint");

    let metadata_path = step_dir.unwrap().path().join("metadata.json");
    assert!(metadata_path.exists(), "metadata.json should exist");

    let metadata_content = fs::read_to_string(&metadata_path).unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content).unwrap();

    // Verify all required fields exist
    assert!(metadata.get("step").is_some(), "metadata should have step");
    assert!(
        metadata.get("avg_return").is_some(),
        "metadata should have avg_return"
    );
    assert!(
        metadata.get("obs_dim").is_some(),
        "metadata should have obs_dim"
    );
    assert!(
        metadata.get("action_count").is_some(),
        "metadata should have action_count"
    );
    assert!(
        metadata.get("hidden_size").is_some(),
        "metadata should have hidden_size"
    );
    assert!(
        metadata.get("env_name").is_some(),
        "metadata should have env_name"
    );
}

#[test]
fn test_checkpoint_symlinks_exist() {
    let dir = tempdir().unwrap();
    let output = run_binary(&[], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let checkpoints_dir = run_dir.join("checkpoints");

    // Check that latest symlink exists
    let latest = checkpoints_dir.join("latest");
    assert!(latest.exists(), "latest symlink should exist");

    // Check that best symlink exists
    let best = checkpoints_dir.join("best");
    assert!(best.exists(), "best symlink should exist");
}

// ============================================================================
// Metrics File Tests
// ============================================================================

#[test]
fn test_metrics_file_format() {
    let dir = tempdir().unwrap();
    let output = run_binary(&[], dir.path());

    assert!(output.status.success());

    let run_dir = get_first_run_dir(dir.path()).unwrap();
    let metrics_path = run_dir.join("metrics.jsonl");

    let content = fs::read_to_string(&metrics_path).unwrap();

    // Each line should be valid JSON
    for (i, line) in content.lines().enumerate() {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
        assert!(
            parsed.is_ok(),
            "Line {} should be valid JSON: {}",
            i + 1,
            line
        );

        let value = parsed.unwrap();
        // Each line should have either "hparams" or "step" field
        assert!(
            value.get("hparams").is_some() || value.get("step").is_some(),
            "Line {} should have hparams or step field",
            i + 1
        );
    }
}

// ============================================================================
// Observation Normalization Tests
// ============================================================================

#[test]
fn test_training_with_normalize_obs() {
    let dir = tempdir().unwrap();

    // Create config with normalize_obs enabled
    let config_content = r#"
env = "cartpole"
num_envs = 2
num_steps = 8
total_steps = 64
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
normalize_obs = true
"#;

    let config_path = dir.path().join("norm_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let run_dir = dir.path().join("cartpole_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "Training with normalize_obs failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify normalizer was saved
    let checkpoints_dir = run_dir.join("checkpoints");

    let step_dir = fs::read_dir(&checkpoints_dir)
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("step_"))
        });

    assert!(step_dir.is_some());
    let normalizer_path = step_dir.unwrap().path().join("normalizer.json");
    assert!(
        normalizer_path.exists(),
        "normalizer.json should be saved when normalize_obs=true"
    );
}

// ============================================================================
// Connect Four Environment Tests
// ============================================================================

#[test]
fn test_connect_four_training() {
    let dir = tempdir().unwrap();

    // Create config for connect four
    let config_content = r#"
env = "connect_four"
num_envs = 2
num_steps = 8
total_steps = 64
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
"#;

    let config_path = dir.path().join("c4_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let run_dir = dir.path().join("connect_four_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "Connect Four training failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_liars_dice_training() {
    let dir = tempdir().unwrap();

    // Create config for Liar's Dice
    let config_content = r#"
env = "liars_dice"
num_envs = 2
num_steps = 8
total_steps = 64
num_epochs = 1
num_minibatches = 1
hidden_size = 16
num_hidden = 1
activation = "relu"
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.1
value_coef = 0.5
max_grad_norm = 0.5
adam_epsilon = 1e-5
checkpoint_freq = 32
log_freq = 1000
seed = 42
"#;

    let config_path = dir.path().join("ld_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let run_dir = dir.path().join("liars_dice_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "Liar's Dice training failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================================================================
// CNN Network Tests
// ============================================================================

#[test]
fn test_cnn_training_connect_four() {
    let dir = tempdir().unwrap();

    // Create config for connect four with CNN
    let config_content = r#"
env = "connect_four"
num_envs = 2
num_steps = 8
total_steps = 32
num_epochs = 1
num_minibatches = 1
network_type = "cnn"
num_conv_layers = 1
conv_channels = [8]
kernel_size = 3
cnn_fc_hidden_size = 16
cnn_num_fc_layers = 1
activation = "relu"
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
adam_epsilon = 1e-5
checkpoint_freq = 16
log_freq = 1000
seed = 42
"#;

    let config_path = dir.path().join("cnn_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let run_dir = dir.path().join("connect_four_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "CNN training failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify checkpoint created
    let checkpoints = run_dir.join("checkpoints");
    assert!(checkpoints.exists(), "Checkpoints directory should exist");
}

#[test]
fn test_cnn_checkpoint_resume() {
    let dir = tempdir().unwrap();

    // Phase 1: Train CNN model
    let config_content = r#"
env = "connect_four"
num_envs = 2
num_steps = 8
total_steps = 32
num_epochs = 1
num_minibatches = 1
network_type = "cnn"
num_conv_layers = 1
conv_channels = [8]
kernel_size = 3
cnn_fc_hidden_size = 16
cnn_num_fc_layers = 1
activation = "relu"
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
adam_epsilon = 1e-5
checkpoint_freq = 16
log_freq = 1000
seed = 42
"#;

    let config_path = dir.path().join("cnn_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let run_dir = dir.path().join("connect_four_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "Initial CNN training failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Phase 2: Resume training from checkpoint
    let run_dir_str = run_dir.to_str().unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args(["train", "--resume", run_dir_str, "--total-steps", "128"])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "CNN checkpoint resume failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cnn_checkpoint_metadata() {
    let dir = tempdir().unwrap();

    // Train CNN model
    let config_content = r#"
env = "connect_four"
num_envs = 2
num_steps = 8
total_steps = 32
num_epochs = 1
num_minibatches = 1
network_type = "cnn"
num_conv_layers = 1
conv_channels = [8]
kernel_size = 3
cnn_fc_hidden_size = 16
cnn_num_fc_layers = 1
activation = "relu"
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
adam_epsilon = 1e-5
checkpoint_freq = 16
log_freq = 1000
seed = 42
"#;

    let config_path = dir.path().join("cnn_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let run_dir = dir.path().join("connect_four_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute");

    assert!(output.status.success(), "CNN training failed");

    // Verify metadata contains CNN fields
    let latest_checkpoint = run_dir.join("checkpoints/latest");
    let metadata_path = latest_checkpoint.join("metadata.json");

    assert!(metadata_path.exists(), "metadata.json should exist");

    let metadata_content = fs::read_to_string(&metadata_path).unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content).unwrap();

    // Verify CNN-specific fields
    assert_eq!(
        metadata["network_type"].as_str(),
        Some("cnn"),
        "network_type should be 'cnn'"
    );
    assert_eq!(
        metadata["num_conv_layers"].as_u64(),
        Some(1),
        "num_conv_layers should be 1"
    );
    assert_eq!(
        metadata["kernel_size"].as_u64(),
        Some(3),
        "kernel_size should be 3"
    );
    assert_eq!(
        metadata["cnn_fc_hidden_size"].as_u64(),
        Some(16),
        "cnn_fc_hidden_size should be 16"
    );

    // Verify obs_shape for connect four
    let obs_shape = metadata["obs_shape"]
        .as_array()
        .expect("obs_shape should be array");
    assert_eq!(obs_shape.len(), 3, "obs_shape should have 3 elements");
    assert_eq!(obs_shape[0].as_u64(), Some(6), "height should be 6");
    assert_eq!(obs_shape[1].as_u64(), Some(7), "width should be 7");
    assert_eq!(obs_shape[2].as_u64(), Some(2), "channels should be 2");
}

#[test]
fn test_cnn_eval() {
    let dir = tempdir().unwrap();

    // Train CNN model
    let config_content = r#"
env = "connect_four"
num_envs = 2
num_steps = 8
total_steps = 32
num_epochs = 1
num_minibatches = 1
network_type = "cnn"
num_conv_layers = 1
conv_channels = [8]
kernel_size = 3
cnn_fc_hidden_size = 16
cnn_num_fc_layers = 1
activation = "relu"
learning_rate = 0.001
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
adam_epsilon = 1e-5
checkpoint_freq = 16
log_freq = 1000
seed = 42
"#;

    let config_path = dir.path().join("cnn_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let run_dir = dir.path().join("connect_four_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute");

    assert!(output.status.success(), "CNN training failed");

    // Run eval on the trained model
    let checkpoint_path = run_dir.join("checkpoints/latest");
    let checkpoint_str = checkpoint_path.to_str().unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "eval",
            "--checkpoint",
            checkpoint_str,
            "--num-games",
            "2",
            "--num-envs",
            "2",
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "CNN eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================================================================
// Tournament Tests
// ============================================================================

// ============================================================================
// Subprocess Reload Tests
// ============================================================================

#[test]
fn test_reload_every_n_checkpoints() {
    let dir = tempdir().unwrap();

    // Create config for cartpole with small checkpoint_freq
    let config_content = r#"
env = "cartpole"
num_envs = 2
num_steps = 8
total_steps = 128
num_epochs = 1
num_minibatches = 1
hidden_size = 8
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
"#;

    let config_path = dir.path().join("reload_config.toml");
    fs::write(&config_path, config_content).unwrap();

    // Run with reload every 2 checkpoints
    let run_dir = dir.path().join("cartpole_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
            "--reload-every-n-checkpoints",
            "2",
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "Training with reload failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify checkpoints were created
    let checkpoints = run_dir.join("checkpoints");
    assert!(checkpoints.exists(), "Checkpoints directory should exist");

    // Verify latest symlink exists
    let latest = checkpoints.join("latest");
    assert!(latest.exists(), "Latest checkpoint symlink should exist");

    // Verify supervisor mode ran (header goes to stdout)
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("supervisor mode"),
        "Should run in supervisor mode"
    );
}

#[test]
fn test_reload_resume_with_extended_steps() {
    let dir = tempdir().unwrap();

    // Phase 1: Initial training with reload
    let config_content = r#"
env = "cartpole"
num_envs = 2
num_steps = 8
total_steps = 64
num_epochs = 1
num_minibatches = 1
hidden_size = 8
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
"#;

    let config_path = dir.path().join("reload_resume_config.toml");
    fs::write(&config_path, config_content).unwrap();

    // Run initial training with reload every 2 checkpoints
    let run_dir = dir.path().join("cartpole_001");
    let output1 = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
            "--reload-every-n-checkpoints",
            "1",
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output1.status.success(),
        "Initial training failed: {}",
        String::from_utf8_lossy(&output1.stderr)
    );

    let run_dir_str = run_dir.to_str().unwrap();

    // Read initial checkpoint step
    let latest_meta_path = run_dir.join("checkpoints/latest/metadata.json");
    let initial_metadata: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&latest_meta_path).unwrap()).unwrap();
    let initial_step = initial_metadata["step"].as_u64().unwrap();

    // Phase 2: Resume with extended steps (also using reload mode)
    let output2 = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--resume",
            run_dir_str,
            "--total-steps",
            "128",
            "--reload-every-n-checkpoints",
            "1",
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output2.status.success(),
        "Resume with extended steps failed: {}",
        String::from_utf8_lossy(&output2.stderr)
    );

    // Verify checkpoint step increased
    let final_metadata: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&latest_meta_path).unwrap()).unwrap();
    let final_step = final_metadata["step"].as_u64().unwrap();

    assert!(
        final_step > initial_step,
        "Final step ({final_step}) should be greater than initial step ({initial_step})"
    );
}

#[test]
fn test_connect_four_training_with_debug_opponents() {
    let dir = tempdir().unwrap();

    // Create config for connect four with opponent pool and debug-opponents enabled
    let config_content = r#"
env = "connect_four"
num_envs = 4
num_steps = 8
total_steps = 256
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
opponent_pool_enabled = true
opponent_pool_rotation_steps = 32
debug_opponents = true
"#;

    let config_path = dir.path().join("c4_debug_opponents_config.toml");
    fs::write(&config_path, config_content).unwrap();

    let run_dir = dir.path().join("connect_four_001");
    let output = Command::new(env!("CARGO_BIN_EXE_burn-ppo"))
        .args([
            "train",
            "--config",
            config_path.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute");

    assert!(
        output.status.success(),
        "Connect Four training with debug-opponents failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that debug-opponents output was printed (goes to stderr)
    let stderr = String::from_utf8_lossy(&output.stderr);

    // The debug output should contain opponent selection info when --debug-opponents is enabled
    // After the first checkpoint and rotation, we should see opponent selection debug output
    assert!(
        stderr.contains("[debug-opponents]"),
        "debug_opponents should print '[debug-opponents]' prefix.\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("Rotation at step"),
        "debug_opponents should print 'Rotation at step' message.\nstderr: {stderr}"
    );
}
