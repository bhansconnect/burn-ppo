/// Checkpointing for training state save/restore
///
/// Features:
/// - Atomic writes (temp file + rename)
/// - `latest` symlink to most recent checkpoint
/// - `best` symlink to highest avg return checkpoint
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::module::Module;
use burn::optim::Optimizer;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Record, Recorder};
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::network::ActorCritic;
use crate::normalization::ObsNormalizer;

/// Training metadata saved alongside model weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub avg_return: f32,
    pub rng_seed: u64,
    /// Best average return seen during training (for CheckpointManager restoration)
    /// Uses Option to handle f32::NEG_INFINITY which serializes as null
    #[serde(default)]
    pub best_avg_return: Option<f32>,
    /// Last 100 episode returns for smoothed metrics
    #[serde(default)]
    pub recent_returns: Vec<f32>,
    /// Observation dimension (for generic model loading)
    #[serde(default)]
    pub obs_dim: usize,
    /// Action count (for generic model loading)
    #[serde(default)]
    pub action_count: usize,
    /// Number of players (for value head dimension)
    #[serde(default = "default_num_players")]
    pub num_players: usize,
    /// Hidden layer size (for network reconstruction)
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Number of hidden layers (for network reconstruction)
    #[serde(default = "default_num_hidden")]
    pub num_hidden: usize,
    /// Parent run name if this run was forked from another
    #[serde(default)]
    pub forked_from: Option<String>,
    /// Environment name for dispatching at eval time
    #[serde(default)]
    pub env_name: String,
}

/// Default num_players for backward compatibility with older checkpoints
fn default_num_players() -> usize {
    1
}

/// Default hidden_size for backward compatibility with older checkpoints
fn default_hidden_size() -> usize {
    64
}

/// Default num_hidden for backward compatibility with older checkpoints
fn default_num_hidden() -> usize {
    2
}

/// Manages checkpointing for a training run
pub struct CheckpointManager {
    checkpoints_dir: PathBuf,
    best_avg_return: f32,
}

impl CheckpointManager {
    /// Create new checkpoint manager for the given run directory
    pub fn new(run_dir: &Path) -> Result<Self> {
        let checkpoints_dir = run_dir.join("checkpoints");
        fs::create_dir_all(&checkpoints_dir)?;

        Ok(Self {
            checkpoints_dir,
            best_avg_return: f32::NEG_INFINITY,
        })
    }

    /// Save a checkpoint with atomic write
    ///
    /// If `update_best` is true, updates the "best" symlink if this checkpoint
    /// has a higher avg_return than the previous best. Set to false when using
    /// challenger evaluation to control best selection manually.
    ///
    /// Returns the path to the saved checkpoint directory
    pub fn save<B: burn::tensor::backend::Backend>(
        &mut self,
        model: &ActorCritic<B>,
        metadata: &CheckpointMetadata,
        update_best: bool,
    ) -> Result<PathBuf> {
        let checkpoint_name = format!("step_{:08}", metadata.step);
        let checkpoint_dir = self.checkpoints_dir.join(&checkpoint_name);

        // Create temp directory for atomic write
        let temp_dir = self
            .checkpoints_dir
            .join(format!(".tmp_{}", checkpoint_name));
        fs::create_dir_all(&temp_dir)?;

        // Save model using Burn's recorder
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model_path = temp_dir.join("model");
        model
            .clone()
            .save_file(model_path, &recorder)
            .context("Failed to save model")?;

        // Save metadata as JSON
        let metadata_path = temp_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        fs::write(&metadata_path, metadata_json)?;

        // Atomic rename
        if checkpoint_dir.exists() {
            fs::remove_dir_all(&checkpoint_dir)?;
        }
        fs::rename(&temp_dir, &checkpoint_dir)?;

        // Update latest symlink
        self.update_symlink("latest", &checkpoint_dir)?;

        // Update best symlink if this is a new best (and auto-update is enabled)
        if update_best && metadata.avg_return > self.best_avg_return {
            self.best_avg_return = metadata.avg_return;
            self.update_symlink("best", &checkpoint_dir)?;
        }

        Ok(checkpoint_dir)
    }

    /// Load a checkpoint from a directory
    ///
    /// Note: config is needed to initialize the model structure before loading weights
    pub fn load<B: burn::tensor::backend::Backend>(
        checkpoint_dir: &Path,
        config: &Config,
        device: &B::Device,
    ) -> Result<(ActorCritic<B>, CheckpointMetadata)> {
        // Load metadata
        let metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json =
            fs::read_to_string(&metadata_path).context("Failed to read checkpoint metadata")?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

        // Use dimensions from metadata if available, otherwise fall back to CartPole defaults
        let obs_dim = if metadata.obs_dim > 0 {
            metadata.obs_dim
        } else {
            4
        };
        let action_count = if metadata.action_count > 0 {
            metadata.action_count
        } else {
            2
        };
        let num_players = if metadata.num_players > 0 {
            metadata.num_players
        } else {
            1
        };
        let default_model: ActorCritic<B> =
            ActorCritic::new(obs_dim, action_count, num_players, config, device);

        // Load model using Burn's recorder
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model_path = checkpoint_dir.join("model");
        let model = default_model
            .load_file(model_path, &recorder, device)
            .context("Failed to load model")?;

        Ok((model, metadata))
    }

    /// Get the path to the latest checkpoint, if any
    ///
    /// Note: Currently unused in training. Reserved for future inference mode.
    #[allow(dead_code)]
    pub fn latest_checkpoint(&self) -> Option<PathBuf> {
        let latest = self.checkpoints_dir.join("latest");
        if latest.exists() {
            fs::read_link(&latest).ok()
        } else {
            None
        }
    }

    /// Get the path to the best checkpoint, if any
    ///
    /// Note: Currently unused in training. Reserved for future inference mode.
    #[allow(dead_code)]
    pub fn best_checkpoint(&self) -> Option<PathBuf> {
        let best = self.checkpoints_dir.join("best");
        if best.exists() {
            fs::read_link(&best).ok()
        } else {
            None
        }
    }

    /// Get the current best average return
    pub fn best_avg_return(&self) -> f32 {
        self.best_avg_return
    }

    /// Set the best average return (used when resuming)
    pub fn set_best_avg_return(&mut self, value: f32) {
        self.best_avg_return = value;
    }

    /// Manually promote a checkpoint to best
    ///
    /// Used by challenger evaluation to update best based on head-to-head win rate
    /// rather than average return.
    pub fn promote_to_best(&mut self, checkpoint_dir: &Path) -> Result<()> {
        self.update_symlink("best", checkpoint_dir)
    }

    /// Get the full path to the best checkpoint symlink
    ///
    /// Returns the path to the "best" symlink (not the resolved target).
    /// Use this to check if a best checkpoint exists.
    pub fn best_checkpoint_path(&self) -> PathBuf {
        self.checkpoints_dir.join("best")
    }

    /// Update a symlink atomically
    fn update_symlink(&self, name: &str, target: &Path) -> Result<()> {
        let link_path = self.checkpoints_dir.join(name);
        let temp_link = self.checkpoints_dir.join(format!(".tmp_{}", name));

        // Create new symlink with temp name
        if temp_link.exists() {
            fs::remove_file(&temp_link)?;
        }

        // Use relative path for symlink target
        let target_name = target.file_name().unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(target_name, &temp_link)?;
        #[cfg(not(unix))]
        fs::write(&temp_link, target_name.to_string_lossy().as_bytes())?;

        // Atomic rename
        fs::rename(&temp_link, &link_path)?;

        Ok(())
    }
}

/// Save optimizer state to a checkpoint directory
///
/// The optimizer record is saved alongside the model in the checkpoint directory.
pub fn save_optimizer<B, O, M>(optimizer: &O, checkpoint_dir: &Path) -> Result<()>
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    O: Optimizer<M, B>,
    O::Record: Record<B>,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let optimizer_path = checkpoint_dir.join("optimizer");
    let record = optimizer.to_record();
    recorder
        .record(record, optimizer_path)
        .context("Failed to save optimizer state")?;
    Ok(())
}

/// Load optimizer state from a checkpoint directory
///
/// Returns the optimizer with restored state if an optimizer checkpoint exists.
/// If no optimizer checkpoint is found, returns the optimizer unchanged.
pub fn load_optimizer<B, O, M>(optimizer: O, checkpoint_dir: &Path, device: &B::Device) -> Result<O>
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    O: Optimizer<M, B>,
    O::Record: Record<B>,
{
    let optimizer_path = checkpoint_dir.join("optimizer.mpk");
    if !optimizer_path.exists() {
        // No optimizer state saved, return as-is
        return Ok(optimizer);
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record: O::Record = recorder
        .load(optimizer_path, device)
        .context("Failed to load optimizer state")?;
    Ok(optimizer.load_record(record))
}

/// Save observation normalizer to a checkpoint directory
///
/// The normalizer is saved as JSON for easy inspection and portability.
pub fn save_normalizer(normalizer: &ObsNormalizer, checkpoint_dir: &Path) -> Result<()> {
    let normalizer_path = checkpoint_dir.join("normalizer.json");
    let json = serde_json::to_string_pretty(normalizer)?;
    fs::write(normalizer_path, json)?;
    Ok(())
}

/// Save RNG state to a checkpoint directory
///
/// Generates a fresh seed from the current RNG state and saves it.
/// This allows resuming training with deterministic continuation.
pub fn save_rng_state(rng: &mut rand::rngs::StdRng, checkpoint_dir: &Path) -> Result<()> {
    use rand::RngCore;

    // Generate a 32-byte seed from current RNG state
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);

    let rng_path = checkpoint_dir.join("rng_state.bin");
    fs::write(rng_path, seed)?;
    Ok(())
}

/// Load RNG state from a checkpoint directory
///
/// Returns a new StdRng initialized from the saved seed.
pub fn load_rng_state(checkpoint_dir: &Path) -> Result<Option<rand::rngs::StdRng>> {
    use rand::SeedableRng;

    let rng_path = checkpoint_dir.join("rng_state.bin");
    if !rng_path.exists() {
        return Ok(None);
    }

    let seed_bytes = fs::read(&rng_path).context("Failed to read RNG state")?;

    if seed_bytes.len() != 32 {
        anyhow::bail!(
            "Invalid RNG state file: expected 32 bytes, got {}",
            seed_bytes.len()
        );
    }

    let mut seed = [0u8; 32];
    seed.copy_from_slice(&seed_bytes);

    Ok(Some(rand::rngs::StdRng::from_seed(seed)))
}

/// Load observation normalizer from a checkpoint directory
///
/// Returns None if no normalizer was saved (older checkpoint or normalize_obs=false).
pub fn load_normalizer(checkpoint_dir: &Path) -> Result<Option<ObsNormalizer>> {
    let normalizer_path = checkpoint_dir.join("normalizer.json");
    if !normalizer_path.exists() {
        return Ok(None);
    }

    let json = fs::read_to_string(&normalizer_path).context("Failed to read normalizer")?;
    let normalizer: ObsNormalizer = serde_json::from_str(&json)?;
    Ok(Some(normalizer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use tempfile::tempdir;

    use crate::config::Config;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_checkpoint_manager_creation() {
        let dir = tempdir().unwrap();
        let manager = CheckpointManager::new(dir.path());
        assert!(manager.is_ok());
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = tempdir().unwrap();
        let mut manager = CheckpointManager::new(dir.path()).unwrap();
        let device = Default::default();
        let config = Config::default();

        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, 1, &config, &device);
        let metadata = CheckpointMetadata {
            step: 1000,
            avg_return: 150.0,
            rng_seed: 42,
            best_avg_return: Some(150.0),
            recent_returns: vec![140.0, 150.0, 160.0],
            obs_dim: 4,
            action_count: 2,
            num_players: 1,
            hidden_size: 64,
            num_hidden: 2,
            forked_from: None,
            env_name: "cartpole".to_string(),
        };

        let checkpoint_path = manager.save(&model, &metadata, true).unwrap();
        assert!(checkpoint_path.exists());

        // Verify latest symlink
        let latest = manager.latest_checkpoint();
        assert!(latest.is_some());

        // Load and verify
        let (loaded_model, loaded_metadata) =
            CheckpointManager::load::<TestBackend>(&checkpoint_path, &config, &device).unwrap();
        assert_eq!(loaded_metadata.step, 1000);
        assert_eq!(loaded_metadata.avg_return, 150.0);

        // Verify model has same structure
        let (logits, values) = loaded_model.forward(burn::tensor::Tensor::zeros([1, 4], &device));
        assert_eq!(logits.dims(), [1, 2]);
        assert_eq!(values.dims(), [1, 1]); // [batch, num_players]
    }

    #[test]
    fn test_best_checkpoint_tracking() {
        let dir = tempdir().unwrap();
        let mut manager = CheckpointManager::new(dir.path()).unwrap();
        let device = Default::default();
        let config = Config::default();

        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, 1, &config, &device);

        // Save first checkpoint with low return
        manager
            .save(
                &model,
                &CheckpointMetadata {
                    step: 1000,
                    avg_return: 100.0,
                    rng_seed: 42,
                    best_avg_return: Some(100.0),
                    recent_returns: vec![100.0],
                    obs_dim: 4,
                    action_count: 2,
                    num_players: 1,
                    hidden_size: 64,
                    num_hidden: 2,
                    forked_from: None,
                    env_name: "cartpole".to_string(),
                },
                true,
            )
            .unwrap();

        // Save second checkpoint with higher return - should become best
        manager
            .save(
                &model,
                &CheckpointMetadata {
                    step: 2000,
                    avg_return: 200.0,
                    rng_seed: 42,
                    best_avg_return: Some(200.0),
                    recent_returns: vec![100.0, 200.0],
                    obs_dim: 4,
                    action_count: 2,
                    num_players: 1,
                    hidden_size: 64,
                    num_hidden: 2,
                    forked_from: None,
                    env_name: "cartpole".to_string(),
                },
                true,
            )
            .unwrap();

        // Save third checkpoint with lower return - best should stay at 2000
        manager
            .save(
                &model,
                &CheckpointMetadata {
                    step: 3000,
                    avg_return: 150.0,
                    rng_seed: 42,
                    best_avg_return: Some(200.0),
                    recent_returns: vec![100.0, 200.0, 150.0],
                    obs_dim: 4,
                    action_count: 2,
                    num_players: 1,
                    hidden_size: 64,
                    num_hidden: 2,
                    forked_from: None,
                    env_name: "cartpole".to_string(),
                },
                true,
            )
            .unwrap();

        // Verify best points to step 2000
        let best = manager.best_checkpoint().unwrap();
        assert!(best.to_string_lossy().contains("step_00002000"));

        // Verify latest points to step 3000
        let latest = manager.latest_checkpoint().unwrap();
        assert!(latest.to_string_lossy().contains("step_00003000"));
    }

    #[test]
    fn test_default_hidden_size() {
        assert_eq!(default_hidden_size(), 64);
    }

    #[test]
    fn test_default_num_hidden() {
        assert_eq!(default_num_hidden(), 2);
    }

    #[test]
    fn test_backward_compat_deserialize_old_checkpoint_metadata() {
        // Simulate old checkpoint JSON without new fields
        let old_json = r#"{
            "step": 5000,
            "avg_return": 100.0,
            "rng_seed": 42,
            "best_avg_return": 100.0,
            "recent_returns": [100.0],
            "obs_dim": 4,
            "action_count": 2,
            "num_players": 1
        }"#;

        let metadata: CheckpointMetadata = serde_json::from_str(old_json).unwrap();

        // Verify defaults are applied for missing fields
        assert_eq!(metadata.hidden_size, 64); // default
        assert_eq!(metadata.num_hidden, 2); // default
        assert_eq!(metadata.env_name, ""); // default for String
        assert!(metadata.forked_from.is_none());

        // Verify existing fields loaded correctly
        assert_eq!(metadata.step, 5000);
        assert_eq!(metadata.obs_dim, 4);
        assert_eq!(metadata.action_count, 2);
    }

    #[test]
    fn test_backward_compat_with_partial_new_fields() {
        // Simulate checkpoint with some new fields but not all
        let partial_json = r#"{
            "step": 5000,
            "avg_return": 100.0,
            "rng_seed": 42,
            "best_avg_return": 100.0,
            "recent_returns": [100.0],
            "obs_dim": 4,
            "action_count": 2,
            "num_players": 1,
            "hidden_size": 128
        }"#;

        let metadata: CheckpointMetadata = serde_json::from_str(partial_json).unwrap();

        // Specified field should be loaded
        assert_eq!(metadata.hidden_size, 128);
        // Missing fields should use defaults
        assert_eq!(metadata.num_hidden, 2);
        assert_eq!(metadata.env_name, "");
    }

    #[test]
    fn test_metadata_roundtrip_with_new_fields() {
        let metadata = CheckpointMetadata {
            step: 1000,
            avg_return: 150.0,
            rng_seed: 42,
            best_avg_return: Some(150.0),
            recent_returns: vec![140.0, 150.0],
            obs_dim: 86,
            action_count: 7,
            num_players: 2,
            hidden_size: 256,
            num_hidden: 3,
            forked_from: Some("parent_run".to_string()),
            env_name: "connect_four".to_string(),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let loaded: CheckpointMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.hidden_size, 256);
        assert_eq!(loaded.num_hidden, 3);
        assert_eq!(loaded.env_name, "connect_four");
        assert_eq!(loaded.forked_from, Some("parent_run".to_string()));
    }
}
