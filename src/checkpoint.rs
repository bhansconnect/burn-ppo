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
///
/// All network architecture fields are required - old checkpoints without
/// these fields will fail to load (by design, to prevent silent mismatches).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub avg_return: f32,
    pub rng_seed: u64,
    /// Best average return seen during training (for `CheckpointManager` restoration)
    /// Uses Option to handle `f32::NEG_INFINITY` which serializes as null
    #[serde(default)]
    pub best_avg_return: Option<f32>,
    /// Last 100 episode returns for smoothed metrics
    #[serde(default)]
    pub recent_returns: Vec<f32>,
    /// Parent run name if this run was forked from another
    #[serde(default)]
    pub forked_from: Option<String>,
    // --- Required fields (no defaults) ---
    /// Observation dimension (for generic model loading)
    pub obs_dim: usize,
    /// Action count (for generic model loading)
    pub action_count: usize,
    /// Number of players (for value head dimension)
    pub num_players: usize,
    /// Hidden layer size (for network reconstruction)
    pub hidden_size: usize,
    /// Number of hidden layers (for network reconstruction)
    pub num_hidden: usize,
    /// Activation function (for network reconstruction)
    pub activation: String,
    /// Whether split actor/critic networks were used
    #[serde(default)]
    pub split_networks: bool,
    /// Environment name for dispatching at eval time
    pub env_name: String,
    /// Training skill rating (Weng-Lin mu) for challenger evaluation
    /// Rating accumulates across promotions
    #[serde(default = "default_rating")]
    pub training_rating: f64,
    /// Training skill uncertainty (Weng-Lin sigma)
    #[serde(default = "default_uncertainty")]
    pub training_uncertainty: f64,
}

fn default_rating() -> f64 {
    25.0
}

fn default_uncertainty() -> f64 {
    25.0 / 3.0 // ~8.333, standard Weng-Lin sigma
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
    /// has a higher `avg_return` than the previous best. Set to false when using
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
        let temp_dir = self.checkpoints_dir.join(format!(".tmp_{checkpoint_name}"));
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
    /// Note: config is needed to initialize the model structure before loading weights.
    /// Old checkpoints without required metadata fields will fail to deserialize.
    pub fn load<B: burn::tensor::backend::Backend>(
        checkpoint_dir: &Path,
        config: &Config,
        device: &B::Device,
    ) -> Result<(ActorCritic<B>, CheckpointMetadata)> {
        // Load metadata - deserialization fails if required fields are missing
        let metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json =
            fs::read_to_string(&metadata_path).context("Failed to read checkpoint metadata")?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
            .context("Failed to parse checkpoint metadata (missing required fields?)")?;

        // Create config with architecture from checkpoint metadata
        // (not current config, which may have different split_networks/hidden_size/etc.)
        let mut load_config = config.clone();
        load_config.hidden_size = metadata.hidden_size;
        load_config.num_hidden = metadata.num_hidden;
        load_config.activation.clone_from(&metadata.activation);
        load_config.split_networks = metadata.split_networks;

        let default_model: ActorCritic<B> = ActorCritic::new(
            metadata.obs_dim,
            metadata.action_count,
            metadata.num_players,
            &load_config,
            device,
        );

        // Load model using Burn's recorder
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model_path = checkpoint_dir.join("model");
        let model = default_model
            .load_file(model_path, &recorder, device)
            .context("Failed to load model")?;

        Ok((model, metadata))
    }
    /// Get the current best average return
    pub const fn best_avg_return(&self) -> f32 {
        self.best_avg_return
    }

    /// Set the best average return (used when resuming)
    pub const fn set_best_avg_return(&mut self, value: f32) {
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
        let temp_link = self.checkpoints_dir.join(format!(".tmp_{name}"));

        // Create new symlink with temp name
        if temp_link.exists() {
            fs::remove_file(&temp_link)?;
        }

        // Use relative path for symlink target
        let target_name = target.file_name().ok_or_else(|| {
            anyhow::anyhow!("Symlink target has no file name: {}", target.display())
        })?;
        #[cfg(unix)]
        std::os::unix::fs::symlink(target_name, &temp_link)?;
        #[cfg(not(unix))]
        fs::write(&temp_link, target_name.to_string_lossy().as_bytes())?;

        // Atomic rename
        fs::rename(&temp_link, &link_path)?;

        Ok(())
    }
}

/// Load checkpoint metadata from a checkpoint directory
///
/// This reads only the metadata.json file without loading the model weights.
pub fn load_metadata(checkpoint_path: &Path) -> Result<CheckpointMetadata> {
    let metadata_path = checkpoint_path.join("metadata.json");
    let metadata_json =
        std::fs::read_to_string(&metadata_path).context("Failed to read checkpoint metadata")?;
    serde_json::from_str(&metadata_json).context("Failed to parse checkpoint metadata")
}

/// Update the training rating fields in a checkpoint's metadata
///
/// This modifies only the metadata.json file without touching model weights.
pub fn update_training_rating(checkpoint_path: &Path, rating: f64, uncertainty: f64) -> Result<()> {
    let mut metadata = load_metadata(checkpoint_path)?;
    metadata.training_rating = rating;
    metadata.training_uncertainty = uncertainty;
    let metadata_path = checkpoint_path.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&metadata_path, metadata_json).context("Failed to write checkpoint metadata")
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
/// Returns a new `StdRng` initialized from the saved seed.
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
/// Returns None if no normalizer was saved (older checkpoint or `normalize_obs=false`).
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
            forked_from: None,
            obs_dim: 4,
            action_count: 2,
            num_players: 1,
            hidden_size: 64,
            num_hidden: 2,
            activation: "tanh".to_string(),
            split_networks: false,
            env_name: "cartpole".to_string(),
            training_rating: 0.0,
            training_uncertainty: 25.0 / 3.0,
        };

        let checkpoint_path = manager.save(&model, &metadata, true).unwrap();
        assert!(checkpoint_path.exists());

        // Verify latest symlink exists
        let latest_symlink = dir.path().join("checkpoints/latest");
        assert!(latest_symlink.exists(), "latest symlink should exist");

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
                    forked_from: None,
                    obs_dim: 4,
                    action_count: 2,
                    num_players: 1,
                    hidden_size: 64,
                    num_hidden: 2,
                    activation: "tanh".to_string(),
                    split_networks: false,
                    env_name: "cartpole".to_string(),
                    training_rating: 0.0,
                    training_uncertainty: 25.0 / 3.0,
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
                    forked_from: None,
                    obs_dim: 4,
                    action_count: 2,
                    num_players: 1,
                    hidden_size: 64,
                    num_hidden: 2,
                    activation: "tanh".to_string(),
                    split_networks: false,
                    env_name: "cartpole".to_string(),
                    training_rating: 0.0,
                    training_uncertainty: 25.0 / 3.0,
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
                    forked_from: None,
                    obs_dim: 4,
                    action_count: 2,
                    num_players: 1,
                    hidden_size: 64,
                    num_hidden: 2,
                    activation: "tanh".to_string(),
                    split_networks: false,
                    env_name: "cartpole".to_string(),
                    training_rating: 0.0,
                    training_uncertainty: 25.0 / 3.0,
                },
                true,
            )
            .unwrap();

        // Verify best points to step 2000
        let best_symlink = dir.path().join("checkpoints/best");
        let best_target = fs::read_link(&best_symlink).unwrap();
        assert!(best_target.to_string_lossy().contains("step_00002000"));

        // Verify latest points to step 3000
        let latest_symlink = dir.path().join("checkpoints/latest");
        let latest_target = fs::read_link(&latest_symlink).unwrap();
        assert!(latest_target.to_string_lossy().contains("step_00003000"));
    }

    #[test]
    fn test_old_checkpoint_without_required_fields_fails() {
        // Old checkpoint JSON missing required fields should fail to deserialize
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

        // This should fail because hidden_size, num_hidden, activation, env_name are missing
        let result: Result<CheckpointMetadata, _> = serde_json::from_str(old_json);
        assert!(
            result.is_err(),
            "Old checkpoint without required fields should fail to deserialize"
        );
    }

    #[test]
    fn test_checkpoint_with_partial_fields_fails() {
        // Checkpoint with some new fields but not all should fail
        let partial_json = r#"{
            "step": 5000,
            "avg_return": 100.0,
            "rng_seed": 42,
            "best_avg_return": 100.0,
            "recent_returns": [100.0],
            "obs_dim": 4,
            "action_count": 2,
            "num_players": 1,
            "hidden_size": 128,
            "num_hidden": 2
        }"#;

        // This should fail because activation and env_name are missing
        let result: Result<CheckpointMetadata, _> = serde_json::from_str(partial_json);
        assert!(
            result.is_err(),
            "Checkpoint with partial fields should fail to deserialize"
        );
    }

    #[test]
    fn test_metadata_roundtrip_with_all_fields() {
        let metadata = CheckpointMetadata {
            step: 1000,
            avg_return: 150.0,
            rng_seed: 42,
            best_avg_return: Some(150.0),
            recent_returns: vec![140.0, 150.0],
            forked_from: Some("parent_run".to_string()),
            obs_dim: 86,
            action_count: 7,
            num_players: 2,
            hidden_size: 256,
            num_hidden: 3,
            activation: "relu".to_string(),
            split_networks: true,
            env_name: "connect_four".to_string(),
            training_rating: 150.5,
            training_uncertainty: 5.0,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let loaded: CheckpointMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.hidden_size, 256);
        assert_eq!(loaded.num_hidden, 3);
        assert_eq!(loaded.activation, "relu");
        assert!(loaded.split_networks);
        assert_eq!(loaded.env_name, "connect_four");
        assert_eq!(loaded.forked_from, Some("parent_run".to_string()));
        assert!((loaded.training_rating - 150.5).abs() < f64::EPSILON);
        assert!((loaded.training_uncertainty - 5.0).abs() < f64::EPSILON);
    }

    // =========================================
    // Normalizer and RNG State Tests
    // =========================================

    #[test]
    fn test_save_load_normalizer_roundtrip() {
        use crate::normalization::ObsNormalizer;

        let dir = tempdir().unwrap();
        let checkpoint_dir = dir.path();

        // Create and update a normalizer
        let mut normalizer = ObsNormalizer::new(4, 10.0);
        normalizer.update_batch(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4);

        // Save it
        save_normalizer(&normalizer, checkpoint_dir).unwrap();

        // Load it back
        let loaded = load_normalizer(checkpoint_dir).unwrap();
        assert!(loaded.is_some(), "Normalizer should be loaded");

        let loaded = loaded.unwrap();
        // Verify by normalizing some data
        let test_data = vec![1.0, 2.0, 3.0, 4.0];
        let norm1 = normalizer.normalize(&test_data);
        let norm2 = loaded.normalize(&test_data);
        assert_eq!(
            norm1, norm2,
            "Loaded normalizer should produce same results"
        );
    }

    #[test]
    fn test_load_normalizer_missing_file() {
        let dir = tempdir().unwrap();
        // No normalizer saved
        let loaded = load_normalizer(dir.path()).unwrap();
        assert!(
            loaded.is_none(),
            "Loading from dir without normalizer should return None"
        );
    }

    #[test]
    fn test_save_load_rng_state_roundtrip() {
        use rand::{Rng, SeedableRng};

        let dir = tempdir().unwrap();
        let checkpoint_dir = dir.path();

        // Create RNG and advance it
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let _: f32 = rng.gen();
        let _: f32 = rng.gen();

        // Save state - this advances the RNG
        save_rng_state(&mut rng, checkpoint_dir).unwrap();

        // Load state - should produce the same sequence as a fresh load
        let loaded1 = load_rng_state(checkpoint_dir).unwrap();
        assert!(loaded1.is_some(), "RNG state should be loaded");

        // Load again - both should produce identical sequences
        let loaded2 = load_rng_state(checkpoint_dir).unwrap();
        assert!(loaded2.is_some());

        let mut rng1 = loaded1.unwrap();
        let mut rng2 = loaded2.unwrap();

        // Both loaded RNGs should produce identical sequences
        for _ in 0..10 {
            assert_eq!(rng1.gen::<f32>(), rng2.gen::<f32>());
        }
    }

    #[test]
    fn test_load_rng_state_missing_file() {
        let dir = tempdir().unwrap();
        // No RNG state saved
        let loaded = load_rng_state(dir.path()).unwrap();
        assert!(
            loaded.is_none(),
            "Loading from dir without rng_state should return None"
        );
    }

    // =========================================
    // Split Network Architecture Mismatch Tests
    // =========================================

    #[test]
    fn test_load_split_checkpoint_with_shared_config() {
        // Save a checkpoint with split_networks=true, load with config that has split_networks=false
        // Should succeed because load uses metadata architecture, not current config
        let dir = tempdir().unwrap();
        let mut manager = CheckpointManager::new(dir.path()).unwrap();
        let device = Default::default();

        // Create config with split networks
        let split_config = Config {
            split_networks: true,
            ..Config::default()
        };

        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, 1, &split_config, &device);

        let metadata = CheckpointMetadata {
            step: 1000,
            avg_return: 100.0,
            rng_seed: 42,
            best_avg_return: Some(100.0),
            recent_returns: vec![100.0],
            forked_from: None,
            obs_dim: 4,
            action_count: 2,
            num_players: 1,
            hidden_size: 64,
            num_hidden: 2,
            activation: "tanh".to_string(),
            split_networks: true, // Saved as split
            env_name: "test".to_string(),
            training_rating: 25.0,
            training_uncertainty: 25.0 / 3.0,
        };

        let checkpoint_path = manager.save(&model, &metadata, true).unwrap();

        // Load with a config that has split_networks=false (mismatched)
        let shared_config = Config::default(); // split_networks defaults to false

        let result =
            CheckpointManager::load::<TestBackend>(&checkpoint_path, &shared_config, &device);
        assert!(
            result.is_ok(),
            "Should load split checkpoint with shared config"
        );

        // Verify model structure matches saved (split), not current config (shared)
        let (loaded_model, loaded_metadata) = result.unwrap();
        assert!(loaded_metadata.split_networks);

        // Forward pass should work
        let (logits, values) = loaded_model.forward(burn::tensor::Tensor::zeros([1, 4], &device));
        assert_eq!(logits.dims(), [1, 2]);
        assert_eq!(values.dims(), [1, 1]);
    }

    #[test]
    fn test_load_shared_checkpoint_with_split_config() {
        // Save a checkpoint with split_networks=false, load with config that has split_networks=true
        // This was the failing case: loading creates wrong structure, causing vec length mismatch
        let dir = tempdir().unwrap();
        let mut manager = CheckpointManager::new(dir.path()).unwrap();
        let device = Default::default();

        // Create config with shared backbone (default)
        let shared_config = Config::default();

        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, 1, &shared_config, &device);

        let metadata = CheckpointMetadata {
            step: 1000,
            avg_return: 100.0,
            rng_seed: 42,
            best_avg_return: Some(100.0),
            recent_returns: vec![100.0],
            forked_from: None,
            obs_dim: 4,
            action_count: 2,
            num_players: 1,
            hidden_size: 64,
            num_hidden: 2,
            activation: "tanh".to_string(),
            split_networks: false, // Saved as shared
            env_name: "test".to_string(),
            training_rating: 25.0,
            training_uncertainty: 25.0 / 3.0,
        };

        let checkpoint_path = manager.save(&model, &metadata, true).unwrap();

        // Load with a config that has split_networks=true (mismatched)
        let split_config = Config {
            split_networks: true,
            ..Config::default()
        };

        let result =
            CheckpointManager::load::<TestBackend>(&checkpoint_path, &split_config, &device);
        assert!(
            result.is_ok(),
            "Should load shared checkpoint with split config (was failing before fix)"
        );

        // Verify model structure matches saved (shared), not current config (split)
        let (loaded_model, loaded_metadata) = result.unwrap();
        assert!(!loaded_metadata.split_networks);

        // Forward pass should work
        let (logits, values) = loaded_model.forward(burn::tensor::Tensor::zeros([1, 4], &device));
        assert_eq!(logits.dims(), [1, 2]);
        assert_eq!(values.dims(), [1, 1]);
    }

    #[test]
    fn test_load_mixed_architecture_checkpoints() {
        // Test loading both split and shared checkpoints simultaneously
        // This simulates eval where one player uses split network and another uses shared
        let dir = tempdir().unwrap();
        let mut manager = CheckpointManager::new(dir.path()).unwrap();
        let device = Default::default();

        // Create and save split network model
        let split_config = Config {
            split_networks: true,
            ..Config::default()
        };
        let split_model: ActorCritic<TestBackend> =
            ActorCritic::new(4, 2, 1, &split_config, &device);

        let split_metadata = CheckpointMetadata {
            step: 1000,
            avg_return: 100.0,
            rng_seed: 42,
            best_avg_return: Some(100.0),
            recent_returns: vec![100.0],
            forked_from: None,
            obs_dim: 4,
            action_count: 2,
            num_players: 1,
            hidden_size: 64,
            num_hidden: 2,
            activation: "tanh".to_string(),
            split_networks: true,
            env_name: "test".to_string(),
            training_rating: 25.0,
            training_uncertainty: 25.0 / 3.0,
        };
        let split_path = manager.save(&split_model, &split_metadata, false).unwrap();

        // Create and save shared network model
        let shared_config = Config::default();
        let shared_model: ActorCritic<TestBackend> =
            ActorCritic::new(4, 2, 1, &shared_config, &device);

        let shared_metadata = CheckpointMetadata {
            step: 2000,
            avg_return: 150.0,
            rng_seed: 43,
            best_avg_return: Some(150.0),
            recent_returns: vec![150.0],
            forked_from: None,
            obs_dim: 4,
            action_count: 2,
            num_players: 1,
            hidden_size: 64,
            num_hidden: 2,
            activation: "tanh".to_string(),
            split_networks: false,
            env_name: "test".to_string(),
            training_rating: 25.0,
            training_uncertainty: 25.0 / 3.0,
        };
        let shared_path = manager
            .save(&shared_model, &shared_metadata, false)
            .unwrap();

        // Load both with a single config (doesn't matter which, metadata should be used)
        let load_config = Config::default();

        let (loaded_split, split_meta) =
            CheckpointManager::load::<TestBackend>(&split_path, &load_config, &device).unwrap();
        let (loaded_shared, shared_meta) =
            CheckpointManager::load::<TestBackend>(&shared_path, &load_config, &device).unwrap();

        // Verify architectures match what was saved
        assert!(split_meta.split_networks);
        assert!(!shared_meta.split_networks);

        // Both models should work with same input
        let input = burn::tensor::Tensor::zeros([1, 4], &device);
        let (split_logits, split_values) = loaded_split.forward(input.clone());
        let (shared_logits, shared_values) = loaded_shared.forward(input);

        // Same output shapes
        assert_eq!(split_logits.dims(), shared_logits.dims());
        assert_eq!(split_values.dims(), shared_values.dims());
    }
}
