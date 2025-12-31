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
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::network::ActorCritic;

/// Training metadata saved alongside model weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub avg_return: f32,
    pub rng_seed: u64,
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
    /// Returns the path to the saved checkpoint directory
    pub fn save<B: burn::tensor::backend::Backend>(
        &mut self,
        model: &ActorCritic<B>,
        metadata: &CheckpointMetadata,
    ) -> Result<PathBuf> {
        let checkpoint_name = format!("step_{:08}", metadata.step);
        let checkpoint_dir = self.checkpoints_dir.join(&checkpoint_name);

        // Create temp directory for atomic write
        let temp_dir = self.checkpoints_dir.join(format!(".tmp_{}", checkpoint_name));
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

        // Update best symlink if this is a new best
        if metadata.avg_return > self.best_avg_return {
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
        let metadata_json = fs::read_to_string(&metadata_path)
            .context("Failed to read checkpoint metadata")?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;

        // Create default model structure (to be populated with loaded weights)
        // We need obs_dim and action_count - for now use CartPole defaults
        // TODO: Save these in metadata for generic loading
        let obs_dim = 4;
        let action_count = 2;
        let default_model: ActorCritic<B> = ActorCritic::new(obs_dim, action_count, config, device);

        // Load model using Burn's recorder
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model_path = checkpoint_dir.join("model");
        let model = default_model
            .load_file(model_path, &recorder, device)
            .context("Failed to load model")?;

        Ok((model, metadata))
    }

    /// Get the path to the latest checkpoint, if any
    pub fn latest_checkpoint(&self) -> Option<PathBuf> {
        let latest = self.checkpoints_dir.join("latest");
        if latest.exists() {
            fs::read_link(&latest).ok()
        } else {
            None
        }
    }

    /// Get the path to the best checkpoint, if any
    pub fn best_checkpoint(&self) -> Option<PathBuf> {
        let best = self.checkpoints_dir.join("best");
        if best.exists() {
            fs::read_link(&best).ok()
        } else {
            None
        }
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

        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, &config, &device);
        let metadata = CheckpointMetadata {
            step: 1000,
            avg_return: 150.0,
            rng_seed: 42,
        };

        let checkpoint_path = manager.save(&model, &metadata).unwrap();
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
        assert_eq!(values.dims(), [1]);
    }

    #[test]
    fn test_best_checkpoint_tracking() {
        let dir = tempdir().unwrap();
        let mut manager = CheckpointManager::new(dir.path()).unwrap();
        let device = Default::default();
        let config = Config::default();

        let model: ActorCritic<TestBackend> = ActorCritic::new(4, 2, &config, &device);

        // Save first checkpoint with low return
        manager
            .save(
                &model,
                &CheckpointMetadata {
                    step: 1000,
                    avg_return: 100.0,
                    rng_seed: 42,
                },
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
                },
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
                },
            )
            .unwrap();

        // Verify best points to step 2000
        let best = manager.best_checkpoint().unwrap();
        assert!(best.to_string_lossy().contains("step_00002000"));

        // Verify latest points to step 3000
        let latest = manager.latest_checkpoint().unwrap();
        assert!(latest.to_string_lossy().contains("step_00003000"));
    }
}
