/// JSON-lines metrics logging
///
/// Logs metrics to metrics.jsonl in append-only format for crash safety
/// and streaming reads by the Aim watcher.
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

/// JSON-lines logger for training metrics
pub struct MetricsLogger {
    writer: BufWriter<File>,
}

/// Metric types for JSON-lines format
#[derive(Serialize)]
#[serde(tag = "type")]
pub enum Metric {
    #[serde(rename = "hparams")]
    Hyperparams {
        step: usize,
        data: serde_json::Value,
    },
    #[serde(rename = "scalar")]
    Scalar {
        step: usize,
        name: String,
        value: f32,
    },
}

impl MetricsLogger {
    /// Create new logger, opening or creating the metrics file
    pub fn new(run_dir: &Path) -> std::io::Result<Self> {
        let metrics_path = run_dir.join("metrics.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(metrics_path)?;
        let writer = BufWriter::new(file);
        Ok(Self { writer })
    }

    /// Log hyperparameters at the start of training
    pub fn log_hparams<T: Serialize>(&mut self, hparams: &T) -> std::io::Result<()> {
        let data = serde_json::to_value(hparams)?;
        let metric = Metric::Hyperparams { step: 0, data };
        self.write_metric(&metric)
    }

    /// Log a scalar metric
    pub fn log_scalar(&mut self, name: &str, value: f32, step: usize) -> std::io::Result<()> {
        let metric = Metric::Scalar {
            step,
            name: name.to_string(),
            value,
        };
        self.write_metric(&metric)
    }

    /// Write metric to file
    fn write_metric(&mut self, metric: &Metric) -> std::io::Result<()> {
        let line = serde_json::to_string(metric)?;
        writeln!(self.writer, "{line}")?;
        Ok(())
    }

    /// Flush buffered writes to disk
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_logger_creation() {
        let dir = tempdir().unwrap();
        let logger = MetricsLogger::new(dir.path());
        assert!(logger.is_ok());
    }

    #[test]
    fn test_log_scalar() {
        let dir = tempdir().unwrap();
        let mut logger = MetricsLogger::new(dir.path()).unwrap();

        logger.log_scalar("train/loss", 0.5, 100).unwrap();
        logger.flush().unwrap();

        let content = std::fs::read_to_string(dir.path().join("metrics.jsonl")).unwrap();
        assert!(content.contains("train/loss"));
        assert!(content.contains("0.5"));
    }

    #[test]
    fn test_log_hparams() {
        use serde_json::json;

        let dir = tempdir().unwrap();
        let mut logger = MetricsLogger::new(dir.path()).unwrap();

        let hparams = json!({
            "lr": 0.00025,
            "gamma": 0.99
        });
        logger.log_hparams(&hparams).unwrap();
        logger.flush().unwrap();

        let content = std::fs::read_to_string(dir.path().join("metrics.jsonl")).unwrap();
        assert!(content.contains("hparams"));
        assert!(content.contains("0.00025"));
    }
}
