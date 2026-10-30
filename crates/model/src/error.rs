use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("cannot read array `{name}` from {path}: {message}")]
    Npz {
        path: PathBuf,
        name: String,
        message: String,
    },
    #[error("invalid model: {0}")]
    InvalidModel(String),
    #[error("expected {expected} features, got {got}")]
    FeatureCount { expected: usize, got: usize },
    #[error("invalid training data: {0}")]
    InvalidTrainingData(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;
