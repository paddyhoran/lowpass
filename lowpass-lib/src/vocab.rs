//! Contains a simple vocab collection implementation.

/// Maps string labels (intents, or BIO slot tags) to/from dense integer ids.
#[derive(Debug, Clone)]
pub struct Vocab {
    labels: Vec<String>,
}

impl Vocab {
    pub fn new(labels: Vec<String>) -> Self {
        Self { labels }
    }

    pub fn id(&self, label: &str) -> Option<u32> {
        self.labels.iter().position(|l| l == label).map(|i| i as u32)
    }

    pub fn label(&self, id: u32) -> &str {
        &self.labels[id as usize]
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }
}
