use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod bio;
pub mod encode;
pub mod model;
pub mod parser;
pub mod tokenizer;
pub mod training_data;
pub mod vocab;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Vec<Parameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
    pub required: bool,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool: String,
    pub params: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_serialization() {
        let tool_call = ToolCall {
            tool: "set_alarm".to_string(),
            params: HashMap::from([("time".to_string(), "7:00 AM".to_string())]),
        };

        let json = serde_json::to_string(&tool_call).unwrap();
        assert!(json.contains("set_alarm"));
    }
}
