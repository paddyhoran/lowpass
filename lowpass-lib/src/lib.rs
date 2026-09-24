use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod infer;
pub mod model;
pub mod tokenizer;
pub mod vocab;

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
