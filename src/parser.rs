use crate::{Tool, ToolCall, Parameter};
use regex::Regex;
use std::collections::HashMap;

pub struct Parser {
    tools: Vec<Tool>,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            tools: Self::initialize_demo_tools(),
        }
    }

    fn initialize_demo_tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "set_alarm".to_string(),
                description: "Set an alarm for a specific time".to_string(),
                parameters: vec![
                    Parameter {
                        name: "time".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Time for the alarm (e.g., '7:00 AM', '7am')".to_string(),
                    },
                ],
            },
            Tool {
                name: "check_weather".to_string(),
                description: "Check the weather for a location".to_string(),
                parameters: vec![
                    Parameter {
                        name: "location".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Location to check weather for".to_string(),
                    },
                ],
            },
            Tool {
                name: "calculate".to_string(),
                description: "Perform a mathematical calculation".to_string(),
                parameters: vec![
                    Parameter {
                        name: "expression".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        description: "Mathematical expression to evaluate".to_string(),
                    },
                ],
            },
        ]
    }

    pub fn parse(&self, input: &str) -> Option<ToolCall> {
        let input_lower = input.to_lowercase();

        if let Some(tool_call) = self.parse_alarm(&input_lower, input) {
            return Some(tool_call);
        }

        if let Some(tool_call) = self.parse_weather(&input_lower, input) {
            return Some(tool_call);
        }

        if let Some(tool_call) = self.parse_calculate(&input_lower, input) {
            return Some(tool_call);
        }

        None
    }

    fn parse_alarm(&self, input_lower: &str, input: &str) -> Option<ToolCall> {
        if !input_lower.contains("alarm") && !input_lower.contains("wake") {
            return None;
        }

        let time_pattern = Regex::new(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))").unwrap();
        
        if let Some(captures) = time_pattern.captures(input) {
            let time = captures.get(1).unwrap().as_str().to_string();
            return Some(ToolCall {
                tool: "set_alarm".to_string(),
                params: HashMap::from([("time".to_string(), time)]),
            });
        }

        let simple_time = Regex::new(r"(\d{1,2})\s*(?:am|pm|AM|PM)").unwrap();
        if let Some(captures) = simple_time.captures(input) {
            let time = captures.get(0).unwrap().as_str().to_string();
            return Some(ToolCall {
                tool: "set_alarm".to_string(),
                params: HashMap::from([("time".to_string(), time)]),
            });
        }

        None
    }

    fn parse_weather(&self, input_lower: &str, input: &str) -> Option<ToolCall> {
        if !input_lower.contains("weather") && !input_lower.contains("temperature") {
            return None;
        }

        let location_patterns = vec![
            Regex::new(r"(?:in|for|at)\s+([A-Z][a-zA-Z\s]+?)(?:\s|$|\?)").unwrap(),
            Regex::new(r"weather\s+([A-Z][a-zA-Z\s]+?)(?:\s|$|\?)").unwrap(),
        ];

        for pattern in location_patterns {
            if let Some(captures) = pattern.captures(input) {
                let location = captures.get(1).unwrap().as_str().trim().to_string();
                return Some(ToolCall {
                    tool: "check_weather".to_string(),
                    params: HashMap::from([("location".to_string(), location)]),
                });
            }
        }

        None
    }

    fn parse_calculate(&self, input_lower: &str, input: &str) -> Option<ToolCall> {
        if !input_lower.contains("calculate") && !input_lower.contains("compute") 
            && !input_lower.contains("what is") && !input_lower.contains("what's") {
            return None;
        }

        let math_pattern = Regex::new(r"(\d+\s*[\+\-\*/]\s*\d+(?:\s*[\+\-\*/]\s*\d+)*)").unwrap();
        
        if let Some(captures) = math_pattern.captures(input) {
            let expression = captures.get(1).unwrap().as_str().to_string();
            return Some(ToolCall {
                tool: "calculate".to_string(),
                params: HashMap::from([("expression".to_string(), expression)]),
            });
        }

        None
    }

    pub fn get_tools(&self) -> &[Tool] {
        &self.tools
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_alarm() {
        let parser = Parser::new();
        
        let result = parser.parse("set alarm for 7am");
        assert!(result.is_some());
        let tool_call = result.unwrap();
        assert_eq!(tool_call.tool, "set_alarm");
        assert!(tool_call.params.contains_key("time"));
    }

    #[test]
    fn test_parse_weather() {
        let parser = Parser::new();
        
        let result = parser.parse("check weather in Seattle");
        assert!(result.is_some());
        let tool_call = result.unwrap();
        assert_eq!(tool_call.tool, "check_weather");
        assert_eq!(tool_call.params.get("location").unwrap(), "Seattle");
    }

    #[test]
    fn test_parse_calculate() {
        let parser = Parser::new();
        
        let result = parser.parse("calculate 5 + 3");
        assert!(result.is_some());
        let tool_call = result.unwrap();
        assert_eq!(tool_call.tool, "calculate");
        assert_eq!(tool_call.params.get("expression").unwrap(), "5 + 3");
    }

    #[test]
    fn test_parse_unknown() {
        let parser = Parser::new();
        
        let result = parser.parse("hello world");
        assert!(result.is_none());
    }
}
