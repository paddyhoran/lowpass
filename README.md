# Low Pass

A local tool calling framework built in Rust, designed to parse natural language commands and convert them into structured tool calls.

## Phase 0: Foundation (Current)

This is a minimal working system with hardcoded pattern matching for demonstration purposes.

### Features

- **3 Demo Tools:**
  - `set_alarm` - Set an alarm for a specific time
  - `check_weather` - Check weather for a location
  - `calculate` - Perform mathematical calculations

- **Pattern-based parsing** using regex
- **JSON output** for structured tool calls
- **CLI interface** for easy testing

### Installation

```bash
cargo build --release
```

### Usage

```bash
# Set an alarm
cargo run --bin tcf -- "set alarm for 7am"

# Check weather
cargo run --bin tcf -- "check weather in Seattle"

# Calculate
cargo run --bin tcf -- "calculate 5 + 3"
```

### Example Output

```bash
$ cargo run --bin tcf -- "set alarm for 7am"
{
  "tool": "set_alarm",
  "params": {
    "time": "7am"
  }
}
```

### Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture
```

### Architecture

```
src/
├── lib.rs          # Core data structures (Tool, ToolCall, Parameter)
├── parser.rs       # Pattern-based parser with regex matching
└── bin/
    └── cli.rs      # Command-line interface
```

### Next Steps (Phase 1)

- Replace hardcoded patterns with ML-based inference using `candle`
- Load pre-trained BERT-Tiny model from HuggingFace
- Implement tokenization pipeline
- Add intent classification head

## Development

### Project Structure

- **Library crate:** Core functionality in `src/lib.rs` and modules
- **Binary crate:** CLI tool in `src/bin/cli.rs`
- **Tests:** Unit tests in each module

### Dependencies

- `serde` - Serialization/deserialization
- `serde_json` - JSON support
- `regex` - Pattern matching (Phase 0 only)
