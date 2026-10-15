base_model_dir := "models/bert-tiny"

dev: clear fmt clippy test

clear:
    clear

# Check formatting (development context)
fmt:
    @echo "==> Formatting..."
    @cargo fmt --all
    @cargo fix --all-targets --allow-dirty

# Lint with Clippy
clippy:
    @echo "==> Linting with Clippy..."
    @cargo clippy --all-targets -- -D warnings

# Runs all tests
test:
    @echo "==> Running tests (debug)..."
    @cargo test -- --nocapture

# Download the base checkpoint (if needed) and scaffold models/<name>/ with an example tool
start name:
    cargo run --bin download_model -- {{base_model_dir}}
    cargo run --bin init_model -- {{name}}

# Generate BIO-tagged training data from models/<name>/data/tools templates
generate name:
    cargo run --bin generator -- models/{{name}}/data/tools models/{{name}}/data/training_data.jsonl

# Train the intent + slot heads for <name> on its generated training data
train name:
    cargo run --bin train -- models/{{name}}/data/training_data.jsonl {{base_model_dir}} models/{{name}}/artifacts

# Package a trained model into a standalone crate at models/<name>/<name>/
create name:
    cargo run --bin create_model -- {{name}}
