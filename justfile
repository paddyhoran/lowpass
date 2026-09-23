base_model_dir := "models/bert-tiny"

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
