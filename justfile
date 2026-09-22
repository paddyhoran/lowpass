tools_dir := "data/tools"
data_path := "data/training_data.jsonl"
model_dir := "models/bert-tiny"
checkpoint_dir := "models/trained"

# Generate BIO-tagged training data from data/tools templates
generate:
    cargo run --bin generator -- {{tools_dir}} {{data_path}}

# Train the intent + slot heads on the generated training data
train:
    cargo run --bin train -- {{data_path}} {{model_dir}} {{checkpoint_dir}}
