# Low Pass

A local tool calling framework built in Rust, designed to parse natural language commands and convert 
them into structured tool calls.

## The Concept / Design

### High Level

The system operates via a single forward pass through a lightweight Transformer encoder, topped with
two independent, parallel classification heads. 

```text
                          ┌─> [Head 1: Intent] ──> Predicts Tool (e.g., `set_alarm`)
[ User Input String ] ────┤
                          └─> [Head 2: Slots]  ──> Predicts Params (e.g., "7:00 AM" = time)
```

The results of these two predictions are then used to craft a tool call.

### Local First

`lowpass` is designed to run locally, both training and inference, so it can be embedded in other 
applications.

