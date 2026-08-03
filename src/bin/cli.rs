use lowpass::parser::Parser;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let parser = Parser::new();

    if args.len() > 1 {
        let input = args[1..].join(" ");
        process_input(&parser, &input);
    } else {
        eprintln!("Usage: tcf \"<your command>\"");
        eprintln!("\nExamples:");
        eprintln!("  tcf \"set alarm for 7am\"");
        eprintln!("  tcf \"check weather in Seattle\"");
        eprintln!("  tcf \"calculate 5 + 3\"");
        std::process::exit(1);
    }
}

fn process_input(parser: &Parser, input: &str) {
    match parser.parse(input) {
        Some(tool_call) => {
            let json =
                serde_json::to_string_pretty(&tool_call).expect("Failed to serialize tool call");
            println!("{}", json);
        }
        None => {
            eprintln!("Error: Could not parse input into a tool call");
            eprintln!("Input: {}", input);
            std::process::exit(1);
        }
    }
}
