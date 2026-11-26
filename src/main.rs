use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use dotenv;
use serde_json::{Value, json};
use std::{env, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

fn execute_tool(name: &str, arguments: &str) -> String {
    let args: Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(e) => return format!("Error: invalid tool arguments: {e}"),
    };

    match name {
        "Read" => match args["file_path"].as_str() {
            Some(path) => std::fs::read_to_string(path)
                .unwrap_or_else(|e| format!("Error reading {path}: {e}")),
            None => "Error: missing file_path".to_string(),
        },

        "Write" => match (args["file_path"].as_str(), args["content"].as_str()) {
            (Some(path), Some(content)) => match std::fs::write(path, content) {
                Ok(_) => "File written successfully".to_string(),
                Err(e) => format!("Error writing {path}: {e}"),
            },
            _ => "Error: missing file_path or content".to_string(),
        },

        _ => format!("Error: unknown tool {name}"),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let tools = json!(
    [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read and return the contents of a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path to the file to read"
                                    }
                                },
                                "required": ["file_path"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "description": "Write content to a file",
                            "parameters": {
                            "type": "object",
                            "required": ["file_path", "content"],
                            "properties": {
                                "file_path": {
                                "type": "string",
                                "description": "The path of the file to write to"
                                },
                                "content": {
                                "type": "string",
                                "description": "The content to write to the file"
                                }
                            }
                        }
                    }
                }
             ]
        );

    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": args.prompt })];

    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "messages": messages,
                "model": "anthropic/claude-haiku-4.5",
                "max_tokens": 4000,
                "tools": tools
            }))
            .await?;

        // append response message to messages
        let message = response["choices"][0]["message"].clone();
        messages.push(message.clone());

        match message["tool_calls"].as_array() {
            Some(tool_calls) if !tool_calls.is_empty() => {
                // for each tool_call: execute and append the result
                for tool_call in tool_calls {
                    let id = tool_call["id"].as_str().unwrap_or_default();
                    let name = tool_call["function"]["name"].as_str().unwrap_or_default();
                    let arguments = tool_call["function"]["arguments"].as_str().unwrap_or("{}");

                    let result = execute_tool(name, arguments);

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": id,
                        "content": result
                    }));
                }
            }
            _ => {
                // no tool_calls: print content and exit
                println!("{}", message["content"].as_str().unwrap_or(""));
                break;
            }
        }
    }

    Ok(())
}
