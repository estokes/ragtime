use anyhow::{bail, Result};
use clap::Parser;
use llama_cpp_2::llama_backend::LlamaBackend;
use ragtime::{
    phi3::{
        llama::{Phi3, Phi3Args},
        prompt::Phi3Prompt,
    },
    QaModel, QaPrompt,
};
use std::{
    io::{stdout, Write},
    path::PathBuf,
    sync::Arc,
    thread::available_parallelism,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, help = "path to the model")]
    model: PathBuf,
    #[arg(long, help = "the number of threads to use (default: all available)")]
    threads: Option<u32>,
    #[arg(long, help = "suppress llama.cpp logging")]
    quiet: bool,
    #[arg(
        long,
        default_value = "8",
        help = "the context divisor (trade context for less memory use)"
    )]
    ctx_divisor: u32,
    #[arg(long, help = "the prompt file")]
    prompt_file: Option<PathBuf>,
    #[arg(long, help = "the prompt string")]
    prompt: Option<String>,
    #[arg(long, help = "the system prompt")]
    system_prompt: Option<String>,
    #[arg(long, default_value = "42", help = "random seed")]
    seed: u32,
}

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    if args.prompt.is_some() && args.prompt_file.is_some() {
        bail!("prompt and prompt_file are mutually exclusive");
    }
    if args.prompt.is_none() && args.prompt_file.is_none() {
        bail!("prompt or prompt_file is required");
    }
    let backend = Arc::new({
        let mut be = LlamaBackend::init()?;
        if args.quiet {
            be.void_logs();
        }
        be
    });
    let mut gen = Phi3::new(Phi3Args {
        backend,
        threads: args
            .threads
            .unwrap_or(available_parallelism()?.get() as u32),
        ctx_divisor: args.ctx_divisor,
        seed: args.seed,
        model: args.model,
    })?;
    let mut prompt = Phi3Prompt::new();
    let prompt_str = if let Some(prompt_file) = args.prompt_file {
        std::fs::read_to_string(prompt_file)?
    } else {
        args.prompt.unwrap()
    };
    if let Some(system_prompt) = args.system_prompt.as_ref() {
        std::fmt::Write::write_str(&mut prompt.system(), system_prompt)?;
    }
    std::fmt::Write::write_str(&mut prompt.user(), &prompt_str)?;
    for tok in gen.ask(prompt.finalize()?, None)? {
        let tok = tok?;
        print!("{tok}");
        stdout().flush()?;
    }
    Ok(())
}
