use anyhow::Result;
use llama_cpp_2::llama_backend::LlamaBackend;
use ragtime::{
    phi3::{
        llama::{Phi3, Phi3Args},
        prompt::Phi3Prompt,
    },
    QaModel, QaPrompt,
};
use std::{
    io::{stdin, stdout, BufRead, BufReader, Write},
    path::PathBuf,
    sync::Arc,
    thread::available_parallelism,
};

fn test_gen(q: &str) -> Result<()> {
    const BASE: &str = "/home/eric/proj/Phi-3-mini-128k-instruct";
    let backend = Arc::new(LlamaBackend::init()?);
    let mut gen = Phi3::new(Phi3Args {
        backend,
        threads: available_parallelism()?.get() as u32,
        model: PathBuf::from(format!("{BASE}/ggml-model-q8_0.gguf")),
    })?;
    let mut prompt = Phi3Prompt::new();
    std::fmt::Write::write_str(&mut prompt.user(), q)?;
    for tok in gen.ask(prompt.finalize()?, None)? {
        let tok = tok?;
        print!("{tok}");
        stdout().flush()?;
    }
    Ok(())
}

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let mut stdin = BufReader::new(stdin());
    let mut line = String::new();
    stdin.read_line(&mut line)?;
    test_gen(&line)
}
