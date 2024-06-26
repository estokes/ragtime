use anyhow::Result;
use llama_cpp_2::llama_backend::LlamaBackend;
use ragtime::qa_llama::QaModel;
use std::{io::{stdin, BufRead, BufReader}, sync::Arc};

fn test_gen(q: &str) -> Result<()> {
    const BASE: &str = "/home/eric/proj/Phi-3-mini-128k-instruct";
    let backend = Arc::new(LlamaBackend::init()?);
    let gen = QaModel::new(backend, &format!("{BASE}/ggml-model-q8_0.gguf"))?;
    let a = gen.ask(q, 1000)?;
    println!("{a}");
    Ok(())
}

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let mut stdin = BufReader::new(stdin());
    let mut line = String::new();
    stdin.read_line(&mut line)?;
    test_gen(&line)
}
