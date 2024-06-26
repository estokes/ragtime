use std::{io::{stdin, BufRead, BufReader}, sync::Arc};
use chrono::prelude::*;
use anyhow::Result;
use llama_cpp_2::llama_backend::LlamaBackend;
use ragtime::RagQa;

const EMB_BASE: &str = "/home/eric/proj";
const QA_BASE: &str =
    "/home/eric/proj/Phi-3-mini-128k-instruct";

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let backend = Arc::new(LlamaBackend::init()?);
    let mut qa = RagQa::new(
        backend,
        64,
        format!("{EMB_BASE}/ggml-sfr-embedding-mistral-q8_0.gguf"),
        format!("{QA_BASE}/ggml-model-q8_0.gguf"),
    )?;
    qa.add_document("/home/eric/Downloads/Fowl Engine.txt", 256, 128)?;
    let mut line = String::new();
    let mut stdin = BufReader::new(stdin());
    loop {
        line.clear();
        stdin.read_line(&mut line)?;
        let start = Utc::now();
        let _res = qa.ask(&line, 1000)?;
        //println!("{}", res);
        println!("\nquery time: {}ms", (Utc::now() - start).num_milliseconds());
    }
}
