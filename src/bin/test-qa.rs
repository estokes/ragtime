use anyhow::Result;
use chrono::prelude::*;
use llama_cpp_2::llama_backend::LlamaBackend;
use ragtime::{bge_m3::onnx::BgeArgs, phi3::llama::Phi3Args, RagQaPhi3BgeM3};
use std::{
    io::{stdin, stdout, BufRead, BufReader, Write},
    path::PathBuf,
    sync::Arc,
};

const EMB_BASE: &str = "/home/eric/proj/bge-m3/onnx";
const QA_BASE: &str = "/home/eric/proj/Phi-3-mini-128k-instruct";

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    ort::init().commit()?;
    let backend = Arc::new(LlamaBackend::init()?);
    let mut qa = RagQaPhi3BgeM3::new(
        64,
        BgeArgs {
            model: PathBuf::from(format!("{EMB_BASE}/model.onnx")),
            tokenizer: PathBuf::from(format!("{EMB_BASE}/tokenizer.json")),
        },
        Phi3Args {
            backend,
            model: PathBuf::from(format!("{QA_BASE}/ggml-model-q8_0.gguf")),
        },
    )?;
    qa.add_document("/home/eric/Downloads/Fowl Engine.txt", 256, 128)?;
    let mut line = String::new();
    let mut stdin = BufReader::new(stdin());
    loop {
        line.clear();
        stdin.read_line(&mut line)?;
        let start = Utc::now();
        for tok in qa.ask(&line, None)? {
            let tok = tok?;
            print!("{tok}");
            stdout().flush()?;
        }
        println!(
            "\nquery time: {}ms",
            (Utc::now() - start).num_milliseconds()
        );
    }
}
