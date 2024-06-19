use std::io::{stdin, BufRead, BufReader};

use anyhow::Result;
use ragtime::RagQa;

const EMB_BASE: &str = "/home/eric/proj/bge-m3/onnx";
const QA_BASE: &str =
    "/home/eric/proj/Phi-3-mini-128k-instruct";

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    ort::init().commit()?;
    let mut qa = RagQa::new(
        64,
        format!("{EMB_BASE}/model.onnx"),
        format!("{EMB_BASE}/tokenizer.json"),
        1024,
        format!("{QA_BASE}/ggml-model-q8_0.gguf"),
    )?;
    qa.add_document("/home/eric/Downloads/Fowl Engine.txt", 256, 128)?;
    let mut line = String::new();
    let mut stdin = BufReader::new(stdin());
    loop {
        line.clear();
        stdin.read_line(&mut line)?;
        let prompt = qa.encode_prompt(&line)?;
        dbg!(&prompt);
        println!("{}", qa.ask(&prompt, 1000)?);
    }
}
