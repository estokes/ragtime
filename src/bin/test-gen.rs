use anyhow::Result;
use ragtime::qa::QaModel;
use std::io::{stdin, BufRead, BufReader};

fn test_gen(q: &str) -> Result<()> {
    const BASE: &str =
        "/home/eric/proj/Phi-3-mini-128k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32";
    let gen = QaModel::new(
        &format!("{BASE}/phi3-mini-128k-instruct-cpu-int4-rtn-block-32.onnx"),
        &format!("{BASE}/tokenizer.json"),
    )?;
    let a = gen.ask(&format!("<|user|>{q} <|end|>\n<|assistant|>"), 1000)?;
    println!("{a}");
    Ok(())
}

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    ort::init().commit()?;
    let mut stdin = BufReader::new(stdin());
    let mut line = String::new();
    stdin.read_line(&mut line)?;
    test_gen(&line)
}
