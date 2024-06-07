use std::thread::available_parallelism;
use anyhow::{anyhow, Result};
use ort::{inputs, Session};
use tokenizers::Tokenizer;
use ndarray::Array1;

pub fn embed() -> Result<()> {
    const BASE: &str = "/home/eric/proj/bge-m3/onnx";
    tracing_subscriber::fmt::init();
    ort::init().commit()?;
    let session = Session::builder()?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_intra_threads(available_parallelism()?.get())?
        .commit_from_file(format!("{BASE}/model.onnx"))?;
    let tokenizer =
        Tokenizer::from_file(format!("{BASE}/tokenizer.json")).map_err(|e| anyhow!("{e:?}"))?;
    let tokens = tokenizer
        .encode("I've got a lovely bunch of coconuts", false)
        .map_err(|e| anyhow!("{e:?}"))?;
    let tokens = Array1::from_iter(tokens.get_ids().iter().copied());
    let outputs = session.run(inputs![tokens]?)?;
    println!("{outputs:?}");
    Ok(())
}
