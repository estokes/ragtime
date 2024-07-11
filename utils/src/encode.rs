use anyhow::Result;
use ragtime::{
    bge_m3::onnx::{BgeArgs, BgeM3},
    doc::ChunkId,
    EmbedModel,
};
use std::path::PathBuf;

fn test_encode() -> Result<()> {
    const BASE: &str = "/home/eric/proj/bge-m3/onnx";
    let mut edb = BgeM3::new(
        (),
        BgeArgs {
            model: PathBuf::from(format!("{BASE}/model.onnx")),
            tokenizer: PathBuf::from(format!("{BASE}/tokenizer.json")),
        },
    )?;
    edb.add(
        "".into(),
        &[
            (ChunkId::new(), "I've got a lovely bunch of coconuts".into()),
            (ChunkId::new(), "I like coconuts very much".into()),
            (
                ChunkId::new(),
                "A goomba is a character from super mario bros".into(),
            ),
        ],
    )?;
    let m = edb.search("who here likes coconuts?".into(), 3)?;
    println!("{m:?}");
    Ok(())
}

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    ort::init().commit()?;
    test_encode()
}
