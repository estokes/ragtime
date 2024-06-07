use anyhow::{anyhow, bail, Result};
use ndarray::{Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use ort::{inputs, Session, SessionOutputs, Value};
use std::{path::Path, thread::available_parallelism};
use tokenizers::Tokenizer;
use usearch::{Index, IndexOptions};

struct EmbedModel {
    session: Session,
    tokenizer: Tokenizer,
}

impl EmbedModel {
    fn new<P: AsRef<Path>>(model: P, tokenizer: P) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(available_parallelism()?.get())?
            .commit_from_file(model)?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{e:?}"))?;
        Ok(Self { session, tokenizer })
    }

    fn embed<'a>(&'a self, text: Vec<String>) -> Result<SessionOutputs<'a>> {
        if text.len() == 0 {
            bail!("can't embed empty batch")
        }
        let tokens = self
            .tokenizer
            .encode_batch(text, false)
            .map_err(|e| anyhow!("{e:?}"))?;
        let longest = tokens
            .iter()
            .map(|t| t.get_ids().len())
            .max()
            .ok_or_else(|| anyhow!("no tokens returned from tokenizer"))?;
        let input = Array2::from_shape_fn((tokens.len(), longest), |(i, j)| {
            let enc = &tokens[i];
            if j >= enc.len() {
                0i64
            } else {
                enc.get_ids()[j] as i64
            }
        });
        let attention_mask = Array2::from_shape_fn((tokens.len(), longest), |(i, j)| {
            let enc = &tokens[i];
            if j >= enc.len() {
                0i64
            } else {
                1i64
            }
        });
        Ok(self.session.run(inputs![input, attention_mask]?)?)
    }
}

pub fn main() -> Result<()> {
    const BASE: &str = "/home/eric/proj/bge-m3/onnx";
    tracing_subscriber::fmt::init();
    ort::init().commit()?;
    let embedding_model = EmbedModel::new(
        &format!("{BASE}/model.onnx"),
        &format!("{BASE}/tokenizer.json"),
    )?;
    let embed = embedding_model.embed(vec![
        "I've got a lovely bunch of coconuts".into(),
        "I like coconuts very much".into(),
    ])?;
    let embed = embed["sentence_embedding"].try_extract_tensor::<f32>()?;
    println!("{:?}", embed.shape());
    Ok(())
}
