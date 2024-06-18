use crate::{doc::ChunkId, session_from_model_file};
use anyhow::{anyhow, bail, Result};
use ndarray::{s, Array2, Axis};
use ort::{inputs, Session, SessionOutputs};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;
use usearch::{ffi::Matches, Index, IndexOptions, MetricKind, ScalarKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Saved {
    model: PathBuf,
    tokenizer: PathBuf,
    dims: usize,
}

pub struct EmbedDb {
    params: Saved,
    session: Session,
    tokenizer: Tokenizer,
    index: Index,
}

fn options(dims: usize) -> IndexOptions {
    let mut opts = IndexOptions::default();
    opts.dimensions = dims;
    opts.metric = MetricKind::Cos;
    opts.quantization = ScalarKind::F32;
    opts
}

impl EmbedDb {
    pub fn new<P: AsRef<Path>>(model: P, tokenizer: P, dims: usize) -> Result<Self> {
        let params = Saved {
            model: PathBuf::from(model.as_ref()),
            tokenizer: PathBuf::from(tokenizer.as_ref()),
            dims,
        };
        let (session, tokenizer) = session_from_model_file(model, tokenizer)?;
        let index = Index::new(&options(dims))?;
        index.reserve(1000)?;
        Ok(Self {
            params,
            session,
            tokenizer,
            index,
        })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut path = PathBuf::from(path.as_ref());
        path.set_extension("json");
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        serde_json::to_writer_pretty(&mut fd, &self.params)?;
        path.set_extension("usearch");
        Ok(self.index.save(&*path.to_string_lossy())?)
    }

    pub fn load<P: AsRef<Path>>(&self, path: P, view: bool) -> Result<Self> {
        let mut fd = File::open(path.as_ref())?;
        let params: Saved = serde_json::from_reader(&mut fd)?;
        let (session, tokenizer) = session_from_model_file(&params.model, &params.tokenizer)?;
        let index = Index::new(&options(params.dims))?;
        let mut path = PathBuf::from(path.as_ref());
        path.set_extension("usearch");
        let path = path.to_string_lossy();
        if view {
            index.view(&*path)?
        } else {
            index.load(&*path)?;
        }
        Ok(Self {
            params,
            session,
            tokenizer,
            index,
        })
    }

    fn embed<'a>(
        tokenizer: &Tokenizer,
        session: &'a Session,
        text: Vec<&str>,
    ) -> Result<SessionOutputs<'a>> {
        if text.len() == 0 {
            bail!("can't add an empty batch")
        }
        let tokens = tokenizer
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
        Ok(session.run(inputs![input, attention_mask]?)?)
    }

    pub fn add(&mut self, text: Vec<(ChunkId, &str)>) -> Result<()> {
        let embed = Self::embed(
            &self.tokenizer,
            &self.session,
            text.iter().map(|(_, t)| *t).collect(),
        )?;
        let embed = embed["sentence_embedding"].try_extract_tensor::<f32>()?;
        for (e, (id, _)) in embed.axis_iter(Axis(0)).zip(text.iter()) {
            let d = e.as_slice().ok_or_else(|| anyhow!("could not get slice"))?;
            self.index.add(id.0, d)?
        }
        Ok(())
    }

    /// The keys component of Matches is a vec of ChunkIds represented as u64s.
    pub fn search(&self, text: &str, n: usize) -> Result<Matches> {
        let qembed = Self::embed(&self.tokenizer, &self.session, vec![text])?;
        let qembed = qembed["sentence_embedding"].try_extract_tensor::<f32>()?;
        let qembed = qembed.slice(s![0, ..]);
        let qembed = qembed
            .as_slice()
            .ok_or_else(|| anyhow!("could not get sentence embedding"))?;
        Ok(self.index.search(qembed, n)?)
    }
}
