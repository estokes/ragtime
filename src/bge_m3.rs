use super::session_from_model_file;
use crate::{doc::ChunkId, EmbedModel, Persistable};
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

const DIMS: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Saved {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

pub struct BgeM3 {
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

impl Persistable for BgeM3 {
    type Ctx = ();

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut path = PathBuf::from(path.as_ref());
        path.set_extension("json");
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        serde_json::to_writer_pretty(&mut fd, &self.params)?;
        path.set_extension("usearch");
        Ok(self.index.save(&*path.to_string_lossy())?)
    }

    fn load<P: AsRef<Path>>(_ctx: (), path: P, view: bool) -> Result<Self> {
        let mut fd = File::open(path.as_ref())?;
        let params: Saved = serde_json::from_reader(&mut fd)?;
        let (session, tokenizer) = session_from_model_file(&params.model, &params.tokenizer)?;
        let index = Index::new(&options(DIMS))?;
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
}

impl EmbedModel for BgeM3 {
    type Args = Saved;

    fn new(params: Self::Args) -> Result<Self> {
        let (session, tokenizer) = session_from_model_file(&params.model, &params.tokenizer)?;
        let index = Index::new(&options(DIMS))?;
        index.reserve(1000)?;
        Ok(Self {
            params,
            session,
            tokenizer,
            index,
        })
    }

    fn add(&mut self, text: &[(ChunkId, &str)]) -> Result<()> {
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
    fn search(&mut self, text: &str, n: usize) -> Result<Matches> {
        let qembed = Self::embed(&self.tokenizer, &self.session, vec![text])?;
        let qembed = qembed["sentence_embedding"].try_extract_tensor::<f32>()?;
        let qembed = qembed.slice(s![0, ..]);
        let qembed = qembed
            .as_slice()
            .ok_or_else(|| anyhow!("could not get sentence embedding"))?;
        Ok(self.index.search(qembed, n)?)
    }
}

impl BgeM3 {
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
}
