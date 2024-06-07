use anyhow::{anyhow, bail, Result};
use ndarray::{s, Array2, Axis};
use ort::{inputs, Session, SessionOutputs};
use std::{path::Path, thread::available_parallelism};
use tokenizers::Tokenizer;
use usearch::{ffi::Matches, Index, IndexOptions, MetricKind, ScalarKind};

pub struct EmbedDb {
    session: Session,
    tokenizer: Tokenizer,
    index: Index,
    embeddings: Vec<String>,
}

impl EmbedDb {
    pub fn new<P: AsRef<Path>>(model: P, tokenizer: P, dims: usize) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(available_parallelism()?.get())?
            .commit_from_file(model)?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{e:?}"))?;
        let mut opts = IndexOptions::default();
        opts.dimensions = dims;
        opts.metric = MetricKind::Cos;
        opts.quantization = ScalarKind::F32;
        let index = Index::new(&opts)?;
        index.reserve(1000)?;
        Ok(Self {
            session,
            tokenizer,
            index,
            embeddings: vec![],
        })
    }

    fn embed<'a>(
        tokenizer: &Tokenizer,
        session: &'a Session,
        text: Vec<String>,
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

    pub fn add(&mut self, text: Vec<String>) -> Result<()> {
        let embed = Self::embed(&self.tokenizer, &self.session, text.clone())?;
        let embed = embed["sentence_embedding"].try_extract_tensor::<f32>()?;
        for (e, s) in embed.axis_iter(Axis(0)).zip(text.into_iter()) {
            let d = e.as_slice().ok_or_else(|| anyhow!("could not get slice"))?;
            self.embeddings.push(s);
            self.index.add((self.embeddings.len() - 1) as u64, d)?
        }
        Ok(())
    }

    pub fn search(&self, text: String, n: usize) -> Result<Matches> {
        let qembed = Self::embed(&self.tokenizer, &self.session, vec![text])?;
        let qembed = qembed["sentence_embedding"].try_extract_tensor::<f32>()?;
        let qembed = qembed.slice(s![0, ..]);
        let qembed = qembed
            .as_slice()
            .ok_or_else(|| anyhow!("could not get sentence embedding"))?;
        Ok(self.index.search(qembed, n)?)
    }

    pub fn get(&self, key: u64) -> Result<&str> {
        let i = key as usize;
        if i >= self.embeddings.len() {
            bail!("no such embedding {i}")
        }
        Ok(&self.embeddings[key as usize])
    }
}
