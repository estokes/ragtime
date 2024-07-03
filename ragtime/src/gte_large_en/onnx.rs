use crate::{doc::ChunkId, l2_normalize, session_from_model_file, EmbedModel, Persistable};
use anyhow::{anyhow, bail, Result};
use ndarray::{Array1, Array2, Axis};
use ort::{inputs, Session};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions}, iter, path::{Path, PathBuf}
};
use tokenizers::Tokenizer;
use usearch::{ffi::Matches, Index, IndexOptions, MetricKind, ScalarKind};

const DIMS: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GteLargeEnArgs {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

pub struct GteLargeEn {
    params: GteLargeEnArgs,
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

impl Persistable for GteLargeEn {
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
        let params: GteLargeEnArgs = serde_json::from_reader(&mut fd)?;
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

impl EmbedModel for GteLargeEn {
    type Args = GteLargeEnArgs;

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

    fn add<S: AsRef<str>>(&mut self, summary: S, text: &[(ChunkId, S)]) -> Result<()> {
        let embed = Self::embed(
            &self.tokenizer,
            &self.session,
            iter::once(summary.as_ref())
                .chain(text.iter().map(|(_, t)| t.as_ref()))
                .collect(),
        )?;
        let mut iter = embed.axis_iter(Axis(0));
        let summary = &iter
            .next()
            .ok_or_else(|| anyhow!("no summary"))? * 0.5;
        let mut tmp = Array1::zeros(summary.shape()[0]);
        for (e, (id, _)) in iter.zip(text.iter()) {
            tmp.assign(&e);
            tmp += &summary;
            l2_normalize(tmp.as_slice_mut().ok_or_else(|| anyhow!("could not get embedding"))?);
            self.index.add(id.0, tmp.as_slice().unwrap())?
        }
        Ok(())
    }

    /// The keys component of Matches is a vec of ChunkIds represented as u64s.
    fn search<S: AsRef<str>>(&mut self, text: S, n: usize) -> Result<Matches> {
        let mut qembed = Self::embed(&self.tokenizer, &self.session, vec![text.as_ref()])?;
        let qembed = qembed
            .as_slice_mut()
            .ok_or_else(|| anyhow!("could not get embedding"))?;
        l2_normalize(qembed);
        Ok(self.index.search(qembed, n)?)
    }
}

impl GteLargeEn {
    fn embed<'a>(
        tokenizer: &Tokenizer,
        session: &'a Session,
        text: Vec<&str>,
    ) -> Result<Array2<f32>> {
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
        let token_type_ids: Array2<i64> = Array2::zeros((tokens.len(), longest));
        let res = session.run(inputs![input, attention_mask, token_type_ids]?)?;
        let res = res["last_hidden_state"].try_extract_tensor::<f32>()?;
        let shape = res.shape();
        let mut out: Array2<f32> = Array2::zeros((shape[0], shape[2]));
        for (i, e) in res.axis_iter(Axis(0)).enumerate() {
            let mut v = out.row_mut(i);
            v.assign(&e.mean_axis(Axis(0)).ok_or_else(|| anyhow!("could not get mean"))?);
        }
        Ok(out)
    }
}
