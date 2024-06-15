use anyhow::{anyhow, bail, Result};
use ndarray::{array, concatenate, s, Array1, Array2, Array4, ArrayBase, Axis, Dim, ViewRepr};
use ort::{inputs, DynValue, Session, SessionInputValue, SessionOutputs};
use std::{collections::HashMap, path::Path, thread::available_parallelism};
use tokenizers::Tokenizer;
use usearch::{ffi::Matches, Index, IndexOptions, MetricKind, ScalarKind};

pub mod doc;

fn session_from_model_file<P: AsRef<Path>>(model: P, tokenizer: P) -> Result<(Session, Tokenizer)> {
    let session = Session::builder()?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_intra_threads(available_parallelism()?.get())?
        .commit_from_file(model)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{e:?}"))?;
    Ok((session, tokenizer))
}

pub struct EmbedDb {
    session: Session,
    tokenizer: Tokenizer,
    index: Index,
    embeddings: Vec<String>,
}

impl EmbedDb {
    pub fn new<P: AsRef<Path>>(model: P, tokenizer: P, dims: usize) -> Result<Self> {
        let (session, tokenizer) = session_from_model_file(model, tokenizer)?;
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

pub struct QaModel {
    session: Session,
    tokenizer: Tokenizer,
}

impl QaModel {
    pub fn new<T: AsRef<Path>>(model: T, tokenizer: T) -> Result<Self> {
        let (session, tokenizer) = session_from_model_file(model, tokenizer)?;
        Ok(Self { session, tokenizer })
    }

    fn encode_args(
        &self,
        len: usize,
        inputs: ArrayBase<ViewRepr<&i64>, Dim<[usize; 2]>>,
        attention_mask: ArrayBase<ViewRepr<&i64>, Dim<[usize; 2]>>,
    ) -> Result<HashMap<String, SessionInputValue>> {
        let mut args: HashMap<String, SessionInputValue> = HashMap::default();
        for inp in &self.session.inputs {
            if inp.name != "input_ids" && inp.name != "attention_mask" {
                let a = Array4::<f32>::zeros((1, 32, len, 96));
                let v = DynValue::try_from(a)?;
                args.insert(inp.name.clone().into(), v.into());
            }
        }
        let inputs = DynValue::try_from(inputs)?;
        let attention_mask = DynValue::try_from(attention_mask)?;
        args.insert("input_ids".into(), inputs.into());
        args.insert("attention_mask".into(), attention_mask.into());
        Ok(args)
    }

    pub fn ask(&self, question: &str, gen: usize) -> Result<String> {
        let encoded = self
            .tokenizer
            .encode(question, true)
            .map_err(|e| anyhow!("{e:?}"))?;
        let mut tokens = Array1::from_iter(encoded.get_ids().iter().map(|t| *t as i64));
        let mut attn_mask = Array1::from_iter(encoded.get_ids().iter().map(|_| 1i64));
        for _ in 0..gen {
            let len = tokens.len();
            let tokens_view = tokens.view().insert_axis(Axis(0));
            let attn_mask_view = attn_mask.view().insert_axis(Axis(0));
            let args = self.encode_args(len, tokens_view, attn_mask_view)?;
            let outputs = self.session.run(args)?;
            let logits = outputs["logits"].try_extract_tensor::<f32>()?;
            let (token, _) = logits.slice(s![0, len - 1, ..]).iter().enumerate().fold(
                (0, -f32::MAX),
                |(ct, cp), (t, p)| if *p > cp { (t, *p) } else { (ct, cp) },
            );
            if token == 32000 {
                // end of text
                break;
            }
            tokens = concatenate![Axis(0), tokens, array![token as i64]];
            attn_mask = concatenate![Axis(0), attn_mask, array![1 as i64]];
        }
        let tokens = tokens.iter().map(|i| *i as u32).collect::<Vec<_>>();
        Ok(self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow!("{e:?}"))?)
    }
}
