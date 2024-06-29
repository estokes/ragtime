/// this is broken at the moment, no model I've been able to find has the proper pooling layers
use crate::doc::ChunkId;
use anyhow::{anyhow, bail, Context, Result};
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    token::LlamaToken,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    iter,
    marker::PhantomPinned,
    num::NonZero,
    ops::Deref,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    thread::available_parallelism,
};
use usearch::{ffi::Matches, Index, IndexOptions, MetricKind, ScalarKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Saved {
    model: PathBuf,
}

pub struct EmbedDbInner {
    params: Saved,
    ctx: Option<LlamaContext<'static>>,
    model: LlamaModel,
    backend: Arc<LlamaBackend>,
    index: Index,
    _pin: PhantomPinned,
}

impl EmbedDbInner {
    fn ctx_mut(self: Pin<&mut Self>) -> &mut LlamaContext<'static> {
        // it's always safe to get a mutable reference to context this
        // way since model is the thing who's address really can't
        // change (because it's held in ctx)
        unsafe { self.get_unchecked_mut().ctx.as_mut().unwrap() }
    }
}

pub struct EmbedDb(Pin<Box<EmbedDbInner>>);

impl Deref for EmbedDb {
    type Target = EmbedDbInner;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

fn options(dims: usize) -> IndexOptions {
    let mut opts = IndexOptions::default();
    opts.dimensions = dims;
    opts.metric = MetricKind::Cos;
    opts.quantization = ScalarKind::F32;
    opts
}

impl EmbedDb {
    fn init_with_index<P: AsRef<Path>>(
        mut new_index: impl FnMut(usize) -> Result<Index>,
        backend: Arc<LlamaBackend>,
        model: P,
    ) -> Result<Self> {
        let params = Saved {
            model: PathBuf::from(model.as_ref()),
        };
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
        let model = LlamaModel::load_from_file(&backend, &params.model, &model_params)?;
        let index = new_index(model.n_embd() as usize)?;
        let mut t = Box::pin(EmbedDbInner {
            params,
            ctx: None,
            backend,
            model,
            index,
            _pin: PhantomPinned,
        });
        let n_ctx = t.model.n_ctx_train();
        /*
                #[cfg(vulkan)]
                let ctx_params = LlamaContextParams::default()
                    .with_n_threads_batch(32)
                    .with_n_ctx(NonZero::new(n_ctx))
                    .with_n_batch(n_ctx)
                    .with_embeddings(true);
                #[cfg(not(vulkan))]
        */
        let ctx_params = LlamaContextParams::default()
            .with_n_threads_batch(available_parallelism()?.get() as u32)
            .with_n_ctx(NonZero::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_embeddings(true);
        // CR estokes: investigate a better LlamaContext api to eliminate this
        let ctx =
            unsafe { &*(&t.model as *const LlamaModel) }.new_context(&t.backend, ctx_params)?;
        unsafe { t.as_mut().get_unchecked_mut() }.ctx = Some(ctx);
        Ok(EmbedDb(t))
    }

    pub fn new<P: AsRef<Path>>(backend: Arc<LlamaBackend>, model: P) -> Result<Self> {
        let new_index = |dims| Ok(Index::new(&options(dims))?);
        Self::init_with_index(new_index, backend, model)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut path = PathBuf::from(path.as_ref());
        path.set_extension("json");
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        serde_json::to_writer_pretty(&mut fd, &self.params)?;
        path.set_extension("usearch");
        Ok(self.index.save(&*path.to_string_lossy())?)
    }

    pub fn load<P: AsRef<Path>>(backend: Arc<LlamaBackend>, path: P, view: bool) -> Result<Self> {
        let mut fd = File::open(path.as_ref())?;
        let params: Saved = serde_json::from_reader(&mut fd)?;
        let new_index = |dims| -> Result<Index> {
            let index = Index::new(&options(dims))?;
            let mut path = PathBuf::from(path.as_ref());
            path.set_extension("usearch");
            let path = path.to_string_lossy();
            if view {
                index.view(&*path)?
            } else {
                index.load(&*path)?;
            }
            Ok(index)
        };
        Self::init_with_index(new_index, backend, &params.model)
    }

    fn embed<'a, I: IntoIterator<Item = &'a str>>(
        &mut self,
        do_norm: bool,
        chunks: I,
    ) -> Result<Vec<Vec<f32>>> {
        fn normalize(input: &[f32]) -> Vec<f32> {
            let magnitude = input
                .iter()
                .fold(0.0, |acc, &val| val.mul_add(val, acc))
                .sqrt();
            input.iter().map(|&val| val / magnitude).collect()
        }
        fn process_batch(
            ctx: &mut LlamaContext,
            batch: &mut LlamaBatch,
            seq: i32,
            output: &mut Vec<Vec<f32>>,
            do_normalize: bool,
        ) -> Result<()> {
            ctx.clear_kv_cache();
            ctx.decode(batch).with_context(|| "llama_decode() failed")?;
            for i in 0..seq {
                let embedding = ctx
                    .embeddings_seq_ith(i)
                    .with_context(|| "Failed to get embeddings")?;
                let output_embeddings = if do_normalize {
                    normalize(embedding)
                } else {
                    embedding.to_vec()
                };

                output.push(output_embeddings);
            }
            batch.clear();
            Ok(())
        }
        let n_ctx = self.model.n_ctx_train() as usize;
        let tokenized_chunks = chunks
            .into_iter()
            .map(|s| {
                let tokens = self
                    .model
                    .str_to_token(s, AddBos::Always)
                    .map_err(anyhow::Error::from)?;
                if tokens.len() > n_ctx {
                    bail!(
                        "chunk too large {} vs context windows of {n_ctx}",
                        tokens.len()
                    )
                }
                Ok(tokens)
            })
            .collect::<Result<Vec<Vec<LlamaToken>>>>()?;
        let mut batch = LlamaBatch::new(n_ctx, 1);
        let mut seq = 0;
        let mut output = vec![];
        for chunk in &tokenized_chunks {
            if (batch.n_tokens() as usize + chunk.len()) > n_ctx {
                process_batch(
                    self.0.as_mut().ctx_mut(),
                    &mut batch,
                    seq,
                    &mut output,
                    do_norm,
                )?;
                seq = 0;
            }
            batch.add_sequence(chunk, seq, false)?;
            seq += 1;
        }
        process_batch(
            self.0.as_mut().ctx_mut(),
            &mut batch,
            seq,
            &mut output,
            do_norm,
        )?;
        Ok(output)
    }

    pub fn add(&mut self, text: &[(ChunkId, &str)]) -> Result<()> {
        let embeds = self.embed(true, text.iter().map(|(_, t)| *t))?;
        for (e, (id, _)) in embeds.iter().zip(text.iter()) {
            self.index.add(id.0, &*e)?
        }
        Ok(())
    }

    /// The keys component of Matches is a vec of ChunkIds represented as u64s.
    pub fn search(&mut self, text: &str, n: usize) -> Result<Matches> {
        let qembed = self.embed(true, iter::once(text))?;
        if qembed.is_empty() {
            bail!("no embed generated")
        }
        Ok(self.index.search(&qembed[0], n)?)
    }
}
