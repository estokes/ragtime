use crate::{doc::ChunkId, l2_normalize, EmbedModel, FormattedPrompt, Persistable, QaModel};
use anyhow::{anyhow, bail, Context, Ok, Result};
use compact_str::CompactString;
use encoding_rs::Decoder;
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::{data_array::LlamaTokenDataArray, LlamaToken},
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    cmp::min,
    fs::{File, OpenOptions},
    iter,
    marker::PhantomPinned,
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    thread::available_parallelism,
};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub trait LlamaQaModel: Default {
    type Prompt: FormattedPrompt;

    fn is_finished(&mut self, model: &LlamaModel, token: LlamaToken) -> bool;
}

pub trait LlamaEmbedModel: Default {
    type EmbedPrompt: FormattedPrompt;
    type SearchPrompt: FormattedPrompt;

    fn get_embedding(&mut self, ctx: &mut LlamaContext, i: i32) -> Result<Vec<f32>>;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Args {
    pub model: PathBuf,
    pub threads: u32,
    pub ctx_divisor: u32,
    pub seed: u32,
    pub gpu_layers: u32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            threads: available_parallelism().map(|n| n.get() as u32).unwrap_or(8),
            ctx_divisor: 1,
            seed: 42,
            gpu_layers: 9999,
        }
    }
}

impl Args {
    pub fn with_model<P: Into<PathBuf>>(mut self, model: P) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_ctx_divisor(mut self, ctx_divisor: u32) -> Self {
        self.ctx_divisor = ctx_divisor;
        self
    }

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_gpu_layers(mut self, layers: u32) -> Self {
        self.gpu_layers = layers;
        self
    }
}

struct LlamaInner {
    args: Args,
    ctx: Option<LlamaContext<'static>>,
    model: LlamaModel,
    backend: Arc<LlamaBackend>,
    n_ctx: u32,
    _pin: PhantomPinned,
}

impl LlamaInner {
    fn ctx(self: Pin<&mut LlamaInner>) -> &mut LlamaContext<'static> {
        unsafe { self.get_unchecked_mut().ctx.as_mut().unwrap() }
    }
}

pub struct Llama<Model> {
    inner: Pin<Box<LlamaInner>>,
    model: Model,
}

impl<Model> Llama<Model>
where
    Model: Default,
{
    fn init(ctx: Arc<LlamaBackend>, embed: bool, args: Args) -> Result<Self> {
        let model_params = LlamaModelParams::default().with_n_gpu_layers(args.gpu_layers);
        let model = LlamaModel::load_from_file(&ctx, &args.model, &model_params)?;
        let n_ctx = model.n_ctx_train() / args.ctx_divisor;
        let mut t = Llama {
            inner: Box::pin(LlamaInner {
                args,
                ctx: None,
                model,
                backend: ctx,
                n_ctx,
                _pin: PhantomPinned,
            }),
            model: Model::default(),
        };
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(
                NonZeroU32::new(n_ctx).ok_or_else(|| anyhow!("trained context size is zero"))?,
            ))
            .with_n_threads(t.inner.args.threads)
            .with_n_threads_batch(t.inner.args.threads)
            .with_embeddings(embed)
            .with_seed(t.inner.args.seed)
            .with_n_batch(n_ctx);
        let ctx = unsafe { &*((&t.inner.model) as *const LlamaModel) }
            .new_context(&t.inner.backend, ctx_params)?;
        unsafe { t.inner.as_mut().get_unchecked_mut() }.ctx = Some(ctx);
        Ok(t)
    }
}

impl<Model> Persistable for Llama<Model>
where
    Model: Default,
{
    type Ctx = Arc<LlamaBackend>;

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        Ok(serde_json::to_writer_pretty(&mut fd, &self.inner.args)?)
    }

    fn load<P: AsRef<Path>>(ctx: Arc<LlamaBackend>, path: P, _view: bool) -> Result<Self> {
        let args: Args = serde_json::from_reader(File::open(path)?)?;
        Llama::init(ctx, false, args)
    }
}

struct TokenIter<'a, Model: LlamaQaModel> {
    t: &'a mut Llama<Model>,
    decoder: Decoder,
    batch: LlamaBatch,
    answer: String,
    gen: usize,
    i: i32,
    question_len: i32,
}

impl<'a, Model> TokenIter<'a, Model>
where
    Model: LlamaQaModel,
{
    fn step(&mut self) -> Result<Option<CompactString>> {
        if self.i as usize >= self.gen + self.question_len as usize {
            return Ok(None);
        }
        if self.i as u32 > self.t.inner.n_ctx {
            return Ok(None);
        }
        self.t.inner.as_mut().ctx().decode(&mut self.batch)?;
        let candidates = {
            let ctx = self.t.inner.as_mut().ctx();
            let mut a = LlamaTokenDataArray::from_iter(
                ctx.candidates_ith(self.batch.n_tokens() - 1),
                false,
            );
            ctx.sample_token_softmax(&mut a);
            a
        };
        let token = candidates.data[0].id();
        if self.t.model.is_finished(&self.t.inner.model, token) {
            return Ok(None);
        }
        let unicode = self
            .t
            .inner
            .model
            .token_to_bytes(token, Special::Tokenize)?;
        let pos = self.answer.len();
        let _ = self
            .decoder
            .decode_to_string(&unicode, &mut self.answer, false);
        self.batch.clear();
        self.batch.add(token, self.i, &[0], true)?;
        self.i += 1;
        Ok(Some(CompactString::from(&self.answer[pos..])))
    }
}

impl<'a, Model> Iterator for TokenIter<'a, Model>
where
    Model: LlamaQaModel,
{
    type Item = Result<CompactString>;

    fn next(&mut self) -> Option<Self::Item> {
        self.step().transpose()
    }
}

impl<Model> QaModel for Llama<Model>
where
    Model: LlamaQaModel,
{
    type Ctx = Arc<LlamaBackend>;
    type Args = Args;
    type Prompt = Model::Prompt;

    fn new(ctx: Arc<LlamaBackend>, args: Self::Args) -> Result<Self> {
        Self::init(ctx, false, args)
    }

    fn ask(
        &mut self,
        question: <Model::Prompt as FormattedPrompt>::FinalPrompt,
        gen: Option<usize>,
    ) -> Result<impl Iterator<Item = Result<CompactString>>> {
        let n_ctx = self.inner.n_ctx as usize;
        // save some of the context for the answer
        let min_answer = n_ctx >> 4;
        self.inner.as_mut().ctx().clear_kv_cache();
        let tokens = self
            .inner
            .model
            .str_to_token(&question.as_ref(), AddBos::Always)?;
        let mut batch = LlamaBatch::new(n_ctx, 1);
        let last_idx: i32 = min((tokens.len() - 1) as i32, (n_ctx - min_answer) as i32);
        for (i, token) in (0i32..).zip(tokens.into_iter()) {
            let last = i == last_idx;
            batch.add(token, i, &[0], last)?;
            if last {
                break;
            }
        }
        Ok(TokenIter {
            t: self,
            decoder: encoding_rs::UTF_8.new_decoder(),
            batch,
            answer: String::with_capacity(gen.unwrap_or(n_ctx) * 32),
            gen: gen.unwrap_or(n_ctx - (last_idx + 1) as usize),
            i: last_idx + 1,
            question_len: last_idx + 1,
        })
    }
}

pub struct LlamaEmbed<Model> {
    index: Index,
    base: Llama<Model>,
}

fn index_options(dims: usize) -> IndexOptions {
    let mut opts = IndexOptions::default();
    opts.dimensions = dims;
    opts.metric = MetricKind::Cos;
    opts.quantization = ScalarKind::F32;
    opts
}

impl<Model> LlamaEmbed<Model>
where
    Model: LlamaEmbedModel,
{
    fn init_with_index(
        mut new_index: impl FnMut(usize) -> Result<Index>,
        backend: Arc<LlamaBackend>,
        args: Args,
    ) -> Result<Self> {
        let base = Llama::init(backend.clone(), true, args)?;
        let index = new_index(base.inner.model.n_embd() as usize)?;
        Ok(Self { index, base })
    }

    fn embed<'a, I: IntoIterator<Item = &'a str>>(&mut self, chunks: I) -> Result<Vec<Vec<f32>>> {
        let n_ctx = self.base.inner.n_ctx as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);
        let mut output = vec![];
        let tokenized_chunks = chunks
            .into_iter()
            .map(|s| {
                let tokens = self
                    .base
                    .inner
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
        for chunk in tokenized_chunks {
            let seq = chunk.len() as i32;
            for (i, token) in (0..).zip(chunk.iter()) {
                batch.add(*token, i, &[0], i == seq - 1)?;
            }
            let ctx = self.base.inner.as_mut().ctx();
            ctx.clear_kv_cache();
            ctx.decode(&mut batch)
                .with_context(|| "llama_decode() failed")?;
            output.push(self.base.model.get_embedding(ctx, seq - 1)?);
            batch.clear();
        }
        Ok(output)
    }
}

impl<Model> Persistable for LlamaEmbed<Model>
where
    Model: LlamaEmbedModel,
{
    type Ctx = Arc<LlamaBackend>;

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut path = PathBuf::from(path.as_ref());
        path.set_extension("json");
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        serde_json::to_writer_pretty(&mut fd, &self.base.inner.args)?;
        path.set_extension("usearch");
        Ok(self.index.save(&*path.to_string_lossy())?)
    }

    fn load<P: AsRef<Path>>(ctx: Arc<LlamaBackend>, path: P, view: bool) -> Result<Self> {
        let args: Args = serde_json::from_reader(File::open(path.as_ref())?)?;
        let new_index = |dims| -> Result<Index> {
            let index = Index::new(&index_options(dims))?;
            index.reserve(10)?;
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
        Self::init_with_index(new_index, ctx, args)
    }
}

impl<Model> EmbedModel for LlamaEmbed<Model>
where
    Model: LlamaEmbedModel,
{
    type Ctx = Arc<LlamaBackend>;
    type Args = Args;
    type EmbedPrompt = Model::EmbedPrompt;
    type SearchPrompt = Model::SearchPrompt;

    fn new(ctx: Arc<LlamaBackend>, args: Self::Args) -> Result<Self> {
        Self::init_with_index(|dims| Ok(Index::new(&index_options(dims))?), ctx, args)
    }

    fn add(
        &mut self,
        summary: <Self::EmbedPrompt as FormattedPrompt>::FinalPrompt,
        text: &[(ChunkId, <Self::EmbedPrompt as FormattedPrompt>::FinalPrompt)],
    ) -> Result<()> {
        let embed =
            self.embed(iter::once(summary.as_ref()).chain(text.iter().map(|(_, t)| t.as_ref())))?;
        let summary = embed[0].iter().map(|elt| *elt * 0.5).collect::<Vec<_>>();
        let mut tmp = vec![];
        for (e, (id, _)) in embed[1..].into_iter().zip(text.iter()) {
            tmp.clear();
            tmp.extend(e.iter().zip(summary.iter()).map(|(elt, selt)| *elt + *selt));
            l2_normalize(&mut tmp);
            if self.index.capacity() == self.index.size() {
                self.index.reserve(max(10, self.index.capacity() * 2))?;
            }
            self.index.add(id.0, &tmp)?;
        }
        Ok(())
    }

    fn search(
        &mut self,
        q: <Self::SearchPrompt as FormattedPrompt>::FinalPrompt,
        n: usize,
    ) -> Result<usearch::ffi::Matches> {
        let embed = self.embed(iter::once(q.as_ref()))?;
        Ok(self.index.search(&embed[0], n)?)
    }
}
