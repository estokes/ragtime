use crate::{FormattedPrompt, Persistable, QaModel};
use anyhow::{anyhow, Ok, Result};
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
    fs::{File, OpenOptions},
    marker::PhantomPinned,
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    thread::available_parallelism,
};

pub trait LlamaQaModel: Default {
    type Prompt: FormattedPrompt;

    fn is_finished(&mut self, model: &LlamaModel, token: LlamaToken) -> bool;
}

pub trait LlamaEmbedModel: Default {
    type Prompt: FormattedPrompt;

    fn get_embedding(&mut self, ctx: &mut LlamaContext) -> Result<Vec<f32>>;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Args {
    model: PathBuf,
    threads: u32,
    ctx_divisor: u32,
    seed: u32,
    embed: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            threads: available_parallelism().map(|n| n.get() as u32).unwrap_or(8),
            ctx_divisor: 1,
            seed: 42,
            embed: false,
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
    fn ctx(self: Pin<&mut Self>) -> &mut LlamaContext<'static> {
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
    fn init(ctx: Arc<LlamaBackend>, embeddings: bool, mut args: Args) -> Result<Self> {
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
        let model = LlamaModel::load_from_file(&ctx, &args.model, &model_params)?;
        let n_ctx = model.n_ctx_train() / args.ctx_divisor;
        args.embed = embeddings;
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
            .with_seed(t.inner.args.seed)
            .with_embeddings(embeddings)
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
        Llama::init(ctx, args.embed, args)
    }
}

struct TokenIter<'a, Model: LlamaQaModel> {
    t: &'a mut Llama<Model>,
    decoder: Decoder,
    batch: LlamaBatch,
    answer: String,
    gen: Option<usize>,
    i: i32,
    question_len: i32,
}

impl<'a, Model> TokenIter<'a, Model>
where
    Model: LlamaQaModel,
{
    fn step(&mut self) -> Result<Option<CompactString>> {
        if let Some(gen) = self.gen {
            if self.i as usize >= gen + self.question_len as usize {
                return Ok(None);
            }
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
        self.inner.as_mut().ctx().clear_kv_cache();
        let tokens = self
            .inner
            .model
            .str_to_token(&question.as_ref(), AddBos::Always)?;
        let mut batch = LlamaBatch::new(self.inner.n_ctx as usize, 1);
        let last_idx: i32 = (tokens.len() - 1) as i32;
        for (i, token) in (0i32..).zip(tokens.into_iter()) {
            batch.add(token, i, &[0], i == last_idx)?;
        }
        Ok(TokenIter {
            t: self,
            decoder: encoding_rs::UTF_8.new_decoder(),
            batch,
            answer: String::with_capacity(gen.unwrap_or(n_ctx) * 32),
            gen,
            i: last_idx + 1,
            question_len: last_idx + 1,
        })
    }
}
