use super::prompt::{Phi3FinalPrompt, Phi3Prompt};
use crate::{Persistable, QaModel};
use anyhow::{anyhow, Ok, Result};
use compact_str::CompactString;
use encoding_rs::Decoder;
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::data_array::LlamaTokenDataArray,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct Saved {
    pub model: PathBuf,
}

pub struct Phi3Args {
    pub model: PathBuf,
    pub backend: Arc<LlamaBackend>,
}

struct Phi3Inner {
    params: Saved,
    ctx: Option<LlamaContext<'static>>,
    model: LlamaModel,
    backend: Arc<LlamaBackend>,
    n_ctx: u32,
    _pin: PhantomPinned,
}

impl Phi3Inner {
    fn ctx(self: Pin<&mut Self>) -> &mut LlamaContext<'static> {
        unsafe { self.get_unchecked_mut().ctx.as_mut().unwrap() }
    }
}

pub struct Phi3(Pin<Box<Phi3Inner>>);

impl Persistable for Phi3 {
    type Ctx = Arc<LlamaBackend>;

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        Ok(serde_json::to_writer_pretty(&mut fd, &self.0.params)?)
    }

    fn load<P: AsRef<Path>>(ctx: Arc<LlamaBackend>, path: P, _view: bool) -> Result<Self> {
        let params: Saved = serde_json::from_reader(File::open(path)?)?;
        Phi3::new(Phi3Args {
            model: params.model,
            backend: ctx,
        })
    }
}

struct TokenIter<'a> {
    model: &'a mut Phi3,
    decoder: Decoder,
    batch: LlamaBatch,
    answer: String,
    gen: Option<usize>,
    i: i32,
    question_len: i32,
}

impl<'a> TokenIter<'a> {
    fn step(&mut self) -> Result<Option<CompactString>> {
        if let Some(gen) = self.gen {
            if self.i as usize >= gen + self.question_len as usize {
                return Ok(None);
            }
        }
        if self.i as u32 > self.model.0.n_ctx {
            return Ok(None);
        }
        self.model.0.as_mut().ctx().decode(&mut self.batch)?;
        let candidates = {
            let ctx = self.model.0.as_mut().ctx();
            let mut a = LlamaTokenDataArray::from_iter(
                ctx.candidates_ith(self.batch.n_tokens() - 1),
                false,
            );
            ctx.sample_token_softmax(&mut a);
            a
        };
        let token = candidates.data[0].id();
        // CR estokes: abstract this
        if token == self.model.0.model.token_eos() || token.0 == 32007 {
            return Ok(None);
        }
        let unicode = self
            .model
            .0
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

impl<'a> Iterator for TokenIter<'a> {
    type Item = Result<CompactString>;

    fn next(&mut self) -> Option<Self::Item> {
        self.step().transpose()
    }
}

impl QaModel for Phi3 {
    type Args = Phi3Args;
    type Prompt = Phi3Prompt;

    fn new(args: Self::Args) -> Result<Self> {
        let params = Saved { model: args.model };
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
        let model = LlamaModel::load_from_file(&args.backend, &params.model, &model_params)?;
        let n_ctx = model.n_ctx_train() / 8;
        let mut t = Phi3(Box::pin(Phi3Inner {
            params,
            ctx: None,
            model,
            backend: args.backend,
            n_ctx,
            _pin: PhantomPinned,
        }));
        let n_par = if cfg!(vulkan) {
            32
        } else {
            available_parallelism()?.get() as u32
        };
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(
                NonZeroU32::new(n_ctx).ok_or_else(|| anyhow!("trained context size is zero"))?,
            ))
            .with_n_threads(n_par)
            .with_n_threads_batch(n_par)
            .with_n_batch(n_ctx);
        let ctx = unsafe { &*((&t.0.model) as *const LlamaModel) }
            .new_context(&t.0.backend, ctx_params)?;
        unsafe { t.0.as_mut().get_unchecked_mut() }.ctx = Some(ctx);
        Ok(t)
    }

    fn ask(
        &mut self,
        question: Phi3FinalPrompt,
        gen: Option<usize>,
    ) -> Result<impl Iterator<Item = Result<CompactString>>> {
        let n_ctx = self.0.n_ctx as usize;
        self.0.as_mut().ctx().clear_kv_cache();
        let tokens = self.0.model.str_to_token(&question.0, AddBos::Always)?;
        let mut batch = LlamaBatch::new(self.0.n_ctx as usize, 1);
        let last_idx: i32 = (tokens.len() - 1) as i32;
        for (i, token) in (0i32..).zip(tokens.into_iter()) {
            batch.add(token, i, &[0], i == last_idx)?;
        }
        Ok(TokenIter {
            model: self,
            decoder: encoding_rs::UTF_8.new_decoder(),
            batch,
            answer: String::with_capacity(gen.unwrap_or(n_ctx) * 32),
            gen,
            i: last_idx + 1,
            question_len: last_idx + 1,
        })
    }
}
