use anyhow::{anyhow, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::data_array::LlamaTokenDataArray,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    num::NonZeroU32,
    path::{Path, PathBuf},
    thread::available_parallelism,
};

#[derive(Debug, Serialize, Deserialize)]
struct Saved {
    model: PathBuf,
}

pub struct QaModel {
    params: Saved,
    backend: LlamaBackend,
    model: LlamaModel,
}

impl QaModel {
    pub fn new<T: AsRef<Path>>(model: T) -> Result<Self> {
        let params = Saved {
            model: PathBuf::from(model.as_ref()),
        };
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model.as_ref(), &model_params)?;
        Ok(Self {
            params,
            backend,
            model,
        })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        Ok(serde_json::to_writer_pretty(&mut fd, &self.params)?)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let params: Saved = serde_json::from_reader(File::open(path)?)?;
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &params.model, &model_params)?;
        Ok(Self {
            params,
            backend,
            model,
        })
    }

    pub fn ask(&self, question: &str, gen: usize) -> Result<String> {
        let n_par = available_parallelism()?.get() as u32;
        let n_ctx = self.model.n_ctx_train();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(
                NonZeroU32::new(n_ctx).ok_or_else(|| anyhow!("trained context size is zero"))?,
            ))
            .with_n_threads(n_par)
            .with_n_threads_batch(n_par);
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;
        let tokens = self.model.str_to_token(question, AddBos::Always)?;
        let mut batch = LlamaBatch::new(self.model.n_ctx_train() as usize, 1);
        let last_idx: i32 = (tokens.len() - 1) as i32;
        for (i, token) in (0i32..).zip(tokens.into_iter()) {
            batch.add(token, i, &[0], i == last_idx)?;
        }
        ctx.decode(&mut batch)?;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut answer = String::with_capacity(gen * 64);
        let mut candidates = LlamaTokenDataArray::new(vec![], false);
        for i in (last_idx + 1)..(last_idx + 1 + gen as i32) {
            if i as u32 > n_ctx {
                break;
            }
            candidates.data.clear();
            candidates
                .data
                .extend(ctx.candidates_ith(batch.n_tokens() - 1));
            ctx.sample_token_softmax(&mut candidates);
            let token = candidates.data[0].id();
            if token == self.model.token_eos() {
                break;
            }
            let unicode = self.model.token_to_bytes(token, Special::Tokenize)?;
            let _ = decoder.decode_to_string(&unicode, &mut answer, false);
            batch.clear();
            batch.add(token, i, &[0], true)?;
            ctx.decode(&mut batch)?;
        }
        Ok(answer)
    }
}
