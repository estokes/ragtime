use anyhow::{anyhow, Result};
use chrono::prelude::*;
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
    io::Write,
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::Arc,
    thread::available_parallelism,
};

#[derive(Debug, Serialize, Deserialize)]
struct Saved {
    model: PathBuf,
}

pub struct QaModel {
    params: Saved,
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
}

impl QaModel {
    pub fn new<T: AsRef<Path>>(backend: Arc<LlamaBackend>, model: T) -> Result<Self> {
        let params = Saved {
            model: PathBuf::from(model.as_ref()),
        };
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
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

    pub fn load<P: AsRef<Path>>(backend: Arc<LlamaBackend>, path: P) -> Result<Self> {
        let params: Saved = serde_json::from_reader(File::open(path)?)?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
        let model = LlamaModel::load_from_file(&backend, &params.model, &model_params)?;
        Ok(Self {
            params,
            backend,
            model,
        })
    }

    pub fn ask(&self, question: &str, gen: usize) -> Result<String> {
        let now = Utc::now();
        let n_par = 16; //available_parallelism()?.get() as u32;
        let n_ctx = self.model.n_ctx_train() / 8;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(
                NonZeroU32::new(n_ctx).ok_or_else(|| anyhow!("trained context size is zero"))?,
            ))
            .with_n_threads(n_par)
            .with_n_threads_batch(n_par)
            .with_n_batch(n_ctx);
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;
        let tokens = self.model.str_to_token(question, AddBos::Always)?;
        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        let last_idx: i32 = (tokens.len() - 1) as i32;
        for (i, token) in (0i32..).zip(tokens.into_iter()) {
            batch.add(token, i, &[0], i == last_idx)?;
        }
        ctx.decode(&mut batch)?;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut answer = String::with_capacity(gen * 64);
        for i in (last_idx + 1)..(last_idx + 1 + gen as i32) {
            if i == last_idx + 1 {
                let elapsed = (Utc::now() - now).num_milliseconds();
                println!("time to first token: {elapsed}ms");
            }
            if i as u32 > n_ctx {
                break;
            }
            let mut candidates =
                LlamaTokenDataArray::from_iter(ctx.candidates_ith(batch.n_tokens() - 1), false);
            ctx.sample_token_softmax(&mut candidates);
            let token = candidates.data[0].id();
            if token == self.model.token_eos() || token.0 == 32007 {
                break;
            }
            let unicode = self.model.token_to_bytes(token, Special::Tokenize)?;
            let pos = answer.len();
            let _ = decoder.decode_to_string(&unicode, &mut answer, false);
            print!("{}", &answer[pos..]);
            std::io::stdout().flush()?;
            batch.clear();
            batch.add(token, i, &[0], true)?;
            ctx.decode(&mut batch)?;
        }
        Ok(answer)
    }
}
