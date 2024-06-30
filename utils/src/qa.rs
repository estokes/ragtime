use anyhow::{anyhow, Result};
use chrono::prelude::*;
use clap::Parser;
use llama_cpp_2::llama_backend::LlamaBackend;
use ragtime::{bge_m3::onnx::BgeArgs, phi3::llama::Phi3Args, RagQaPhi3BgeM3};
use std::{
    io::{stdin, stdout, BufRead, BufReader, Write},
    path::PathBuf,
    sync::Arc,
    thread::available_parallelism,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    emb_model: Option<PathBuf>,
    emb_tokenizer: Option<PathBuf>,
    qa_model: Option<PathBuf>,
    threads: Option<u32>,
    checkpoint: Option<PathBuf>,
    #[arg(default_value = "256")]
    chunk_size: usize,
    #[arg(default_value = "128")]
    overlap_size: usize,
    #[arg(default_value = "64")]
    max_mapped: usize,
    quiet: bool,
    add_documents: Vec<PathBuf>,
}

impl Args {
    fn init(&self) -> Result<RagQaPhi3BgeM3> {
        tracing_subscriber::fmt::init();
        ort::init().commit()?;
        let backend = Arc::new({
            let mut be = LlamaBackend::init()?;
            if self.quiet {
                be.void_logs();
            }
            be
        });
        if let Some(cp) = &self.checkpoint {
            let view = self.add_documents.is_empty();
            if let Ok(qa) = RagQaPhi3BgeM3::load((), Arc::clone(&backend), cp, view) {
                return Ok(qa);
            }
        }
        let emb_model = self
            .emb_model
            .as_ref()
            .ok_or_else(|| anyhow!("emb model is required"))?;
        let emb_tokenizer = self
            .emb_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("emb tokenizer is required"))?;
        let qa_model = self
            .qa_model
            .as_ref()
            .ok_or_else(|| anyhow!("qa model is required"))?;
        let npar = available_parallelism()?.get() as u32;
        let threads = self.threads.unwrap_or_else(|| npar);
        let qa = RagQaPhi3BgeM3::new(
            self.max_mapped,
            BgeArgs {
                model: emb_model.clone(),
                tokenizer: emb_tokenizer.clone(),
            },
            Phi3Args {
                backend,
                threads,
                model: qa_model.clone(),
            },
        )?;
        Ok(qa)
    }
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let mut qa = args.init()?;
    for doc in &args.add_documents {
        qa.add_document(doc, args.chunk_size, args.overlap_size)?;
    }
    if let Some(cp) = &args.checkpoint {
        qa.save(cp)?;
    }
    println!("ready");
    let mut line = String::new();
    let mut stdin = BufReader::new(stdin());
    loop {
        line.clear();
        stdin.read_line(&mut line)?;
        let start = Utc::now();
        for tok in qa.ask(&line, None)? {
            let tok = tok?;
            print!("{tok}");
            stdout().flush()?;
        }
        println!(
            "\nquery time: {}ms",
            (Utc::now() - start).num_milliseconds()
        );
    }
}
