use anyhow::{anyhow, Result};
use chrono::prelude::*;
use clap::Parser;
use llama_cpp_2::llama_backend::LlamaBackend;
use ragtime::{
    gte_large_en::onnx::GteLargeEnArgs, llama, RagQaPhi3GteLargeEn,
};
use std::{
    io::{stdin, stdout, BufRead, BufReader, Write},
    path::PathBuf,
    sync::Arc,
    thread::available_parallelism,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, help = "path to the embedding model (onnx)")]
    emb_model: Option<PathBuf>,
    #[arg(long, help = "path to the embedding tokenizer (e.g. tokenizer.json)")]
    emb_tokenizer: Option<PathBuf>,
    #[arg(long, help = "path to the QA model (gguf)")]
    qa_model: Option<PathBuf>,
    #[arg(long, help = "the number of threads to use (default: all available)")]
    threads: Option<u32>,
    #[arg(
        long,
        help = "path to a checkpoint directory (created if it doesn't exist)"
    )]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "256", help = "the size of each chunk in tokens")]
    chunk_size: usize,
    #[arg(long, default_value = "128", help = "the chunk overlap size in tokens")]
    overlap_size: usize,
    #[arg(
        long,
        default_value = "8",
        help = "the context divisor (trade context for less memory use)"
    )]
    ctx_divisor: u32,
    #[arg(
        long,
        default_value = "64",
        help = "the maximum number of documents to keep open at once"
    )]
    max_mapped: usize,
    #[arg(long, help = "suppress llama.cpp logging")]
    quiet: bool,
    #[arg(long, help = "document to add to the index, may be repeated")]
    add_document: Vec<PathBuf>,
    #[arg(long, default_value = "42", help = "random seed")]
    seed: u32,
    #[arg(
        long,
        help = "do not question, only retreive and display matching document chunks"
    )]
    retrieve_only: bool,
}

impl Args {
    fn init(&self) -> Result<(bool, RagQaPhi3GteLargeEn)> {
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
            let view = self.add_document.is_empty();
            if let Ok(qa) = RagQaPhi3GteLargeEn::load((), Arc::clone(&backend), cp, view) {
                return Ok((view, qa));
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
        let qa = RagQaPhi3GteLargeEn::new(
            self.max_mapped,
            (),
            GteLargeEnArgs {
                model: emb_model.clone(),
                tokenizer: emb_tokenizer.clone(),
            },
            backend,
            llama::Args::default()
                .with_threads(threads)
                .with_ctx_divisor(self.ctx_divisor)
                .with_seed(self.seed)
                .with_model(qa_model.clone()),
        )?;
        Ok((false, qa))
    }
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let (view, mut qa) = args.init()?;
    for doc in &args.add_document {
        qa.add_document(doc, args.chunk_size, args.overlap_size)?;
    }
    if let Some(cp) = &args.checkpoint {
        if !view {
            qa.save(cp)?;
        }
    }
    let mut line = String::new();
    let mut stdin = BufReader::new(stdin());
    loop {
        line.clear();
        print!("ready> ");
        stdout().flush()?;
        stdin.read_line(&mut line)?;
        let start = Utc::now();
        if args.retrieve_only {
            for res in qa.search(&line, 3)? {
                println!("{res:?}")
            }
        } else {
            for tok in qa.ask(&line, None)? {
                let tok = tok?;
                print!("{tok}");
                stdout().flush()?;
            }
        }
        println!(
            "\nquery time: {}ms",
            (Utc::now() - start).num_milliseconds()
        );
    }
}
