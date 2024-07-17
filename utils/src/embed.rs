use anyhow::Result;
use clap::{Parser, ValueEnum};
use ragtime::{
    bge_m3::onnx::{BgeArgs, BgeM3},
    doc::ChunkId,
    EmbedModel,
};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelType {
    BgeM3,
    GteLargeEn,
    GteQwen27bInstruct
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, help = "path to the model")]
    model_path: PathBuf,
    #[arg(long, help = "tokenizer path")]
    tokenizer_path: PathBuf,
    #[arg(long, help = "The type of model to use")]
    model_type: ModelType,
    #[arg(long, help = "the number of threads to use (default: all available)")]
    threads: Option<u32>,
    #[arg(long, help = "suppress llama.cpp logging")]
    quiet: bool,
    #[arg(
        long,
        default_value = "1",
        help = "the context divisor (trade context for less memory use)"
    )]
    ctx_divisor: u32,
    #[arg(long, default_value = "42", help = "random seed")]
    seed: u32,
    #[arg(long, help = "add the specified document to the embedding database, may be specified more than once")]
    add_document: Vec<PathBuf>,
    #[arg(long, help = "break the document into chunks of this size (default: no)")]
    chunk: Option<usize>,
    #[arg(long, help = "overlap the chunks by this many tokens (default: no)")]
    overlap: Option<usize>,
}

fn run_bgem3(args: Args) -> Result<()> {
    ort::init().commit()?;
    let mut model = BgeM3::new((), BgeArgs {
        model: args.model_path,
        tokenizer: args.tokenizer_path,
    })?;
    for doc in args.add_document {
        let chunk_size = args.chunk.unwrap_or(0);
        let overlap = args.overlap.unwrap_or(0);
        unimplemented!()
    }
    Ok(())
}

fn run_gtelargeen(args: Args) -> Result<()> {
    unimplemented!()
}

fn run_gteqwen27binstruct(args: Args) -> Result<()> {
    unimplemented!()
}

pub fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    match args.model_type {
        ModelType::BgeM3 => run_bgem3(args),
        ModelType::GteLargeEn => run_gtelargeen(args),
        ModelType::GteQwen27bInstruct => run_gteqwen27binstruct(args),
    }
}
