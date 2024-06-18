use crate::{db::EmbedDb, qa::QaModel, doc::DocStore};
use anyhow::{anyhow, Result};
use ort::Session;
use std::{path::Path, thread::available_parallelism};
use tokenizers::Tokenizer;

pub mod db;
pub mod doc;
pub mod qa;

fn session_from_model_file<P: AsRef<Path>>(model: P, tokenizer: P) -> Result<(Session, Tokenizer)> {
    let session = Session::builder()?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_intra_threads(available_parallelism()?.get())?
        .commit_from_file(model)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{e:?}"))?;
    Ok((session, tokenizer))
}

pub struct RagQa {
    docs: DocStore,
    db: EmbedDb,
    qa: QaModel,
}

impl RagQa {
    pub fn new<P: AsRef<Path>>(
        max_mapped: usize,
        embed_model: P,
        embed_tokenizer: P,
        embed_dims: usize,
        qa_model: P,
        qa_tokenizer: P,
    ) -> Result<Self> {
        let docs = DocStore::new(max_mapped);
        let db = EmbedDb::new(embed_model, embed_tokenizer, embed_dims)?;
        let qa = QaModel::new(qa_model, qa_tokenizer)?;
        Ok(Self { docs, db, qa })
    }

    pub fn add_document<P: AsRef<Path>>(
        &mut self,
        doc: P,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<()> {
        
        unimplemented!()
    }
}
