use crate::{db::EmbedDb, doc::DocStore, qa_llama::QaModel};
use anyhow::{anyhow, bail, Result};
use ort::Session;
use std::{cmp::min, fs, path::Path, thread::available_parallelism};
use tokenizers::Tokenizer;

pub mod db;
pub mod doc;
pub mod qa_llama;
pub mod qa_onnx;

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
    ) -> Result<Self> {
        let docs = DocStore::new(max_mapped);
        let db = EmbedDb::new(embed_model, embed_tokenizer, embed_dims)?;
        let qa = QaModel::new(qa_model)?;
        Ok(Self { docs, db, qa })
    }

    /// save the state to the specified directory, which will be
    /// created if it does not exist.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if path.exists() {
            if !path.is_dir() {
                bail!("save path exists but is not a directory")
            }
        } else {
            fs::create_dir_all(path)?
        }
        self.docs.save(path.join("docs.json"))?;
        self.db.save(path.join("db.json"))?;
        self.qa.save(path.join("model.json"))?;
        Ok(())
    }

    /// load the state from the specified directory
    pub fn load<P: AsRef<Path>>(&self, path: P, view: bool) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_dir() {
            bail!("save directory could not be found")
        }
        let docs = DocStore::load(path.join("docs.json"))?;
        let db = EmbedDb::load(path.join("db.json"), view)?;
        let qa = QaModel::load(path.join("model.json"))?;
        Ok(Self { docs, db, qa })
    }

    pub fn add_document<P: AsRef<Path>>(
        &mut self,
        doc: P,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<()> {
        let chunks = self
            .docs
            .add_document(doc, chunk_size, overlap)?
            .collect::<Result<Vec<_>>>()?;
        self.db.add(chunks)
    }

    pub fn encode_prompt<S: AsRef<str>>(
        &mut self,
        context: &mut String,
        rag: Option<S>,
        q: S,
    ) -> Result<String> {
        use std::fmt::Write;
        let q = q.as_ref();
        let mut prompt = String::with_capacity(min(4 * 1024 * 1024, q.len() * 10));
        write!(prompt, "<|system|>RAG is available if you have questions. If you want to search using RAG, say <RAG>your question</RAG> End your text there, and the system will search the database for your question. The answer to your question will be given using additional <|system|>prompts. Only answer the user's question once you have the information you need, multiple RAG queries may be required. If there isn't any relevant information available the database will tell you. <|end|>\n")?;
        write!(prompt, "{context}\n")?;
        if let Some(rag) = rag {
            let rag = rag.as_ref();
            let matches = self.db.search(rag, 3)?;
            for dst in [&mut prompt, context] {
                if matches.distances.len() == 0 || matches.distances[0] > 0.7 {
                    write!(dst, "<|system|>There was no relevant information available about \"{rag}\" in the database<|end|>\n")?;
                } else {
                    write!(dst, "<|system|>")?;
                    for (id, dist) in matches.keys.iter().zip(matches.distances.iter()) {
                        if *dist <= 0.7 {
                            let chunk = self.docs.get_chunk(*id)?;
                            let s = self.docs.get(&chunk)?;
                            write!(dst, "{s}\n\n")?;
                        }
                    }
                    write!(dst, " <|end|>\n")?;
                };
            }
        }
        write!(prompt, "<|user|>{q} <|end|><|assistant|>")?;
        Ok(prompt)
    }

    pub fn ask<S: AsRef<str>>(&mut self, q: S, gen_max: usize) -> Result<String> {
        let mut context = String::new();
        let mut prompt = self.encode_prompt(&mut context, None, q.as_ref())?;
        loop {
            let res = self.qa.ask(&prompt, gen_max)?;
            match res
                .trim()
                .strip_prefix("<RAG>")
                .and_then(|s| s.strip_suffix("</RAG>"))
            {
                None => break Ok(res),
                Some(rag) => {
                    prompt = self.encode_prompt(&mut context, Some(rag), q.as_ref())?;
                    dbg!(&prompt);
                }
            }
        }
    }
}
