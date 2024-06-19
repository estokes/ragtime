use crate::{db::EmbedDb, doc::DocStore, qa_onnx::QaModel};
use anyhow::{anyhow, bail, Result};
use ort::Session;
use std::{fs, path::Path, thread::available_parallelism, cmp::min};
use tokenizers::Tokenizer;

pub mod db;
pub mod doc;
pub mod qa_onnx;
pub mod qa_llama;

fn session_from_model_file<P: AsRef<Path>>(model: P, tokenizer: P) -> Result<(Session, Tokenizer)> {
    let session = Session::builder()?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_intra_threads(available_parallelism()?.get())?
        .commit_from_file(model)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{e:?}"))?;
    Ok((session, tokenizer))
}

#[derive(Debug, Clone)]
pub struct Prompt(String);

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

    pub fn encode_prompt<S: AsRef<str>>(&mut self, q: S) -> Result<Prompt> {
        use std::fmt::Write;
        let q = q.as_ref();
        let matches = self.db.search(q, 3)?;
        let mut prompt = String::with_capacity(min(4 * 1024 * 1024, q.len() * 10));
        if matches.distances.len() == 0 || matches.distances[0] > 0.7 {
            write!(prompt, "<|system|>There was no relevant information available about this question<|end|>\n")?;
        } else {
            write!(prompt, "<|system|>Context found in the database about this question\n")?;
            for (id, dist) in matches.keys.iter().zip(matches.distances.iter()) {
                if *dist <= 0.7 {
                    let chunk = self.docs.get_chunk(*id)?;
                    let s = self.docs.get(&chunk)?;
                    write!(prompt, "{s}\n\n")?;
                }
            }
            write!(prompt, " <|end|>\n")?;
        };
        write!(prompt, "<|user|>{q} <|end|><|assistant|>")?;
        Ok(Prompt(prompt))
    }

    pub fn ask(&self, prompt: &Prompt, gen_max: usize) -> Result<String> {
        self.qa.ask(&prompt.0, gen_max)
    }
}
