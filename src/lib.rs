use crate::doc::{ChunkId, DocStore};
use anyhow::{anyhow, bail, Result};
use compact_str::CompactString;
use ort::Session;
use std::{cmp::min, fs, path::Path, thread::available_parallelism};
use tokenizers::Tokenizer;
use usearch::ffi::Matches;

pub mod bge_m3;
pub mod doc;
pub mod phi3;

fn session_from_model_file<P: AsRef<Path>>(model: P, tokenizer: P) -> Result<(Session, Tokenizer)> {
    let session = Session::builder()?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_intra_threads(available_parallelism()?.get())?
        .commit_from_file(model)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{e:?}"))?;
    Ok((session, tokenizer))
}

pub trait Persistable: Sized {
    type Ctx;

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;
    fn load<P: AsRef<Path>>(ctx: Self::Ctx, path: P, view: bool) -> Result<Self>;
}

pub trait EmbedModel: Sized {
    type Args;

    fn new(args: Self::Args) -> Result<Self>;
    fn add(&mut self, text: &[(ChunkId, &str)]) -> Result<()>;
    fn search(&mut self, q: &str, n: usize) -> Result<Matches>;
}

pub trait QaPrompt {
    type FinalPrompt;

    fn new() -> Self;
    fn with_capacity(n: usize) -> Self;
    fn system<'a>(&'a mut self) -> impl std::fmt::Write + 'a;
    fn user<'a>(&'a mut self) -> impl std::fmt::Write + 'a;
    fn finalize(self) -> Result<Self::FinalPrompt>;
    fn clear(&mut self);
}

pub trait QaModel: Sized {
    type Args;
    type Prompt: QaPrompt;

    fn new(args: Self::Args) -> Result<Self>;
    fn ask<'a>(
        &'a mut self,
        q: <Self::Prompt as QaPrompt>::FinalPrompt,
        gen_max: Option<usize>,
    ) -> Result<impl Iterator<Item = Result<CompactString>> + 'a>;
}

pub struct RagQa<E, Q> {
    docs: DocStore,
    db: E,
    qa: Q,
}

impl<E, Q> RagQa<E, Q>
where
    E: Persistable,
    Q: Persistable,
{
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
    pub fn load<P: AsRef<Path>>(
        embed_ctx: E::Ctx,
        qa_ctx: Q::Ctx,
        path: P,
        view: bool,
    ) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_dir() {
            bail!("save directory could not be found")
        }
        let docs = DocStore::load(path.join("docs.json"))?;
        let db = E::load(embed_ctx, path.join("db.json"), view)?;
        let qa = Q::load(qa_ctx, path.join("model.json"), view)?;
        Ok(Self { docs, db, qa })
    }
}

impl<E, Q> RagQa<E, Q>
where
    E: EmbedModel,
    Q: QaModel,
{
    pub fn new(max_mapped: usize, embed_args: E::Args, qa_args: Q::Args) -> Result<Self> {
        let docs = DocStore::new(max_mapped);
        let db = E::new(embed_args)?;
        let qa = Q::new(qa_args)?;
        Ok(Self { docs, db, qa })
    }

    pub fn add_document<P: AsRef<Path>>(
        &mut self,
        doc: P,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<()> {
        println!("adding document {:?}", doc.as_ref());
        let chunks = self
            .docs
            .add_document(doc, chunk_size, overlap)?
            .collect::<Result<Vec<_>>>()?;
        self.db.add(&chunks)?;
        println!("document added");
        Ok(())
    }

    fn encode_prompt(&mut self, q: &str) -> Result<<Q::Prompt as QaPrompt>::FinalPrompt> {
        use std::fmt::Write;
        let mut prompt = Q::Prompt::with_capacity(min(4 * 1024 * 1024, q.len() * 10));
        let matches = self.db.search(q, 3)?;
        {
            let mut dst = prompt.system();
            if matches.distances.len() == 0 || matches.distances[0] > 0.7 {
                write!(
                    dst,
                    "There was no relevant information available about \"{q}\" in the database\n"
                )?;
            } else {
                for (id, dist) in matches.keys.iter().zip(matches.distances.iter()) {
                    if *dist <= 0.7 {
                        let chunk = self.docs.get_chunk(*id)?;
                        let s = self.docs.get(&chunk)?;
                        write!(dst, "{s}\n\n")?;
                    }
                }
            }
        }
        write!(prompt.user(), "{q}")?;
        Ok(prompt.finalize()?)
    }

    pub fn ask<'a, S: AsRef<str>>(
        &'a mut self,
        q: S,
        gen_max: Option<usize>,
    ) -> Result<impl Iterator<Item = Result<CompactString>> + 'a> {
        let prompt = self.encode_prompt(q.as_ref())?;
        self.qa.ask(prompt, gen_max)
    }
}

pub type RagQaPhi3BgeM3 = RagQa<bge_m3::onnx::BgeM3, phi3::llama::Phi3>;
