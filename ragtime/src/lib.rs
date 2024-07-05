use crate::doc::{ChunkId, DocStore};
use anyhow::{anyhow, bail, Result};
use compact_str::CompactString;
use ort::Session;
use smallvec::SmallVec;
use std::{cmp::min, fmt::Debug, fs, path::Path, thread::available_parallelism};
use tokenizers::Tokenizer;
use usearch::ffi::Matches;

pub mod bge_m3;
pub mod doc;
pub mod gte_large_en;
pub mod phi3;

fn session_from_model_file<P: AsRef<Path>>(model: P, tokenizer: P) -> Result<(Session, Tokenizer)> {
    let session = Session::builder()?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_intra_threads(available_parallelism()?.get())?
        .commit_from_file(model)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{e:?}"))?;
    Ok((session, tokenizer))
}

fn l2_normalize(input: &mut [f32]) {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();
    for val in input {
        *val /= magnitude;
    }
}

pub trait Persistable: Sized {
    type Ctx;

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;
    fn load<P: AsRef<Path>>(ctx: Self::Ctx, path: P, view: bool) -> Result<Self>;
}

pub trait EmbedModel: Sized {
    type Args;

    fn new(args: Self::Args) -> Result<Self>;
    fn add<S: AsRef<str>>(&mut self, summary: S, text: &[(ChunkId, S)]) -> Result<()>;
    fn search<S: AsRef<str>>(&mut self, q: S, n: usize) -> Result<Matches>;
}

pub trait QaPrompt {
    type FinalPrompt: Debug;

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

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub distance: f32,
    pub summary: String,
    pub text: String,
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
        use std::fmt::Write;
        let mut prompt = Q::Prompt::new();
        write!(prompt.system(), "Please produce a brief summary of the text. Try to squeeze all the major concepts in under 300 words.")?;
        write!(prompt.user(), "{}", fs::read_to_string(doc.as_ref())?)?;
        let mut summary = String::new();
        for tok in self.qa.ask(prompt.finalize()?, None)? {
            summary.push_str(&tok?);
        }
        let chunks = self
            .docs
            .add_document(doc, &summary, chunk_size, overlap)?
            .collect::<Result<SmallVec<[_; 128]>>>()?;
        self.db.add(summary.as_str(), &chunks)?;
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
                        let (summary, text) = self.docs.get(&chunk)?;
                        write!(
                            dst,
                            "Document Summary\n{summary}\n\nDocument Section\n{text}\n\n"
                        )?;
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
        tracing::debug!("{:?}", prompt);
        self.qa.ask(prompt, gen_max)
    }

    pub fn search<S: AsRef<str>>(&mut self, q: S, n: usize) -> Result<Vec<SearchResult>> {
        let matches = self.db.search(q, n)?;
        matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .map(|(id, dist)| {
                let chunk = self.docs.get_chunk(*id)?;
                let (summary, text) = self.docs.get(&chunk)?;
                Ok(SearchResult {
                    distance: *dist,
                    summary: summary.to_string(),
                    text: text.to_string(),
                })
            })
            .collect::<Result<_>>()
    }
}

pub type RagQaPhi3BgeM3 = RagQa<bge_m3::onnx::BgeM3, phi3::llama::Phi3>;
pub type RagQaPhi3GteLargeEn = RagQa<gte_large_en::onnx::GteLargeEn, phi3::llama::Phi3>;
