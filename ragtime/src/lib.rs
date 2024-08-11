use crate::doc::{ChunkId, DocStore};
use anyhow::{anyhow, bail, Result};
use compact_str::CompactString;
use ort::Session;
use smallvec::SmallVec;
use std::{
    cmp::min,
    fmt::Debug,
    fs,
    path::{Path, PathBuf},
    thread::available_parallelism,
};
use tokenizers::Tokenizer;
use usearch::ffi::Matches;

pub mod bge_m3;
pub mod decoder;
pub mod doc;
pub mod gte_large_en;
pub mod gte_qwen2_7b_instruct;
pub mod llama;
pub mod phi3;
pub mod simple_prompt;

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

pub trait FormattedPrompt {
    type FinalPrompt: Debug + AsRef<str>;

    fn new() -> Self;
    fn with_capacity(n: usize) -> Self;
    fn system<'a>(&'a mut self) -> impl std::fmt::Write + 'a;
    fn user<'a>(&'a mut self) -> impl std::fmt::Write + 'a;
    fn finalize(self) -> Result<Self::FinalPrompt>;
    fn clear(&mut self);
}

pub trait EmbedModel: Sized {
    type Ctx;
    type Args;
    type SearchPrompt: FormattedPrompt;
    type EmbedPrompt: FormattedPrompt;

    fn new(ctx: Self::Ctx, args: Self::Args) -> Result<Self>;
    fn add(
        &mut self,
        summary: <Self::EmbedPrompt as FormattedPrompt>::FinalPrompt,
        text: &[(ChunkId, <Self::EmbedPrompt as FormattedPrompt>::FinalPrompt)],
    ) -> Result<()>;
    fn search(
        &mut self,
        q: <Self::SearchPrompt as FormattedPrompt>::FinalPrompt,
        n: usize,
    ) -> Result<Matches>;
}

pub trait QaModel: Sized {
    type Ctx;
    type Args;
    type Prompt: FormattedPrompt;

    fn new(ctx: Self::Ctx, args: Self::Args) -> Result<Self>;
    fn ask<'a>(
        &'a mut self,
        q: <Self::Prompt as FormattedPrompt>::FinalPrompt,
        gen_max: Option<usize>,
    ) -> Result<impl Iterator<Item = Result<CompactString>> + 'a>;
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub distance: f32,
    pub path: PathBuf,
    pub summary: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone)]
pub enum SummarySpec {
    None,
    Generate,
    Summary(String),
}

impl Default for SummarySpec {
    fn default() -> Self {
        SummarySpec::Generate
    }
}

impl From<String> for SummarySpec {
    fn from(value: String) -> Self {
        SummarySpec::Summary(value)
    }
}

impl<'a> From<&'a str> for SummarySpec {
    fn from(value: &'a str) -> Self {
        SummarySpec::Summary(value.into())
    }
}

/** RagQa encapsulates the RAG workflow into a simple api that makes
the core operations single method calls.

```no_run
use ragtime::{llama, RagQaPhi3GteQwen27bInstruct};
use llama_cpp_2::llama_backend::LlamaBackend;
use anyhow::Result;
use std::{io::{stdout, Write}, sync::Arc};
# fn main() -> Result<()> {
let backend = Arc::new(LlamaBackend::init()?);
let mut qa = RagQaPhi3GteQwen27bInstruct::new(
    64,
    backend.clone(),
    llama::Args::default().with_model("gte-Qwen2-7B-instruct/ggml-model-q8_0.gguf"),
    backend,
    llama::Args::default().with_model("Phi-3-mini-128k-instruct/ggml-model-q8_0.gguf")
)?;

// add documents
qa.add_document("doc0", 256, 128)?;
qa.add_document("doc1", 256, 128)?;

// query
for tok in qa.ask("question about your docs", None)? {
    let tok = tok?;
    print!("{tok}");
    stdout().flush()?;
}
# Ok(())
# }
```
**/
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
    pub fn new(max_mapped: usize, db: E, qa: Q) -> Result<Self> {
        Ok(Self {
            docs: DocStore::new(max_mapped)?,
            db,
            qa,
        })
    }

    pub fn add_document<P: AsRef<Path>, S: Into<SummarySpec>>(
        &mut self,
        doc: P,
        summary: S,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<()> {
        use std::fmt::Write;
        dbg!("adding document", doc.as_ref());
        if self.docs.contains(&doc) {
            dbg!("skipping; already added");
            return Ok(());
        }
        let decoded = self.docs.decoder_mut().decode(doc.as_ref())?;
        let summary = match summary.into() {
            SummarySpec::None => None,
            SummarySpec::Summary(s) => Some(s),
            SummarySpec::Generate => {
                let txt = fs::read_to_string(dbg!(decoded.decoded_path()))?;
                if txt.len() >= 128 {
                    let mut summary = String::new();
                    let mut prompt = Q::Prompt::new();
                    write!(prompt.system(), "Please write a brief summary of the text. Try to squeeze all the major concepts in under 300 words.")?;
                    write!(prompt.user(), "{}", txt)?;
                    for tok in self.qa.ask(prompt.finalize()?, None)? {
                        summary.push_str(&tok?);
                    }
                    Some(summary)
                } else {
                    None
                }
            }
        };
        let chunks = self
            .docs
            .add_document(doc, summary.as_ref(), chunk_size, overlap)?
            .map(|r| {
                r.and_then(|(id, s)| {
                    let mut prompt = E::EmbedPrompt::new();
                    write!(prompt.user(), "{s}")?;
                    Ok((id, prompt.finalize()?))
                })
            })
            .collect::<Result<SmallVec<[_; 128]>>>()?;
        let mut p = E::EmbedPrompt::new();
        if let Some(summary) = summary {
            write!(p.user(), "{summary}")?;
        }
        self.db.add(p.finalize()?, &chunks)?;
        Ok(())
    }

    /// add or override a custom decoder for a specific mime type. The decoder takes the path
    /// to the original file and the path to the temp file it should write to and returns a Result
    /// indicating success or failure.
    pub fn add_decoder(
        &mut self,
        mime_type: &'static str,
        decoder: Box<dyn FnMut(&Path, &Path) -> Result<()>>,
    ) {
        self.docs.decoder_mut().add_decoder(mime_type, decoder)
    }

    fn encode_prompt(&mut self, q: &str) -> Result<<Q::Prompt as FormattedPrompt>::FinalPrompt> {
        use std::fmt::Write;
        let mut prompt = Q::Prompt::with_capacity(min(128 * 1024, q.len() * 10));
        let mut sprompt = E::SearchPrompt::new();
        write!(sprompt.user(), "{q}")?;
        let matches = self.db.search(sprompt.finalize()?, 3)?;
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
                        let doc = self.docs.get(&chunk)?;
                        write!(dst, "Document Path {:?}\n", doc.original_path)?;
                        if let Some(summary) = doc.summary.as_ref() {
                            write!(dst, "Document Summary\n{}\n", summary)?;
                        }
                        write!(dst, "Document Section\n{}\n\n", doc.text)?;
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
        use std::fmt::Write;
        let mut sprompt = E::SearchPrompt::new();
        write!(sprompt.user(), "{}", q.as_ref())?;
        let matches = self.db.search(sprompt.finalize()?, n)?;
        matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .map(|(id, dist)| {
                let chunk = self.docs.get_chunk(*id)?;
                let doc = self.docs.get(&chunk)?;
                Ok(SearchResult {
                    distance: *dist,
                    path: doc.original_path.to_owned(),
                    summary: doc.summary.map(|s| s.to_string()),
                    text: doc.text.to_string(),
                })
            })
            .collect::<Result<_>>()
    }
}
