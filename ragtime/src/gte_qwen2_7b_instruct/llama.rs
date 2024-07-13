use crate::{
    llama::{Llama, LlamaEmbedModel},
    FormattedPrompt,
};

#[derive(Debug)]
pub struct GteFinalEmbedPrompt(String);

impl AsRef<str> for GteFinalEmbedPrompt {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

pub struct GteEmbedPrompt(String);

impl FormattedPrompt for GteEmbedPrompt {
    type FinalPrompt = GteFinalEmbedPrompt;

    fn system<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        &mut self.0
    }

    fn user<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        &mut self.0
    }

    fn finalize(self) -> anyhow::Result<Self::FinalPrompt> {
        Ok(GteFinalEmbedPrompt(self.0))
    }

    fn with_capacity(n: usize) -> Self {
        Self(String::with_capacity(n))
    }

    fn new() -> Self {
        Self(String::new())
    }

    fn clear(&mut self) {
        self.0.clear()
    }
}

#[derive(Debug)]
pub struct GteFinalSearchPrompt(String);

impl AsRef<str> for GteFinalSearchPrompt {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

pub struct GteSearchPrompt(String);

impl FormattedPrompt for GteSearchPrompt {
    type FinalPrompt = GteFinalSearchPrompt;

    fn system<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        &mut self.0
    }

    fn user<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        use std::fmt::Write;
        const TASK: &str =
            "Given a web search query, retrieve relevant passages that answer the query";
        write!(&mut self.0, "Instruct: {TASK}\nQuery: ").unwrap();
        &mut self.0
    }

    fn finalize(self) -> anyhow::Result<Self::FinalPrompt> {
        Ok(GteFinalSearchPrompt(self.0))
    }

    fn with_capacity(n: usize) -> Self {
        Self(String::with_capacity(n))
    }

    fn new() -> Self {
        Self(String::new())
    }

    fn clear(&mut self) {
        self.0.clear()
    }
}

#[derive(Default)]
pub struct GteQwen27bInstructModel;

impl LlamaEmbedModel for GteQwen27bInstructModel {
    type EmbedPrompt = GteEmbedPrompt;
    type SearchPrompt = GteSearchPrompt;

    fn get_embedding(
        &mut self,
        ctx: &mut llama_cpp_2::context::LlamaContext,
        i: i32,
    ) -> anyhow::Result<Vec<f32>> {
        Ok(ctx.embeddings_ith(i)?.to_vec())
    }
}

pub type GteQwen27bInstruct = Llama<GteQwen27bInstructModel>;
