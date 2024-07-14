use crate::{
    llama::{LlamaEmbed, LlamaEmbedModel}, FormattedPrompt
};

struct Writer<'a>(&'a mut String);

impl<'a> std::fmt::Write for Writer<'a> {
    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> std::fmt::Result {
        self.0.write_fmt(args)
    }

    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write_str(s)
    }

    fn write_char(&mut self, c: char) -> std::fmt::Result {
        self.0.write_char(c)
    }
}

impl<'a> Drop for Writer<'a> {
    fn drop(&mut self) {}
}

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
        Writer(&mut self.0)
    }

    fn user<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        Writer(&mut self.0)
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
        Writer(&mut self.0)
    }

    fn user<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        use std::fmt::Write;
        const TASK: &str =
            "Given a web search query, retrieve relevant passages that answer the query";
        write!(&mut self.0, "Instruct: {TASK}\nQuery: ").unwrap();
        Writer(&mut self.0)
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

pub type GteQwen27bInstruct = LlamaEmbed<GteQwen27bInstructModel>;
