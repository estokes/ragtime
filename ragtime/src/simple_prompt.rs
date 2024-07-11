use crate::FormattedPrompt;
use anyhow::Result;

#[derive(Debug)]
pub struct SimpleFinalPrompt(pub String);

impl AsRef<str> for SimpleFinalPrompt {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&str> for SimpleFinalPrompt {
    fn from(s: &str) -> Self {
        Self(s.into())
    }
}

impl From<String> for SimpleFinalPrompt {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[derive(Debug)]
pub struct SimplePrompt(String);


impl FormattedPrompt for SimplePrompt {
    type FinalPrompt = SimpleFinalPrompt;

    fn system<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        &mut self.0
    }

    fn user<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        &mut self.0
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

    fn finalize(self) -> Result<SimpleFinalPrompt> {
        Ok(SimpleFinalPrompt(self.0))
    }
}
