use crate::QaPrompt;
use std::fmt::Write;

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
    fn drop(&mut self) {
        write!(self.0, "<|end|>").unwrap()
    }
}

pub struct Phi3Prompt(pub(super) String);

impl QaPrompt for Phi3Prompt {
    fn system<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        write!(self.0, "<|system|>").unwrap();
        Writer(&mut self.0)
    }

    fn user<'a>(&'a mut self) -> impl std::fmt::Write + 'a {
        write!(self.0, "<|user|>").unwrap();
        Writer(&mut self.0)
    }

    fn with_capacity(n: usize) -> Self {
        Self(String::with_capacity(n))
    }

    fn clear(&mut self) {
        self.0.clear()
    }

    fn finalize(&mut self) {
        write!(self.0, "<|assistant|>").unwrap()
    }
}
