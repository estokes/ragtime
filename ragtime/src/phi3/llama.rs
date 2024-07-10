use super::prompt::Phi3Prompt;
use crate::llama::{Llama, LlamaQaModel};
use llama_cpp_2::{model::LlamaModel, token::LlamaToken};

#[derive(Default)]
pub struct Phi3Model;

impl LlamaQaModel for Phi3Model {
    type Prompt = Phi3Prompt;

    fn is_finished(&mut self, model: &LlamaModel, token: LlamaToken) -> bool {
        token == model.token_eos() || token.0 == 32007
    }
}

pub type Phi3 = Llama<Phi3Model>;
