use crate::{phi3::prompt::Phi3Prompt, session_from_model_file, Persistable, QaModel};
use anyhow::{anyhow, Result};
use compact_str::CompactString;
use ndarray::{array, concatenate, s, Array1, Array4, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};
use ort::{DynValue, Session, SessionInputValue};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

use super::prompt::Phi3FinalPrompt;

#[derive(Debug, Serialize, Deserialize)]
pub struct Saved {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

pub struct Phi3 {
    params: Saved,
    session: Session,
    tokenizer: Tokenizer,
}

impl Persistable for Phi3 {
    type Ctx = ();

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        Ok(serde_json::to_writer_pretty(&mut fd, &self.params)?)
    }

    fn load<P: AsRef<Path>>(_ctx: (), path: P, _view: bool) -> Result<Self> {
        let params: Saved = serde_json::from_reader(File::open(path)?)?;
        let (session, tokenizer) = session_from_model_file(&params.model, &params.tokenizer)?;
        Ok(Self {
            params,
            session,
            tokenizer,
        })
    }
}

struct TokenIter<'a> {
    model: &'a mut Phi3,
    tokens: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    attn_mask: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    gen: Option<usize>,
    n: usize,
}

impl<'a> TokenIter<'a> {
    fn step(&mut self) -> Result<Option<CompactString>> {
        let len = self.tokens.len();
        let tokens_view = self.tokens.view().insert_axis(Axis(0));
        let attn_mask_view = self.attn_mask.view().insert_axis(Axis(0));
        let args = self.model.encode_args(len, tokens_view, attn_mask_view)?;
        let outputs = self.model.session.run(args)?;
        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let (token, _) = logits.slice(s![0, len - 1, ..]).iter().enumerate().fold(
            (0, -f32::MAX),
            |(ct, cp), (t, p)| if *p > cp { (t, *p) } else { (ct, cp) },
        );
        // CR estokes: abstract this
        if token == 32000 {
            // end of text
            return Ok(None);
        }
        self.tokens = concatenate![Axis(0), self.tokens, array![token as i64]];
        self.attn_mask = concatenate![Axis(0), self.attn_mask, array![1 as i64]];
        let t = self
            .model
            .tokenizer
            .decode(&[token as u32], true)
            .map_err(|e| anyhow!("{e:?}"))?;
        Ok(Some(CompactString::from(t)))
    }
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = Result<CompactString>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.gen.is_some() && self.n >= self.gen.unwrap() {
            None
        } else {
            self.n += 1;
            self.step().transpose()
        }
    }
}

impl QaModel for Phi3 {
    type Args = Saved;
    type Prompt = Phi3Prompt;

    fn new(params: Saved) -> Result<Self> {
        let (session, tokenizer) = session_from_model_file(&params.model, &params.tokenizer)?;
        Ok(Self {
            params,
            session,
            tokenizer,
        })
    }

    fn ask(
        &mut self,
        question: Phi3FinalPrompt,
        gen: Option<usize>,
    ) -> Result<impl Iterator<Item = Result<CompactString>>> {
        let encoded = self
            .tokenizer
            .encode(question.0, true)
            .map_err(|e| anyhow!("{e:?}"))?;
        Ok(TokenIter {
            model: self,
            tokens: Array1::from_iter(encoded.get_ids().iter().map(|t| *t as i64)),
            attn_mask: Array1::from_iter(encoded.get_ids().iter().map(|_| 1i64)),
            gen,
            n: 0,
        })
    }
}

impl Phi3 {
    fn encode_args(
        &self,
        len: usize,
        inputs: ArrayBase<ViewRepr<&i64>, Dim<[usize; 2]>>,
        attention_mask: ArrayBase<ViewRepr<&i64>, Dim<[usize; 2]>>,
    ) -> Result<HashMap<String, SessionInputValue>> {
        let mut args: HashMap<String, SessionInputValue> = HashMap::default();
        for inp in &self.session.inputs {
            if inp.name != "input_ids" && inp.name != "attention_mask" {
                let a = Array4::<f32>::zeros((1, 32, len, 96));
                let v = DynValue::try_from(a)?;
                args.insert(inp.name.clone().into(), v.into());
            }
        }
        let inputs = DynValue::try_from(inputs)?;
        let attention_mask = DynValue::try_from(attention_mask)?;
        args.insert("input_ids".into(), inputs.into());
        args.insert("attention_mask".into(), attention_mask.into());
        Ok(args)
    }
}
