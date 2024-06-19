use crate::session_from_model_file;
use anyhow::{anyhow, Result};
use ndarray::{array, concatenate, s, Array1, Array4, ArrayBase, Axis, Dim, ViewRepr};
use ort::{DynValue, Session, SessionInputValue};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap, fs::{File, OpenOptions}, io::{stdout, Write}, path::{Path, PathBuf}
};
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Deserialize)]
struct Saved {
    model: PathBuf,
    tokenizer: PathBuf,
}

pub struct QaModel {
    params: Saved,
    session: Session,
    tokenizer: Tokenizer,
}

impl QaModel {
    pub fn new<T: AsRef<Path>>(model: T, tokenizer: T) -> Result<Self> {
        let params = Saved {
            model: PathBuf::from(model.as_ref()),
            tokenizer: PathBuf::from(tokenizer.as_ref()),
        };
        let (session, tokenizer) = session_from_model_file(&params.model, &params.tokenizer)?;
        Ok(Self {
            params,
            session,
            tokenizer,
        })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut fd = OpenOptions::new().create(true).write(true).open(&path)?;
        Ok(serde_json::to_writer_pretty(&mut fd, &self.params)?)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let params: Saved = serde_json::from_reader(File::open(path)?)?;
        let (session, tokenizer) = session_from_model_file(&params.model, &params.tokenizer)?;
        Ok(Self {
            params,
            session,
            tokenizer,
        })
    }

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

    pub fn ask(&self, question: &str, gen: usize) -> Result<String> {
        let encoded = self
            .tokenizer
            .encode(question, true)
            .map_err(|e| anyhow!("{e:?}"))?;
        let mut tokens = Array1::from_iter(encoded.get_ids().iter().map(|t| *t as i64));
        let mut attn_mask = Array1::from_iter(encoded.get_ids().iter().map(|_| 1i64));
        for _ in 0..gen {
            let len = tokens.len();
            let tokens_view = tokens.view().insert_axis(Axis(0));
            let attn_mask_view = attn_mask.view().insert_axis(Axis(0));
            let args = self.encode_args(len, tokens_view, attn_mask_view)?;
            let outputs = self.session.run(args)?;
            let logits = outputs["logits"].try_extract_tensor::<f32>()?;
            let (token, _) = logits.slice(s![0, len - 1, ..]).iter().enumerate().fold(
                (0, -f32::MAX),
                |(ct, cp), (t, p)| if *p > cp { (t, *p) } else { (ct, cp) },
            );
            // CR estokes: abstract this
            if token == 32000 {
                // end of text
                break;
            }
            print!(
                "{}",
                self.tokenizer
                    .decode(&[token as u32], false)
                    .map_err(|e| anyhow!("{e:?}"))?
            );
            stdout().flush()?;
            tokens = concatenate![Axis(0), tokens, array![token as i64]];
            attn_mask = concatenate![Axis(0), attn_mask, array![1 as i64]];
        }
        let tokens = tokens.iter().map(|i| *i as u32).collect::<Vec<_>>();
        Ok(self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow!("{e:?}"))?)
    }
}
