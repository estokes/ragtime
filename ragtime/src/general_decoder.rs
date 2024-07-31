/// A FileDecoder for a lot of well known file formats
use crate::DecodedFile;
use anyhow::{Ok, Result};
use fs3::FileExt;
use fxhash::FxHashMap;
use regex::bytes::Regex;
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug)]
struct DecodedInner {
    original_path: PathBuf,
    decoded_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Decoded(Arc<DecodedInner>);

impl DecodedFile for Decoded {
    fn original_path(&self) -> &Path {
        &self.0.original_path
    }

    fn decoded_path(&self) -> &Path {
        &self.0.decoded_path
    }
}

#[derive(Debug)]
pub struct GeneralDecoder {
    decoded: FxHashMap<PathBuf, Decoded>,
    tmpdir: PathBuf,
    next_id: u64,
    lock_file: File,
    tmp_file_regex: Regex,
}

impl Drop for GeneralDecoder {
    fn drop(&mut self) {
        let _ = self.cleanup();
        let _ = self.lock_file.unlock();
    }
}

impl GeneralDecoder {
    fn cleanup(&self) -> Result<()> {
        for f in self.tmpdir.read_dir()? {
            let f = f?;
            if f.file_type()?.is_file() {
                let name = f.file_name();
                if self.tmp_file_regex.is_match(name.as_encoded_bytes()) {
                    fs::remove_file(f.path())?;
                }
            }
        }
        Ok(())
    }

    pub fn new<S: AsRef<Path>>(tmpdir: S) -> Result<Self> {
        let tmpdir = tmpdir.as_ref().to_path_buf();
        let lock_file = File::create(tmpdir.join(".gdlock"))?;
        lock_file.try_lock_exclusive()?;
        let tmp_file_regex = Regex::new(r"^gd[0-9]+$")?;
        let t = Self {
            decoded: FxHashMap::default(),
            tmpdir,
            next_id: 0,
            lock_file,
            tmp_file_regex,
        };
        t.cleanup()?;
        Ok(t)
    }
}
