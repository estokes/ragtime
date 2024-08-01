use anyhow::{anyhow, bail, Result};
use core::fmt;
use fs3::FileExt;
use fxhash::FxHashMap;
use regex::bytes::Regex;
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Weak,
    },
};

#[derive(Debug)]
struct DecodedInner {
    original_path: PathBuf,
    decoded_path: PathBuf,
    open: Arc<AtomicU32>,
}

impl Drop for DecodedInner {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.decoded_path);
        self.open.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct Decoded(Arc<DecodedInner>);

impl Decoded {
    pub fn original_path(&self) -> &Path {
        &self.0.original_path
    }

    pub fn decoded_path(&self) -> &Path {
        &self.0.decoded_path
    }
}

pub struct Decoder {
    decoded: FxHashMap<PathBuf, Weak<DecodedInner>>,
    decoders: FxHashMap<&'static str, Box<dyn FnMut(&Path, &Path) -> Result<()>>>,
    tmpdir: PathBuf,
    open: Arc<AtomicU32>,
    next_id: u64,
    lock_file: File,
    tmp_file_regex: Regex,
}

impl fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GeneralDecoder")
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.cleanup();
        let _ = self.lock_file.unlock();
    }
}

impl Decoder {
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

    pub fn tmp_dir(&self) -> &Path {
        &self.tmpdir
    }

    /// Add a custom decoder for a specific mime type. The decoder is a
    /// closure that takes the path to the original file and the path to
    /// the decoded file and decodes the file to text.
    pub fn add_decoder(
        &mut self,
        mime_type: &'static str,
        decoder: Box<dyn FnMut(&Path, &Path) -> Result<()>>,
    ) {
        self.decoders.insert(mime_type, decoder);
    }

    pub fn new<S: AsRef<Path>>(tmpdir: S) -> Result<Self> {
        let tmpdir = tmpdir.as_ref().to_path_buf();
        let lock_file = File::create(tmpdir.join(".gdlock"))?;
        lock_file.try_lock_exclusive()?;
        let tmp_file_regex = Regex::new(r"^gd[0-9]+$")?;
        let t = Self {
            decoded: FxHashMap::default(),
            decoders: FxHashMap::default(),
            tmpdir,
            open: Arc::new(AtomicU32::new(0)),
            next_id: 0,
            lock_file,
            tmp_file_regex,
        };
        t.cleanup()?;
        Ok(t)
    }

    pub fn decode<P: AsRef<Path>>(&mut self, path: P) -> Result<Decoded> {
        let path = path.as_ref();
        if let Some(weak) = self.decoded.get(path) {
            if let Some(decoded) = weak.upgrade() {
                return Ok(Decoded(decoded));
            }
        }
        let decoded_path = self.tmpdir.join(format!("gd{}", self.next_id));
        let typ = infer::get_from_path(path)?
            .ok_or_else(|| anyhow!("unknown mime type"))?
            .mime_type();
        match self.decoders.get_mut(typ) {
            Some(decoder) => decoder(path, &decoded_path)?,
            None => generic_decode(typ, path, &decoded_path)?,
        }
        let decoded = Arc::new(DecodedInner {
            original_path: path.to_path_buf(),
            decoded_path,
            open: Arc::clone(&self.open),
        });
        self.decoded
            .insert(path.to_path_buf(), Arc::downgrade(&decoded));
        self.next_id += 1;
        self.open.fetch_add(1, Ordering::Relaxed);
        self.gc();
        Ok(Decoded(decoded))
    }

    fn gc(&mut self) {
        if (self.decoded.len() as u32) >= self.open.load(Ordering::Relaxed) << 1 {
            self.decoded.retain(|_, weak| weak.strong_count() > 0);
        }
    }
}

fn loconvert(ext: &str, path: &Path, decoded_path: &Path) -> Result<()> {
    use std::process::Command;
    let name = path.file_name().ok_or_else(|| anyhow!("no file name"))?;
    let outfile = PathBuf::from("/tmp")
        .join(name)
        .with_extension(ext.split_once(":").map(|(s, _)| s).unwrap_or(ext));
    let mut cmd = Command::new("libreoffice");
    cmd.arg("--convert-to")
        .arg(ext)
        .arg(path)
        .arg("--outdir")
        .arg("/tmp")
        .status()?;
    fs::copy(&outfile, decoded_path)?;
    fs::remove_file(&outfile)?;
    Ok(())
}

#[cfg(unix)]
fn generic_decode(typ: &str, path: &Path, decoded_path: &Path) -> Result<()> {
    use std::process::Command;
    match typ {
        "application/vnd.oasis.opendocument.text"
        | "application/vnd.ms-powerpoint"
        | "application/vnd.oasis.opendocument.presentation"
        | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        | "application/msword" => loconvert("txt:Text", path, decoded_path),
        "application/vnd.ms-excel"
        | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        | "application/vnd.oasis.opendocument.spreadsheet" => loconvert("csv", path, decoded_path),
        _ => bail!("no decoder for mime type {}", typ),
    }
}

#[cfg(windows)]
fn generic_decode(typ: &str, path: &Path, decoded_path: &Path) -> Result<()> {
    unimplemented!()
}
