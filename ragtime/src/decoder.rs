use anyhow::{anyhow, bail, Result};
use compact_str::{format_compact, CompactString};
use core::fmt;
use fxhash::FxHashMap;
use infer::{Infer, Matcher};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Weak,
    },
};
use tempfile::TempDir;

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
    tmpdir: TempDir,
    open: Arc<AtomicU32>,
    next_id: u64,
    infer: Infer,
}

impl fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GeneralDecoder")
    }
}

impl Decoder {
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

    /// Add a custom matcher to determine the mime type of a file.
    /// The infer library will be tried first when trying to determine the
    /// mime type of a file. On unix systems, the `file` command will be tried
    /// if the infer library does not find a match. On windows systems it will
    /// assume text/plain for unknown file types.
    pub fn add_mime_inferer(
        &mut self,
        mime: &'static str,
        extension: &'static str,
        matcher: Matcher,
    ) {
        self.infer.add(mime, extension, matcher)
    }

    pub fn new() -> Result<Self> {
        Ok(Self {
            decoded: FxHashMap::default(),
            decoders: FxHashMap::default(),
            tmpdir: TempDir::new()?,
            open: Arc::new(AtomicU32::new(0)),
            next_id: 0,
            infer: Infer::new(),
        })
    }

    pub fn decode<P: AsRef<Path>>(&mut self, path: P) -> Result<Decoded> {
        let path = path.as_ref();
        if let Some(weak) = self.decoded.get(path) {
            if let Some(decoded) = weak.upgrade() {
                return Ok(Decoded(decoded));
            }
        }
        let decoded_path = self
            .tmpdir
            .path()
            .join(&*format_compact!("{}", self.next_id));
        self.next_id += 1;
        let typ = dbg!(infer_from_path(&self.infer, path))?;
        match self.decoders.get_mut(&*typ) {
            Some(decoder) => decoder(path, &decoded_path)?,
            None => generic_decode(&self.infer, &typ, path, &decoded_path)?,
        }
        let decoded = Arc::new(DecodedInner {
            original_path: path.to_path_buf(),
            decoded_path,
            open: Arc::clone(&self.open),
        });
        self.decoded
            .insert(path.to_path_buf(), Arc::downgrade(&decoded));
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

fn infer_from_path<P: AsRef<Path>>(infer: &Infer, path: P) -> Result<CompactString> {
    let path = path.as_ref();
    let typ = infer
        .get_from_path(path)?
        .map(|t| CompactString::from(t.mime_type()));
    match typ {
        Some(typ) => Ok(typ),
        None => {
            if cfg!(unix) {
                use std::process::Command;
                let out = Command::new("file")
                    .arg("-b")
                    .arg("--mime-type")
                    .arg(path)
                    .output()?;
                Ok(CompactString::from_utf8_lossy(out.stdout.trim_ascii()))
            } else {
                Ok(CompactString::from("text/plain"))
            }
        }
    }
}

#[cfg(unix)]
fn loconvert(ext: &str, path: &Path, decoded_path: &Path) -> Result<()> {
    use std::process::Command;
    use tempfile::tempdir;
    let name = path.file_name().ok_or_else(|| anyhow!("no file name"))?;
    let dir = tempdir()?;
    fs::copy(path, dir.path().join(name))?;
    let outfile = dir
        .path()
        .join(name)
        .with_extension(ext.split_once(":").map(|(s, _)| s).unwrap_or(ext));
    Command::new("libreoffice")
        .arg("--convert-to")
        .arg(ext)
        .arg(path)
        .arg("--outdir")
        .arg(dir.path())
        .status()?;
    fs::rename(&outfile, decoded_path)?;
    Ok(())
}

#[cfg(unix)]
fn uncompress(infer: &Infer, cmd: &str, path: &Path, decoded_path: &Path) -> Result<()> {
    use std::process::Command;
    let (out, tmp) = tempfile::NamedTempFile::new()?.into_parts();
    Command::new(cmd).arg(path).stdout(out).status()?;
    let typ = infer_from_path(infer, &tmp)?;
    generic_decode(infer, &typ, &tmp, decoded_path)
}

#[cfg(unix)]
fn generic_decode(infer: &Infer, typ: &str, path: &Path, decoded_path: &Path) -> Result<()> {
    use std::process::Command;
    match typ {
        "text/plain"
        | "text/csv"
        | "text/xml"
        | "text/x-Algol68"
        | "text/x-lisp"
        | "text/x-c"
        | "text/x-c++"
        | "text/x-objective-c"
        | "text/x-ruby"
        | "text/x-asm"
        | "text/x-fortran"
        | "text/x-java"
        | "text/x-pascal"
        | "text/x-haskell"
        | "text/x-erlang"
        | "text/x-lua"
        | "text/x-php"
        | "text/x-m4"
        | "application/x-msdos-batch"
        | "text/x-script.python"
        | "text/x-perl"
        | "text/x-shellscript"
        | "text/x-tcl"
        | "application/json" => {
            fs::copy(path, decoded_path)?;
            Ok(())
        }
        "application/vnd.oasis.opendocument.text"
        | "text/html"
        | "application/vnd.ms-powerpoint"
        | "application/vnd.oasis.opendocument.presentation"
        | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        | "application/msword" => loconvert("txt:Text", path, decoded_path),
        "application/vnd.ms-excel"
        | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        | "application/vnd.oasis.opendocument.spreadsheet" => loconvert("csv", path, decoded_path),
        "application/epub+zip" => {
            Command::new("epub2txt")
                .arg(path)
                .stdout(File::create(decoded_path)?)
                .status()?;
            Ok(())
        }
        "application/pdf" => {
            Command::new("pdftotext")
                .arg(path)
                .arg(decoded_path)
                .status()?;
            Ok(())
        }
        "application/postscript" => {
            Command::new("ps2txt")
                .arg(path)
                .stdout(File::create(decoded_path)?)
                .status()?;
            Ok(())
        }
        "application/x-bzip2" => uncompress(infer, "bzcat", path, decoded_path),
        "application/gzip" => uncompress(infer, "zcat", path, decoded_path),
        "application/x-xz" => uncompress(infer, "xzcat", path, decoded_path),
        "application/zstd" => uncompress(infer, "zstdcat", path, decoded_path),
        "application/x-lzma" => uncompress(infer, "lzcat", path, decoded_path),
        "application/x-executable"
        | "application/x-pie-executable"
        | "application/x-sharedlib"
        | "application/octet-stream" => {
            Command::new("strings")
                .arg(path)
                .stdout(File::create(decoded_path)?)
                .status()?;
            Ok(())
        }
        "text/troff" => {
            Command::new("man")
                .arg(path)
                .stdout(File::create(decoded_path)?)
                .status()?;
            Ok(())
        }
        _ => bail!("no decoder for mime type {}", typ),
    }
}

#[cfg(windows)]
fn generic_decode(_infer: &Infer, typ: &str, path: &Path, decoded_path: &Path) -> Result<()> {
    match typ {
        "text/plain" => {
            fs::copy(path, decoded_path)?;
            Ok(())
        }
        typ => bail!("no decoder for mime type {}", typ),
    }
}
