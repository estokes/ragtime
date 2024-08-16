/// Document handling
use crate::decoder::{Decoded, Decoder};
use anyhow::{anyhow, bail, Result};
use chrono::prelude::*;
use core::fmt;
use fxhash::{FxBuildHasher, FxHashMap};
use indexmap::IndexMap;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::error::Error;
use std::fmt::Display;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct DocumentChanged(pub PathBuf);

impl Display for DocumentChanged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "The document {:?} has changed since it was indexed",
            &self.0
        )
    }
}

impl Error for DocumentChanged {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DocId(u64);

static NEXT_DOCID: AtomicU64 = AtomicU64::new(0);

impl DocId {
    pub fn new() -> Self {
        Self(NEXT_DOCID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub u64);

impl fmt::Display for ChunkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

static NEXT_CHUNKID: AtomicU64 = AtomicU64::new(0);

fn word_boundry(x: u8) -> bool {
    x == 32 || x == 10 || x == 9
}

impl ChunkId {
    pub fn new() -> Self {
        Self(NEXT_CHUNKID.fetch_add(1, Ordering::Relaxed))
    }
}

impl From<u64> for ChunkId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl Into<u64> for ChunkId {
    fn into(self) -> u64 {
        self.0
    }
}

struct ChunkIter<'a> {
    data: &'a [u8],
    id: DocId,
    pos: usize,
    chunk_size: usize,
    overlap: usize,
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = ChunkDescriptor;

    fn next(&mut self) -> Option<Self::Item> {
        let mut ntok = 0;
        let mut overlap = 0;
        let mut pos = self.pos;
        loop {
            if pos >= self.data.len() {
                if pos == self.pos {
                    break None;
                } else {
                    let start = self.pos;
                    self.pos = self.data.len();
                    let cd = ChunkDescriptor {
                        id: ChunkId::new(),
                        doc: self.id,
                        start,
                        end: self.data.len() - 1,
                    };
                    break Some(cd);
                }
            }
            if ntok == self.chunk_size - self.overlap {
                overlap = pos;
            }
            if ntok >= self.chunk_size {
                let start = self.pos;
                self.pos = overlap;
                let cd = ChunkDescriptor {
                    id: ChunkId::new(),
                    doc: self.id,
                    start,
                    end: pos,
                };
                break Some(cd);
            }
            while pos < self.data.len() && !word_boundry(self.data[pos]) {
                pos += 1
            }
            while pos < self.data.len() && word_boundry(self.data[pos]) {
                pos += 1;
            }
            ntok += 1
        }
    }
}

/// Return an iterator over the chunks in this
/// document. [chunk_size] is the number of words that should be
/// in a chunk, and [overlap] is the number of words of overlap
/// that should exist between chunks. e.g. 512 and 256 would
/// produce 512 word chunks that overlap with each other by 256
/// words.
fn iter_chunks<'a>(
    id: DocId,
    data: &'a [u8],
    chunk_size: usize,
    overlap: usize,
) -> Result<ChunkIter<'a>> {
    if chunk_size == 0 || overlap >= chunk_size {
        bail!("chunk_size must be > 0, overlap must be < tokens")
    }
    Ok(ChunkIter {
        id,
        pos: 0,
        chunk_size,
        overlap,
        data,
    })
}

#[derive(Debug)]
struct Doc {
    decoded: Decoded,
    _file: File,
    map: Mmap,
    last_used: DateTime<Utc>,
    saved: SavedDoc,
}

impl Doc {
    fn new(decoded: Decoded, saved: SavedDoc, file: File, map: Mmap) -> Self {
        Self {
            decoded,
            _file: file,
            map,
            last_used: Utc::now(),
            saved,
        }
    }

    fn get<'a>(&'a self, chunk: &ChunkDescriptor) -> Result<&'a str> {
        let data = &*self.map;
        if chunk.end >= data.len() {
            bail!(
                "chunk out of bounds, doc: {:?} has changed since it was indexed?",
                chunk.doc
            )
        }
        let res = std::str::from_utf8(&data[chunk.start..chunk.end])?;
        Ok(res)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkRef<'a> {
    pub original_path: &'a Path,
    pub decoded_path: &'a Path,
    pub summary: Option<&'a str>,
    pub text: &'a str,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ChunkDescriptor {
    id: ChunkId,
    doc: DocId,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedDoc {
    id: DocId,
    path: PathBuf,
    summary: Option<String>,
    md5sum: [u8; 16],
    chunks: Vec<ChunkId>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Saved {
    docs: Vec<SavedDoc>,
    chunks: Vec<ChunkDescriptor>,
    next_docid: u64,
    next_chunkid: u64,
    max_mapped: usize,
}

#[derive(Debug)]
pub struct DocStore {
    mapped: IndexMap<DocId, Doc, FxBuildHasher>,
    unmapped: FxHashMap<DocId, SavedDoc>,
    by_path: FxHashMap<PathBuf, DocId>,
    chunks: FxHashMap<ChunkId, ChunkDescriptor>,
    max_mapped: usize,
    decoder: Decoder,
}

impl DocStore {
    pub fn new(max_mapped: usize) -> Result<Self> {
        if max_mapped < 1 {
            bail!("max_mapped must be at least 1")
        }
        Ok(Self {
            mapped: IndexMap::default(),
            unmapped: HashMap::default(),
            by_path: HashMap::default(),
            chunks: HashMap::default(),
            max_mapped,
            decoder: Decoder::new()?,
        })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut fd = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path.as_ref())?;
        let next_docid = NEXT_DOCID.load(Ordering::Relaxed);
        let next_chunkid = NEXT_CHUNKID.load(Ordering::Relaxed);
        let docs = self
            .mapped
            .iter()
            .map(|(_, d)| d.saved.clone())
            .chain(self.unmapped.values().cloned())
            .collect();
        let chunks = self.chunks.iter().map(|(_, c)| *c).collect();
        let saved = Saved {
            docs,
            chunks,
            next_chunkid,
            next_docid,
            max_mapped: self.max_mapped,
        };
        Ok(serde_json::to_writer_pretty(&mut fd, &saved)?)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let saved: Saved = serde_json::from_reader(File::open(path.as_ref())?)?;
        let mut t = Self::new(saved.max_mapped)?;
        for sd in saved.docs {
            t.by_path.insert(sd.path.clone(), sd.id);
            t.unmapped.insert(sd.id, sd);
        }
        for chunk in saved.chunks {
            t.chunks.insert(chunk.id, chunk);
        }
        macro_rules! update {
            ($var:ident, $field:ident) => {
                let _ = $var.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                    if saved.$field > cur {
                        Some(saved.$field)
                    } else {
                        None
                    }
                });
            };
        }
        update!(NEXT_DOCID, next_docid);
        update!(NEXT_CHUNKID, next_chunkid);
        Ok(t)
    }

    pub fn contains<P: AsRef<Path>>(&self, doc: P) -> bool {
        self.by_path.contains_key(doc.as_ref())
    }

    /// Add the specified document to the collection. The returned
    /// iterator contains the chunks of the document, the size and
    /// overlap of the chunks are controlled by the chunk_size and
    /// overlap parameters.
    ///
    /// The model should consume the iterator and embed each of the
    /// chunks, indexing them by ChunkId, which will uniquely identify
    /// both the document and the chunk.
    pub fn add_document<P: AsRef<Path>, S: AsRef<str>, F: FnMut(ChunkId, &str) -> Result<()>>(
        &mut self,
        path: P,
        summary: Option<S>,
        chunk_size: usize,
        overlap: usize,
        mut embed: F,
    ) -> Result<()> {
        self.gc();
        let decoded = self.decoder.decode(path.as_ref())?;
        let id = match self.by_path.entry(PathBuf::from(path.as_ref())) {
            Entry::Vacant(e) => *e.insert(DocId::new()),
            Entry::Occupied(_) => bail!("document {:?} already loaded", path.as_ref()),
        };
        let file = File::open(decoded.decoded_path())?;
        let map = unsafe { Mmap::map(&file)? };
        let saved = SavedDoc {
            id,
            path: decoded.original_path().to_path_buf(),
            summary: summary.map(|s| s.as_ref().into()),
            md5sum: md5::compute(&*map).0,
            chunks: vec![],
        };
        let chunks = &mut self.chunks;
        let mapped = &mut self.mapped;
        let mut doc = Doc::new(decoded, saved, file, map);
        macro_rules! fail {
            ($e:expr) => {{
                self.by_path.remove(path.as_ref());
                for cid in doc.saved.chunks {
                    chunks.remove(&cid);
                }
                return Err($e);
            }};
        }
        for chunk in iter_chunks(id, &*doc.map, chunk_size, overlap)? {
            let s = match doc.get(&chunk) {
                Ok(s) => s,
                Err(e) => fail!(e),
            };
            if let Err(e) = embed(chunk.id, s) {
                fail!(e)
            }
            doc.saved.chunks.push(chunk.id);
            chunks.insert(chunk.id, chunk);
        }
        mapped.insert(id, doc);
        Ok(())
    }

    /// Remove the specified document from the document store
    pub fn remove_document<P: AsRef<Path>>(&mut self, path: P) -> Vec<ChunkId> {
        let id = match self.by_path.remove(path.as_ref()) {
            Some(id) => id,
            None => return vec![],
        };
        let saved = match self.mapped.swap_remove(&id) {
            Some(doc) => {
                self.unmapped.remove(&id);
                doc.saved
            }
            None => match self.unmapped.remove(&id) {
                Some(saved) => saved,
                None => return vec![]
            }
        };
        for id in &saved.chunks {
            self.chunks.remove(id);
        }
        saved.chunks
    }

    fn gc(&mut self) {
        if self.mapped.len() > self.max_mapped {
            self.mapped
                .sort_by(|_, d0, _, d1| d1.last_used.cmp(&d0.last_used));
            while self.mapped.len() > 0 && self.mapped.len() > self.max_mapped {
                let (id, doc) = self.mapped.pop().unwrap();
                self.unmapped.insert(id, doc.saved);
            }
        }
    }

    /// Get a reference to the chunk identified by [id]
    pub fn get_chunk_ref<'a, I: Into<ChunkId>>(&'a mut self, id: I) -> Result<ChunkRef<'a>> {
        let id = id.into();
        let chunk = *self
            .chunks
            .get(&id)
            .ok_or_else(|| anyhow!("no such chunk {id:?}"))?;
        match self.mapped.get_mut(&chunk.doc) {
            Some(doc) => {
                doc.last_used = Utc::now();
            }
            None => match self.unmapped.get(&chunk.doc) {
                None => bail!("no document for chunk {id}"),
                Some(saved) => {
                    let decoded = self.decoder.decode(&saved.path)?;
                    let file = File::open(decoded.decoded_path())?;
                    let map = unsafe { Mmap::map(&file)? };
                    let md5sum = md5::compute(&*map).0;
                    if md5sum != saved.md5sum {
                        bail!(DocumentChanged(decoded.original_path().to_path_buf()))
                    }
                    let doc = Doc::new(
                        decoded,
                        self.unmapped.remove(&chunk.doc).unwrap(),
                        file,
                        map,
                    );
                    self.mapped.insert(chunk.doc, doc);
                    self.gc()
                }
            },
        };
        let doc = self
            .mapped
            .get(&chunk.doc)
            .ok_or_else(|| anyhow!("document isn't loaded"))?;
        Ok(ChunkRef {
            summary: doc.saved.summary.as_ref().map(|s| s.as_str()),
            text: doc.get(&chunk)?,
            original_path: doc.decoded.original_path(),
            decoded_path: doc.decoded.decoded_path(),
        })
    }

    pub(super) fn decoder_mut(&mut self) -> &mut Decoder {
        &mut self.decoder
    }
}
