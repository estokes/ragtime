/// Document handling
use anyhow::{anyhow, bail, Result};
use chrono::prelude::*;
use fxhash::{FxBuildHasher, FxHashMap};
use indexmap::IndexMap;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

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

static NEXT_CHUNKID: AtomicU64 = AtomicU64::new(0);

impl ChunkId {
    pub fn new() -> Self {
        Self(NEXT_CHUNKID.fetch_add(1, Ordering::Relaxed))
    }
}

struct ChunkIter<'a> {
    id: DocId,
    data: &'a [u8],
    pos: usize,
    chunk_size: usize,
    overlap: usize,
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = Chunk;

    fn next(&mut self) -> Option<Self::Item> {
        fn word_boundry(x: u8) -> bool {
            x == 32 || x == 10 || x == 9
        }
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
                    break Some(Chunk {
                        id: ChunkId::new(),
                        doc: self.id,
                        start,
                        end: self.data.len() - 1,
                    });
                }
            }
            if ntok == self.chunk_size - self.overlap {
                overlap = pos;
            }
            if ntok >= self.chunk_size {
                let start = self.pos;
                self.pos = overlap;
                break Some(Chunk {
                    id: ChunkId::new(),
                    doc: self.id,
                    start,
                    end: pos,
                });
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

#[derive(Debug)]
struct Doc {
    path: PathBuf,
    id: DocId,
    _file: File,
    map: Mmap,
    last_used: DateTime<Utc>,
}

impl Doc {
    fn new(id: DocId, path: PathBuf) -> Result<Self> {
        let file = File::open(&path)?;
        let map = unsafe { Mmap::map(&file)? };
        Ok(Self {
            path,
            id,
            _file: file,
            map,
            last_used: Utc::now(),
        })
    }

    /// Return an iterator over the chunks in this
    /// document. [chunk_size] is the number of words that should be
    /// in a chunk, and [overlap] is the number of words of overlap
    /// that should exist between chunks. e.g. 512 and 256 would
    /// produce 512 word chunks that overlap with each other by 256
    /// words.
    fn chunks<'a>(&'a self, chunk_size: usize, overlap: usize) -> Result<ChunkIter<'a>> {
        let id = self.id;
        if chunk_size == 0 || overlap >= chunk_size {
            bail!("chunk_size must be > 0, overlap must be < tokens")
        }
        Ok(ChunkIter {
            data: &*self.map,
            id,
            pos: 0,
            chunk_size,
            overlap,
        })
    }

    fn get<'a>(&'a self, chunk: &Chunk) -> Result<&'a str> {
        let data = &*self.map;
        let res = std::str::from_utf8(&data[chunk.start..chunk.end])?;
        Ok(res)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Chunk {
    id: ChunkId,
    doc: DocId,
    start: usize,
    end: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Saved {
    docs: Vec<(DocId, PathBuf)>,
    chunks: Vec<Chunk>,
    next_docid: u64,
    next_chunkid: u64,
    max_mapped: usize,
}

#[derive(Debug)]
pub struct DocStore {
    mapped: IndexMap<DocId, Doc, FxBuildHasher>,
    unmapped: FxHashMap<DocId, PathBuf>,
    by_path: FxHashMap<PathBuf, DocId>,
    chunks: FxHashMap<ChunkId, Chunk>,
    max_mapped: usize,
}

impl DocStore {
    pub fn new(max_mapped: usize) -> Self {
        Self {
            mapped: IndexMap::default(),
            unmapped: HashMap::default(),
            by_path: HashMap::default(),
            chunks: HashMap::default(),
            max_mapped,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut fd = OpenOptions::new()
            .create(true)
            .write(true)
            .open(path.as_ref())?;
        let next_docid = NEXT_DOCID.load(Ordering::Relaxed);
        let next_chunkid = NEXT_CHUNKID.load(Ordering::Relaxed);
        let docs = self
            .by_path
            .iter()
            .map(|(p, id)| (*id, p.clone()))
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
        let mut t = Self::new(saved.max_mapped);
        for (id, path) in saved.docs {
            t.unmapped.insert(id, path.clone());
            t.by_path.insert(path, id);
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

    pub fn add_document<'a, S: AsRef<Path>>(
        &'a mut self,
        path: S,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<impl Iterator<Item = Result<(ChunkId, &'a str)>> + 'a> {
        let path = PathBuf::from(path.as_ref());
        let id = *self.by_path.entry(path.clone()).or_insert_with(DocId::new);
        let chunks = &mut self.chunks;
        let mapped = &mut self.mapped;
        let doc = if mapped.contains_key(&id) {
            mapped.get(&id).unwrap()
        } else {
            self.unmapped.remove(&id);
            mapped.insert(id, Doc::new(id, path)?);
            mapped.get(&id).unwrap()
        };
        Ok(doc.chunks(chunk_size, overlap)?.map(move |chunk| {
            chunks.insert(chunk.id, chunk);
            Ok((chunk.id, doc.get(&chunk)?))
        }))
    }

    pub fn get_chunk(&mut self, id: u64) -> Result<Chunk> {
        let chunk = *self
            .chunks
            .get(&ChunkId(id))
            .ok_or_else(|| anyhow!("no such chunk {id:?}"))?;
        match self.mapped.get_mut(&chunk.doc) {
            Some(doc) => {
                doc.last_used = Utc::now();
            }
            None => match self.unmapped.remove(&chunk.doc) {
                None => bail!("no document for chunk {id}"),
                Some(path) => {
                    self.mapped.insert(chunk.doc, Doc::new(chunk.doc, path)?);
                }
            },
        }
        if self.mapped.len() > self.max_mapped {
            self.mapped
                .sort_by(|_, d0, _, d1| d1.last_used.cmp(&d0.last_used));
            while self.mapped.len() > 0 && self.mapped.len() > self.max_mapped {
                let (id, doc) = self.mapped.pop().unwrap();
                self.unmapped.insert(id, doc.path);
            }
        }
        Ok(chunk)
    }

    pub fn get<'a>(&'a self, chunk: &Chunk) -> Result<&'a str> {
        let doc = self
            .mapped
            .get(&chunk.doc)
            .ok_or_else(|| anyhow!("document isn't loaded"))?;
        doc.get(chunk)
    }
}
