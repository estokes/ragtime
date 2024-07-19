# Ragtime

Ragtime is a rust library that intends to make self hosted retrieval
augmented generation (RAG) applications easier to deploy and
use. Currently it uses Phi3 for question answering and summarizing,
and supports multiple embedding models. It has a generic model
interface to facilitate integrating additional models as they become
available.

Currently onnx and llama.cpp backends are supported for running
models, additional backends (such as burn or candle) may be added as
they mature.

At the moment the best results are obtained with the
gte-Qwen2-instruct family of embedding models and Phi3. This
combination can index both source code and text documentation in the
same vector database for question answering or simple retrieval.

```rust
use ragtime::{llama, RagQaPhi3GteQwen27bInstruct};
use llama_cpp_2::llama_backend::LlamaBackend;
use anyhow::Result;
use std::{io::{stdout, Write}, sync::Arc};

let backend = Arc::new(LlamaBackend::init()?);
let mut qa = RagQaPhi3GteQwen27bInstruct::new(
    64,
    backend.clone(),
    llama::Args::default().with_model("gte-Qwen2-7B-instruct/ggml-model-q8_0.gguf"),
    backend,
    llama::Args::default().with_model("Phi-3-mini-128k-instruct/ggml-model-q8_0.gguf")
)?;

// add documents
qa.add_document("doc0", 256, 128)?;
qa.add_document("doc1", 256, 128)?;

// query
for tok in qa.ask("question about your docs", None)? {
    let tok = tok?;
    print!("{tok}");
    stdout().flush()?;
}
```

While ragtime does not directly use any async runtime, it's iterator
based return mechanism makes it very simple to stick in a background
thread and push tokens into an IPC mechanism such as an mpsc
channel. This allows it to be async runtime agnostic, and easier to
use in non async applications such as command line tools.

## Utils

As well as the core crate there are several simple command line
wrappers in the utils folder.

- qa: a question answering command line utility that can ingest
  documents, persist and load the index, and provide a question
  answering REPL
- gen: a driver to run a Phi3 session with no RAG support
- chunk: a utility to display how a document will be split into chunks

## Models

Both the Phi3 and all of the supported embedding model weights are
available on hugging face. In some cases you will need to convert them
to gguf format for llama.cpp using the python script included in the
llama.cpp repository. In the case of onnx models, many are available
for direct download from hugging face, otherwise you will have to
convert them from hugging face format.

## Deployment

Ragtime applications can be deployed with minimal dependencies if
using CPU or Vulkan acceleration. In the case of cuda, only the cuda
runtime is required.
