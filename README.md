# Astra VectorDB — A Vector Database in C++ from Scratch

> **Astra VectorDB** is a from-scratch vector database and local **RAG engine** written in **C++17**. It is designed to make the internals of modern vector-search systems explicit instead of hiding them behind a database SDK or managed vector-database API.

The project implements and compares three nearest-neighbor search strategies — **Brute Force, KD-Tree, and HNSW** — and exposes them through a lightweight **REST API** and a browser-based interface.

On top of the search engine, it adds a real document pipeline where documents are:

**Ingested → Chunked → Embedded → Indexed → Retrieved → Passed to an LLM**

using **Ollama** for embeddings and local generation.

The result is a small but complete system covering:

**Vector Indexing • Similarity Search • HNSW • KD-Tree • Benchmarking • Document Ingestion • Persistence • Hybrid Retrieval • REST APIs • RAG • Local LLM Question Answering**

---

## 🚀 Why I Built This

Most applications that use semantic search interact with a vector database through a high-level SDK. That hides the internal mechanisms responsible for vector indexing and retrieval.

I built Astra VectorDB to understand those internals by implementing the major components directly in **C++**.

The project focuses on understanding:

- **How vectors are represented and compared**
- **How exact and approximate nearest-neighbor search differ**
- **How a KD-Tree partitions a vector space**
- **How an HNSW multilayer graph is constructed and searched**
- **Why KD-Trees become less effective with high-dimensional embeddings**
- **How documents are converted into chunks and embeddings**
- **How semantic similarity can be combined with keyword matching**
- **How retrieved chunks are passed to an LLM in a RAG pipeline**
- **How indexing, persistence, concurrency, and HTTP APIs fit together**

---

# ✨ Key Features

### 🔎 Multiple Vector Search Algorithms

Astra VectorDB implements and compares:

- **Brute Force**
- **KD-Tree**
- **HNSW**

The same query can be executed using different indexing strategies and their latency can be compared.

### 📐 Multiple Distance Metrics

The system supports:

- **Cosine Distance**
- **Euclidean Distance**
- **Manhattan Distance**

### 📄 Document Ingestion

The system can ingest supported text and code files, extract their content, split them into chunks, generate embeddings, and index them.

### 🧠 HNSW-Based Semantic Search

Document embeddings are indexed using a custom **HNSW (Hierarchical Navigable Small World)** graph for approximate nearest-neighbor retrieval.

### 🤖 Local RAG Pipeline

The project implements a complete RAG pipeline using:

- **Ollama `nomic-embed-text`** for embeddings
- **Ollama `llama3.2:1b`** for generation

### 🔀 Hybrid Retrieval

Document ranking combines:

**72% semantic similarity + 28% keyword relevance**

This allows the system to use both semantic meaning and exact keyword overlap.

### ⚡ Benchmarking

The system compares:

**Brute Force vs KD-Tree vs HNSW**

under the same query conditions and reports measured search latency.

### 💾 Persistent Storage

Document chunks and embeddings are persisted using **JSONL**, allowing the in-memory indexes to be rebuilt on application startup.

### 👀 Real-Time File Watcher

A background file watcher scans the document directory every **3 seconds** and automatically processes:

- New files
- Modified files
- Deleted files

### 🌐 REST API

The entire vector database and RAG pipeline are exposed through lightweight JSON-based HTTP endpoints using **cpp-httplib**.

### 📊 Browser-Based Visualization

The project includes a browser UI with:

- Vector search
- Algorithm comparison
- Benchmarking
- PCA visualization
- Document management
- RAG question answering
- HNSW graph inspection

---

# 🏗️ High-Level Architecture

```text
                           Browser UI
                               │
                               │ HTTP / JSON
                               ▼
                     ┌─────────────────────┐
                     │   C++ HTTP Server  │
                     │     cpp-httplib     │
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
        Demo VectorDB       DocumentDB        FileWatcher
              │                 │                  │
       ┌──────┼──────┐          │                  │
       ▼      ▼      ▼          ▼                  ▼
    Brute   KD-Tree  HNSW      HNSW            documents/
    Force                      + BF fallback        │
                                  │                  │
                                  ▼                  │
                           Ollama embeddings ◄──────┘
                                  │
                                  ▼
                            Vector search
                                  │
                                  ▼
                         Retrieved text chunks
                                  │
                                  ▼
                             Ollama LLM
                                  │
                                  ▼
                               Answer
