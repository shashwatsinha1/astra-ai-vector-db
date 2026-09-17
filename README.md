# Astra VectorDB — A Vector Database in C++ from Scratch

**Astra VectorDB** is a from-scratch **vector database** and local **RAG engine** written in **C++17**. It is designed to make the internals of modern vector-search systems explicit instead of hiding them behind a database SDK or managed vector-database API.

The project implements and compares three nearest-neighbor search strategies — **Brute Force, KD-Tree, and HNSW** — and exposes them through a lightweight REST API and a browser-based interface. On top of the search engine, it adds a real document pipeline: documents are chunked, converted into embeddings using Ollama, indexed in an HNSW graph, and retrieved for a **Retrieval-Augmented Generation (RAG)** workflow.

The result is a small but complete system covering **vector indexing, similarity search, benchmarking, document ingestion, persistence, hybrid retrieval, REST APIs, and local LLM question answering**.

---

## Why I Built This

Most applications that use semantic search call a vector database through a high-level SDK. That hides the core ideas that actually make vector search work.

This project was built to understand those internals by implementing the main pieces directly in **C++**:

- **How vectors are represented and compared.**
- **How exact and approximate nearest-neighbor indexes differ.**
- **How an HNSW multilayer graph is constructed and searched.**
- **Why KD-Trees become less effective as dimensionality increases.**
- **How real documents are converted into chunks and embeddings.**
- **How semantic retrieval can be combined with keyword signals.**
- **How retrieved chunks are passed into an LLM to build a RAG pipeline.**
- **How indexing, persistence, concurrency, and HTTP APIs fit together in one backend.**

---

## What the System Does

### 1. Compare vector-search algorithms

The demo index contains **20 preloaded 16-dimensional vectors** representing four semantic categories:

- **Computer Science**
- **Mathematics**
- **Food**
- **Sports**

The same query can be searched using:

- **Brute Force** — exact scan of every vector.
- **KD-Tree** — exact spatial partitioning with subtree pruning.
- **HNSW** — approximate nearest-neighbor search using a multilayer graph.

The web UI can run all three algorithms for the same query and report their measured search latency.

### 2. Search with multiple distance metrics

The backend supports:

- **Cosine distance**
- **Euclidean distance**
- **Manhattan distance**

The search API lets the caller choose both the algorithm and the metric.

### 3. Index real documents

The application can ingest text and supported document files, split the content into overlapping chunks, generate embeddings with Ollama's `nomic-embed-text` model, and store those embeddings in a dedicated document index.

The file-watching pipeline can detect newly created, modified, or deleted files and update the document index automatically.

### 4. Ask questions using RAG

Once documents are indexed, a user can ask a natural-language question.

The system:

1. **Embeds the question.**
2. **Retrieves the most relevant document chunks.**
3. **Builds a prompt containing the retrieved chunks.**
4. **Sends that prompt to a local Ollama language model.**
5. **Returns the generated answer together with the retrieved context metadata.**

This makes the project more than a vector-search demo: the vector database becomes the retrieval layer of an end-to-end RAG system.

---

## High-Level Architecture

```text
                            Browser UI
                              │
                              │ HTTP / JSON
                              ▼
                     ┌─────────────────────┐
                     │  C++ HTTP Server    │
                     │    cpp-httplib      │
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
    Force                     + BF fallback        │
                              │                    │
                              ▼                    │
                       Ollama embeddings ◄────────┘
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
```

There are effectively two vector-database workloads:

### Demo VectorDB

```text
16D vectors
 ├── Brute Force
 ├── KD-Tree
 └── HNSW
```

### DocumentDB

```text
Real embeddings
 ├── HNSW for retrieval
 └── Brute Force fallback for very small collections
```

---

## End-to-End RAG Flow

```text
Document
   │
   ▼
Text extraction / normalization
   │
   ▼
Chunking (250 words with overlap)
   │
   ▼
Ollama: nomic-embed-text
   │
   ▼
Embedding vector (dimension determined from model output)
   │
   ▼
DocumentDB
   │
   ▼
HNSW index + persistent JSONL storage
   │
   │
   │        User Question
   │              │
   │              ▼
   │        Ollama embedding
   │              │
   │              ▼
   └────────► semantic retrieval
                  │
                  ▼
             relevant chunks
                  │
                  ▼
             prompt construction
                  │
                  ▼
          Ollama: llama3.2:1b
                  │
                  ▼
                Answer
```

---

## Retrieval Scoring

Document retrieval is not purely vector-based. The implementation combines **semantic similarity** with a lightweight **keyword signal**.

For a query containing text, the final score is computed as:

```text
finalScore = 0.72 × semanticScore + 0.28 × keywordScore

semanticScore = max(0, 1 - cosineDistance)
```

The keyword component tokenizes the query and checks how many query terms appear in the document **title, source, type, or text**.

This gives the retriever a simple hybrid behavior:

**Semantic similarity handles meaning while keyword overlap helps exact-term matching.**

---

# Core Components

## 1. Distance Metrics

The backend implements the three distance functions directly rather than using a numerical library.

### Euclidean Distance

```text
d(a,b) = sqrt( Σ (ai - bi)² )
```

### Manhattan Distance

```text
d(a,b) = Σ |ai - bi|
```

### Cosine Distance

```text
cosineDistance = 1 - (a · b) / (||a|| ||b||)
```

Cosine distance is used for document retrieval because the important signal is the **direction of the embedding vector** rather than only its magnitude.

---

## 2. Brute-Force Search

The baseline implementation stores vectors in a simple collection and computes the distance from the query to every vector.

For **N** vectors of dimensionality **D**, the dominant work is approximately:

```text
O(N × D)
```

### Why Keep It?

Brute Force is the reference implementation used to reason about **correctness** and to benchmark the optimized approaches.

It is also useful for **small collections** where building or traversing a more complicated index is unnecessary.

---

## 3. KD-Tree

The KD-Tree recursively partitions the vector space using one coordinate at each level.

The implementation cycles through dimensions:

```text
axis = depth % dimensions
```

During search, it first explores the subtree containing the query and only explores the opposite subtree when the current distance bound indicates that it could still contain a better result.

### Important Trade-Off

KD-Trees work well for **lower-dimensional spatial data**, but their pruning effectiveness degrades as dimensionality increases.

That is especially relevant for modern embeddings, which often have hundreds of dimensions.

The project therefore uses the KD-Tree primarily as a **contrast to HNSW** rather than as the main index for 768-dimensional document embeddings.

---

## 4. HNSW — Hierarchical Navigable Small World

**HNSW is the main Approximate Nearest Neighbor (ANN) structure in the project.**

The implementation builds a graph with multiple layers:

```text
Layer 3       • -------- •
               \        /
Layer 2       • --- • -- •
               | \     /
Layer 1       • - • - • - •
             / | \ | / | \
Layer 0     • • • • • • • • • •
```

Higher layers contain fewer nodes and provide **long-range navigation**.

Layer 0 contains the full graph and performs the detailed neighborhood exploration.

### Insertion

For each vector:

1. Randomly choose a maximum layer.
2. Start from the current entry point at the top layer.
3. Greedily move toward a closer node on higher layers.
4. At each applicable layer, perform a beam search with the construction parameter `ef_build = 200`.
5. Connect the new node to the selected nearest neighbors.
6. Add reciprocal connections and prune oversized neighbor lists.

The graph uses:

```text
M  = 16 connections on higher layers
M0 = 32 connections on layer 0
```

### Search

The query starts at the top layer and greedily descends toward the nearest neighborhood.

At layer 0, the implementation performs a wider candidate search using:

```text
ef = 50
```

for the demo index.

The objective is to avoid comparing the query against every vector while still returning high-quality nearest neighbors.

### Why HNSW Is Used for Documents

Real document embeddings are much higher-dimensional than the 16D demo vectors.

HNSW provides a **graph-based approximate search strategy** that remains useful when KD-Tree pruning becomes ineffective because of high-dimensional geometry.

---

## 5. DocumentDB

`DocumentDB` is a separate document-oriented index built on top of the **HNSW implementation**.

Each stored document chunk contains:

```text
id
title
text
source
type
embedding
```

Internally it maintains:

- An **ID → document map**
- A **source → chunk-ID index**
- An **HNSW index** for semantic retrieval
- A **Brute-Force index** used for very small document collections
- A **mutex** for thread-safe access
- A **JSONL persistence file** for document text and embeddings

### Small-Collection Fallback

When the number of stored chunks is below **10**, retrieval uses **Brute Force**.

Otherwise it uses **HNSW** to generate candidates.

This avoids unnecessary approximate-index traversal for tiny collections.

---

## 6. Real-Time File Ingestion

The backend includes a polling-based **FileWatcher** that scans the configured documents directory recursively every **3 seconds**.

For each supported file, it tracks a lightweight change stamp derived from the file's **modification time and size**.

When a file is created or changed:

```text
File changed
    ↓
Extract text
    ↓
Normalize content
    ↓
Split into chunks
    ↓
Generate embeddings
    ↓
Replace all chunks belonging to that source
    ↓
Persist updated vector data
```

When a watched file is deleted, its associated chunks are removed from the document index.

### Supported Input Types

The implementation supports common text/code formats such as:

```text
.txt .md .markdown .csv .tsv
.json .jsonl .yaml .yml .log
.cpp .cc .cxx .c .h .hpp .hh
.py .js .ts .tsx .jsx
.java .cs .go .rs
.php .rb .swift .kt .sql
.sh .ps1 .bat .toml .ini .cfg .env
.html .htm .xml
```

It also contains dedicated extraction paths for:

- **`.pdf`** using `pdftotext`
- **`.docx`** using the Windows/PowerShell ZIP/XML structure of Office documents

---

## 7. Text Chunking

Long documents are divided into **overlapping word-based chunks** before embedding.

The chunker is designed around:

```text
Chunk size  ≈ 250 words
Overlap     ≈ 30–40 words
```

The manual document-insert API uses a **30-word overlap**, while file-watcher ingestion uses a **40-word overlap**.

The overlap prevents information at a chunk boundary from being completely separated from the surrounding context.

---

## 8. Persistence

Document text and embeddings are persisted as **newline-delimited JSON** in:

```text
data/vectors.jsonl
```

At startup:

```text
vectors.jsonl
    ↓
load documents + embeddings
    ↓
rebuild DocumentDB indexes
    ↓
HNSW becomes searchable
```

The project therefore persists the data needed to reconstruct the index, rather than serializing the complete in-memory HNSW graph.

---

## 9. Thread Safety

The server can receive HTTP requests while the background file watcher is running.

To protect shared state, VectorDB, DocumentDB, and watcher state use:

```text
std::mutex
std::atomic
```

where appropriate.

The watcher runs on a **background thread** and can be started/stopped safely, while database operations lock the shared collections during mutation/search.

---

# Browser Interface

The UI is a single `index.html` served directly by the C++ HTTP server.

It provides three main tabs.

## Search

Use the demo 16D vector collection to:

- Enter a query vector
- Select **HNSW, KD-Tree, or Brute Force**
- Select **Cosine, Euclidean, or Manhattan** distance
- Run **K-NN search**
- Inspect matching vectors and distances
- Compare latency across all algorithms

The page also renders a **2D PCA projection** of the demo vectors so the semantic clusters can be inspected visually.

## Documents

The Documents tab allows users to:

- Insert text manually
- View stored chunks
- Delete chunks
- Monitor the live file-watcher state
- Trigger a folder rescan

## Ask AI

The Ask AI tab exposes the RAG workflow:

```text
Question
   ↓
Embedding
   ↓
Top-k retrieval
   ↓
Context construction
   ↓
LLM generation
   ↓
Answer + retrieved contexts
```

The UI also exposes the retrieved context information so it is possible to see **which document chunks were used for the answer**.

---

# REST API

The server is implemented with the single-header **cpp-httplib** library and exposes JSON-based HTTP endpoints.

## Demo Vector API

| **Method** | **Endpoint** | **Purpose** |
|---|---|---|
| `GET` | `/search` | K-NN search over demo vectors |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Remove a demo vector |
| `GET` | `/items` | List demo vectors |
| `GET` | `/benchmark` | Compare Brute Force, KD-Tree, and HNSW latency |
| `GET` | `/hnsw-info` | Inspect HNSW nodes, edges, and layer statistics |
| `GET` | `/stats` | Return demo-vector statistics |

### Example

```http
GET /search?v=0.9,0.8,...&k=5&metric=cosine&algo=hnsw
```

## Document / RAG API

| **Method** | **Endpoint** | **Purpose** |
|---|---|---|
| `POST` | `/doc/insert` | Chunk, embed, and store a document |
| `GET` | `/doc/list` | List stored document chunks |
| `DELETE` | `/doc/delete/:id` | Delete a document chunk |
| `POST` | `/doc/search` | Embed a query and retrieve relevant chunks |
| `POST` | `/doc/ask` | Full RAG: retrieve + generate |
| `GET` | `/status` | Ollama, model, document, and watcher status |
| `GET` | `/watch/status` | Show watched files and ingestion status |
| `POST` | `/watch/rescan` | Force a watch-folder rescan |

### Example RAG Request

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

The `/doc/ask` response contains:

- **Generated answer**
- **Generation model**
- **Retrieved context chunks**
- **Context titles and sources**
- **Similarity distances**
- **Current document count**

---

# Benchmarking

One of the goals of the project is not only to implement multiple indexes, but to measure them under the same query conditions.

The `/benchmark` endpoint runs:

- **Brute Force K-NN**
- **KD-Tree K-NN**
- **HNSW K-NN**

for the same query, `k`, and distance metric and returns the elapsed time in **microseconds** for each search.

The UI exposes this as **COMPARE ALL ALGOS**.

This makes the project useful for discussing the practical trade-off between:

```text
Exact search
   ↓
Simple, predictable, expensive as N grows

Approximate search
   ↓
More complex index, faster candidate retrieval at scale
```

The benchmark is intended as an **engineering comparison tool**, not as a claim that one algorithm will always be faster for every dataset, dimensionality, or hardware configuration.

---

# Design Decisions

## Why implement all three indexes?

Because each one demonstrates a different point in the design space:

- **Brute Force** → simplest exact baseline
- **KD-Tree** → spatial partitioning + pruning
- **HNSW** → graph-based approximate ANN

## Why separate VectorDB and DocumentDB?

The demo vectors and real document embeddings have different purposes.

**VectorDB** is fixed at **16 dimensions** so the algorithms can be visualized and compared easily.

**DocumentDB** determines the embedding dimension at runtime from Ollama and keeps a separate HNSW index for the real embedding workload.

## Why keep a Brute-Force fallback in DocumentDB?

For very small collections, a full ANN traversal adds complexity without much benefit.

The fallback also gives the implementation a simple **exact retrieval path**.

## Why persist JSONL instead of the HNSW graph?

The project stores document content and embeddings in a simple, inspectable format.

The in-memory HNSW structure can then be rebuilt when the application starts.

This keeps the persistence layer easy to understand while leaving the indexing structure focused on runtime search.

## Why integrate Ollama instead of a hosted AI API?

The project is intended to demonstrate the full retrieval pipeline locally:

```text
Text → embedding → vector index → retrieval → LLM
```

Ollama provides both the **embedding model** and the **generation model** without requiring a hosted inference API.

---

# Technology Stack

| **Layer** | **Technology** |
|---|---|
| **Language** | C++17 |
| **HTTP server** | cpp-httplib |
| **ANN index** | Custom HNSW implementation |
| **Exact indexes** | Custom Brute Force + KD-Tree |
| **Embeddings** | Ollama `nomic-embed-text` |
| **LLM** | Ollama `llama3.2:1b` |
| **Persistence** | JSONL |
| **Frontend** | HTML + CSS + JavaScript |
| **Visualization** | HTML Canvas + browser-side PCA |
| **Build** | g++ / CMake |
| **Containerization** | Docker |

---

# Project Structure

```text
Your_Own_AI/
├── main.cpp             # C++ backend and all core database/index logic
├── httplib.h            # Single-header HTTP server/client library
├── index.html           # Browser UI, PCA visualization, benchmark UI, RAG UI
├── CMakeLists.txt       # CMake build configuration
├── Dockerfile            # Multi-stage Docker build
├── render.yaml           # Render deployment configuration
├── documents/            # Watched document directory
├── data/
│   └── vectors.jsonl     # Persisted document chunks + embeddings
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

---

# Build and Run Locally

## Prerequisites

- **C++17 compiler**
- **CMake** (optional if building directly with g++)
- **Git**
- **Ollama** for embeddings and RAG

## Pull the Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

## Build with g++

### Windows / MinGW

```bash
g++ -std=c++17 -O2 main.cpp -o db -lws2_32
```

### Or build with CMake

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Run

```bash
ollama serve
./db
```

Then open:

```text
http://localhost:8080
```

When Ollama is available, the server reports the configured embedding and generation models at startup.

---

# Runtime Configuration

The server supports environment variables so local and containerized deployments can use the same binary.

| **Variable** | **Default** | **Purpose** |
|---|---|---|
| `VECTORDB_HOST` | `0.0.0.0` | HTTP bind address |
| `PORT` | `8080` | Primary HTTP port, useful for platforms such as Render |
| `VECTORDB_PORT` | `8080` | Fallback HTTP port when `PORT` is not set |
| `VECTORDB_DATA_PATH` | `data/vectors.jsonl` | Persistence file |
| `VECTORDB_WATCH_DIR` | `documents` | Directory monitored for automatic ingestion |
| `OLLAMA_HOST` | `127.0.0.1` | Ollama host |
| `OLLAMA_PORT` | `11434` | Ollama port |

---

# Docker / Deployment Support

The repository includes:

- A **multi-stage Dockerfile** that compiles the C++ backend and creates a smaller runtime image.
- A `render.yaml` configuration for running the service as a Render Docker web service.
- Environment-variable configuration for the server port, persistence path, watched directory, and external Ollama host.

> **Ollama itself is not bundled into the application image.** The C++ server treats Ollama as an external/local inference dependency.

### For local development

```text
C++ server  →  127.0.0.1:11434  →  Ollama
```

### For a cloud deployment

```text
C++ server  →  external Ollama host  →  embedding / generation models
```

The vector database can start even when Ollama is unavailable; the `/status` endpoint reports whether the AI dependency is currently reachable.

---

# Interesting Implementation Details

## HNSW Configuration

```text
M         = 16
M0        = 32
ef_build  = 200
search ef = 50 (demo)
```

Document retrieval uses a larger candidate exploration setting during HNSW search and then applies filtering/scoring before returning the final top-k results.

## Search Candidate Strategy

For documents, the system requests an expanded candidate set before final ranking:

```text
candidateLimit = clamp(max(50, k × 10), k, 250)
```

The final ranking can then incorporate the semantic and keyword components.

## HNSW Graph Inspection

The `/hnsw-info` endpoint exposes graph-level information such as:

- **Number of nodes**
- **Top layer**
- **Nodes per layer**
- **Edges per layer**
- **Node metadata**
- **Graph edges**

This makes the internal data structure observable instead of treating HNSW as a black box.

---

# Limitations and Future Improvements

This project intentionally focuses on understanding the core vector-search pipeline rather than reproducing every feature of a production vector database.

### Natural Next Improvements Include

- **Persistent serialization of the HNSW graph itself** instead of rebuilding it from stored vectors.
- **Batch embedding requests** for higher ingestion throughput.
- **Better HNSW neighbor selection / graph optimization.**
- More formal **recall@k and latency benchmarking** on larger datasets.
- More scalable storage instead of a single JSONL persistence file.
- **Authentication and authorization** for the REST API.
- Production-grade **background job management** for ingestion.
- A dedicated **file-upload endpoint** for browser-based document uploads.
- More advanced metadata filtering and hybrid ranking.
- Distributed indexing and horizontal scaling.

These would move the implementation closer to a production-oriented vector-search service while preserving the same core architecture.

---

# What This Project Demonstrates

This project brings together several systems concepts in one implementation:

```text
Data Structures & Algorithms
        ↓
Vector representations
        ↓
Nearest-neighbor indexing
        ↓
HNSW graph traversal
        ↓
REST API design
        ↓
Concurrent background ingestion
        ↓
Persistence
        ↓
Embedding generation
        ↓
Semantic + keyword retrieval
        ↓
RAG + local LLM generation
        ↓
Interactive visualization + benchmarking
```

The main learning objective was to understand what happens between a user's natural-language query and the final LLM answer, including the vector representation, index traversal, retrieval, and context construction steps in between.

---

# License

**MIT License.**
