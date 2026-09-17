# VectorDB - Building a Vector Database in C++ from Scratch

This project is a fully functional vector database developed in C++ with a simple web interface. The goal of this project is to understand how modern vector databases like Pinecone or Weaviate work internally by implementing everything from scratch.

The system includes multiple search algorithms such as HNSW, KD-Tree, and brute-force search. It also integrates a Retrieval-Augmented Generation (RAG) pipeline using a local language model through Ollama.

## Real-Time Vector Database Upgrade

This version is no longer only a manual demo. The backend runs a live ingestion service that watches a local `documents/` folder for `.txt`, `.md`, and `.markdown` files. When a file is created, edited, or deleted, the server automatically:

1. Reads the file.
2. Splits it into overlapping chunks.
3. Generates real embeddings with Ollama `nomic-embed-text`.
4. Inserts or replaces those chunks in the document vector index.
5. Makes them searchable immediately through the RAG API and web UI.

This gives the project a real-world shape: a local vector search service with automatic ingestion, approximate nearest-neighbor retrieval, REST endpoints, a live status UI, and local LLM question answering.

### Runtime Configuration

You can configure the service with environment variables:

| Variable | Default | Description |
|---|---|---|
| `VECTORDB_HOST` | `0.0.0.0` | Bind address. Defaults to all interfaces. |
| `PORT` | `8080` | HTTP server port (standard Render port env var). |
| `VECTORDB_PORT` | `8080` | Fallback HTTP server port if `PORT` is unset. |
| `VECTORDB_DATA_PATH` | `data/vectors.jsonl` | Persistent storage path for ingested documents and vectors. |
| `VECTORDB_WATCH_DIR` | `documents` | Folder scanned for live document ingestion. |
| `OLLAMA_HOST` | `127.0.0.1` | Hostname or IP of the external/local Ollama server. |
| `OLLAMA_PORT` | `11434` | Port of the external/local Ollama server. |

### Real-Time Workflow (Local Development)

```powershell
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2

g++ -std=c++17 -O2 main.cpp -o vectordb -lws2_32
./vectordb
```

Then open `http://127.0.0.1:8080`, drop `.txt` or `.md` files into `documents/`, and watch the Live File Ingestion panel update. Ask questions in the Ask AI tab; the answer is generated from the indexed file chunks.

### Deploying to Render (Docker Web Service)

You can deploy this service reliably on [Render](https://render.com) using Render's official **Docker** environment and a persistent disk.

#### Important Note on Ollama
> **Ollama is NOT bundled** in the Render service. The embedding and generation models (`nomic-embed-text` and `llama3.2:1b`) require significant RAM and compute, making them too heavy for standard free or small Render instances. Ollama must run on a separate host or VPS (with GPU/sufficient CPU) reachable over the network at `OLLAMA_HOST` and `OLLAMA_PORT`. If Ollama is unreachable, VectorDB starts up gracefully with `Ollama: OFFLINE` status.

#### Render Configuration

- **Environment / Runtime**: `Docker` (uses the included multi-stage [`Dockerfile`](file:///d:/Your_Own_AI/Dockerfile))
- **Persistent Disk**:
  - Name: `vectordb-data`
  - Mount Path: `/data`
  - Size: 1 GB (or more)
  - *Note: Render persistent disks require a Starter plan or higher.*
- **Environment Variables**:
  - `OLLAMA_HOST`: `<external-ollama-host-ip-or-domain>`
  - `OLLAMA_PORT`: `11434`
  - `VECTORDB_DATA_PATH`: `/data/vectors.jsonl`
  - `VECTORDB_WATCH_DIR`: `/data/documents`

You can deploy automatically using the included `render.yaml` Blueprint file via Render's **Blueprints** dashboard.

### CMake Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

---

## What This Project Does

| Feature | Description |
|---|---|
| **Multiple Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all algorithms and compare speed |
| **Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (C++)                   ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

You need **3 things** installed on your Windows laptop:

1. **MSYS2** (gives you g++ compiler)
2. **Git**
3. **Ollama** (runs the local AI models)

---

## Step-by-Step Setup (Windows)

### Step 1 — Install MSYS2 (C++ Compiler)

1. Go to **https://www.msys2.org** and download the installer
2. Run the installer, keep default path (`C:\msys64`)
3. After install, open **MSYS2 UCRT64** from Start Menu (the orange icon)
4. Run these commands inside the MSYS2 terminal:

```bash
pacman -Syu
```
*(Close and reopen the terminal if it asks you to)*

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```

5. Add g++ to your Windows PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click **Advanced** → **Environment Variables**
   - Under **System variables**, find **Path**, click **Edit**
   - Click **New** and add: `C:\msys64\ucrt64\bin`
   - Click OK on all windows
   - **Open a new PowerShell** and verify:
   ```
   g++ --version
   ```
   You should see something like `g++ (GCC) 15.x.x`

---

### Step 2 — Install Git

1. Go to **https://git-scm.com/download/win** and download Git for Windows
2. Run the installer with default settings
3. Verify in PowerShell:
```
git --version
```

---

### Step 3 — Install Ollama (Local AI Models)

1. Go to **https://ollama.com** and click **Download for Windows**
2. Run the installer
3. Ollama starts automatically in the system tray
4. Open **PowerShell** and pull the two required models:

```powershell
ollama pull nomic-embed-text
```
*(~274 MB — this is the embedding model)*

```powershell
ollama pull llama3.2
```
*(~2 GB — this is the language model)*

5. Verify Ollama is running:
```powershell
ollama list
```
You should see both models listed.

> **Minimum specs for Ollama:** 8GB RAM recommended. The models will use ~3GB total.

---

### Step 4 — Clone the Repository

Open **PowerShell** and run:

```powershell
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
```

*(Replace `YOUR_USERNAME` with the actual GitHub username)*

---

### Step 5 — Compile the C++ Server

Inside the `VectorDB` folder, run:

```powershell
g++ -std=c++17 -O2 main.cpp -o db -lws2_32
```

This produces `db.exe`. It takes about 10–20 seconds.

> **Troubleshooting:**
> - `g++: command not found` → MSYS2 not in PATH, redo Step 1 point 5
> - `undefined reference to WSA...` → missing `-lws2_32` flag, add it
> - Takes too long? Remove `-O2` for faster (but slower executable) compile

---

### Step 6 — Run Everything

**Terminal 1** — Start Ollama (if not already running):
```powershell
ollama serve
```
*(If Ollama is already in the system tray, skip this)*

**Terminal 2** — Start the VectorDB server:
```powershell
./db
```

You should see:
```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

**Open your browser** and go to:
```
http://localhost:8080
```

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

**The scatter plot** shows all 20 vectors projected to 2D using PCA. Notice how the 4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)

This uses Ollama to generate **real 768-dimensional embeddings** from any text.

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Make sure you have inserted some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:
```
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2 → generates an answer based only on your documents
```

The answer streams in with a typewriter effect. Click the **context chips** to see exactly which chunks the AI used.

### Tab 4: Benchmark (Scaling Suite)

A rigorous engineering suite for evaluating vector search performance across scaling dataset sizes ($N = 20$ to $50,000$):
- **Dataset Size Selection**: Toggle benchmark sizes ($20, 100, 500, 1000, 5000, 10000$).
- **Statistical Rigor**: Configurable repetitions (e.g. 100 runs) and warm-up cycles (10 runs).
- **Interactive Scaling Chart**: Visualizes Dataset Size ($N$) vs Median Search Latency ($\mu$s) across Brute Force, KD-Tree, and HNSW.
- **Detailed Metrics Table**: Displays Index Build Time, Mean, Median, P95, Min, Max, and ground-truth $\text{Recall}@K$.

---

## REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector & Benchmark Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Legacy single-vector 20-item benchmark |
| `GET` | `/benchmark?sizes=20,100,500,1000&repetitions=100&k=5&metric=cosine` | Multi-size scaling benchmark |
| `POST` | `/benchmark` | Configurable multi-size benchmark with JSON payload |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Example: Search via curl

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Run Scaling Benchmark via curl (GET)

```powershell
curl "http://localhost:8080/benchmark?sizes=20,100,500,1000,5000,10000&repetitions=100&k=5&metric=cosine"
```

### Example: Run Scaling Benchmark via curl (POST)

```powershell
curl -X POST http://localhost:8080/benchmark `
  -H "Content-Type: application/json" `
  -d '{"sizes":[20,100,500,1000,5000],"repetitions":100,"k":5,"metric":"cosine","m":16,"ef_build":200,"ef_search":50}'
```

### Example: Ask a question via curl

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Engineering Benchmark & Algorithm Scaling Analysis

### 1. The 20-Vector Anomaly: Why HNSW Can Be Slower on Small N

On the default 20-vector demo dataset (20 vectors × 16 dimensions), running a benchmark often reveals that **Brute Force (exhaustive scan) is faster than HNSW**.

This is not an implementation bug—it is a textbook demonstration of algorithmic constant factors versus asymptotic complexity on modern hardware:

- **Brute Force ($O(N \cdot d)$)**:
  At $N=20, d=16$, Brute Force performs exactly $20 \times 16 = 320$ float multiply-accumulates. The entire dataset occupies $20 \times 16 \times 4\text{ bytes} \approx 1.28\text{ KB}$, fitting entirely inside the CPU's fastest **L1 data cache** (32–48 KB). The CPU executes this contiguous linear array scan using vectorized SIMD instructions (AVX/SSE) with near-zero cache misses, zero pointer indirection, and branch prediction accuracy approaching 100%.

- **HNSW ($O(\log N)$)**:
  Although HNSW is asymptotically logarithmic, it incurs substantial constant structural overhead per search:
  1. **Dynamic Priority Queues**: Maintains candidate and result min/max heaps with dynamic balancing.
  2. **Visited Set Tracking**: Lookups and inserts into a visited hash map (`std::unordered_map` or hash table) to avoid cycles.
  3. **Pointer Chasing**: Navigates graph nodes via adjacency list lookups across different memory addresses.
  4. **Multi-layer Routing**: Greedily traverses from the sparse top layer down to Layer 0.

Mathematically, query time is modeled as:
$$T_{\text{BF}}(N) = C_{\text{stream}} \cdot N \cdot d$$
$$T_{\text{HNSW}}(N) = C_{\text{graph\_overhead}} + C_{\text{search}} \cdot \log(N)$$

When $N = 20$, $C_{\text{graph\_overhead}} \gg C_{\text{stream}} \cdot 20 \cdot 16$. The overhead of initializing priority queues and visited sets dwarfs the cost of computing 20 vector dot products.

### 2. Empirical Crossover Point & Scaling Behavior

As dataset size $N$ increases ($20 \to 100 \to 500 \to 1,000 \to 5,000 \to 10,000 \dots$):
- **Brute Force latency scales strictly linearly ($O(N)$)**: At $N=10,000$, brute force must evaluate 160,000 float multiplications and sort 10,000 distance candidates.
- **HNSW latency scales logarithmically ($O(\log N)$)**: Upper highway layers quickly prune 99%+ of the search space in a handful of hops, bounding search time by the constant parameter `ef_search`.

**Empirical Crossover**:
Around $N \approx 500 \dots 1,000$ vectors, the linear cost of brute force overtakes the constant overhead of HNSW. At $N=10,000$, HNSW is **>6x faster** than Brute Force, and at $N=50,000$, HNSW is orders of magnitude faster.

### 3. Fair Benchmarking Methodology

To ensure scientifically valid and defensible comparisons:
1. **Identical Dataset**: All algorithms benchmarked on the exact same synthetic vectors generated deterministically with a fixed PRNG seed (`std::mt19937`).
2. **Identical Query Vector**: The exact same query vector $q$ is issued to Brute Force, KD-Tree, and HNSW.
3. **Identical Parameters**: Same $k$, same distance metric (`cosine`, `euclidean`, `manhattan`), same repetition count.
4. **Isolated Search Timing**: Dataset generation, index construction, memory allocations, network I/O, and JSON serialization are strictly excluded from search latency measurements.

### 4. Warm-Up Runs and Repeated Measurements

- **Why Warm-Up Runs are Essential (Default: 10 runs)**:
  Cold cache queries incur artificial latency due to instruction cache misses, OS page faulting, TLB translation caching, and dynamic branch predictor training. Executing unmeasured warm-up queries ensures measurements reflect steady-state operational performance.
- **Why Repeated Runs are Required (Default: 100 runs)**:
  Single-shot timings are prone to OS scheduler preemption, background interrupts, and CPU frequency scaling (Intel Turbo Boost / AMD Precision Boost). Running 100 iterations provides a statistically meaningful distribution.

### 5. Statistical Percentiles: Why Median and P95 Matter

- **Arithmetic Mean**: Susceptible to extreme outliers caused by OS context switching.
- **Median ($P_{50}$)**: Represents true typical latency under steady-state execution.
- **95th Percentile ($P_{95}$)**: Crucial for production SLAs; reveals tail latency and worst-case performance under load.
- **Minimum & Maximum**: Bounds the full observed distribution.

### 6. Ground-Truth & Recall@K for Approximate Search

Unlike exact search algorithms, HNSW is an **Approximate Nearest Neighbor (ANN)** algorithm. Benchmarking speed without accuracy is meaningless:
- **Brute Force acts as the exact ground-truth reference** ($\text{Recall} = 1.0$).
- HNSW top-$k$ results are compared against Brute Force top-$k$:
  $$\text{Recall}@k = \frac{|\text{Results}_{\text{HNSW}} \cap \text{Results}_{\text{BruteForce}}|}{k}$$
- Measuring $\text{Recall}@k$ validates that speedups do not sacrifice retrieval quality (e.g. maintaining $>98-100\%$ recall with default $M=16, ef_{\text{search}}=50$).

### 7. Decoupling Index Construction Time from Search Latency

- **Index Build Time**: An offline batch operation amortized across millions of queries.
  - Brute Force: $\sim 0\text{ ms}$ (no structural index).
  - KD-Tree: Moderate recursive build cost.
  - HNSW: Higher construction cost due to multi-layer greedy search and bidirectional edge rewiring ($ef_{\text{build}}=200$).
- **Search Latency**: The real-time, SLA-critical online query path.
Reporting build time separately ensures engineering visibility without penalizing query latency.

---

## Interview Cheat Sheet: Defending VectorDB

### Q: "Why was HNSW slower than brute force on your original 20-vector benchmark?"
> "At N=20 and 16 dimensions, the entire dataset is only 1.28 KB, which fits entirely within the L1 CPU cache. Brute force simply does a contiguous linear scan utilizing SIMD vectorization with zero branch mispredictions and zero pointer indirection.
> 
> HNSW, while asymptotically logarithmic $O(\log N)$, has a non-trivial constant factor: it must initialize a priority queue, track visited nodes in a hash map, and traverse graph adjacency pointers. When $N$ is small, the graph traversal overhead $C_{\text{graph}}$ is significantly larger than the brute force scan $C_{\text{stream}} \cdot N \cdot d$. As $N$ scales past the empirical crossover point ($N \approx 500-1,000$), brute force scales linearly while HNSW scales logarithmically, making HNSW over 6x faster at $N=10,000$."

### Q: "How did you design a fair and statistically sound benchmark?"
> "A fair benchmark requires three core pillars:
> 1. **Parity**: All three algorithms receive the exact same dataset, identical query vector, identical metric, and identical $k$.
> 2. **Purity of Timed Path**: Index construction, memory allocations, and network/JSON overhead are completely excluded. Only the algorithmic query traversal is timed using `std::chrono::steady_clock`.
> 3. **Statistical Rigor & ANN Quality**: We run 10 warm-up queries to warm the CPU cache and branch predictor, followed by 100 repeated measured iterations to compute median and P95 tail latencies. Furthermore, because HNSW is approximate, we use Brute Force as ground truth to calculate $\text{Recall}@k$, ensuring that speed gains do not compromise search accuracy."

---

## Project Structure

```
VectorDB/
├── main.cpp        ← C++ backend (HNSW, KD-Tree, BruteForce, Benchmark Suite, REST API, RAG)
├── httplib.h       ← Single-header HTTP server library (cpp-httplib)
├── index.html      ← Frontend (PCA scatter plot, chat UI, scaling benchmark suite)
└── README.md       ← This file
```

### Architecture (main.cpp)

```
BruteForce          O(N·d)      Exact, baseline
KDTree              O(log N)    Exact, axis-aligned partitioning
HNSW                O(log N)    Approximate, multilayer small-world graph

VectorDB            Unified interface over all 3 (16D demo vectors)
DocumentDB          HNSW-only index for real Ollama embeddings (768D)
OllamaClient        HTTP client → /api/embeddings + /api/generate
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer. Layer 0 has all nodes with many connections; higher layers have fewer nodes (exponentially fewer) with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from your assigned max down to 0, run a beam search (ef_construction=200) and connect to the M nearest neighbors bidirectionally.

**Search:** Same greedy descent from top layer. At layer 0, expand to ef nearest candidates using a priority queue.

**Why it's fast:** The upper layers act like a highway — you quickly get to the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree can't beat the current best — the "ball within hyperslab" check.

**Weakness:** Degrades with high dimensions (curse of dimensionality). Works well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all the space is near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| Embedding takes forever | Ollama is downloading the model on first use, wait 2 min |
| `g++: command not found` | Add `C:\msys64\ucrt64\bin` to Windows PATH |
| Port 8080 already in use | Kill the process: `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| LLM answer is slow | Normal — llama3.2 takes 10–30s on a laptop CPU. Use llama3.2:1b for faster answers |

### Use a Smaller/Faster LLM

If llama3.2 is too slow on your laptop, switch to the 1B model:

```powershell
ollama pull llama3.2:1b
```

Then edit [main.cpp](main.cpp) line where `genModel` is set:
```cpp
std::string genModel = "llama3.2:1b";   // change this
```
Recompile and restart.

---

## License

MIT — use this however you want.
