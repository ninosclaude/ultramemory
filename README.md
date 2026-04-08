# ultramemory

Portable knowledge graph builder for local documents. Single Go binary, no cloud, no setup.

## Dependencies

- **Go** (build time)
- One runtime profile:
  - **Default build**: local Ollama with `gemma3:4b` and `mxbai-embed-large`
  - **`mistral` build tag**: `MISTRAL_API_KEY` plus Mistral API access for extraction, embeddings and OCR fallback

### Default build (`go build .`)

```bash
ollama pull gemma3:4b          # entity/edge extraction
ollama pull mxbai-embed-large  # embeddings (1024-dim)
```

By default Ollama keeps models loaded in VRAM indefinitely. Set a keep-alive timeout to free GPU memory when idle:

```bash
OLLAMA_KEEP_ALIVE=30m ollama serve
```

### Optional: PDF support

| Tool | Purpose | Install |
|------|---------|---------|
| `pdftotext` (poppler) | Extract text from digital PDFs | `brew install poppler` |
| `tesseract` | OCR for scanned PDFs (**recommended**) | `brew install tesseract` |

Without these tools, PDFs are skipped. Without `tesseract`, scanned PDFs fall back to the active model OCR backend — this works but produces lower accuracy output and logs a warning:

```
using model OCR fallback — accuracy is lower than Tesseract; install tesseract for better results
```

## Install

```bash
go install github.com/sharpner/ultramemory@latest
```

Or build from source:

```bash
# Default build: pure Ollama binary
go build -o ultramemory .

# Pure Mistral binary
go build -tags mistral -o ultramemory .
```

## Supported formats

| Format | Extensions | Method |
|--------|-----------|--------|
| Text / Markup | `.txt`, `.md`, `.html`, `.css` | direct UTF-8 read |
| Code | `.go`, `.ts`, `.tsx`, `.js`, `.jsx`, `.py`, `.rs`, `.sh` | direct UTF-8 read |
| Data / Config | `.json`, `.yaml`, `.yml`, `.toml`, `.env`, `.sql`, `.graphql`, `.proto` | direct UTF-8 read |
| LaTeX / BibTeX | `.tex`, `.bib` | direct UTF-8 read |
| PDF (digital) | `.pdf` | `pdftotext` (poppler) |
| PDF (scanned) | `.pdf` | `tesseract` OCR, active model OCR fallback |

## Usage

```bash
# Ingest a directory and start the worker
ultramemory run ./my-docs

# Or separately:
ultramemory ingest ./my-docs   # enqueue files into SQLite job queue
ultramemory worker             # process queue (runs until Ctrl+C)

# Ingest with custom source label (e.g. arXiv URL)
ultramemory ingest -source "https://arxiv.org/abs/2511.01815" ./paper/
ultramemory run -source "https://arxiv.org/abs/2511.01815" ./paper/

# Pure Mistral build
go build -tags mistral -o ultramemory .
MISTRAL_API_KEY=sk-... ./ultramemory run ./my-docs

# Search the graph
ultramemory search "Alice Schmidt TechCorp"

# Search with token budget (LLM-friendly output)
ultramemory search -max-tokens 200 "Alice Schmidt TechCorp"
ultramemory search -format json -max-tokens 500 "Alice Schmidt TechCorp"

# Build an experimental keyed episode index from existing episode embeddings
MEMORY_PERSONAL_KEY=secret ultramemory personal-index

# Search episodes in the keyed semantic space
MEMORY_PERSONAL_KEY=secret ultramemory personal-search "documents about cats"

# Cluster episodes in the authorized keyed space
MEMORY_PERSONAL_KEY=secret ultramemory personal-cluster

# List detected communities
ultramemory communities
ultramemory communities -format json

# Status
ultramemory status
```

## Environment

### Shared

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_DB` | `memory-local.db` | SQLite database path |
| `MEMORY_GROUP` | `default` | Namespace for graph isolation |
| `MEMORY_PERSONAL_KEY` | unset | Key for experimental personal episode search/clustering |
| `MEMORY_RESOLVE_THRESHOLD` | `0.92` | Cosine similarity threshold for entity deduplication (0–1) |
| `MEMORY_LLM_PARALLEL` | build-dependent | Concurrent extraction calls |

### Default Ollama build

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_OLLAMA` | `http://localhost:11434` | Ollama base URL |
| `MEMORY_MODEL` | `gemma3:4b` | Entity/edge extraction model |
| `MEMORY_EMBED_MODEL` | `mxbai-embed-large` | Embedding model |

### `mistral` build tag

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_MODEL` | `mistral-3b-latest` | Entity/edge extraction model |
| `MEMORY_EMBED_MODEL` | `mistral-embed` | Embedding model |
| `MISTRAL_API_KEY` | required | Mistral API key |

## Build profiles

- `go build .`
  - Pure Ollama binary
  - Default DB format: `ollama-v1`
  - Uses `gemma3:4b` + `mxbai-embed-large`
- `go build -tags mistral .`
  - Pure Mistral binary
  - Default DB format: `mistral-v1`
  - Uses `mistral-3b-latest` + `mistral-embed`

The database format is stamped on first open. An `ollama-v1` binary refuses a `mistral-v1` database, and vice versa:

```bash
error: open db: wrong db format: expected mistral-v1, found ollama-v1
```

## JSON API

Both `search` and `status` support `-format json` for machine-readable output.

### search -format json

Each result is emitted as a newline-delimited JSON object (one per line).
Use `-max-tokens N` to cap output size — the stream stops before the budget is exceeded:

```bash
ultramemory search -format json "Alice Schmidt TechCorp"
ultramemory search -format json -max-tokens 500 "Alice Schmidt TechCorp"
```

```json
{"rank":1,"type":"entity","title":"Alice Schmidt","body":"person","score":0.0312}
{"rank":2,"type":"edge","title":"WORKS_AT","body":"Alice Schmidt works at TechCorp Berlin","score":0.0294}
{"rank":3,"type":"entity","title":"TechCorp Berlin","body":"organisation","score":0.0278}
```

| Field   | Type    | Description                                      |
|---------|---------|--------------------------------------------------|
| `rank`  | int     | 1-based result rank                              |
| `type`  | string  | `"entity"` or `"edge"`                           |
| `title` | string  | Entity name or edge relation type                |
| `body`  | string  | Entity type (for entities) or fact (for edges)   |
| `score` | float64 | RRF fusion score (higher = more relevant)        |

### status -format json

```bash
ultramemory status -format json
```

```json
{
  "graph": {"episodes": 42, "entities": 187, "edges": 312},
  "queue": {"pending": 0, "processing": 0, "done": 156, "failed": 2}
}
```

| Field                | Type | Description                              |
|----------------------|------|------------------------------------------|
| `graph.episodes`     | int  | Number of ingested text chunks           |
| `graph.entities`     | int  | Named entities extracted                 |
| `graph.edges`        | int  | Relationships extracted                  |
| `queue.pending`      | int  | Chunks waiting for extraction            |
| `queue.processing`   | int  | Chunks currently being processed         |
| `queue.done`         | int  | Successfully processed chunks            |
| `queue.failed`       | int  | Chunks that failed extraction            |

## How it works

1. **Ingest**: Files are chunked (1500 chars, 150 overlap) and enqueued as SQLite jobs
2. **Extract**: the active build profile extracts named entities and relationships from each chunk
3. **Embed**: the active embedding backend generates vectors for semantic search
4. **Search**: Hybrid FTS5 + cosine similarity fused via RRF, then extended by MAGMA graph traversal

### Experimental personal secure index

The repository now also includes an experimental keyed episode layer for personal memory workflows:

- `personal-index` builds a `bregman-v1` secure episode index from existing episode embeddings
- `personal-search` scores queries directly in the keyed semantic space
- `personal-cluster` builds communities over keyed episode similarity rather than the extracted entity graph

This path is intentionally separate from the graph pipeline. The existing graph search and community detection continue to work as before; the personal secure index is an opt-in layer for "search/clustering only with the right key" experiments.

### Search pipeline detail

```
Query
 ├─ FTS5 (full-text search on entities + edges)
 ├─ Vector search (cosine similarity, threshold 0.3)
 ├─ RRF fusion (Reciprocal Rank Fusion, k=60)
 └─ MAGMA graph traversal (arxiv.org/abs/2601.03236)
      ├─ Top-5 RRF entities become seed nodes
      ├─ Beam search (width=10, depth=5, budget=200 nodes)
      ├─ Transition: S = exp(λ₁·φ(edge_type, intent) + λ₂·cos_sim)
      │    ├─ Intent classified from query keywords (Why/When/Entity/What)
      │    └─ Edge types routed: Causal/Temporal/Relational/Attributive
      └─ Graph-traversal results appended after direct RRF matches
```

Default build uses one concurrent local extraction call. The `mistral` build defaults to four concurrent extraction calls.

## Acknowledgements

Inspired by [Graphiti](https://github.com/getzep/graphiti) — a full-featured Python knowledge graph library by [Zep AI](https://github.com/getzep). ultramemory ports the core episode→entity→edge model to a single, dependency-free Go binary.

The graph traversal algorithm is based on **MAGMA: Multi-hop Activation Graph Memory Algorithm**:

> Ge, T. et al. (2026). *MAGMA: Multi-hop Activation Graph Memory Algorithm for Efficient Memory Retrieval in Long-Context LLMs.* arXiv:2601.03236. https://arxiv.org/abs/2601.03236

Key paper contributions implemented here:
- Transition score `S(n_j|n_i,q) = exp(λ₁·φ(edge_type, intent) + λ₂·cos_sim)` (Eq. 5)
- Additive score propagation `score_v = score_u · γ + S` (Algorithm 1)
- Intent classification T_q ∈ {Why, When, Entity} steering edge-type weights
- Beam search with visited-set cycle prevention, budget-based termination

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
