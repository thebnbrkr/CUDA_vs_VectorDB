#  Custom CUDA vs ChromaDB: Vector Search Benchmark

This notebook benchmarks the performance of a **custom CUDA-based vector search implementation** against **ChromaDB**, a popular open-source vector database.

The test focuses on **100,000 64-dimensional vectors**, evaluating both retrieval speed and relevance of results.

---

##  What It Demonstrates

1.  **Low-level GPU acceleration** using a custom `.cu` kernel with `nvcc`
2.  **Vector similarity search** using dot product scoring
3.  Benchmarks vs. ChromaDB for retrieval time on the same dataset
4.  Performance comparison with actual wall clock timings

---

## How It Works

###  CUDA Implementation

- Defines a custom CUDA kernel to compute vector similarity
- Uses tiled memory access and 256-thread blocks for optimization
- Compiled with `nvcc` and executed on an NVIDIA GPU
- Measures total time for upload + search

###  ChromaDB Setup

- Uses `chromadb` Python library
- Loads the same 100K vectors
- Uses in-memory storage for fair comparison
- Queries a random test vector and retrieves top-1 result
- Measures total vector search time

---

##  Tech Stack

| Component       | Tool/Library              |
|----------------|---------------------------|
| GPU Benchmark  | Custom `.cu` kernel       |
| CPU Baseline   | [ChromaDB](https://www.trychroma.com) |
| Language       | Python + CUDA C++         |
| Hardware       | NVIDIA CUDA-compatible GPU|

---

##  Results

| Method      | Time (Upload + Search) | Best Match Index | Distance / Score |
|-------------|------------------------|------------------|------------------|
| **CUDA**    | ~13.1 ms               | 0                | 0.0000 (exact)   |
| **ChromaDB**| ~45,751 ms             | 42864            | vector approx.   |

###  Interpretation:

- The CUDA method is ~3500× faster for top-1 search on this dataset.
- ChromaDB, while slower, supports more advanced functionality:
  - Metadata filtering
  - LLM integration
  - Disk-backed persistence
- CUDA is ideal for **raw speed** and performance-critical search.

---

##  Requirements

- Python 3.8+
- `chromadb`
- NVIDIA GPU + CUDA Toolkit (`nvcc`)

