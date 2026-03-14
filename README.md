# ANNex — ANN with Filtering

**Fast, in-memory approximate nearest-neighbor search with real-time metadata filtering.**

ANNex is a vector search library built on [HNSW](https://arxiv.org/abs/1603.09320) (C++ core with Python bindings). It packages the HNSW index together with a metadata index so that ANN retrieval and filtering happen in the same memory space — no network hops required.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Creating an Index](#creating-an-index)
- [Querying](#querying)
- [Creating a Custom Filter Function](#creating-a-custom-filter-function)
- [Parameter Reference](#parameter-reference)
  - [Index-Time Parameters](#index-time-parameters)
  - [Query-Time Parameters](#query-time-parameters)
- [Limitations](#limitations)
- [Links](#links)

---

## Features

- ANN search with in-graph metadata filtering
- Built-in Product Quantization for reduced memory footprint
- Deploys as a standard PyTorch model (SageMaker, EC2, ECS, etc.)
- Cloud-agnostic, no vendor lock-in
- Custom filter functions in pure Python

---

## Installation

```bash
pip install annex-vec
```

---

## Quick Start

```python
import numpy as np
from annex_vec import ANNex

# 1. Build an index
index = ANNex(space='l2', dim=512, enable_pq=True, pq_sub_vectors=8, num_bits_sub_vector=8)
index.init_index(max_elements=10_000, ef_construction=200, M=16)
index.set_num_threads(8)
index.add_items(embeddings, ids=ids)

# 2. Query with filters
results = index.knn_query(
    data=np.float32(np.random.random(512)),
    k=10,
    filter_func_name='myfilters.filter_by_channel',
    filter_params={'channel': 'us'},
    timeout_ms=20,
)
```

---

## Creating an Index

```python
from annex_vec import ANNex

# Initialize with distance metric and optional Product Quantization
index = ANNex(
    space='l2',              # 'l2', 'ip', or 'cosine'
    dim=512,
    enable_pq=True,          # enable product quantization
    pq_sub_vectors=8,        # sub-vectors to split each embedding into
    num_bits_sub_vector=8,   # bits per sub-vector codebook
    dtype='float32',
)

# Build the HNSW graph
index.init_index(max_elements=10_000, ef_construction=200, M=16)

# Parallelise indexing across available CPU cores
index.set_num_threads(8)

# Add embeddings and their IDs
index.add_items(data=embeddings, ids=ids)

# Save the index artifact for deployment
index.save("hnswindex")
```

The saved artifact can be compressed as a `tar.gz` and uploaded to S3 (or any object store) for endpoint deployment.

---

## Querying

```python
import numpy as np

query_vec = np.float32(np.random.random(512))

filter_params = {'channel': 'us', 'eligible': 'paid'}

results = index.knn_query(
    data=query_vec,
    k=10,
    filter_func_name='myfilters.filter_by_channel_and_eligible',
    filter_params=filter_params,
    timeout_ms=20,
    explain_res=False,   # set True for debug stats
)
```

When `explain_res=True`, the response includes traversal time, filter time, iteration count, and full data objects with scores — useful for debugging filter logic.

---

## Creating a Custom Filter Function

Filter functions are plain Python callables that receive a data object and a params dict, returning a boolean:

```python
def filter_by_channel_and_eligible(obj, filter_params):
    """Return True to include this node in results."""
    return (
        obj['channel'] == filter_params['channel']
        and obj['eligible'] == filter_params['eligible']
    )
```

**Steps to register a new filter:**

1. Clone the repo.
2. Add your function file to the `filter_functions/` directory.
3. Register it in `filter_functions/filter_map.py` with a unique name.
4. Increment the version number in `setup.py`.
5. Merge to master and build the pipeline.
6. Install the updated package:

```bash
# Specific version
pip install annex-vec==1.0.2 --index-url https://your-artifact-registry/simple

# Latest version
pip install annex-vec --index-url https://your-artifact-registry/simple
```

---
## Parameter Reference

### Index-Time Parameters

| Param | Description | Example |
|---|---|---|
| `space` | Distance metric. `'l2'`: Euclidean `d = sum((Ai-Bi)^2)`. `'ip'`: inner product `d = 1.0 - sum(Ai*Bi)`. `'cosine'`: cosine similarity. | `'l2'` |
| `dim` | Dimensionality of input vectors (must match at query time). | `512` |
| `dtype` | Input vector type. Default `'float32'`. | `'float32'` |
| `max_elements` | Total number of elements in the dataset. | `10000` |
| `ef_construction` | Index build quality vs speed tradeoff. Higher = better recall, slower build. If recall at `ef = ef_construction` is < 0.9, increase it. | `200` |
| `M` | Bi-directional links per element. Higher M = better recall for high-dim data, more memory (~`M × 8–10` bytes/element). Range 12–48 works for most cases. | `16` |
| `enable_pq` | Enable product quantization. Default `False`. | `True` |
| `pq_sub_vectors` | Sub-vectors per embedding (only with `enable_pq=True`). More = better recall, more memory. | `8` |
| `num_bits_sub_vector` | Bits per sub-vector codebook (only with `enable_pq=True`). | `8` |
| `set_num_threads()` | CPU cores for parallel index construction. | `8` |

### Query-Time Parameters

| Param | Description | Example |
|---|---|---|
| `data` | Query vector (same dim as index). | `np.float32(np.random.random(512))` |
| `k` | Number of results to return. | `10` |
| `timeout_ms` | Max graph traversal time in ms. Search stops when K results are found **or** the timeout is reached — whichever comes first. | `20` |
| `filter_func_name` | Registered name of the filter function. | `'myfilters.filter_by_channel_and_eligible'` |
| `filter_params` | Dict passed to the filter function alongside each node's data object. | `{'channel': 'us', 'eligible': 'paid'}` |
| `explain_res` | Return debug stats (traversal time, filter time, iterations, full objects + scores). Default `False`. | `True` |

---

## Limitations

- **Index size is bounded by machine RAM.** The entire index must fit in memory.
- **No incremental updates or deletes.** The full index must be rebuilt (e.g., on a daily Airflow cadence).
- **Python GIL limits multi-threading.** The PyTorch deployment uses multi-process workers; each loads its own index copy.

---

## Links

| Resource | Link |
|---|---|
| HNSW Paper | [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320) |
| hnswlib (C++) | [github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib) |
| FAISS | [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) |

---

## License

Apache 2.0 License
