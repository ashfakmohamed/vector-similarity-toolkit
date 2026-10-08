# Vector Similarity Toolkit

Pure-Python dot product, cosine similarity, and deterministic top-k ranking for small embedding experiments.

## Highlights

- Zero runtime dependencies
- Explicit dimension validation
- Stable score-and-name ordering
- Useful for learning and small local prototypes

```python
from vector_similarity_toolkit import top_k

matches = top_k(query_vector, document_vectors, limit=5)
```

Use a vector database or optimized numerical library for large production workloads.

Run `pip install -r requirements-dev.txt`, then `ruff check .` and `python -m pytest -q`.

No license has been selected; all rights are reserved.
