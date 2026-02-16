# Attzoo 🔍

A modern, extensible PyTorch library for attention mechanisms in transformer models and deep learning architectures. Designed for educational and research purposes with clean, well-documented, and modular code.

## 📋 Attention Mechanisms

### ✅ Currently Implemented

| Mechanism | Class | Description | Use Case |
|-----------|-------|-------------|----------|
| **Vanilla Self-Attention** | `VanillaSelfAttention` | Standard scaled dot-product self-attention | Basic transformer building block |
| **Multi-Head Self-Attention** | `MultiHeadSelfAttention` | Parallel attention heads with different representations | Standard transformer layers |
| **Local Self-Attention** | `LocalSelfAttention` | Windowed attention with configurable window size | Long sequences, O(n×w) complexity |
| **Grouped Self-Attention** | `GroupedSelfAttention` | Memory-efficient attention with shared K,V heads | Efficient transformers (GQA/MQA) |
| **Dilated Self-Attention** | `DilatedSelfAttention` | Sparse attention with dilation patterns | Structured sequences, long-range deps |
| **Linear Self-Attention** | `LinearSelfAttention` | Linear complexity attention using kernel methods | O(n) complexity for very long sequences |
| **Block Self-Attention** | `BlockSelfAttention` | Block-wise sparse attention patterns | Hierarchical attention, document modeling |
| **ALiBi Self-Attention** | `ALiBiSelfAttention` | Attention with linear bias for positions | Length extrapolation capabilities |
| **LSH Self-Attention** | `LSHSelfAttention` | Hash-based bucketed attention within buckets | Approximate global attention for long sequences |
| **Gated Self-Attention (Residual)** | `GatedSelfAttention` | Highway-style gate mixing attention output with the input | Learnable residual strength per token |
| **Combined Attention (Mixture)** | `CombinedAttention` | Learned gate mixes outputs of two attention modules | Softly combine local/global or different patterns |

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/yourusername/attzoo.git
cd attzoo
uv sync

# For development
uv sync --all-extras --dev
```

### Install from PyPI

Install the latest release from PyPI:

```bash
uv add attzoo
```

### Publish to PyPI (Maintainers)

```bash
# Build distributions
uv build

# Upload to PyPI
UV_PUBLISH_TOKEN=pypi-<your-token> uv publish
```

Notes:
- For a brand-new project on PyPI, use an account-scoped token for the first upload.
- Bump `version` in `pyproject.toml` before every release.

### Basic Usage

```python
import torch
from attzoo import MultiHeadSelfAttention

# Initialize model and input
d_model = 128
seq_len = 512
batch_size = 4

attn = MultiHeadSelfAttention(d_model=d_model, num_heads=8)
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
out, weights = attn(x)
print(out.shape)     # [4, 512, 128]
print(weights.shape) # [4, 8, 512, 512]
```

## 📝 Documentation

- Browse the generated docs locally:

  ```bash
  uv run -m mkdocs serve
  ```

  The site is served at `http://127.0.0.1:8000/` with hot reload enabled.

- Produce a production build:

  ```bash
  uv run -m mkdocs build
  ```

  Static assets are written to `site/` and published automatically by CI as part of the GitHub workflow.

## 📊 Performance Comparison

| Mechanism | Time Complexity | Memory Complexity | Best Use Case |
|-----------|----------------|-------------------|---------------|
| Vanilla | O(n²) | O(n²) | Short sequences (< 512) |
| Multi-Head | O(n²) | O(n²) | Standard transformer layers |
| Local | O(n×w) | O(n×w) | Long sequences with local patterns |
| Grouped | O(n²) | O(n²/g) | Memory-constrained scenarios |
| Dilated | O(n×d) | O(n×d) | Structured/periodic patterns |
| Linear | O(n) | O(n) | Very long sequences (> 4K tokens) |
| Block | O(b×(n/b)²) | O(b×(n/b)²) | Memory-efficient long sequences |
| ALiBi | O(n²) | O(n²) | Length extrapolation tasks |
| LSH | Sub-quadratic (~O(n×w×h)) | Sub-quadratic | Approximate long-range attention |

*Where n=sequence length, w=window size, g=group ratio, d=dilation connections, b=number of blocks*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
