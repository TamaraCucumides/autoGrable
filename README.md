# autoGrable

Stage-1 structure selection for table-to-graph learning.

## Installation

### From GitHub

**CPU only** — torch installs automatically:

```bash
pip install git+https://github.com/your-username/autoGrable.git
```

**GPU (CUDA)** — install torch with the right CUDA version first, then install autoGrable:

```bash
# example for CUDA 12.1 — see https://pytorch.org/get-started/locally for other versions
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/your-username/autoGrable.git
```

### Local development

```bash
git clone https://github.com/your-username/autoGrable.git
cd autoGrable
pip install torch --index-url https://download.pytorch.org/whl/cu121  # skip for CPU
pip install -e ".[dev]"
```

Or use the provided `requirements.txt` for a pinned CPU environment:

```bash
pip install -r requirements.txt
pip install -e .
```