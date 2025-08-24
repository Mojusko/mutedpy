
#   <img src="icon.png" alt="Repository Icon" width="180"> MUTEDPY: Mutational and Embedding Data Processing Library for Python



**mutedpy** is a Python library for analyzing mutational datasets of protein sequences and structures. It provides a suite of machine learning models and tools for datasets ranging from $10^2$ to $10^6$ samples, with a focus on both shallow and deep learning approaches.

---

## Reference

If you use mutedpy in your research, please cite our paper:

- [Vornholt & Mutny (2024)](https://pubs.acs.org/doi/full/10.1021/acscentsci.4c00258)

---

---

## Features

- **Neural Networks:** Feed-forward, convolutional, and graph-based architectures
- **Linear Models**
- **Gaussian Processes:**
  - Multiple similarity metrics
  - Amino acid embeddings
  - ESM and data-driven embeddings
  - Geometric features
- **Directed Evolution Simulation:** Tools for simulating directed evolution campaigns
- **Sequence Dataset Manipulation:**
  - Format conversion
  - Scanning and enumeration search
  - Train/test data splitting

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Mojusko/mutedpy
```

Install in editable mode:

```bash
pip install -e .
```

> The `-e` flag installs the package in "editable" mode, so updates to your local copy are immediately reflected. Requires Python 3.6+.

---

## 🗂️ Project Structure

// ... Add a brief description of the main folders and their purpose if desired ...

---

## 📝 Updates

- **21/05/2024** – initial commit
- **24/08/2025** - project structure improved
---

## 📋 Requirements

**Classical:**
- pytorch, cvxpy, numpy, scipy, sklearn, pymanopt, mosek, pandas

**Special:**
1. [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize)
2. [stpy](https://github.com/mojusko/stpy)

---

## ✅ Test Coverage

We use `pytest` for testing. To run the test suite and check coverage:

```bash
pytest --cov=mutedpy tests/
```

Test coverage reports are generated using the `pytest-cov` plugin. Please ensure new contributions are covered by tests.

---

## 🤝 Contributions

Contributions are welcome! Please open an issue or pull request.

**Author:** Mojmir Mutny