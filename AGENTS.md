# AGENTS.md

This document provides guidelines for running and contributing to the models within the Cerebras ModelZoo, with a specific focus on the Graph Neural Network (GNN) models.

## Introduction to the Cerebras ModelZoo

The Cerebras ModelZoo is a collection of deep learning models and utilities optimized to run on Cerebras hardware. The repository provides reference implementations, configuration files, and utilities that demonstrate best practices for training and deploying models using Cerebras systems.

### Key Features and Components

*   [**CLI**](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/cli-overview): The ModelZoo CLI is a comprehensive command-line interface that serves as a single entry point for all ModelZoo-related tasks. It streamlines workflows such as data preprocessing, model training, and validation.
*   [**Models**](./src/cerebras/modelzoo/models): Includes configuration files and reference implementations for a wide range of NLP, vision, and multimodal models, including Llama, Mixtral, DINOv2, and Llava. These are optimized for Cerebras hardware and follow best practices for performance and scalability.
*   [**Data Preprocessing Tools**](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/core-workflows/quickstart-guide-for-data-preprocessing): Scripts and utilities for preparing datasets for training, including tokenization, formatting, and batching for supported models.
*   [**Checkpoint Converters and Porting Tools**](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/migration/convert-checkpoints-and-model-configs/convert-checkpoints-and-model-configs): Tools for converting between checkpoint formats (e.g., Cerebras ↔ HuggingFace) and porting PyTorch models to run on Cerebras systems.
*   **Advanced Features**: Support for training optimizations such as custom training loops, custom model implementations, µParam (μP) scaling, rotary position embedding (RoPE) scaling for extended sequence lengths, and more.

---

## GNN Model Guidelines

This section provides specific guidelines for running and contributing to the Graph Neural Network (GNN) models located in the `src/cerebras/modelzoo/models/gnn` directory.

### Environment Setup

We use the `uv` tool for managing the Python 3.8 virtual environment, installing dependencies, and running scripts.

#### Quick Start (Recommended)

The easiest way to prepare the environment is to run the `setup.sh` script from the repository root.

```bash
./setup.sh
```

This script will automatically perform the following steps:
- Create a Python 3.8 virtual environment in `.venv/` (`uv venv`).
- Install the project in editable mode (`uv pip install --editable .`).
- Install all required dependencies from `requirements.txt`.
- Pre-download the necessary datasets by running `download.py`.

#### Manual Setup

If you prefer to set up the environment manually, follow these steps using `uv`:

1.  **Install `uv`** if it is not already available.
2.  **Create the virtual environment:**
    ```bash
    uv venv .venv -p python3.8
    ```
3.  **Install the project in editable mode:**
    ```bash
    uv pip install --editable .
    ```
4.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

### Running the Models

Once the environment is set up, navigate to the GNN model directory. **All subsequent commands should be run from this directory.**

```bash
cd src/cerebras/modelzoo/models/gnn
```

#### Dataset Preparation

The `download.py` helper prepares the PubMed and Reddit datasets with checksum verification and resumable downloads.

```bash
uv run download.py --root data/datasets
```

#### Using the Helper Script

A convenience script `run.sh` is provided for common training and evaluation tasks.
```bash
./run.sh
```

#### Running Directly with `uv`

You can also execute the scripts directly using `uv`.

- Launch training or evaluation with the configuration that matches your experiment (for example, `configs/params.yaml` for PubMed or `configs/params_graphsage_reddit.yaml` for Reddit).
- Use `uv run download.py --root data/datasets` to refresh processed datasets when needed. (Note: `setup.sh` does this automatically.)

---

### Contribution Guidelines for GNN Models

The following guidelines must be followed when contributing to the GNN models.

#### Project Structure & Scope

- The upstream ModelZoo, located under `src/cerebras/modelzoo/...`, is read-only and must not be modified.
- All changes must be confined to the `src/cerebras/modelzoo/models/gnn` directory.
  - **Key Files:** `data.py`, `model.py`, `run.py`, `download.py`
  - **Configuration:** `configs/params_graphsage_reddit.yaml`
  - **Artifacts (Not for commit):** The `model_dir_gnn/` directory is used for logs and checkpoints. Do not commit its contents.

#### Coding Style & Naming Conventions

- **Style:** Adhere to PEP 8 with 4-space indents.
- **Type Hints:** Add type hints for all public-facing APIs.
- **Naming:**
  - `snake_case` for functions and variables.
  - `PascalCase` for classes.
  - `UPPER_SNAKE_CASE` for constants.
- **Imports:** Keep imports scoped to the `models/gnn` directory or documented public APIs from the ModelZoo. Do not import or modify core ModelZoo components.
- **Docstrings:** Write concise docstrings. For functions handling data, include expected tensor shapes/dtypes and file path structures.

#### Commit & Pull Request Guidelines

- **Commits:** Use the [Conventional Commits](https://www.conventionalcommits.org/) specification with a `gnn` scope.
  - *Example:* `feat(gnn): integrate reddit dataset pipeline`
  - *Example:* `fix(gnn): handle isolated nodes in graph processing`
- **Pull Requests:** Every PR must include:
  - A clear summary of the changes.
  - Confirmation that changes are confined to the `models/gnn/` directory.
  - A link to any relevant issue(s).
  - Snippets of the configuration used for testing.
  - Relevant logs or metrics from the `model_dir_gnn/` output directory.

#### Security & Data Hygiene

- **Do Not Commit Large Files:** Never commit datasets, checkpoints, or other large artifacts.
- **Secrets Management:** Do not hardcode secrets (e.g., API keys) in configuration files or source code. If needed, pass them via environment variables.

### Capturing Compile Trace Artifacts (Debugging)

You can force the Cerebras runtime to dump Lazy Tensor (CIRH) diagnostics and
the corresponding executor performance trace by running the standard GNN
workflow in compile-only mode with debug flags enabled:

```bash
mkdir -p model_dir_gnn/ir_debug
export CSTORCH_DEBUG=1
export LTC_IR_DEBUG=1
export LTC_IR_DEBUG_ROOT_PATH="$(pwd)/model_dir_gnn/ir_debug"
export LTC_SAVE_TENSORS_FMT=dot
export LTC_SAVE_TENSORS_FILE="$(pwd)/model_dir_gnn/ir_debug/tensors.dot"

uv run cszoo fit src/cerebras/modelzoo/models/gnn/configs/params_graphsage_reddit.yaml \
  --target_device=CSX \
  --compile_only \
  --model_dir=$(pwd)/model_dir_gnn/reddit_graphsage_debug
```

This creates `executors/000001` with `track.json` / `performance.json` plus IR
debug artifacts (when the compile runs against a CSX backend). Delete the
generated `executors/` directory and any contents of
`model_dir_gnn/ir_debug/` after inspection to keep the workspace clean.
