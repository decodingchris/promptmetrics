# PromptMetrics

A modular toolkit for the rigorous evaluation and metric generation of LLM prompts.

---

`PromptMetrics` is a professional-grade Python toolkit for the rigorous evaluation of Large Language Model (LLM) prompts. It provides a flexible, reproducible, and cost-effective framework for measuring prompt performance against academic benchmarks like [Humanity's Last Exam (HLE)](https://huggingface.co/datasets/cais/hle).

This tool is designed for serious prompt engineering research. It allows you to:
-   **Test any prompt** against standardized, version-controlled datasets.
-   Use powerful **LLM-based judges** for accurate, semantic grading with guaranteed structured output.
-   Benefit from a **two-step pipeline** that separates expensive generation from repeatable judging, saving time and API credits.
-   Produce **self-contained, timestamped, and auditable artifacts** for every experiment, ensuring perfect reproducibility.

## Core Philosophy

`PromptMetrics` is designed as a scientific instrument. It is built on three key principles:
1.  **Focused Responsibility:** It does one thing and does it well: it **measures prompt performance**. It is designed to work with, but remain separate from, tools that generate prompt ideas.
2.  **Iterative Research:** The architecture allows you to easily re-run evaluations with new judging models or prompts, creating a non-destructive audit trail of your research.
3.  **Modern Engineering:** Built with a state-of-the-art toolchain (`uv`, `pytest`, `ruff`, `mypy`) for reliability, speed, and long-term maintainability.

## Installation & Setup

`PromptMetrics` uses `uv`, a next-generation Python package manager, for a fast and reliable setup.

1.  **Install `uv`:**
    Follow the official instructions at [astral.sh/uv](https://astral.sh/uv). For macOS/Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/decodingchris/promptmetrics.git
    cd promptmetrics
    ```

3.  **Create Environment & Install Dependencies:**
    This single command creates a virtual environment in `.venv` and installs all necessary packages from `pyproject.toml` and `uv.lock`.
    ```bash
    uv sync --dev
    ```

4.  **Set Up Your API Keys:**
    Create a file named `.env` in the project root by copying the example. Then, add your API keys.
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file with your keys:
    ```ini
    # .env
    # Get from https://openrouter.ai/keys
    OPENROUTER_API_KEY="sk-or-v1-..."
    # Get from https://huggingface.co/settings/tokens (needed for gated datasets)
    HF_TOKEN="hf_..."
    ```

## How to Use `PromptMetrics`

The evaluation process is a simple two-step pipeline. All commands are run with `uv run` to ensure they execute within the project's managed environment.

### Step 1: Generate Predictions

This step runs your chosen prompt against a benchmark and saves the model's raw responses into a timestamped, self-contained JSON file.

The `--prompt_source` argument is flexible:
-   Use a simple name (e.g., `chain_of_thought`) to load a built-in prompt from the `prompts/public/{benchmark}/generation/` directory.
-   Provide a file path (e.g., `my_prompts/custom.txt`) to use your own external prompt.

```bash
# Example using a built-in prompt on 3 samples
uv run pm-generate \
  --model "mistralai/mistral-small-3.2-24b-instruct:free" \
  --benchmark "hle" \
  --prompt_source "chain_of_thought" \
  --max_samples 3
```
The script will print the full path to the generated artifact, which you will use in the next step.

### Step 2: Judge the Predictions

This step takes the generated artifact and uses a powerful "judge" LLM to grade each response.

```bash
# Paste the full path from the previous command as the --input_file
uv run pm-judge \
  --input_file "results/hle/mistralai_.../public-chain_of_thought/predictions/20250808..._predictions.json" \
  --judge_model "openai/gpt-4o"
```
This command will:
1.  Read the predictions file.
2.  Use the specified judge model to evaluate each answer with guaranteed structured output.
3.  Save the detailed verdicts to a new timestamped file in the `judged/` directory.
4.  Print the final, trustworthy accuracy score to your console.

```
--- Final Score ---
Model: mistralai/mistral-small-3.2-24b-instruct:free
Prompt Source: chain_of_thought
Judged By: openai/gpt-4o (with prompt 'judge_v1')
Accuracy: 33.33% (1/3 correct)
```

## Understanding the Results

The `results/` directory is organized hierarchically for clarity and scalability:
`results/{benchmark_name}/{model_name}/{experiment_name}/`

Each experiment folder contains two subdirectories:
-   `predictions/`: Contains the raw, timestamped output from the model being tested.
-   `judged/`: Contains one or more timestamped, judged artifacts, each corresponding to a specific judge model and judge prompt.

The final `judged_...json` file is a complete record, containing all metadata, summary metrics, and the detailed verdicts for each question.

## For Developers

We use a suite of modern tools to ensure code quality.

#### Running Tests

To run the full test suite with coverage reporting:
```bash
uv run pytest --cov
```

#### Code Quality Checks

To automatically format your code and run the linter:
```bash
# Format code
uv run ruff format .

# Check for linting errors and style issues
uv run ruff check .
```

#### Type Checking

To run the static type checker:
```bash
uv run mypy src
```
All these checks are also run automatically in our CI pipeline via GitHub Actions.