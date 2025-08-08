# PromptMetrics

A modular toolkit for the rigorous evaluation and metric generation of LLM prompts.</p>

---

`PromptMetrics` is a professional-grade Python toolkit for the rigorous evaluation of LLM prompts. It provides a flexible, reproducible, and cost-effective framework for measuring prompt performance against academic benchmarks.

This tool is designed for prompt engineering research. It allows you to:
-   **Test any prompt** against standardized datasets like Humanity's Last Exam (HLE).
-   Use powerful **LLM-based judges** for accurate, semantic grading.
-   Benefit from a **two-step pipeline** that separates expensive generation from repeatable judging, saving time and money.
-   Produce **self-contained, portable artifacts** for every experiment, ensuring reproducibility.

## Core Philosophy

`PromptMetrics` is a scientific instrument. It is built on three key principles:
1.  **Focused Responsibility:** It does one thing exceptionally well: it measures prompt performance.
2.  **Iterative Research:** The architecture is designed to let you easily re-run evaluations with new judging criteria without re-generating answers.
3.  **Modern Engineering:** Built with a state-of-the-art toolchain (`uv`, `pytest`, `ruff`, `mypy`) for reliability, speed, and maintainability.

## Installation & Setup

`PromptMetrics` uses `uv`, a next-generation Python package manager, for a fast and reliable setup.

1.  **Install `uv`:**
    Follow the official instructions at [astral.sh/uv](https://astral.sh/uv). For macOS/Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/promptmetrics.git
    cd promptmetrics
    ```

3.  **Create Environment & Install Dependencies:**
    This single command creates a virtual environment and installs all necessary packages.
    ```bash
    uv sync --dev
    ```

4.  **Set Up Your API Key:**
    Create a file named `.env` in the project root and add your [OpenRouter API key](https://openrouter.ai/keys).
    ```ini
    # .env
    OPENROUTER_API_KEY="sk-or-v1-..."
    ```

## How to Use `PromptMetrics`

The evaluation process is a simple two-step pipeline. All commands are run with `uv run` to ensure they execute within the project's managed environment.

### Step 1: Generate Predictions

This step runs your prompt against a benchmark and saves the model's raw responses.

The `--prompt_source` argument can be a **built-in prompt name** (like `chain_of_thought`) or a **path to your own custom `.txt` file**.

```bash
# Example using a built-in prompt
uv run pm-generate \
  --model "google/gemini-1.5-flash" \
  --benchmark "hle" \
  --prompt_source "chain_of_thought" \
  --max_samples 10 # Optional: for a quick test run
```
This command will create a self-contained results file, for example: `results/hle/google_gemini-1.5-flash/chain_of_thought/raw_predictions.json`.

### Step 2: Judge the Predictions

This step takes the generated artifact and uses a powerful "judge" LLM to grade each response.

```bash
# Point the judge to the generated artifact
uv run pm-judge \
  --input_file "results/hle/google_gemini-1.5-flash/chain_of_thought/raw_predictions.json" \
  --judge_model "openai/gpt-4o" # Optional: specify a different judge model
```
This will:
1.  Read the `raw_predictions.json` file.
2.  Use the specified judge model to evaluate each answer.
3.  Save the detailed verdicts to `judged_results.json` in the same directory.
4.  Print the final accuracy score to your console.

```
--- Final Score ---
Model: google/gemini-1.5-flash
Prompt: chain_of_thought
Accuracy: 80.00% (8/10 correct)
```

## For Developers

We use a suite of modern tools to ensure code quality.

#### Running Tests

To run the full test suite and get a coverage report:
```bash
uv run pytest --cov=src
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

All these checks are also run automatically via GitHub Actions on every push and pull request.