# PromptMetrics

A modular toolkit for the rigorous evaluation and metric generation of LLM prompts.

---

`PromptMetrics` is a professional-grade Python toolkit for the rigorous evaluation of Large Language Model (LLM) prompts. It provides a flexible, reproducible, and cost-effective framework for measuring prompt performance against academic benchmarks like [Humanity's Last Exam (HLE)](https://huggingface.co/datasets/cais/hle), including **full support for multi-modal (vision) questions**.

This tool is designed for serious prompt engineering research. It allows you to:
-   **Test any prompt** against standardized, version-controlled datasets, including vision benchmarks.
-   **Replicate official benchmark methodologies** with support for separate `system` and `user` prompts.
-   Use powerful **LLM-based evaluators** for accurate, semantic grading with guaranteed structured output.
-   Benefit from a **two-step pipeline** that separates expensive generation from repeatable evaluation, saving time and API credits.
-   Produce **self-contained, timestamped, and auditable artifacts** for every experiment, ensuring perfect reproducibility.

## Core Philosophy

`PromptMetrics` is designed as a scientific instrument. It is built on three key principles:
1.  **Focused Responsibility:** It does one thing and does it well: it **measures prompt and model performance**. It is designed to work with, but remain separate from, tools that generate prompt ideas.
2.  **Iterative Research:** The architecture allows you to easily re-run evaluations with new evaluator models or prompts, creating a non-destructive audit trail of your research.
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
    This single command creates a virtual environment in `.venv` and installs all necessary packages.
    ```bash
    uv sync --dev
    ```

4.  **Set Up Your API Keys:**
    Create a file named `.env` in the project root. Then, add your API keys.
    ```bash
    # Create the file
    touch .env
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

### Step 1: Generate Model Outputs

This step runs your chosen prompt and model against a benchmark, saving the raw responses into a timestamped, self-contained JSON file.

```bash
# Example running the multi-modal HLE benchmark with a vision model
uv run pm-generate \
  --model "openai/gpt-4o" \
  --benchmark "hle" \
  --generation_prompt_source "official_generation_v1" \
  --max_samples 3
```
The script will print the full path to the generated artifact, which you will use in the next step.

### Step 2: Evaluate the Generations

This step takes the generated artifact and uses a powerful "evaluator" LLM to grade each response. The `--evaluation_prompt_source` argument is required.

```bash
# Paste the full path from the previous command as the --input_file
uv run pm-evaluate \
  --input_file "results/hle/openai_gpt-4o/public-official_generation_v1/generations/20250810..._generations.json" \
  --evaluator_model "openai/gpt-4o" \
  --evaluation_prompt_source "official_evaluation_v1"
```
This command will:
1.  Read the generations file.
2.  Use the specified evaluator model to evaluate each answer with guaranteed structured output.
3.  Save the detailed evaluations to a new timestamped file in the `evaluations/` directory.
4.  Print the final, trustworthy score to your console, including advanced metrics for official evaluations.

```
--- Final Score ---
Model: openai/gpt-4o
Generation Prompt: official_generation_v1
Evaluated By: openai/gpt-4o (with prompt 'official_evaluation_v1')
Accuracy: 66.67% +/- 38.49% (CI 95%)
Correct: 2 / 3
Expected Calibration Error (ECE): 25.50%
```

## Advanced Usage

#### Writing Custom Prompts (System & User Roles)

For full control over the model's behavior, you can create your own prompt files. `PromptMetrics` uses simple separators to define message roles, giving you maximum flexibility. The `{question}` placeholder is always required in the user message.

**Example: `my_custom_prompt.txt`**
```txt
---[SYSTEM]---
You are a helpful assistant specializing in astrophysics.

---[USER]---
Please answer the following question: {question}
```
You can use any custom prompt by providing the file path to the `--generation_prompt_source` argument.

#### Running on Full Benchmarks (Safety First)

To prevent accidental, large, and costly API jobs, `PromptMetrics` includes a safety check. If you run `pm-generate` without `--max_samples` or `pm-evaluate` on a large file, it will display a warning and ask for confirmation.

```
--- ⚠️  Warning: Full Benchmark Run ---
You have not specified --max_samples. This will run generation on the entire 'hle' benchmark.

This will result in approximately 2500 API calls to the model 'openai/gpt-4o'.
This may lead to significant API costs and could take a long time to complete.

Are you sure you want to continue? (y/N):
```

For automated scripts (e.g., in a CI/CD pipeline) or for expert users who want to bypass this check, use the `--allow-full-run` flag.

```bash
# This will run the full benchmark without asking for confirmation
uv run pm-generate --benchmark "hle" ... --allow-full-run
```

#### Replicating Official Benchmarks

`PromptMetrics` allows for high-fidelity replication of official benchmarks by using their specific prompts and evaluation criteria.

**1. Use Official Prompts:** Built-in official prompts can be selected via `--generation_prompt_source` and `--evaluation_prompt_source`. For example, to run the official HLE workflow:
```bash
# Generate using the official HLE system/user prompt format
uv run pm-generate \
  --model "openai/gpt-4o" \
  --benchmark "hle" \
  --generation_prompt_source "official_generation_v1" \
  --max_samples 3

# Evaluate using the official HLE evaluation prompt for advanced metrics
uv run pm-evaluate \
  --input_file "results/hle/openai_gpt-4o/public-official_generation_v1/generations/...(new file).json" \
  --evaluation_prompt_source "official_evaluation_v1"
```

**2. Get Advanced Metrics:** When using an official evaluation prompt, the final output will include specialized metrics like **Expected Calibration Error (ECE)**.

```
--- Final Score ---
Model: openai/gpt-4o
Generation Prompt: official_generation_v1
Evaluated By: openai/gpt-4o (with prompt 'official_evaluation_v1')
Accuracy: 66.67% +/- 38.49% (CI 95%)
Correct: 2 / 3
Expected Calibration Error (ECE): 25.50%
```

## Understanding the Results

The `results/` directory is organized hierarchically for clarity and scalability:
`results/{benchmark_name}/{model_name}/{experiment_name}/`

Each experiment folder contains two subdirectories:
-   `generations/`: Contains the raw, timestamped output from the model being tested.
-   `evaluations/`: Contains one or more timestamped, evaluated artifacts, each corresponding to a specific evaluator model and evaluation prompt.

The final `evaluations_...json` file is a complete record. Its `metadata` section is nested to clearly separate the parameters of the generation and evaluation steps, ensuring perfect auditability.

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