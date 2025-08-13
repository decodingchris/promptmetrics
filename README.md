# PromptMetrics

A modular toolkit for the rigorous evaluation and metric generation of LLM prompts.

---

`PromptMetrics` is a professional-grade Python toolkit for the rigorous evaluation of Large Language Model (LLM) prompts. It provides a flexible, reproducible, and cost-effective framework for measuring prompt performance against academic benchmarks like **[AIME 2025](https://huggingface.co/datasets/yentinglin/aime_2025)**, **[FACTS Grounding](https://huggingface.co/datasets/google/FACTS-grounding-public)**, **[GPQA](https://huggingface.co/datasets/idavidrein/gpqa)**, **[Humanity's Last Exam (HLE)](https://huggingface.co/datasets/cais/hle)**, and the comprehensive **[MMMU](https://huggingface.co/datasets/MMMU/MMMU)** benchmark, including full support for multi-modal (vision) questions where applicable.

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
    This single command creates a virtual environment in `.venv` and installs all necessary packages, including development tools.
    ```bash
    uv sync --all-extras
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

**Example 1: Running the multi-modal HLE benchmark with a vision model**
```bash
uv run pm-generate \
  --model "openai/gpt-4o" \
  --benchmark "hle" \
  --generation_prompt_source "official" \
  --max_samples 3
```

**Example 2: Running the text-only GPQA benchmark**
```bash
uv run pm-generate \
  --model "google/gemini-pro" \
  --benchmark "gpqa_diamond" \
  --generation_prompt_source "official" \
  --max_samples 5
```
**Example 3: Running the long-context FACTS Grounding benchmark**
```bash
uv run pm-generate \
  --model "anthropic/claude-3.5-sonnet-3.5-20240620" \
  --benchmark "facts" \
  --generation_prompt_source "official" \
  --max_samples 3
```

**Example 4: Running the AIME 2025 mathematical reasoning benchmark**
```bash
uv run pm-generate \
  --model "anthropic/claude-3.5-sonnet-3.5-20240620" \
  --benchmark "aime_2025" \
  --generation_prompt_source "official" \
  --max_samples 3
```

**Example 5: Running the full, multi-subject MMMU benchmark**
This command runs the entire MMMU benchmark, which includes questions from all 30 of its subjects.
```bash
uv run pm-generate \
  --model "openai/gpt-4o" \
  --benchmark "mmmu" \
  --generation_prompt_source "official" \
  --max_samples 10
```

**Example 6: Running a single subject from the MMMU benchmark**
For faster, targeted tests, you can run on a single subject like "Art" or "Computer_Science".
```bash
uv run pm-generate \
  --model "openai/gpt-4o" \
  --benchmark "mmmu_art" \
  --generation_prompt_source "official" \
  --max_samples 3
```

By default, this saves outputs to `results/` and `logs/` subdirectories in your current location. You can redirect this using the `--output_dir` flag (e.g., `--output_dir ./my-experiments`).

The script will print the full path to the generated artifact, which you will use in the next step.

### Step 2: Evaluate the Generations

This step takes the generated artifact and uses a powerful "evaluator" LLM to grade each response. The `--evaluation_prompt_source` argument is required.

```bash
# Paste the full path from the previous command as the --input_file
uv run pm-evaluate \
  --input_file "results/hle/openai_gpt-4o/public-official_generation_v1/generations/<TIMESTAMP>_generations.json" \
  --evaluator_model "openai/gpt-4o" \
  --evaluation_prompt_source "official"
```
This command will:
1.  Read the generations file.
2.  Use the specified evaluator model to evaluate each answer with guaranteed structured output.
3.  Save the detailed evaluations to a new timestamped file in the `evaluations/` directory, parallel to the `generations/` directory.
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

-   **Structured evaluation portability:** We attempt multiple structured-output strategies (native parsing, JSON Schema constrained responses, and JSON mode). Some evaluator models on OpenRouter may not support these features; in that case PromptMetrics falls back gracefully and suggests compatible evaluators.

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

`PromptMetrics` allows for high-fidelity replication of official benchmarks by using their specific prompts and evaluation criteria. The `--generation_prompt_source "official"` and `--evaluation_prompt_source "official"` flags automatically select the correct prompt for the specified benchmark.

**1. Use Official Prompts:** To run the official HLE workflow:
```bash
# Generate using the official HLE system/user prompt format
uv run pm-generate \
  --benchmark "hle" --model "openai/gpt-4o" --generation_prompt_source "official" --max_samples 3

# Evaluate using the official HLE evaluation prompt for advanced metrics
uv run pm-evaluate \
  --input_file "results/hle/openai_gpt-4o/public-official_generation_v1/generations/...(new_file).json" \
  --evaluation_prompt_source "official"
```

**2. Get Advanced Metrics:** When an evaluation uses the benchmark's specialized Pydantic model for grading, the final output will include advanced metrics like **Expected Calibration Error (ECE)**. This is automatically triggered when using an official evaluation prompt.

**Crucially, this also works for benchmarks like FACTS, GPQA, and MMMU that do not have an *official* evaluation prompt.** When you use `--evaluation_prompt_source "official"`, `PromptMetrics` intelligently falls back to a compatible, high-quality community prompt (`non_official_evaluation_v1.txt`) while still using the benchmark's specialized Pydantic model (`OfficialFACTSEvaluation`, `OfficialGPQAEvaluation`, or `OfficialMMMU_V1Evaluation`). This ensures you get advanced metrics even when an official grading prompt isn't published.
```
--- Final Score ---
Model: openai/gpt-4o
Generation Prompt: non_official_generation_v1
Evaluated By: openai/gpt-4o (with prompt 'non_official_evaluation_v1')
Accuracy: 66.67% +/- 38.49% (CI 95%)
Correct: 2 / 3
Expected Calibration Error (ECE): 25.50%
```

## Understanding the Results

By default, `PromptMetrics` saves data into `results/` and `logs/` directories. The `results/` directory is organized hierarchically for clarity and scalability:
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
