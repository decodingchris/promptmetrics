import argparse
import json
import asyncio
import datetime
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from typing import List, Dict, Any

from promptmetrics.registry import get_benchmark_instance
from promptmetrics.llm_providers.openrouter import OpenRouterLLM
from promptmetrics.logging_utils import setup_logger
from promptmetrics.utils import load_prompt_template


def adapt_messages_for_text_only(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    text_only_messages = []
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            text_only_messages.append(msg)
            continue

        new_content_parts = [
            part["text"] for part in msg["content"] if part.get("type") == "text"
        ]
        full_text = "\n".join(new_content_parts)

        has_image = any(part.get("type") == "image_url" for part in msg["content"])
        if has_image:
            full_text += "\n\n[NOTE: An image was part of this question but has been omitted as the current model does not support image input.]"

        text_only_messages.append({"role": msg["role"], "content": full_text})
    return text_only_messages


async def main_async():
    parser = argparse.ArgumentParser(
        description="Generate model responses for an evaluation."
    )
    parser.add_argument("--model", required=True, help="OpenRouter model name.")
    parser.add_argument(
        "--benchmark", required=True, help="Name of the benchmark to run (e.g., 'hle')."
    )
    parser.add_argument(
        "--generation_prompt_source",
        required=True,
        help="Name of a built-in generation prompt, path to a custom prompt file, or 'official' to use the benchmark's default.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the model.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum number of completion tokens for the model.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of concurrent generation requests.",
    )
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Bypass the confirmation prompt when running on a full benchmark without --max_samples.",
    )
    args = parser.parse_args()

    benchmark = get_benchmark_instance(args.benchmark)

    generation_prompt_source = args.generation_prompt_source
    if generation_prompt_source.lower() == "official":
        if benchmark.official_generation_prompt_name:
            generation_prompt_source = benchmark.official_generation_prompt_name
            print(f"Using official generation prompt for '{benchmark.name}': '{generation_prompt_source}'")
        else:
            raise ValueError(
                f"The benchmark '{benchmark.name}' does not have an official generation prompt defined."
            )

    if not args.allow_full_run and args.max_samples is None:
        total_samples = benchmark.get_size()
        print("\n--- ⚠️  Warning: Full Benchmark Run ---")
        print(
            f"You have not specified --max_samples. This will run generation on the entire '{args.benchmark}' benchmark."
        )
        print(
            f"\nThis will result in approximately {total_samples} API calls to the model '{args.model}'."
        )
        print(
            "This may lead to significant API costs and could take a long time to complete."
        )
        confirm = input("\nAre you sure you want to continue? (y/N): ").lower().strip()
        if confirm not in ["y", "yes"]:
            print("Operation cancelled by user.")
            return

    llm = OpenRouterLLM(model_name=args.model)

    modality_handling_info = None

    if benchmark.is_multimodal and not llm.supports_vision:
        modality_handling_info = {
            "status": "text_only_fallback",
            "note": "The benchmark is multi-modal, but the model is text-only. Images were omitted from prompts.",
        }
        print("\n--- ⚠️  Warning: Modality Mismatch ---")
        print(modality_handling_info["note"])
        confirm = (
            input("Do you want to continue with this text-only evaluation? (y/N): ")
            .lower()
            .strip()
        )
        if confirm not in ["y", "yes"]:
            print("Evaluation cancelled.")
            return

    prompt_template, resolved_prompt_path, source_type = load_prompt_template(
        generation_prompt_source, benchmark.name, "generation"
    )
    prompt_name = Path(generation_prompt_source).stem
    if source_type == "public":
        experiment_name = f"public-{prompt_name}"
    elif source_type == "private":
        experiment_name = f"private-{prompt_name}"
    else:
        experiment_name = prompt_name

    sanitized_model_name = args.model.replace("/", "_").replace(":", "-")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    log_dir = Path(
        f"logs/{benchmark.name}/{sanitized_model_name}/{experiment_name}/generation"
    )
    setup_logger(log_dir, f"{timestamp}_generation.log")

    output_dir = Path(
        f"results/{benchmark.name}/{sanitized_model_name}/{experiment_name}/generations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_filepath = output_dir / f"{timestamp}_generations.json"

    questions = benchmark.load_data(max_samples=args.max_samples)
    generations = {}

    semaphore = asyncio.Semaphore(args.num_workers)

    async def generate_item(question):
        async with semaphore:
            messages = benchmark.format_prompt_messages(question, prompt_template)

            if modality_handling_info:
                messages = adapt_messages_for_text_only(messages)

            response_data = await llm.generate(
                messages, temperature=args.temperature, max_tokens=args.max_tokens
            )
            generation = {
                "response": response_data.get("content"),
                "reasoning": response_data.get("reasoning"),
            }
            return question["id"], generation

    tasks = [generate_item(question) for question in questions]
    print(
        f"\nGenerating {len(tasks)} responses with {args.num_workers} concurrent workers..."
    )
    results = await tqdm_asyncio.gather(*tasks)
    for q_id, generation_data in results:
        if q_id:
            generations[q_id] = generation_data

    generation_metadata = {
        "model": args.model,
        "benchmark": args.benchmark,
        "prompt_source": args.generation_prompt_source,
        "prompt_source_type": source_type,
        "prompt_file": str(resolved_prompt_path),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "generated_at_utc": timestamp,
    }

    if modality_handling_info:
        generation_metadata["modality_handling"] = modality_handling_info

    output_data = {
        "metadata": {"generation": generation_metadata},
        "generations": generations,
    }

    with open(generations_filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nGenerations saved to:\n{generations_filepath.resolve()}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()