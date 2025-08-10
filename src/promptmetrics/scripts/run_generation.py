import argparse
import json
import os
import asyncio
import datetime
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from typing import Tuple, List, Dict, Any

from promptmetrics.benchmarks.hle import HLEBenchmark
from promptmetrics.llm_providers.openrouter import OpenRouterLLM
from promptmetrics.logging_utils import setup_logger

def get_benchmark_instance(name: str):
    if name.lower() == "hle":
        return HLEBenchmark()
    else:
        raise ValueError(f"Unknown benchmark: {name}")

def load_prompt_template(prompt_source: str, benchmark_name: str) -> Tuple[str, Path, str]:
    if os.path.isfile(prompt_source):
        path = Path(prompt_source)
        return path.read_text(encoding='utf-8'), path, "external"

    prompt_name_with_ext = f"{prompt_source}.txt"
    private_path = Path("prompts") / "private" / benchmark_name / "generation" / prompt_name_with_ext
    if private_path.exists():
        return private_path.read_text(encoding='utf-8'), private_path, "private"

    public_path = Path("prompts") / "public" / benchmark_name / "generation" / prompt_name_with_ext
    if public_path.exists():
        return public_path.read_text(encoding='utf-8'), public_path, "public"

    raise FileNotFoundError(
        f"Prompt '{prompt_source}' not found as a file or in public/private generation directories."
    )

def adapt_messages_for_text_only(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strips images from messages and adds a note, for text-only models."""
    text_only_messages = []
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            text_only_messages.append(msg)
            continue
        
        new_content_parts = [part["text"] for part in msg["content"] if part.get("type") == "text"]
        full_text = "\n".join(new_content_parts)
        
        has_image = any(part.get("type") == "image_url" for part in msg["content"])
        if has_image:
            full_text += "\n\n[NOTE: An image was part of this question but has been omitted as the current model does not support image input.]"
        
        text_only_messages.append({"role": msg["role"], "content": full_text})
    return text_only_messages

async def main_async():
    parser = argparse.ArgumentParser(description="Generate model responses for an evaluation.")
    parser.add_argument("--model", required=True, help="OpenRouter model name.")
    parser.add_argument("--benchmark", required=True, help="Name of the benchmark to run (e.g., 'hle').")
    parser.add_argument("--prompt_source", required=True, help="Name of a built-in generation prompt or path to a custom prompt file.")
    parser.add_argument("--temperature", type=float, default=0.0, help="The sampling temperature for the model.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of completion tokens for the model.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of concurrent generation requests.")
    args = parser.parse_args()

    benchmark = get_benchmark_instance(args.benchmark)
    llm = OpenRouterLLM(model_name=args.model)

    is_mismatched_run = False 
    
    if benchmark.is_multimodal and not llm.supports_vision:
        is_mismatched_run = True
        print("\n--- ⚠️  Warning: Modality Mismatch ---\n"
              f"Benchmark '{args.benchmark}' is multi-modal (contains images).\n"
              f"Model '{args.model}' does not support image input.\n\n"
              "Images will be omitted from the prompts. This may significantly affect the model's performance "
              "and the validity of the benchmark score.\n")
        
        confirm = input("Do you want to continue with this text-only evaluation? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Evaluation cancelled.")
            return

    prompt_template, resolved_prompt_path, source_type = load_prompt_template(args.prompt_source, benchmark.name)
    prompt_name = Path(args.prompt_source).stem
    if source_type == "public":
        experiment_name = f"public-{prompt_name}"
    elif source_type == "private":
        experiment_name = f"private-{prompt_name}"
    else:
        experiment_name = prompt_name

    sanitized_model_name = args.model.replace("/", "_").replace(":", "-")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    log_dir = Path(f"logs/{benchmark.name}/{sanitized_model_name}/{experiment_name}/generation")
    setup_logger(log_dir, f"{timestamp}_generation.log")

    output_dir = Path(f"results/{benchmark.name}/{sanitized_model_name}/{experiment_name}/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_filepath = output_dir / f"{timestamp}_predictions.json"

    questions = benchmark.load_data(max_samples=args.max_samples)
    predictions = {}
    
    semaphore = asyncio.Semaphore(args.num_workers)
    
    async def generate_item(question):
        async with semaphore:
            messages = benchmark.format_prompt_messages(question, prompt_template)
            
            if is_mismatched_run:
                messages = adapt_messages_for_text_only(messages)

            response_data = await llm.generate(messages, temperature=args.temperature, max_tokens=args.max_tokens)
            prediction = {
                "response": response_data.get("content"),
                "reasoning": response_data.get("reasoning"),
            }
            return question['id'], prediction

    tasks = [generate_item(question) for question in questions]
    print(f"\nGenerating {len(tasks)} responses with {args.num_workers} concurrent workers...")
    results = await tqdm_asyncio.gather(*tasks)
    for q_id, prediction_data in results:
        if q_id:
            predictions[q_id] = prediction_data

    output_data = {
        "metadata": {
            "model": args.model,
            "benchmark": args.benchmark,
            "prompt_source": args.prompt_source,
            "prompt_source_type": source_type,
            "prompt_file": str(resolved_prompt_path),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "generated_at_utc": timestamp,
            "is_mismatched_run": is_mismatched_run,
        },
        "predictions": predictions
    }

    with open(predictions_filepath, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nPredictions saved to:\n{predictions_filepath.resolve()}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()