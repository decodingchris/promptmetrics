import argparse
import json
import os
import asyncio
import datetime
import math
import numpy as np
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel, Field
from typing import Tuple, Dict, Any, Literal

from promptmetrics.benchmarks.hle import HLEBenchmark
from promptmetrics.llm_providers.openrouter import OpenRouterLLM
from promptmetrics.logging_utils import setup_logger

class EvaluationVerdict(BaseModel):
    is_correct: bool | None
    extracted_answer: str | None
    reasoning: str

class OfficialHLEEvaluation(BaseModel):
    extracted_final_answer: str | None
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int = Field(ge=0, le=100)

def get_benchmark_instance(name: str):
    if name.lower() == "hle":
        return HLEBenchmark()
    else:
        raise ValueError(f"Unknown benchmark: {name}")

def load_evaluation_prompt_template(prompt_source: str, benchmark_name: str) -> Tuple[str, Path, str]:
    if Path(prompt_source).is_file():
        path = Path(prompt_source)
        return path.read_text(encoding='utf-8'), path, "external"
    
    prompt_name_with_ext = f"{prompt_source}.txt"
    private_path = Path("prompts") / "private" / benchmark_name / "evaluation" / prompt_name_with_ext
    if private_path.exists():
        return private_path.read_text(encoding='utf-8'), private_path, "private"

    public_path = Path("prompts") / "public" / benchmark_name / "evaluation" / prompt_name_with_ext
    if public_path.exists():
        return public_path.read_text(encoding='utf-8'), public_path, "public"

    raise FileNotFoundError(
        f"Evaluation prompt '{prompt_source}' not found as a file or in any of the evaluation search paths."
    )

def calculate_ece(confidence: np.ndarray, correct: np.ndarray, n_bins=10) -> float:
    if len(confidence) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(confidence[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

async def main_async():
    parser = argparse.ArgumentParser(description="Evaluate model responses and save a complete artifact with summary metrics.")
    parser.add_argument("--input_file", required=True, type=Path, help="Path to the timestamped generations.json file.")
    parser.add_argument("--evaluator_model", default="mistralai/mistral-small-3.2-24b-instruct:free", help="LLM to use as the evaluator.")
    parser.add_argument("--evaluation_prompt_source", required=True, help="Name of a built-in evaluation prompt or path to a custom one.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of concurrent evaluation requests.")
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Bypass the confirmation prompt when evaluating a large number of generations."
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file) as f:
        data = json.load(f)
        
    generation_metadata = data['metadata']['generation']
    generations = data['generations']
    was_mismatched = generation_metadata.get("is_mismatched_run", False)

    if not args.allow_full_run:
        num_to_evaluate = len(generations)
        print(f"\n--- ⚠️  Warning: Full Evaluation Run ---")
        print(f"This will evaluate all {num_to_evaluate} generated responses from the input file.")
        print(f"\nThis will result in approximately {num_to_evaluate} API calls to the evaluator model '{args.evaluator_model}'.")
        print(f"This may lead to significant API costs and could take a long time to complete.")
        confirm = input("\nAre you sure you want to continue? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled by user.")
            return

    benchmark = get_benchmark_instance(generation_metadata['benchmark'])
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    log_base_str = str(args.input_file.parent.parent).replace('results', 'logs', 1)
    log_dir = Path(log_base_str) / 'evaluation'
    setup_logger(log_dir, f"{timestamp}_evaluation.log")

    evaluator_llm = OpenRouterLLM(model_name=args.evaluator_model)
    evaluation_prompt_template, evaluation_prompt_path, evaluation_prompt_type = load_evaluation_prompt_template(args.evaluation_prompt_source, benchmark.name)
    questions_map = {q['id']: q for q in benchmark.load_data()}

    evaluation_prompt_name = Path(args.evaluation_prompt_source).stem
    if evaluation_prompt_name == "official_evaluation_v1":
        verdict_model = OfficialHLEEvaluation
        is_official_evaluation = True
        print("Using Official HLE Evaluation prompt and metrics.")
    else:
        verdict_model = EvaluationVerdict
        is_official_evaluation = False

    evaluations_dir = args.input_file.parent.parent / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    
    sanitized_evaluator_model = args.evaluator_model.replace("/", "_").replace(":", "-")
    evaluation_filename = f"{timestamp}_evaluations_by_{sanitized_evaluator_model}_with_{evaluation_prompt_name}.json"
    evaluation_filepath = evaluations_dir / evaluation_filename
    
    evaluations: Dict[str, Any] = {}
    items_to_evaluate = generations

    semaphore = asyncio.Semaphore(args.num_workers)

    async def evaluate_item(q_id, gen_data):
        async with semaphore:
            question_data = questions_map[q_id]
            eval_prompt = evaluation_prompt_template.format(
                answer_type=question_data['answer_type'],
                question=question_data['question'],
                correct_answer=question_data['answer'],
                model_response=gen_data['response']
            )
            verdict_obj = await evaluator_llm.generate_structured(eval_prompt, response_model=verdict_model)
            
            evaluation_dict = verdict_obj if isinstance(verdict_obj, dict) else verdict_obj.model_dump()

            return q_id, {
                "evaluation": evaluation_dict,
                "generation_data": gen_data,
                "correct_answer": question_data['answer']
            }

    tasks = [evaluate_item(q_id, gen_data) for q_id, gen_data in items_to_evaluate.items()]
    print(f"Evaluating {len(tasks)} items with {args.num_workers} concurrent workers...")
    results = await tqdm_asyncio.gather(*tasks)
    for q_id, evaluated_data in results:
        if q_id:
            evaluations[q_id] = evaluated_data

    correct_flags = []
    confidences = []
    for res in evaluations.values():
        evaluation = res.get('evaluation', {})
        if is_official_evaluation:
            is_correct = evaluation.get('correct') == 'yes'
            confidences.append(evaluation.get('confidence', 100))
        else:
            is_correct = evaluation.get('is_correct') is True
        
        if is_correct is not None:
             correct_flags.append(is_correct)

    total_evaluated = len(correct_flags)
    correct_count = sum(correct_flags)
    accuracy = (correct_count / total_evaluated) * 100 if total_evaluated > 0 else 0
    
    summary_metrics = {
        "accuracy": round(accuracy, 2),
        "correct_count": correct_count,
        "total_evaluated": total_evaluated,
    }

    if is_official_evaluation and total_evaluated > 0:
        p_hat = accuracy / 100.0
        ci_half_width = 1.96 * math.sqrt((p_hat * (1 - p_hat)) / total_evaluated)
        summary_metrics["accuracy_ci_95"] = round(ci_half_width * 100, 2)
        
        conf_array = np.array(confidences) / 100.0
        correct_array = np.array(correct_flags)
        ece = calculate_ece(conf_array, correct_array)
        summary_metrics["expected_calibration_error"] = round(ece * 100, 2)

    evaluation_metadata = {
        "model": args.evaluator_model,
        "prompt_source": args.evaluation_prompt_source,
        "prompt_source_type": evaluation_prompt_type,
        "prompt_file": str(evaluation_prompt_path),
        "source_generations_file": args.input_file.name,
        "evaluated_at_utc": timestamp
    }

    final_evaluation_data = {
        "metadata": {
            "generation": generation_metadata,
            "evaluation": evaluation_metadata
        },
        "summary_metrics": summary_metrics,
        "evaluations": evaluations,
    }
    
    with open(evaluation_filepath, "w", encoding='utf-8') as f:
        json.dump(final_evaluation_data, f, indent=2)
    print(f"\nEvaluation artifact with summary saved to {evaluation_filepath.resolve()}")

    print("\n--- Final Score ---")
    print(f"Model: {generation_metadata['model']}")
    print(f"Generation Prompt: {generation_metadata['prompt_source']}")
    print(f"Evaluated By: {evaluation_metadata['model']} (with prompt '{evaluation_metadata['prompt_source']}')")
    
    if is_official_evaluation and total_evaluated > 0:
        print(f"Accuracy: {summary_metrics['accuracy']}% +/- {summary_metrics['accuracy_ci_95']}% (CI 95%)")
        print(f"Correct: {summary_metrics['correct_count']} / {summary_metrics['total_evaluated']}")
        print(f"Expected Calibration Error (ECE): {summary_metrics['expected_calibration_error']}%")
    else:
        print(f"Accuracy: {summary_metrics['accuracy']}% ({summary_metrics['correct_count']}/{summary_metrics['total_evaluated']} correct)")
    
    if was_mismatched:
        print("\n⚠️  NOTE: This score was generated from a text-only run on a multi-modal benchmark.")
        print("   The results may not be directly comparable to standard, multi-modal evaluations.")


def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()