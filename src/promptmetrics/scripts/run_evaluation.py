import argparse
import asyncio
import datetime
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from promptmetrics.llm_providers.openrouter import OpenRouterLLM
from promptmetrics.logging_utils import setup_logger
from promptmetrics.registry import get_benchmark_instance
from promptmetrics.utils import load_prompt_template


class EvaluationVerdict(BaseModel):
    is_correct: bool | None = None
    extracted_answer: str | None = None
    reasoning: str


def calculate_ece(confidence: np.ndarray, correct: np.ndarray, n_bins=10) -> float:
    if len(confidence) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Include 0.0 confidence in the first bin
        if i == 0:
            in_bin = (confidence >= bin_lower) & (confidence <= bin_upper)
        else:
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(confidence[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


async def main_async():
    parser = argparse.ArgumentParser(
        description="Evaluate model responses and save a complete artifact with summary metrics."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        type=Path,
        help="Path to the timestamped generations.json file.",
    )
    parser.add_argument(
        "--evaluator_model",
        default="mistralai/mistral-small-3.2-24b-instruct:free",
        help="LLM to use as the evaluator.",
    )
    parser.add_argument(
        "--evaluation_prompt_source",
        required=True,
        help="Name of a built-in evaluation prompt, path to a custom prompt file, or 'official' to use the benchmark's default.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of concurrent evaluation requests.",
    )
    parser.add_argument(
        "--evaluator_max_tokens",
        type=int,
        default=4096,
        help="Maximum number of completion tokens for the evaluator model.",
    )
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Bypass the confirmation prompt when evaluating a large number of generations.",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file) as f:
        data = json.load(f)

    generation_metadata = data["metadata"]["generation"]
    generations = data["generations"]
    modality_info = generation_metadata.get("modality_handling")

    required_ids = list(generations.keys())

    if not args.allow_full_run:
        num_to_evaluate = len(generations)
        print("\n--- ⚠️  Warning: Full Evaluation Run ---")
        print(
            f"This will evaluate all {num_to_evaluate} generated responses from the input file."
        )
        print(
            f"\nThis will result in approximately {num_to_evaluate} API calls to the evaluator model '{args.evaluator_model}'."
        )
        print(
            "This may lead to significant API costs and could take a long time to complete."
        )
        confirm = input("\nAre you sure you want to continue? (y/N): ").lower().strip()
        if confirm not in ["y", "yes"]:
            print("Operation cancelled by user.")
            return

    benchmark = get_benchmark_instance(generation_metadata["benchmark"])
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    evaluation_prompt_source = args.evaluation_prompt_source
    if evaluation_prompt_source.lower() == "official":
        if benchmark.official_evaluation_prompt_name:
            evaluation_prompt_source = benchmark.official_evaluation_prompt_name
            print(
                f"Using official evaluation prompt for '{benchmark.name}': '{evaluation_prompt_source}'"
            )
        else:
            # Official name is None, try the non-official fallback
            fallback_prompt_name = "non_official_evaluation_v1"
            try:
                # Check if the fallback exists by trying to load it
                load_prompt_template(fallback_prompt_name, benchmark.name, "evaluation")
                print(
                    f"WARNING: Benchmark '{benchmark.name}' has no official evaluation prompt. "
                    f"Falling back to use '{fallback_prompt_name}'."
                )
                evaluation_prompt_source = fallback_prompt_name
            except FileNotFoundError:
                raise ValueError(
                    f"The benchmark '{benchmark.name}' does not have an official evaluation prompt defined, "
                    f"and the fallback prompt '{fallback_prompt_name}' could not be found."
                )

    log_base_str = str(args.input_file.parent.parent).replace("results", "logs", 1)
    log_dir = Path(log_base_str) / "evaluation"
    setup_logger(log_dir, f"{timestamp}_evaluation.log")

    evaluator_llm = OpenRouterLLM(model_name=args.evaluator_model)
    evaluation_prompt_template, evaluation_prompt_path, evaluation_prompt_type = (
        load_prompt_template(evaluation_prompt_source, benchmark.name, "evaluation")
    )

    loaded_questions = benchmark.load_data(ids_to_load=required_ids)
    questions_map = {q["id"]: q for q in loaded_questions}

    resolved_eval_prompt_stem = Path(evaluation_prompt_path).stem
    is_official_prompt = (
        resolved_eval_prompt_stem == benchmark.official_evaluation_prompt_name
    )
    is_official_fallback = (
        benchmark.official_evaluation_prompt_name is None
        and resolved_eval_prompt_stem == "non_official_evaluation_v1"
    )

    supports_advanced_evaluation = benchmark.official_evaluation_model is not None and (
        is_official_prompt or is_official_fallback
    )

    if supports_advanced_evaluation:
        verdict_model = benchmark.official_evaluation_model
        print(
            f"Using specialized evaluation format for '{benchmark.name.upper()}' benchmark."
        )
    else:
        verdict_model = EvaluationVerdict

    evaluations_dir = args.input_file.parent.parent / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    evaluation_prompt_name = Path(evaluation_prompt_source).stem
    sanitized_evaluator_model = args.evaluator_model.replace("/", "_").replace(":", "-")
    evaluation_filename = f"{timestamp}_evaluations_by_{sanitized_evaluator_model}_with_{evaluation_prompt_name}.json"
    evaluation_filepath = evaluations_dir / evaluation_filename

    evaluations: Dict[str, Any] = {}
    items_to_evaluate = generations

    semaphore = asyncio.Semaphore(args.num_workers)

    async def evaluate_item(q_id, gen_data):
        async with semaphore:
            question_data = questions_map[q_id]
            correct_answer = question_data.get(benchmark.answer_key)

            format_map = defaultdict(str)
            format_map.update(question_data)
            # MMMU fidelity: ensure choices_block is present for evaluation prompts
            # This improves evaluator quality by showing the actual options.
            if benchmark.name.startswith("mmmu"):
                choices_list = []
                parsed_choices = question_data.get("parsed_choices")
                if isinstance(parsed_choices, dict):
                    for letter, text in sorted(parsed_choices.items()):
                        choices_list.append(f"({letter}) {text}")
                format_map["choices_block"] = "\n".join(choices_list)
                # The MMMU evaluation template references {answer} for the correct letter,
                # but we also populate correct_answer_letter consistently.
                if "answer" in question_data:
                    format_map["correct_answer_letter"] = question_data["answer"]

            # Unpack shuffled choices if they exist, for rich eval prompts
            if "shuffled_choices" in question_data and isinstance(
                question_data["shuffled_choices"], dict
            ):
                for key, value in question_data["shuffled_choices"].items():
                    format_map[f"choice_{key}"] = value
            format_map["correct_answer"] = correct_answer
            format_map["correct_answer_letter"] = correct_answer
            format_map["model_response"] = gen_data.get("response")

            eval_prompt = evaluation_prompt_template.format_map(format_map)

            verdict_obj = await evaluator_llm.generate_structured(
                eval_prompt,
                response_model=verdict_model,
                max_tokens=args.evaluator_max_tokens,
            )

            evaluation_dict = verdict_obj.model_dump()

            return q_id, {
                "evaluation": evaluation_dict,
                "generation_data": gen_data,
                "correct_answer": correct_answer,
            }

    tasks = [
        evaluate_item(q_id, gen_data) for q_id, gen_data in items_to_evaluate.items()
    ]
    print(
        f"Evaluating {len(tasks)} items with {args.num_workers} concurrent workers..."
    )
    results = await tqdm_asyncio.gather(*tasks)
    for q_id, evaluated_data in results:
        if q_id is not None:
            evaluations[q_id] = evaluated_data

    correct_flags = []
    confidences = []
    for res in evaluations.values():
        evaluation = res.get("evaluation", {})
        is_correct = None
        if supports_advanced_evaluation:
            is_correct_str = evaluation.get("correct")
            if is_correct_str is not None:
                is_correct = is_correct_str == "yes"
            confidences.append(evaluation.get("confidence", 100))
        else:
            is_correct_bool = evaluation.get("is_correct")
            if is_correct_bool is not None:
                is_correct = is_correct_bool is True

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

    if supports_advanced_evaluation and total_evaluated > 0:
        p_hat = accuracy / 100.0
        ci_half_width = 1.96 * math.sqrt((p_hat * (1 - p_hat)) / total_evaluated)
        summary_metrics["accuracy_ci_95"] = round(ci_half_width * 100, 2)

        conf_array = np.array(confidences) / 100.0
        correct_array = np.array(correct_flags)
        ece = calculate_ece(conf_array, correct_array)
        summary_metrics["expected_calibration_error"] = round(ece * 100, 2)

    evaluation_metadata = {
        "model": args.evaluator_model,
        "reasoning_model": evaluator_llm.supports_reasoning,
        # Use the resolved prompt name, not the user's input argument.
        "prompt_source": evaluation_prompt_source,
        "prompt_source_type": evaluation_prompt_type,
        "prompt_file": str(evaluation_prompt_path),
        "source_generations_file": args.input_file.name,
        "max_tokens": args.evaluator_max_tokens,
        "evaluated_at_utc": timestamp,
    }

    final_evaluation_data = {
        "metadata": {
            "generation": generation_metadata,
            "evaluation": evaluation_metadata,
        },
        "summary_metrics": summary_metrics,
        "evaluations": evaluations,
    }

    with open(evaluation_filepath, "w", encoding="utf-8") as f:
        json.dump(final_evaluation_data, f, indent=2)
    print(
        f"\nEvaluation artifact with summary saved to {evaluation_filepath.resolve()}"
    )

    print("\n--- Final Score ---")
    print(f"Model: {generation_metadata['model']}")
    # Use the resolved name here as well for the final console output
    print(f"Generation Prompt: {generation_metadata['prompt_source']}")
    print(
        f"Evaluated By: {evaluation_metadata['model']} (with prompt '{evaluation_prompt_source}')"
    )

    if supports_advanced_evaluation and total_evaluated > 0:
        print(
            f"Accuracy: {summary_metrics['accuracy']}% +/- {summary_metrics['accuracy_ci_95']}% (CI 95%)"
        )
        print(
            f"Correct: {summary_metrics['correct_count']} / {summary_metrics['total_evaluated']}"
        )
        print(
            f"Expected Calibration Error (ECE): {summary_metrics['expected_calibration_error']}%"
        )
    else:
        print(
            f"Accuracy: {summary_metrics['accuracy']}% ({summary_metrics['correct_count']}/{summary_metrics['total_evaluated']} correct)"
        )

    if modality_info:
        print(f"\n⚠️  NOTE: {modality_info['note']}")
        print(
            "   The results may not be directly comparable to standard, multi-modal evaluations."
        )


def main():
    try:
        asyncio.run(main_async())
    except (ValueError, FileNotFoundError) as e:
        # Catch expected user errors and print a clean message.
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
