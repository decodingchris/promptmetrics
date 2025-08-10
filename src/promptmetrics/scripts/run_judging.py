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

class JudgeVerdict(BaseModel):
    is_correct: bool | None
    extracted_answer: str | None
    reasoning: str

class OfficialHLEVerdict(BaseModel):
    extracted_final_answer: str | None
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int = Field(ge=0, le=100)

def get_benchmark_instance(name: str):
    if name.lower() == "hle":
        return HLEBenchmark()
    else:
        raise ValueError(f"Unknown benchmark: {name}")

def load_judge_prompt_template(prompt_source: str, benchmark_name: str) -> Tuple[str, Path, str]:
    if Path(prompt_source).is_file():
        path = Path(prompt_source)
        return path.read_text(encoding='utf-8'), path, "external"
    
    prompt_name_with_ext = f"{prompt_source}.txt"
    private_path = Path("prompts") / "private" / benchmark_name / "judging" / prompt_name_with_ext
    if private_path.exists():
        return private_path.read_text(encoding='utf-8'), private_path, "private"

    public_path = Path("prompts") / "public" / benchmark_name / "judging" / prompt_name_with_ext
    if public_path.exists():
        return public_path.read_text(encoding='utf-8'), public_path, "public"

    raise FileNotFoundError(
        f"Judge prompt '{prompt_source}' not found as a file or in any of the judging search paths."
    )

def calculate_ece(confidence: np.ndarray, correct: np.ndarray, n_bins=10) -> float:
    """Calculates Expected Calibration Error (ECE)."""
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
    parser = argparse.ArgumentParser(description="Judge model responses and save a complete artifact with summary metrics.")
    parser.add_argument("--input_file", required=True, type=Path, help="Path to the timestamped predictions.json file.")
    parser.add_argument("--judge_model", default="mistralai/mistral-small-3.2-24b-instruct:free", help="LLM to use as the judge.")
    parser.add_argument("--judge_prompt_source", default="official_judge_v1", help="Name of a built-in judge prompt or path to a custom one.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of concurrent judging requests.")
    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file) as f:
        data = json.load(f)
    generation_metadata = data['metadata']
    predictions = data['predictions']
    was_mismatched = generation_metadata.get("is_mismatched_run", False)

    benchmark = get_benchmark_instance(generation_metadata['benchmark'])
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    log_base_str = str(args.input_file.parent.parent).replace('results', 'logs', 1)
    log_dir = Path(log_base_str) / 'judging'
    setup_logger(log_dir, f"{timestamp}_judging.log")

    judge_llm = OpenRouterLLM(model_name=args.judge_model)
    judge_prompt_template, judge_prompt_path, judge_prompt_type = load_judge_prompt_template(args.judge_prompt_source, benchmark.name)
    questions_map = {q['id']: q for q in benchmark.load_data()}

    judge_prompt_name = Path(args.judge_prompt_source).stem
    if judge_prompt_name == "official_judge_v1":
        verdict_model = OfficialHLEVerdict
        is_official_judge = True
        print("Using Official HLE Judge prompt and metrics.")
    else:
        verdict_model = JudgeVerdict
        is_official_judge = False

    judged_dir = args.input_file.parent.parent / "judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    
    sanitized_judge_model = args.judge_model.replace("/", "_").replace(":", "-")
    judged_filename = f"{timestamp}_judged_by_{sanitized_judge_model}_with_{judge_prompt_name}.json"
    judged_filepath = judged_dir / judged_filename
    
    verdicts: Dict[str, Any] = {}
    items_to_judge = predictions

    semaphore = asyncio.Semaphore(args.num_workers)

    async def judge_item(q_id, pred_data):
        async with semaphore:
            question_data = questions_map[q_id]
            judge_prompt = judge_prompt_template.format(
                answer_type=question_data['answer_type'],
                question=question_data['question'],
                correct_answer=question_data['answer'],
                model_response=pred_data['response']
            )
            verdict_obj = await judge_llm.generate_structured(judge_prompt, response_model=verdict_model)
            
            judgement_dict = verdict_obj if isinstance(verdict_obj, dict) else verdict_obj.model_dump()

            return q_id, {
                "judgement": judgement_dict,
                "prediction_data": pred_data,
                "correct_answer": question_data['answer']
            }

    tasks = [judge_item(q_id, pred_data) for q_id, pred_data in items_to_judge.items()]
    print(f"Judging {len(tasks)} items with {args.num_workers} concurrent workers...")
    results = await tqdm_asyncio.gather(*tasks)
    for q_id, judged_data in results:
        if q_id:
            verdicts[q_id] = judged_data

    correct_flags = []
    confidences = []
    for res in verdicts.values():
        judgement = res.get('judgement', {})
        if is_official_judge:
            is_correct = judgement.get('correct') == 'yes'
            confidences.append(judgement.get('confidence', 100))
        else:
            is_correct = judgement.get('is_correct') is True
        
        if is_correct is not None:
             correct_flags.append(is_correct)

    total_judged = len(correct_flags)
    correct_count = sum(correct_flags)
    accuracy = (correct_count / total_judged) * 100 if total_judged > 0 else 0
    
    summary_metrics = {
        "accuracy": round(accuracy, 2),
        "correct_count": correct_count,
        "total_judged": total_judged,
    }

    if is_official_judge and total_judged > 0:
        p_hat = accuracy / 100.0
        ci_half_width = 1.96 * math.sqrt((p_hat * (1 - p_hat)) / total_judged)
        summary_metrics["accuracy_ci_95"] = round(ci_half_width * 100, 2)
        
        conf_array = np.array(confidences) / 100.0
        correct_array = np.array(correct_flags)
        ece = calculate_ece(conf_array, correct_array)
        summary_metrics["expected_calibration_error"] = round(ece * 100, 2)

    final_judged_data = {
        "metadata": {
            **generation_metadata,
            "source_file": args.input_file.name,
            "judged_by": args.judge_model,
            "judge_prompt_source": args.judge_prompt_source,
            "judge_prompt_source_type": judge_prompt_type,
            "judge_prompt_file": str(judge_prompt_path),
            "judged_at_utc": timestamp,
        },
        "summary_metrics": summary_metrics,
        "verdicts": verdicts,
    }
    
    with open(judged_filepath, "w", encoding='utf-8') as f:
        json.dump(final_judged_data, f, indent=2)
    print(f"\nJudged artifact with summary saved to {judged_filepath.resolve()}")

    print("\n--- Final Score ---")
    print(f"Model: {generation_metadata['model']}")
    print(f"Prompt Source: {generation_metadata['prompt_source']}")
    print(f"Judged By: {args.judge_model} (with prompt '{args.judge_prompt_source}')")
    
    if is_official_judge and total_judged > 0:
        print(f"Accuracy: {summary_metrics['accuracy']}% +/- {summary_metrics['accuracy_ci_95']}% (CI 95%)")
        print(f"Correct: {summary_metrics['correct_count']} / {summary_metrics['total_judged']}")
        print(f"Expected Calibration Error (ECE): {summary_metrics['expected_calibration_error']}%")
    else:
        print(f"Accuracy: {summary_metrics['accuracy']}% ({summary_metrics['correct_count']}/{summary_metrics['total_judged']} correct)")
    
    if was_mismatched:
        print("\n⚠️  NOTE: This score was generated from a text-only run on a multi-modal benchmark.")
        print("   The results may not be directly comparable to standard, multi-modal evaluations.")


def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()