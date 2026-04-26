import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "metamath_gsm8k_benchmark.py"
SPEC = importlib.util.spec_from_file_location("metamath_gsm8k_benchmark", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

extract_answer = benchmark.extract_answer
score_predictions = benchmark.score_predictions


def test_extract_answer_handles_gsm8k_numeric_forms():
    assert extract_answer("#### 42") == "42"
    assert extract_answer("The answer is 1,234.") == "1234"
    assert extract_answer("So x = -17") == "-17"


def test_score_predictions_uses_continuation_answers_only():
    predictions = [
        "We compute it. The answer is 42.",
        "No final number here",
        "Prompt said #### 5. But the continuation answer is 6.",
    ]
    references = [
        "reasoning\n#### 42",
        "reasoning\n#### 9",
        "reasoning\n#### 5",
    ]
    scored = score_predictions(predictions, references)
    assert scored["correct"] == 1
    assert scored["total"] == 3
    assert scored["accuracy"] == 1 / 3