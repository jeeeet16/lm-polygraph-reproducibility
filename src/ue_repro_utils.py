import json
import re
import string
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


METHOD_SPECS = {
    "seq_prob": {
        "label": "Seq Prob PRR",
        "long_label": "Maximum Sequence Probability",
        "result_key": "seq_prob_prr",
    },
    "perplexity": {
        "label": "Perplexity PRR",
        "long_label": "Perplexity",
        "result_key": "perplexity_prr",
    },
    "token_entropy": {
        "label": "Token Entropy PRR",
        "long_label": "Mean Token Entropy",
        "result_key": "token_entropy_prr",
    },
    "lexical_similarity": {
        "label": "Lexical Similarity PRR",
        "long_label": "Lexical Similarity",
        "result_key": "lexical_similarity_prr",
    },
    "eccentricity": {
        "label": "Eccentricity PRR",
        "long_label": "Eccentricity",
        "result_key": "eccentricity_prr",
    },
}

SAMPLING_INVARIANT_METHODS: Tuple[str, ...] = (
    "seq_prob",
    "perplexity",
    "token_entropy",
)

SAMPLING_DEPENDENT_METHODS: Tuple[str, ...] = (
    "lexical_similarity",
    "eccentricity",
)

DEFAULT_METHODS: Tuple[str, ...] = (
    "seq_prob",
    "token_entropy",
    "eccentricity",
)

EXPANDED_METHODS: Tuple[str, ...] = (
    "seq_prob",
    "perplexity",
    "token_entropy",
    "lexical_similarity",
    "eccentricity",
)


def set_reproducibility(seed: int = 42) -> None:
    """Set deterministic seeds for numpy/torch where possible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/extra spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 with SQuAD-style normalization and token multiplicity."""
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_coqa(num_samples: int, seed: int = 42) -> List[dict]:
    """
    Load CoQA validation split and return unbiased sampled prompt/reference pairs.
    Sampling is done across the full QA pool (not only the first few stories).
    """
    from datasets import load_dataset

    raw = load_dataset("coqa", split="validation")
    examples = []

    for story in raw:
        passage = story["story"]
        questions = story["questions"]
        answers = story["answers"]["input_text"]

        for i, (question, answer) in enumerate(zip(questions, answers)):
            history = ""
            for j in range(i):
                history += f"Q: {questions[j]}\nA: {answers[j]}\n"

            prompt = (
                "Read the passage and answer the question concisely.\n\n"
                f"Passage: {passage}\n\n"
                f"{history}Q: {question}\nA:"
            )
            examples.append({"prompt": prompt, "reference": answer.strip()})

    rng = np.random.default_rng(seed)
    sample_size = min(num_samples, len(examples))
    indices = rng.choice(len(examples), size=sample_size, replace=False)
    return [examples[i] for i in indices]


def compute_prr(
    uncertainty_scores: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Prediction Rejection Ratio.
    Higher score means better ranking of uncertain predictions.
    """
    n = len(uncertainty_scores)
    if n == 0 or n != len(correctness):
        raise ValueError("uncertainty_scores and correctness must be non-empty and same length.")

    order = np.argsort(uncertainty_scores)
    sorted_correct = correctness[order]

    oracle_order = np.argsort(correctness)  # 0 (wrong) first, 1 (correct) last
    oracle_sorted = correctness[oracle_order]

    thresholds = np.linspace(0, 1, n_bins + 1)[:-1]
    model_accs = []
    oracle_accs = []
    for frac in thresholds:
        n_keep = max(1, int(n * (1 - frac)))
        model_accs.append(sorted_correct[:n_keep].mean())
        oracle_accs.append(oracle_sorted[n - n_keep :].mean())

    baseline_acc = correctness.mean()
    oracle_gain = float(np.mean(oracle_accs) - baseline_acc)
    if oracle_gain <= 0:
        return 0.0

    model_gain = float(np.mean(model_accs) - baseline_acc)
    return model_gain / oracle_gain


def _polygraph_imports():
    try:
        from lm_polygraph.defaults.register_default_stat_calculators import (
            register_default_stat_calculators,
        )
        from lm_polygraph.estimators import (
            Eccentricity,
            LexicalSimilarity,
            MaximumSequenceProbability,
            MeanTokenEntropy,
            Perplexity,
        )
        from lm_polygraph.utils.builder_enviroment_stat_calculator import (
            BuilderEnvironmentStatCalculator,
        )
        from lm_polygraph.utils.dataset import Dataset
        from lm_polygraph.utils.generation_parameters import GenerationParameters
        from lm_polygraph.utils.manager import UEManager
        from lm_polygraph.utils.model import WhiteboxModel
    except ImportError as exc:
        raise ImportError(
            "lm-polygraph is required for these notebooks now. "
            "Install it with `pip install lm-polygraph` or `pip install -r requirements.txt`."
        ) from exc

    return {
        "BuilderEnvironmentStatCalculator": BuilderEnvironmentStatCalculator,
        "Dataset": Dataset,
        "Eccentricity": Eccentricity,
        "GenerationParameters": GenerationParameters,
        "LexicalSimilarity": LexicalSimilarity,
        "MaximumSequenceProbability": MaximumSequenceProbability,
        "MeanTokenEntropy": MeanTokenEntropy,
        "Perplexity": Perplexity,
        "UEManager": UEManager,
        "WhiteboxModel": WhiteboxModel,
        "register_default_stat_calculators": register_default_stat_calculators,
    }


def get_supported_methods() -> Dict[str, str]:
    """Return the supported lm-polygraph method names and human-readable labels."""
    return {name: spec["long_label"] for name, spec in METHOD_SPECS.items()}


def get_method_result_key(method_name: str) -> str:
    """Return the result key used in saved payloads for a given method."""
    return METHOD_SPECS[method_name]["result_key"]


def get_method_plot_label(method_name: str) -> str:
    """Return the short display label used in tables and plots."""
    return METHOD_SPECS[method_name]["label"]


def _split_methods_by_sampling_dependence(
    method_names: Sequence[str],
) -> Tuple[List[str], List[str]]:
    invariant = []
    dependent = []
    for name in method_names:
        if name in SAMPLING_INVARIANT_METHODS:
            invariant.append(name)
        elif name in SAMPLING_DEPENDENT_METHODS:
            dependent.append(name)
        else:
            raise ValueError(f"Unknown sampling dependence for method '{name}'.")
    return invariant, dependent


def _build_estimators(method_names: Sequence[str]):
    api = _polygraph_imports()
    registry = {
        "seq_prob": {
            "result_key": METHOD_SPECS["seq_prob"]["result_key"],
            "factory": api["MaximumSequenceProbability"],
        },
        "perplexity": {
            "result_key": METHOD_SPECS["perplexity"]["result_key"],
            "factory": api["Perplexity"],
        },
        "token_entropy": {
            "result_key": METHOD_SPECS["token_entropy"]["result_key"],
            "factory": api["MeanTokenEntropy"],
        },
        "lexical_similarity": {
            "result_key": METHOD_SPECS["lexical_similarity"]["result_key"],
            "factory": api["LexicalSimilarity"],
        },
        "eccentricity": {
            "result_key": METHOD_SPECS["eccentricity"]["result_key"],
            "factory": api["Eccentricity"],
        },
    }

    estimators = []
    metadata = []
    for name in method_names:
        if name not in registry:
            raise ValueError(
                f"Unsupported method '{name}'. Supported methods: {sorted(registry)}"
            )
        spec = registry[name]
        estimator = spec["factory"]()
        estimators.append(estimator)
        metadata.append({"name": name, "result_key": spec["result_key"], "estimator": estimator})
    return estimators, metadata


def _prepare_generation_defaults(model, tokenizer) -> None:
    """
    Sanitize model-level generation defaults in place so inherited config values
    do not trigger repeated warnings on every generate() call.
    """
    if getattr(model, "generation_config", None) is None:
        return

    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = None
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id


def load_model(
    model_id: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """
    Load a Hugging Face causal LM and wrap it in lm-polygraph's WhiteboxModel.
    Uses 4-bit quantization on GPU when available and supported.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = _polygraph_imports()
    GenerationParameters = api["GenerationParameters"]
    WhiteboxModel = api["WhiteboxModel"]

    print(f"Loading {model_id} with lm-polygraph ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = {
        "trust_remote_code": True,
    }

    model = None
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                **common_kwargs,
            )
            print("  -> using 4-bit quantization")
        except Exception as exc:
            print(f"  -> 4-bit load unavailable ({exc})")
            print("  -> falling back to standard GPU loading")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                **common_kwargs,
            )

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            **common_kwargs,
        )

    _prepare_generation_defaults(model, tokenizer)
    model.eval()

    generation_parameters = GenerationParameters(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=False,
    )
    polygraph_model = WhiteboxModel(
        model=model,
        tokenizer=tokenizer,
        model_path=model_id,
        model_type="CausalLM",
        generation_parameters=generation_parameters,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  -> loaded ({n_params:.1f}B params)")
    return polygraph_model, tokenizer


def _configure_sampling_stat_calculators(stat_calculators, n_stochastic: int):
    for sc in stat_calculators:
        if sc.name in {"SamplingGenerationCalculator", "BlackboxSamplingGenerationCalculator"}:
            sc.cfg["samples_n"] = int(n_stochastic)
    return stat_calculators


def _set_generation_parameters(
    model,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    params = model.generation_parameters
    params.max_new_tokens = max_new_tokens
    params.temperature = temperature
    params.top_p = top_p
    params.do_sample = True
    params.num_beams = 1

def _build_manager(
    model,
    data: List[dict],
    estimators,
    *,
    max_new_tokens: int,
    n_stochastic: int,
    temperature: float,
    top_p: float,
    batch_size: int,
):
    api = _polygraph_imports()
    Dataset = api["Dataset"]
    UEManager = api["UEManager"]
    BuilderEnvironmentStatCalculator = api["BuilderEnvironmentStatCalculator"]
    register_default_stat_calculators = api["register_default_stat_calculators"]

    _set_generation_parameters(
        model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    dataset = Dataset(
        [ex["prompt"] for ex in data],
        [ex["reference"] for ex in data],
        batch_size=batch_size,
    )

    available_stat_calculators = register_default_stat_calculators(
        "Whitebox",
        output_attentions=False,
        output_hidden_states=False,
    )
    available_stat_calculators = _configure_sampling_stat_calculators(
        available_stat_calculators,
        n_stochastic=n_stochastic,
    )

    return UEManager(
        data=dataset,
        model=model,
        estimators=estimators,
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        available_stat_calculators=available_stat_calculators,
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=False,
        verbose=True,
        max_new_tokens=max_new_tokens,
        save_stats=["greedy_texts"],
    )


def run_evaluation(
    model,
    tokenizer,
    data: List[dict],
    max_new_tokens: int,
    n_stochastic: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    f1_threshold: float = 0.3,
    method_names: Sequence[str] = DEFAULT_METHODS,
    batch_size: int = 1,
) -> Dict:
    """
    Evaluate lm-polygraph estimators on the provided dataset and return PRR scores.
    """
    del tokenizer  # kept for notebook compatibility

    estimators, metadata = _build_estimators(method_names)
    manager = _build_manager(
        model,
        data,
        estimators,
        max_new_tokens=max_new_tokens,
        n_stochastic=n_stochastic,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )
    manager()

    generations = list(manager.stats["greedy_texts"])
    method_scores = {}
    for meta in metadata:
        estimator = meta["estimator"]
        key = (estimator.level, str(estimator))
        if key not in manager.estimations:
            raise KeyError(f"Missing lm-polygraph output for estimator {key}")
        method_scores[meta["result_key"]] = np.asarray(manager.estimations[key], dtype=float)

    records = []
    for i, ex in enumerate(data):
        generation = generations[i]
        f1 = token_f1(generation, ex["reference"])
        correct = int(f1 >= f1_threshold)
        record = {
            "prompt": ex["prompt"],
            "reference": ex["reference"],
            "generation": generation,
            "f1": f1,
            "correct": correct,
        }
        for result_key, values in method_scores.items():
            record[result_key.replace("_prr", "")] = float(values[i])
        records.append(record)

    corr = np.array([r["correct"] for r in records])
    results = {
        "mean_f1": float(np.mean([r["f1"] for r in records])) if records else 0.0,
        "raw": records,
        "method_prrs": {},
    }

    for result_key, values in method_scores.items():
        prr = compute_prr(values, corr)
        results["method_prrs"][result_key] = prr
        results[result_key] = prr

    return results


def _run_greedy_generations(
    model,
    data: List[dict],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    manager = _build_manager(
        model,
        data,
        estimators=[],
        max_new_tokens=max_new_tokens,
        n_stochastic=1,
        temperature=0.7,
        top_p=0.95,
        batch_size=batch_size,
    )
    manager()
    return list(manager.stats["greedy_texts"])


def sweep_eccentricity_prr(
    model,
    tokenizer,
    data: List[dict],
    temperatures: List[float],
    top_p_values: List[float],
    n_samples_range: List[int],
    max_new_tokens: int,
    f1_threshold: float = 0.3,
    batch_size: int = 1,
) -> Dict[str, Dict[float, float]]:
    """
    Sweep lm-polygraph Eccentricity PRR over sampling hyperparameters.
    """
    del tokenizer  # kept for notebook compatibility

    greedy_generations = _run_greedy_generations(
        model,
        data,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    labels = np.array(
        [
            int(token_f1(generation, ex["reference"]) >= f1_threshold)
            for generation, ex in zip(greedy_generations, data)
        ]
    )

    def _ecc_prr(temperature: float, top_p: float, n_samples: int) -> float:
        estimator, _ = _build_estimators(["eccentricity"])
        estimator = estimator[0]
        manager = _build_manager(
            model,
            data,
            estimators=[estimator],
            max_new_tokens=max_new_tokens,
            n_stochastic=n_samples,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        manager()
        key = (estimator.level, str(estimator))
        scores = np.asarray(manager.estimations[key], dtype=float)
        return compute_prr(scores, labels)

    temp_prrs = {t: _ecc_prr(t, 0.95, 5) for t in temperatures}
    topp_prrs = {tp: _ecc_prr(0.7, tp, 5) for tp in top_p_values}
    n_prrs = {n: _ecc_prr(0.7, 0.95, n) for n in n_samples_range}

    return {"temperature": temp_prrs, "top_p": topp_prrs, "n_samples": n_prrs}


def sweep_method_prrs(
    model,
    tokenizer,
    data: List[dict],
    method_names: Sequence[str],
    temperatures: List[float],
    top_p_values: List[float],
    n_samples_range: List[int],
    max_new_tokens: int,
    f1_threshold: float = 0.3,
    batch_size: int = 1,
    default_temperature: float = 0.7,
    default_top_p: float = 0.95,
    default_n_samples: int = 5,
) -> Dict[str, Dict[str, Dict[float, float]]]:
    """
    Sweep PRR scores for multiple lm-polygraph estimators across sampling settings.

    Greedy-only methods are computed once and repeated across sweep settings, while
    sample-based methods are recomputed for each setting.
    """
    del tokenizer  # kept for notebook compatibility

    invariant_methods, dependent_methods = _split_methods_by_sampling_dependence(method_names)

    results = {
        "temperature": {method: {} for method in method_names},
        "top_p": {method: {} for method in method_names},
        "n_samples": {method: {} for method in method_names},
    }

    if invariant_methods:
        invariant_eval = run_evaluation(
            model=model,
            tokenizer=None,
            data=data,
            max_new_tokens=max_new_tokens,
            n_stochastic=default_n_samples,
            temperature=default_temperature,
            top_p=default_top_p,
            f1_threshold=f1_threshold,
            method_names=invariant_methods,
            batch_size=batch_size,
        )

        for method in invariant_methods:
            result_key = get_method_result_key(method)
            invariant_value = invariant_eval[result_key]
            for temperature in temperatures:
                results["temperature"][method][temperature] = invariant_value
            for top_p in top_p_values:
                results["top_p"][method][top_p] = invariant_value
            for n_samples in n_samples_range:
                results["n_samples"][method][n_samples] = invariant_value

    def _dependent_eval(
        *,
        temperature: float,
        top_p: float,
        n_samples: int,
    ) -> Dict[str, float]:
        if not dependent_methods:
            return {}
        evaluation = run_evaluation(
            model=model,
            tokenizer=None,
            data=data,
            max_new_tokens=max_new_tokens,
            n_stochastic=n_samples,
            temperature=temperature,
            top_p=top_p,
            f1_threshold=f1_threshold,
            method_names=dependent_methods,
            batch_size=batch_size,
        )
        return {
            method: evaluation[get_method_result_key(method)]
            for method in dependent_methods
        }

    for temperature in temperatures:
        dependent_values = _dependent_eval(
            temperature=temperature,
            top_p=default_top_p,
            n_samples=default_n_samples,
        )
        for method, value in dependent_values.items():
            results["temperature"][method][temperature] = value

    for top_p in top_p_values:
        dependent_values = _dependent_eval(
            temperature=default_temperature,
            top_p=top_p,
            n_samples=default_n_samples,
        )
        for method, value in dependent_values.items():
            results["top_p"][method][top_p] = value

    for n_samples in n_samples_range:
        dependent_values = _dependent_eval(
            temperature=default_temperature,
            top_p=default_top_p,
            n_samples=n_samples,
        )
        for method, value in dependent_values.items():
            results["n_samples"][method][n_samples] = value

    return results


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
