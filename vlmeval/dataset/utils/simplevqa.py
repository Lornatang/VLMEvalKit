import re

model_keys = ["model_response"]
mapper = {"正确": "is_correct", "错误": "is_incorrect", "未尝试": "is_not_attempted"}

_PUNCTUATION = re.compile(r"[\s`*\"'“”‘’\[\](){}<>。．.，,！!？?：:；;、\-_/\\]+")
_VERDICT_ALIASES = {
    "正确": "正确",
    "correct": "正确",
    "a": "正确",
    "错误": "错误",
    "incorrect": "错误",
    "wrong": "错误",
    "b": "错误",
    "未尝试": "未尝试",
    "notattempted": "未尝试",
    "noattempt": "未尝试",
    "c": "未尝试",
}
_CHINESE_VERDICTS = ("未尝试", "正确", "错误")


def normalize_verdict(value):
    """Normalize a judge verdict to one of 正确 / 错误 / 未尝试.

    Accepts the canonical Chinese labels, the A/B/C letters requested by the
    prompt, common English labels, and tolerates surrounding punctuation or
    Markdown emphasis. Returns ``None`` when no verdict can be recognized.
    """
    if not isinstance(value, str):
        return None

    text = _PUNCTUATION.sub("", value).lower()
    if not text:
        return None

    if text in _VERDICT_ALIASES:
        return _VERDICT_ALIASES[text]

    # "不正确" contains "正确", so rule it out before prefix matching.
    if text.startswith(("不正确", "不对", "不是")):
        return "错误"

    # Tolerate short explanations such as "结论：正确".
    for verdict in _CHINESE_VERDICTS:
        if text.startswith(verdict) or text.endswith(verdict):
            return verdict

    for alias in ("notattempted", "noattempt"):
        if text.startswith(alias) or text.endswith(alias):
            return "未尝试"

    return None


def SimpleVQAEval(data):
    amount = len(data)
    result_metrics = {}

    for line in data:
        judge_res = line.get("judge_res") if isinstance(line, dict) else None
        if not isinstance(judge_res, dict) or not judge_res:
            judge_res = {"model_response": None}

        for key, value in judge_res.items():
            if isinstance(value, dict):
                value = value.get("conclusion")
            verdict = normalize_verdict(value)

            metrics = result_metrics.setdefault(
                key,
                {
                    "is_correct": 0,
                    "is_incorrect": 0,
                    "is_not_attempted": 0,
                    "is_failed": 0,
                },
            )
            if verdict is None:
                metrics["is_failed"] += 1
            else:
                metrics[mapper[verdict]] += 1

    result_metrics.setdefault(
        "model_response",
        {"is_correct": 0, "is_incorrect": 0, "is_not_attempted": 0, "is_failed": amount},
    )

    print("ALL data count: ", amount)
    divisor = amount if amount else 1
    aggregates = {}

    for model_key, metrics in result_metrics.items():
        aggregate_metrics = {
            "LVLM_name": model_key,
            "is_correct": round(metrics["is_correct"] / divisor, 4),
            "is_incorrect": round(metrics["is_incorrect"] / divisor, 4),
            "is_not_attempted": round(metrics["is_not_attempted"] / divisor, 4),
            "is_failed": round(metrics["is_failed"] / divisor, 4),
        }
        aggregate_metrics["is_given_attempted"] = round(
            aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"], 4)

        denominator = aggregate_metrics["is_given_attempted"]
        aggregate_metrics["accuracy_given_attempted"] = (
            aggregate_metrics["is_correct"] / denominator if denominator > 0 else 0
        )

        f1_denominator = (
            aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"]
        )
        aggregate_metrics["f1"] = (
            2 * aggregate_metrics["accuracy_given_attempted"] * aggregate_metrics["is_correct"]
            / f1_denominator
            if f1_denominator > 0
            else 0
        )

        aggregates[model_key] = aggregate_metrics
        print(model_key + ": AGGREGATE METRICS")
        print(aggregate_metrics)
        print("##################")

        print(f"Accuracy Given Attempted: {aggregate_metrics['accuracy_given_attempted']:.3f}")
        print(f"F1 Score: {aggregate_metrics['f1']:.3f}")

    return aggregates.get("model_response", next(iter(aggregates.values())))
