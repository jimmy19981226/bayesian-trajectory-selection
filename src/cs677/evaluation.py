"""The paper's strict first-boxed-answer metric, with nested LaTeX braces."""

from collections import defaultdict


def first_boxed_answer(text):
    marker = r"\boxed{"
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    depth = 1
    for i in range(start, len(text)):
        # Escaped braces are literal LaTeX characters, not grouping braces.
        backslashes = 0
        j = i - 1
        while j >= 0 and text[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2:
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def reference_answer(row, benchmark):
    if benchmark == "gsm8k":
        if "####" not in row['answer']:
            raise ValueError("GSM8K reference is missing ####")
        return row['answer'].rsplit('####', 1)[1].strip()
    return str(row['answer'])


def summarize_predictions(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("No predictions to evaluate")
    def stats(items):
        correct = sum(bool(r['correct']) for r in items)
        return {"correct": correct, "total": len(items), "accuracy_percent": 100 * correct / len(items)}
    result = {"overall": stats(rows)}
    for field in ('subject', 'level'):
        groups = defaultdict(list)
        for row in rows:
            if row.get(field) is not None:
                groups[str(row[field])].append(row)
        result[field] = {k: stats(v) for k, v in sorted(groups.items())}
    return result
