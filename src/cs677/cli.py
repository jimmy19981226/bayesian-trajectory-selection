import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description='CS 677 Bayesian trajectory selection (reconstructed)')
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'generate', 'score', 'select', 'train', 'evaluate', 'demo'):
        s = sub.add_parser(command)
        s.add_argument('--output', required=command != 'demo', default='runs/demo')
        if command in ('prepare', 'generate', 'score', 'evaluate'):
            s.add_argument('--start', type=int, default=0)
            s.add_argument('--limit', type=int)
        if command in ('generate', 'score', 'train', 'evaluate'):
            s.add_argument('--model', default='Qwen/Qwen2.5-72B-Instruct' if command == 'generate' else 'Qwen/Qwen2.5-3B')
            s.add_argument('--revision')
            s.add_argument('--quantize', action='store_true')
            s.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
            s.add_argument('--seed', type=int, default=42)
        if command in ('generate', 'score', 'select'):
            s.add_argument('--input', required=True)
        if command in ('generate', 'select'):
            s.add_argument('--candidates', type=int, default=5)
        if command in ('generate', 'evaluate'):
            s.add_argument('--max-new-tokens', type=int, default=2048)
        if command in ('score', 'train'):
            s.add_argument('--max-length', type=int, default=4096)
        if command == 'prepare':
            s.add_argument('--revision')
            s.add_argument('--math-revision')
        elif command == 'generate':
            s.add_argument('--temperature', type=float, default=0.7)
            s.add_argument('--top-p', type=float, default=0.95)
        elif command == 'score':
            s.add_argument('--passes', type=int, default=10)
            s.add_argument('--dropout', type=float, default=0.1)
            s.add_argument('--top-m', type=int, default=50)
            s.add_argument('--token-chunk-size', type=int, default=32)
        elif command == 'train':
            s.add_argument('--train-file', required=True)
            s.add_argument('--rank', type=int, default=16)
            s.add_argument('--alpha', type=int, default=32)
            s.add_argument('--lora-dropout', type=float, default=0.05)
            s.add_argument('--epochs', type=float, default=2)
            s.add_argument('--batch-size', type=int, default=4)
            s.add_argument('--gradient-accumulation', type=int, default=4)
            s.add_argument('--learning-rate', type=float, default=2e-4)
        elif command == 'evaluate':
            s.add_argument('--adapter')
            s.add_argument('--benchmark', choices=['math500', 'gsm8k'], required=True)
            s.add_argument('--dataset-revision')
    return p


def demo(args):
    import numpy as np
    from .uncertainty import decompose, top_m_mean
    from .selection import select_winners, chi_squared_weights
    from .evaluation import first_boxed_answer
    from .io import write_json, write_jsonl
    if Path(args.output).exists():
        raise FileExistsError('Choose a new demo output folder')
    # Candidate 0: high within-pass entropy, zero disagreement.
    # Candidate 1: confident but disagreeing passes. Deliberately synthetic.
    distributions = [np.array([[[.5, .5]], [[.5, .5]]]),
                     np.array([[[.99, .01]], [[.1, .9]]])]
    rows = []
    for i, probabilities in enumerate(distributions):
        u = decompose(probabilities)
        rows.append({'problem_id': 'synthetic-0', 'candidate_id': str(i), 'problem': 'What is 1 + 1?',
                     'response': r'Add one and one to obtain \boxed{2}.',
                     'total_entropy': top_m_mean(u['total_entropy']),
                     'mutual_information': top_m_mean(u['mutual_information'])})
    baseline, epistemic, report = select_winners(rows, expected_candidates=2)
    report.update(status='synthetic demonstration; not benchmark results',
                  first_boxed_answer=first_boxed_answer(rows[0]['response']),
                  optional_proposal_weights=chi_squared_weights([r['mutual_information'] for r in rows]).tolist())
    for name, values in [('scored', rows), ('baseline', baseline), ('epistemic', epistemic)]:
        write_jsonl(Path(args.output) / f'{name}.jsonl', values)
    write_json(Path(args.output) / 'summary.json', report)
    print(report)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for key in ('limit', 'candidates', 'max_new_tokens', 'max_length', 'top_m', 'token_chunk_size',
                'batch_size', 'gradient_accumulation', 'rank', 'alpha', 'epochs', 'learning_rate', 'temperature'):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if getattr(args, 'start', 0) < 0:
        p.error('--start must be nonnegative')
    if getattr(args, 'passes', 2) < 2:
        p.error('--passes must be at least two')
    if not 0 < getattr(args, 'dropout', .1) < 1:
        p.error('--dropout must be between zero and one')
    if not 0 <= getattr(args, 'lora_dropout', .05) < 1:
        p.error('--lora-dropout must be in [0, 1)')
    if not 0 < getattr(args, 'top_p', .95) <= 1:
        p.error('--top-p must be in (0, 1]')
    if args.command == 'demo':
        demo(args)
    else:
        from . import pipeline
        getattr(pipeline, args.command)(args)
