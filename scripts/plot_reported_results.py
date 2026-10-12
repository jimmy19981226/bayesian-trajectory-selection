"""Plot archived counts; does not run a model or imply a fresh benchmark."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'results/reported/results.json').read_text())
for section in ('math500_subjects', 'math500_levels'):
    assert sum(r['total'] for r in data[section]) == 500
    assert sum(r['baseline_correct'] for r in data[section]) == 231
    assert sum(r['mi_correct'] for r in data[section]) == 232

fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
x = np.arange(2)
for offset, key, label, color in [(-.18, 'baseline_correct', 'Total entropy', '#61769B'),
                                  (.18, 'mi_correct', 'Epistemic MI', '#317F78')]:
    values = [100 * r[key] / r['total'] for r in data['overall']]
    bars = ax.bar(x + offset, values, width=.34, label=label, color=color)
    ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=5, fontsize=11)
ax.set(xticks=x, xticklabels=['MATH-500', 'GSM8K'], ylim=(0, 100), ylabel='Strict ACC@1 (%)',
       title='Reported benchmark accuracy · CS 677 final paper')
ax.set_axisbelow(True)
ax.grid(axis='y', color='#E5E7EB')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', frameon=False)
fig.supxlabel('Historical results transcribed from the paper; not a restoration rerun.', fontsize=10, color='#555555')
destination = root / 'docs/assets/reported-accuracy.png'
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=180)
print(destination)
