import json
from pathlib import Path
import numpy as np
import pytest
from cs677.uncertainty import entropy, decompose, top_m_mean
from cs677.selection import chi_squared_weights, select_winners
from cs677.evaluation import first_boxed_answer, reference_answer, summarize_predictions
from cs677.pipeline import assert_no_overlap
from cs677.cli import main


def test_known_entropy_and_zero_probability():
    np.testing.assert_allclose(entropy([[1, 0], [.5, .5]]), [0, np.log(2)])


def test_disagreement_separates_epistemic_from_aleatoric():
    agreement = decompose([[[.5, .5]], [[.5, .5]]])
    disagreement = decompose([[[1, 0]], [[0, 1]]])
    assert agreement['mutual_information'][0] == 0
    np.testing.assert_allclose(disagreement['mutual_information'], [np.log(2)])
    np.testing.assert_allclose(agreement['total_entropy'], disagreement['total_entropy'])


@pytest.mark.parametrize('p', [[[.2, .2]], [[-1, 2]], [[float('nan'), 1]]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError):
        entropy(p)


def test_top_m_short_trace_and_order():
    assert top_m_mean([1, 8, 3, 6], 2) == 7
    assert top_m_mean([1, 3], 50) == 2
    with pytest.raises(ValueError):
        top_m_mean([], 50)


def candidates():
    return [{'problem_id': 'p', 'problem': 'question', 'candidate_id': str(i),
             'total_entropy': a, 'mutual_information': b}
            for i, (a, b) in enumerate([(1, .1), (.5, .3)])]


def test_winners_and_disagreement():
    a, b, stats = select_winners(candidates(), 2)
    assert a[0]['candidate_id'] == '0' and b[0]['candidate_id'] == '1'
    assert stats['disagreement_percent'] == 100


def test_selection_incomplete_duplicate_nonfinite_and_ties():
    with pytest.raises(ValueError):
        select_winners(candidates(), 5)
    with pytest.raises(ValueError):
        select_winners(candidates() * 2, 4)
    rows = candidates()
    rows[1]['total_entropy'] = float('nan')
    with pytest.raises(ValueError):
        select_winners(rows, 2)
    rows = candidates()
    rows[1]['total_entropy'] = 1
    assert select_winners(rows[::-1], 2)[0][0]['candidate_id'] == '0'


def test_chi_squared_kkt_and_invariance():
    np.testing.assert_allclose(chi_squared_weights([4, 4]), [.5, .5])
    np.testing.assert_allclose(chi_squared_weights([0, 1]), [.25, .75])
    np.testing.assert_allclose(chi_squared_weights([0, 10]), [0, 1])
    np.testing.assert_allclose(chi_squared_weights([2, 2], prior=[.2, .8]), [.2, .8])
    u = np.array([.1, .7, .4])
    np.testing.assert_allclose(chi_squared_weights(u), chi_squared_weights(u + 100))
    q = chi_squared_weights(u)
    # Interior KKT derivative is the same across all coordinates.
    gradient = u - (q - 1/3) / (1/3)
    np.testing.assert_allclose(gradient, np.repeat(gradient[0], 3))


@pytest.mark.parametrize('text, expected', [
    (r'First \boxed{\frac{1}{2}} then \boxed{9}', r'\frac{1}{2}'),
    (r'\boxed{\{1,2\}}', r'\{1,2\}'),
    (r'No box: 2', None), (r'\boxed{unclosed', None),
    (r'\boxed{ 2 }', ' 2 '), (r'\boxed{2}', '2')])
def test_first_boxed_answer(text, expected):
    assert first_boxed_answer(text) == expected


def test_strict_metric_and_reference():
    assert first_boxed_answer(r'\boxed{0.5}') != r'\frac{1}{2}'
    assert reference_answer({'answer': 'reasoning\n#### 12'}, 'gsm8k') == '12'
    result = summarize_predictions([{'correct': True, 'subject': 'Algebra', 'level': 1},
                                    {'correct': False, 'subject': 'Algebra', 'level': 2}])
    assert result['overall']['accuracy_percent'] == 50


def test_overlap_check_normalizes_whitespace():
    with pytest.raises(ValueError, match='overlap'):
        assert_no_overlap([{'problem_id': '1', 'problem': 'A  B'}], [{'problem': 'A\nB'}])
    assert_no_overlap([{'problem_id': '1', 'problem': 'C'}], [{'problem': 'A B'}])


def test_offline_cli_demo(tmp_path):
    destination = tmp_path / 'demo'
    main(['demo', '--output', str(destination)])
    report = json.loads((destination / 'summary.json').read_text())
    assert report['different_winners'] == 1
    assert 'synthetic' in report['status']


def test_reported_counts_and_original_checksums():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / 'results/reported/results.json').read_text())
    for section in ('math500_subjects', 'math500_levels'):
        assert sum(r['total'] for r in data[section]) == 500
        assert sum(r['baseline_correct'] for r in data[section]) == 231
        assert sum(r['mi_correct'] for r in data[section]) == 232
    for source in json.loads((root / 'sources/manifest.json').read_text()):
        assert hashlib.sha256((root / source['archived_path']).read_bytes()).hexdigest() == source['sha256']
