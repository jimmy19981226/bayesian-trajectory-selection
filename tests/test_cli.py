"""Reject invalid run settings before downloading models or writing outputs."""

import pytest

from cs677.cli import main


@pytest.mark.parametrize('command, required, option', [
    ('generate', ['--input', 'missing.jsonl'], '--temperature'),
    ('train', ['--train-file', 'missing.jsonl'], '--epochs'),
    ('train', ['--train-file', 'missing.jsonl'], '--learning-rate'),
])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '0', '-1'])
def test_invalid_positive_settings_fail_before_dispatch(
    command, required, option, value, tmp_path, monkeypatch, capsys
):
    from cs677 import pipeline

    def unexpected_dispatch(args):
        pytest.fail('Invalid settings reached the model pipeline')

    monkeypatch.setattr(pipeline, command, unexpected_dispatch)
    destination = tmp_path / 'output'
    with pytest.raises(SystemExit) as error:
        main([command, *required, '--output', str(destination), f'{option}={value}'])
    assert error.value.code == 2
    assert f'{option} must be finite and positive' in capsys.readouterr().err
    assert not destination.exists()
