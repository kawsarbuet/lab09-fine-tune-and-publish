"""Offline data and baseline checks."""
import pytest

from baselines import fit_baselines
from prepare import sample_train, split_tickets
from tickets_data import LABELS


@pytest.mark.parametrize('seed', [42, 123, 2026])
def test_split(seed):
    train, val, test = split_tickets(seed)
    assert [len(train), len(val), len(test)] == [720, 240, 240]
    ids = [set(part.ticket_id) for part in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(set.union(*ids)) == 1200


def test_sample():
    train, _, _ = split_tickets()
    for n in (50, 100, 300):
        rows = sample_train(train, n, 42)
        assert len(rows) == n
        assert set(rows.label) == set(LABELS)
        assert set(rows.ticket_id) <= set(train.ticket_id)
        assert rows.equals(sample_train(train, n, 42))


def test_baseline():
    train, val, _ = split_tickets()
    models = fit_baselines(sample_train(train, 300, 42), 42)
    assert models['tfidf'].score(val.text, val.label) > models['majority'].score(val.text, val.label)
