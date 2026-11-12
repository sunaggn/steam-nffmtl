"""History filter, unknown identifiers, empty relevance sets and shared candidates."""
import numpy as np
import pytest

from steam_nffmtl.baselines import PopularityScorer, RandomScorer
from steam_nffmtl.evaluate import comparison_table, evaluate_ranking
from steam_nffmtl.protocol import build_split, profile_matrix


def _ids(data, idx):
    return data.user_ids[idx].tolist()


def test_dedup_and_unknown_ids(tiny_data) -> None:
    d = tiny_data
    assert d.card["duplicates_removed"]["train"] == 1
    assert d.user_ids.tolist() == [1, 2, 3] and d.item_ids.tolist() == [10, 20, 30, 40]
    # cold user 4 (val) and 5 (test), cold game 99 (test) map to -1
    assert d.val["u"][d.val["user_id"] == 4].tolist() == [-1]
    assert d.test["i"][d.test["item_id"] == 99].tolist() == [-1]
    assert d.card["test"]["rows_cold_user"] == 1 and d.card["test"]["rows_known_user_cold_game"] == 1


def test_test_positive_protocol(tiny_data) -> None:
    s = build_split(tiny_data, "test", "positive")
    # u1: 40 is validation history, 99 is a cold game -> no eligible item; u3 only has a negative review
    assert _ids(tiny_data, s.users) == [2]
    i_of = {g: j for j, g in enumerate(tiny_data.item_ids)}
    assert s.relevant[0].indices.tolist() == [i_of[30]]
    # history filter uses training years and the validation year for test
    assert sorted(tiny_data.item_ids[s.exclude[0].indices].tolist()) == [10, 20, 40]
    r = s.report
    assert r["rows_cold_user_excluded"] == 1 and r["rows_cold_game_excluded"] == 1
    assert r["rows_relevant_but_in_history_excluded"] == 2
    assert r["users_with_eligible_relevant"] == 1


def test_interaction_relevance_and_val_cutoff(tiny_data) -> None:
    s = build_split(tiny_data, "test", "interaction")
    assert _ids(tiny_data, s.users) == [2, 3]
    v = build_split(tiny_data, "val", "positive")
    assert _ids(tiny_data, v.users) == [1]
    # validation history = training years only (2021 itself is the target)
    assert sorted(tiny_data.item_ids[v.exclude[0].indices].tolist()) == [10, 20, 30]


def test_same_protocol_for_all_models(tiny_data) -> None:
    s = build_split(tiny_data, "test", "interaction")
    pop = np.asarray(profile_matrix(tiny_data, "interaction").sum(0)).ravel()
    a = evaluate_ranking(PopularityScorer(profile_matrix(tiny_data, "positive")), s, [2], 8, pop, None)
    b = evaluate_ranking(RandomScorer(tiny_data.n_items, 0), s, [2], 1, pop, None)
    assert a.split_fingerprint == b.split_fingerprint == s.fingerprint()
    assert a.per_user["u"].tolist() == b.per_user["u"].tolist()
    table = comparison_table([a, b])
    assert set(table.index) == {"Popularity", "Random"}
    assert build_split(tiny_data, "test", "interaction").fingerprint() == s.fingerprint()
    # history games never appear in any recommendation list
    for res in (a, b):
        r = evaluate_ranking(PopularityScorer(profile_matrix(tiny_data, "positive")) if res.model == "Popularity"
                             else RandomScorer(tiny_data.n_items, 0), s, [2], 8, pop, None, keep_topk=True)
        ex = s.exclude.toarray() > 0
        top = r.topk
        assert not (np.take_along_axis(ex, np.maximum(top, 0), axis=1) & (top >= 0)).any()


class _Broken:
    name = "Broken"

    def score(self, users):
        raise RuntimeError("boom")


def test_errors_are_not_silent(tiny_data) -> None:
    s = build_split(tiny_data, "test", "interaction")
    pop = np.ones(tiny_data.n_items)
    with pytest.raises(RuntimeError):
        evaluate_ranking(_Broken(), s, [2], 8, pop, None, fail_on_error=True)
    res = evaluate_ranking(_Broken(), s, [2], 8, pop, None, fail_on_error=False)
    assert not res.valid and res.coverage["failed_users"] == 2
    good = evaluate_ranking(RandomScorer(tiny_data.n_items, 0), s, [2], 8, pop, None)
    with pytest.raises(ValueError):
        comparison_table([good, res])


def test_tags_from_training_catalog_only(tiny_data) -> None:
    # game 99 exists only in the test year; its tag 'Z' must not enter the vocabulary (n_tags=4)
    assert tiny_data.tag_names == ["A", "B", "C"]
    assert tiny_data.tag_matrix.shape == (4, 3)
    assert tiny_data.tag_matrix[3].sum() == 0  # game 40 has no tags
