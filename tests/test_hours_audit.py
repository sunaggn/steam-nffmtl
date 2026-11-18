"""Small fixtures for the published hours-audit boundary conditions."""

import pandas as pd

from scripts.check_hours import hours_profile, release_check


def test_exactly_1000_hours_does_not_imply_a_value_above_1000():
    exact = hours_profile(pd.DataFrame({"hours": [0.0, 999.9, 1000.0]}))
    above = hours_profile(pd.DataFrame({"hours": [0.0, 1000.1]}))

    assert exact["rows_ge_1000"] == 1
    assert exact["no_values_above_1000h"] is True
    assert above["no_values_above_1000h"] is False


def test_release_window_uses_end_of_day_limit_and_excludes_early_access():
    recs = pd.DataFrame({
        "app_id": [1, 1, 1, 2],
        "date": pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-15", "2022-01-01"]),
        "hours": [24.0, 25.0, 361.0, 100.0],
    })
    games = pd.DataFrame({"app_id": [1, 2], "date_release": ["2022-01-01", "2022-01-01"]})

    report = release_check(recs, games, early_access={2})

    assert report["reviews_in_window"] == 3
    assert report["reviews_with_more_hours_than_possible"] == 2
    assert report["share_impossible_by_days_since_release"] == {0: 0.5, 14: 1.0}
