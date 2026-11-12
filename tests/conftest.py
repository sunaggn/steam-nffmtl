"""Shared fixtures: a hand-built dataset with known expected protocol output."""
import pandas as pd
import pytest

from steam_nffmtl.config import DataConfig
from steam_nffmtl.data import build_prepared


def _row(u: int, i: int, date: str, pos: bool, hours: float = 5.0) -> dict:
    return {"user_id": u, "app_id": i, "date": pd.Timestamp(date), "is_recommended": int(pos), "hours": hours}


@pytest.fixture()
def tiny_data():
    rows = [
        # training years (2019); u1-10 appears twice (dedup keeps the later row)
        _row(1, 10, "2019-01-01", True), _row(1, 10, "2019-06-01", True), _row(1, 20, "2019-02-01", True),
        _row(1, 30, "2019-03-01", False), _row(2, 10, "2019-01-05", True), _row(2, 40, "2019-01-06", True),
        _row(3, 20, "2019-04-01", True), _row(3, 30, "2019-04-02", True),
        # validation year
        _row(1, 40, "2021-01-01", True), _row(2, 20, "2021-01-01", False), _row(4, 10, "2021-02-01", True),
        # test year
        _row(1, 40, "2022-01-01", True), _row(1, 99, "2022-01-02", True), _row(2, 30, "2022-01-03", True),
        _row(2, 20, "2022-01-04", True), _row(3, 10, "2022-01-05", False), _row(5, 10, "2022-01-06", True),
    ]
    recs = pd.DataFrame(rows)
    recs["hours"] = recs["hours"].astype("float32")
    meta = pd.DataFrame({"app_id": [10, 20, 30, 40, 99],
                         "tags": [["A", "B"], ["A"], ["C"], [], ["Z"]]})
    cfg = DataConfig(train_years=[2010, 2020], kcore=1, n_tags=4)
    return build_prepared(recs, meta, cfg)
