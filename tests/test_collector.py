"""Tests pour data.collector.filter_gainers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.collector import filter_gainers


def _df_with_gain(gain_pct: float) -> pd.DataFrame:
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    trough_at = n // 2
    close = np.concatenate([
        np.linspace(100, 50, trough_at),
        np.linspace(50, 50 * (1 + gain_pct / 100), n - trough_at),
    ])
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.ones(n) * 10000,
    }, index=idx)


def test_filter_keeps_big_gainers() -> None:
    panels = {
        "BIG": _df_with_gain(150),
        "SMALL": _df_with_gain(30),
    }
    kept = filter_gainers(panels, min_gain_pct=100)
    assert "BIG" in kept
    assert "SMALL" not in kept


def test_filter_empty() -> None:
    kept = filter_gainers({}, min_gain_pct=100)
    assert kept == {}
