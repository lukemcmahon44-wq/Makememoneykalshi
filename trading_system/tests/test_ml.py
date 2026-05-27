"""
Unit tests for ML pipeline: features lookahead-safety + walk-forward gating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.ml.features import (ALL_FEATURES, assert_no_lookahead,
                                          build_feature_frame,
                                          compute_calendar_features,
                                          compute_price_features,
                                          make_target)
from trading_system.ml.meta_learner import StackingMetaLearner


@pytest.fixture
def synth_df():
    rng = np.random.default_rng(0)
    n = 600
    log_prices = np.cumsum(rng.normal(0, 0.001, n)) + np.log(100)
    prices = np.exp(log_prices)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.001, "low": prices * 0.999,
        "close": prices, "volume": rng.uniform(100, 1000, n),
    }, index=idx)
    return df


class TestFeaturePipeline:
    def test_price_features_shape(self, synth_df):
        out = compute_price_features(synth_df)
        assert "log_ret_1m" in out.columns
        assert len(out) == len(synth_df)

    def test_calendar_features(self, synth_df):
        out = compute_calendar_features(synth_df)
        assert {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}.issubset(out.columns)

    def test_all_features_present(self, synth_df):
        features = build_feature_frame(synth_df)
        # Every advertised feature exists (even if mostly NaN)
        for name in ALL_FEATURES:
            assert name in features.columns

    def test_shift_removes_lookahead(self, synth_df):
        """The returned feature frame is shifted by 1, so feature value at
        time t is computed from data <= t-1."""
        features = build_feature_frame(synth_df)
        # First row entirely NaN after shift
        assert features.iloc[0].isna().all()

    def test_no_perfect_lookahead(self, synth_df):
        features = build_feature_frame(synth_df)
        target = make_target(synth_df, horizon=5)
        # If a feature had perfect correlation, this would raise.
        assert_no_lookahead(features, target, threshold=0.95)


class TestStackingMetaLearner:
    def test_fit_predict(self):
        rng = np.random.default_rng(0)
        n = 200
        # Two base predictions that together predict y
        base = pd.DataFrame({
            "lgbm": rng.uniform(0, 1, n),
            "lstm": rng.uniform(0, 1, n),
        })
        y = (0.6 * base["lgbm"] + 0.4 * base["lstm"] > 0.5).astype(int)
        meta = StackingMetaLearner()
        ok = meta.fit(base, y)
        if not ok:
            pytest.skip("sklearn not installed")
        probs = meta.predict_proba(base)
        assert probs.shape[0] == n
        sig = meta.signal({"lgbm": 0.8, "lstm": 0.7})
        assert -1.0 <= sig <= 1.0
