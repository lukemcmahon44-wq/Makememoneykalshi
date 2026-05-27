"""
Unit tests for every signal function. Every signal returns float in [-1, +1].
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_system.data.store import LOBSnapshot, TradeTick
from trading_system.signals.microstructure.flow import (aggressor_ratio,
                                                              composite_flow,
                                                              cumulative_volume_delta,
                                                              cvd_zscore_signal)
from trading_system.signals.microstructure.hawkes import HawkesCache, HawkesProcess
from trading_system.signals.microstructure.lob_signals import (lob_composite,
                                                                  micro_price,
                                                                  micro_price_signal,
                                                                  order_book_imbalance,
                                                                  vamp_signal,
                                                                  weighted_obi)
from trading_system.signals.regime.macro_overlay import (compute_macro_multiplier,
                                                            equity_signal_gate)
from trading_system.signals.sentiment.fear_greed import fear_greed_signal
from trading_system.signals.sentiment.reddit_nlp import reddit_signal_for_asset
from trading_system.signals.stat_arb.cointegration import (cointegrated_pair_score,
                                                              half_life)
from trading_system.signals.stat_arb.kalman_pairs import KalmanHedge
from trading_system.signals.stat_arb.ou_process import OrnsteinUhlenbeck
from trading_system.signals.technical.momentum import (chaikin_money_flow,
                                                          chande_momentum,
                                                          composite_momentum,
                                                          rate_of_change,
                                                          williams_r)
from trading_system.signals.technical.reversion import (bollinger_signal,
                                                          composite_reversion,
                                                          rsi_signal,
                                                          stochastic_signal)
from trading_system.signals.technical.trend import (adx_regime,
                                                       composite_trend,
                                                       dual_ema_cross,
                                                       ichimoku,
                                                       macd_signal,
                                                       triple_ema_alignment)
from trading_system.signals.technical.volume import (composite_volume,
                                                        money_flow_index,
                                                        obv_zscore,
                                                        volume_surge)


def _check_signal(val: float) -> None:
    assert isinstance(val, float)
    assert np.isfinite(val), f"non-finite signal {val}"
    assert -1.0 <= val <= 1.0, f"out of range: {val}"


class TestTechnicalTrend:
    def test_dual_ema_cross_range(self, synthetic_candles):
        _check_signal(dual_ema_cross(synthetic_candles))

    def test_triple_ema_alignment_range(self, synthetic_candles):
        _check_signal(triple_ema_alignment(synthetic_candles))

    def test_macd_signal_range(self, synthetic_candles):
        _check_signal(macd_signal(synthetic_candles))

    def test_ichimoku_range(self, synthetic_candles):
        _check_signal(ichimoku(synthetic_candles))

    def test_adx_regime_keys(self, synthetic_candles):
        info = adx_regime(synthetic_candles)
        assert info["regime"] in ("trending", "ranging", "transition")
        assert 0.5 <= info["multiplier"] <= 1.5

    def test_composite_range(self, synthetic_candles):
        _check_signal(composite_trend(synthetic_candles))

    def test_empty_candles_returns_zero(self):
        assert dual_ema_cross([]) == 0.0
        assert macd_signal([]) == 0.0


class TestTechnicalReversion:
    def test_bollinger(self, synthetic_candles):
        _check_signal(bollinger_signal(synthetic_candles))

    def test_rsi(self, synthetic_candles):
        _check_signal(rsi_signal(synthetic_candles))

    def test_stochastic(self, synthetic_candles):
        _check_signal(stochastic_signal(synthetic_candles))

    def test_composite(self, synthetic_candles):
        _check_signal(composite_reversion(synthetic_candles, session_vwap=100.0))


class TestTechnicalMomentum:
    def test_roc(self, synthetic_candles):
        _check_signal(rate_of_change(synthetic_candles))

    def test_williams(self, synthetic_candles):
        _check_signal(williams_r(synthetic_candles))

    def test_cmf(self, synthetic_candles):
        _check_signal(chaikin_money_flow(synthetic_candles))

    def test_chande(self, synthetic_candles):
        _check_signal(chande_momentum(synthetic_candles))

    def test_composite(self, synthetic_candles):
        _check_signal(composite_momentum(synthetic_candles))


class TestTechnicalVolume:
    def test_obv_zscore(self, synthetic_candles):
        _check_signal(obv_zscore(synthetic_candles))

    def test_volume_surge(self, synthetic_candles):
        _check_signal(volume_surge(synthetic_candles))

    def test_mfi(self, synthetic_candles):
        _check_signal(money_flow_index(synthetic_candles))

    def test_composite(self, synthetic_candles):
        _check_signal(composite_volume(synthetic_candles))


class TestMicrostructureLOB:
    def setup_method(self):
        self.lob = LOBSnapshot(
            bids=[(99.5, 10), (99.4, 20), (99.3, 30), (99.2, 40), (99.1, 50)],
            asks=[(100.0, 8), (100.1, 15), (100.2, 25), (100.3, 35), (100.4, 45)],
            timestamp=0.0,
        )

    def test_obi_range(self):
        _check_signal(order_book_imbalance(self.lob.bids, self.lob.asks, 5))

    def test_obi_sign(self):
        # Bid-heavy book: positive imbalance
        bid_heavy = [(99.5, 100), (99.4, 100)]
        ask_light = [(100.0, 10), (100.1, 10)]
        assert order_book_imbalance(bid_heavy, ask_light, 2) > 0

    def test_weighted_obi(self):
        _check_signal(weighted_obi(self.lob.bids, self.lob.asks, 5))

    def test_micro_price(self):
        mp = micro_price(99.5, 10, 100.0, 8)
        assert 99.5 <= mp <= 100.0

    def test_micro_price_signal(self):
        _check_signal(micro_price_signal(self.lob))

    def test_vamp_signal(self):
        _check_signal(vamp_signal(self.lob))

    def test_lob_composite(self):
        _check_signal(lob_composite(self.lob))


class TestMicrostructureHawkes:
    def test_branching_ratio(self):
        h = HawkesProcess(mu=1.0, alpha=0.3, beta=1.0)
        assert h.branching_ratio() == pytest.approx(0.3)

    def test_signal_in_range(self):
        h = HawkesProcess(mu=1.0, alpha=0.3, beta=1.0)
        ts = [float(i) for i in range(100)]
        _check_signal(h.momentum_signal(ts, now=ts[-1]))

    def test_cache_update(self):
        cache = HawkesCache(refit_every_n_trades=50)
        # Generate 200 timestamps clustered
        ts = sorted(np.random.default_rng(0).exponential(2.0, 200).cumsum().tolist())
        s = cache.update_and_signal("BTCUSDT", ts, now=ts[-1])
        _check_signal(s)


class TestMicrostructureFlow:
    def setup_method(self):
        rng = np.random.default_rng(1)
        self.trades = []
        t = 0.0
        for _ in range(200):
            t += rng.exponential(1.0)
            self.trades.append(TradeTick(
                ts=t, price=100 + rng.normal(0, 0.1),
                qty=float(rng.uniform(1, 5)),
                is_buyer_maker=bool(rng.random() < 0.5),
            ))

    def test_cvd(self):
        cvd = cumulative_volume_delta(self.trades)
        assert isinstance(cvd, float)

    def test_cvd_zscore(self):
        _check_signal(cvd_zscore_signal(self.trades))

    def test_aggressor_ratio(self):
        _check_signal(aggressor_ratio(self.trades))

    def test_composite(self):
        _check_signal(composite_flow(self.trades))


class TestStatArb:
    def test_ou_fit_synthetic(self):
        rng = np.random.default_rng(3)
        n = 300
        mu = 0.0; theta = 0.05; sigma = 0.5
        x = np.zeros(n); x[0] = 0
        for i in range(1, n):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.standard_normal()
        ou = OrnsteinUhlenbeck().fit(x)
        assert ou.theta > 0
        assert ou.half_life is not None
        _check_signal(ou.signal(x[-1]))

    def test_kalman_update(self):
        rng = np.random.default_rng(2)
        kh = KalmanHedge()
        for _ in range(60):
            pB = float(rng.normal(100, 1))
            pA = 1.2 * pB + 5 + rng.normal(0, 0.1)
            h, intercept, s, _ = kh.update(pA, pB)
        assert 1.0 < h < 1.5
        _check_signal(kh.continuous_signal())

    def test_cointegration_handles_short(self):
        result = cointegrated_pair_score(np.zeros(20), np.zeros(20))
        assert result["cointegrated"] is False


class TestRegime:
    def test_macro_multiplier_extreme_vix(self):
        m, flags = compute_macro_multiplier({"VIXCLS": 50})
        assert m <= 0.25
        assert "VIX_EXTREME" in flags

    def test_macro_multiplier_normal(self):
        m, flags = compute_macro_multiplier({"VIXCLS": 18, "T10Y2Y": 0.5})
        assert 0.8 <= m <= 1.1

    def test_equity_gate(self):
        assert equity_signal_gate({"T10Y2Y": 0.5}) is True
        assert equity_signal_gate({"T10Y2Y": -0.5, "UNRATE_CHANGE_3M": 0.5}) is False


class TestSentiment:
    def test_fear_greed_extremes(self):
        assert fear_greed_signal(5) > 0.5    # extreme fear → bullish
        assert fear_greed_signal(95) < -0.5  # extreme greed → bearish

    def test_reddit_empty(self):
        assert reddit_signal_for_asset([]) == 0.0
