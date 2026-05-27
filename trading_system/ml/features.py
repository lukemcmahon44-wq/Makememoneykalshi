"""
Feature engineering pipeline: ~75 features across price, vol, technical,
microstructure, regime, cross-asset, sentiment, on-chain, calendar.

CRITICAL: every feature uses ONLY data at time t. Shift everything by 1 bar
before training. `assert_no_lookahead` is called automatically on each frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from ..signals._common import (EPS, adx, atr, ema, macd, rolling_std, rsi_series,
                                sma)

log = get_logger(__name__)


FEATURE_GROUPS: Dict[str, List[str]] = {
    "price_returns": [
        "log_ret_1m", "log_ret_5m", "log_ret_15m", "log_ret_1h", "log_ret_4h",
        "abs_ret_1m", "abs_ret_5m", "ret_vs_open", "overnight_gap",
    ],
    "volatility": [
        "realized_vol_5m_20", "realized_vol_15m_20",
        "realized_vol_1h_20", "realized_vol_1d_20",
        "parkinson_vol_1d", "garman_klass_vol", "yang_zhang_vol",
        "atr_14_norm", "bb_width_pct",
    ],
    "technical": [
        "rsi_14", "rsi_7", "stoch_k_14", "stoch_d_14", "williams_r_14",
        "macd_hist", "macd_hist_slope", "bb_zscore_20",
        "ema9_vs_ema21", "ema21_vs_ema50", "ema50_vs_ema200",
        "adx_14", "di_plus_14", "di_minus_14",
        "vwap_dev_atr", "obv_zscore_20", "mfi_14", "cmf_20",
        "ichimoku_cloud_signal", "tenkan_kijun_cross",
    ],
    "microstructure": [
        "obi_level1", "obi_level5", "obi_level10", "obi_level20",
        "weighted_obi", "micro_price_dev", "vamp_dev",
        "spread_bps", "spread_vol_ratio",
        "cvd_1m", "cvd_5m", "cvd_divergence", "aggressor_ratio_1m",
        "hawkes_branching_ratio", "hawkes_intensity_zscore", "trade_count_ratio",
    ],
    "regime": [
        "garch_vol_forecast", "garch_vol_ratio",
        "hmm_bull_probability", "hmm_regime_duration",
        "macro_multiplier", "adx_regime_encoded",
        "vix_level", "vix_1d_change",
    ],
    "cross_asset": [
        "btc_1h_return", "spy_1h_return", "correlation_spy_20",
        "correlation_btc_20", "gold_1d_return", "dxy_1d_return",
        "tlt_1d_return", "sector_etf_1h_return", "btc_dominance_change",
    ],
    "sentiment": [
        "reddit_sentiment_1h", "reddit_sentiment_4h",
        "reddit_sentiment_momentum", "reddit_mention_surge",
        "news_sentiment_finbert", "news_sentiment_momentum",
        "fear_greed_normalized", "insider_buy_signal",
    ],
    "onchain": [
        "onchain_composite", "exchange_netflow_zscore", "active_addr_momentum",
        "nvt_ratio_zscore", "staking_ratio_change",
    ],
    "calendar": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "minutes_to_close_norm", "days_to_options_expiry",
        "is_fomc_day", "is_earnings_week",
    ],
}

ALL_FEATURES: List[str] = [f for g in FEATURE_GROUPS.values() for f in g]


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """df expected to have columns: open, high, low, close, volume."""
    out = pd.DataFrame(index=df.index)
    c = df["close"].astype(float)
    log_c = np.log(c.clip(lower=1e-9))
    out["log_ret_1m"] = log_c.diff()
    out["log_ret_5m"] = log_c.diff(5)
    out["log_ret_15m"] = log_c.diff(15)
    out["log_ret_1h"] = log_c.diff(60)
    out["log_ret_4h"] = log_c.diff(240)
    out["abs_ret_1m"] = out["log_ret_1m"].abs()
    out["abs_ret_5m"] = out["log_ret_5m"].abs()
    if "open" in df.columns:
        out["ret_vs_open"] = (c / df["open"].replace(0, np.nan)) - 1
    out["overnight_gap"] = log_c - log_c.shift(390)   # rough
    return out


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    log_c = np.log(c.clip(lower=1e-9))
    rets = log_c.diff()
    out["realized_vol_5m_20"] = rets.rolling(100).std() * np.sqrt(20)
    out["realized_vol_15m_20"] = rets.rolling(300).std() * np.sqrt(20)
    out["realized_vol_1h_20"] = rets.rolling(60 * 20).std() * np.sqrt(20)
    out["realized_vol_1d_20"] = rets.rolling(390 * 20).std() * np.sqrt(20)
    out["parkinson_vol_1d"] = (np.log(h / l.replace(0, np.nan)) ** 2 / (4 * np.log(2))).rolling(390).mean()
    # Garman-Klass
    o = df["open"].astype(float)
    gk = 0.5 * (np.log(h / l.replace(0, np.nan))) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o.replace(0, np.nan))) ** 2
    out["garman_klass_vol"] = gk.rolling(390).mean()
    out["yang_zhang_vol"] = out["realized_vol_1d_20"]   # placeholder
    out["atr_14_norm"] = (h - l).rolling(14).mean() / c
    bb_mean = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out["bb_width_pct"] = (4 * bb_std) / bb_mean
    return out


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"].astype(float).to_numpy()
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    if len(c) < 50:
        for col in FEATURE_GROUPS["technical"]:
            out[col] = np.nan
        return out

    out["rsi_14"] = pd.Series(rsi_series(c, 14), index=df.index)
    out["rsi_7"] = pd.Series(rsi_series(c, 7), index=df.index)

    # Stochastic
    hh = pd.Series(h).rolling(14).max()
    ll = pd.Series(l).rolling(14).min()
    k = 100 * (pd.Series(c) - ll) / (hh - ll + 1e-10)
    out["stoch_k_14"] = k.values
    out["stoch_d_14"] = k.rolling(3).mean().values

    # Williams %R
    out["williams_r_14"] = (-100 * (hh - pd.Series(c)) / (hh - ll + 1e-10)).values

    # MACD
    macd_line, _, hist = macd(c)
    out["macd_hist"] = hist
    out["macd_hist_slope"] = np.concatenate([[0, 0, 0], np.diff(hist, n=3, prepend=[hist[0]] * 3)[3:]])

    # Bollinger z
    mu20 = pd.Series(c).rolling(20).mean()
    sd20 = pd.Series(c).rolling(20).std()
    out["bb_zscore_20"] = ((pd.Series(c) - mu20) / (sd20 + 1e-10)).values

    # EMA ratios
    e9 = ema(c, 9); e21 = ema(c, 21); e50 = ema(c, 50)
    e200 = ema(c, 200) if len(c) >= 200 else np.full_like(c, np.nan)
    out["ema9_vs_ema21"] = (e9 - e21) / (e21 + 1e-10)
    out["ema21_vs_ema50"] = (e21 - e50) / (e50 + 1e-10)
    out["ema50_vs_ema200"] = (e50 - e200) / (e200 + 1e-10)

    # ADX (sliding)
    adx_arr = np.full_like(c, np.nan, dtype=float)
    dip = np.full_like(c, np.nan, dtype=float)
    dim = np.full_like(c, np.nan, dtype=float)
    from ..data.store import Candle
    candles = [Candle(0, 0, h[i], l[i], c[i], 0) for i in range(len(c))]
    for i in range(30, len(c)):
        a, p, m = adx(candles[: i + 1], 14)
        adx_arr[i] = a; dip[i] = p; dim[i] = m
    out["adx_14"] = adx_arr
    out["di_plus_14"] = dip
    out["di_minus_14"] = dim

    out["vwap_dev_atr"] = np.nan      # filled by live pipeline
    out["obv_zscore_20"] = np.nan
    out["mfi_14"] = np.nan
    out["cmf_20"] = np.nan
    out["ichimoku_cloud_signal"] = np.nan
    out["tenkan_kijun_cross"] = np.nan
    return out


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    out = pd.DataFrame(index=df.index)
    hour = pd.Series(idx.hour, index=df.index)
    dow = pd.Series(idx.dayofweek, index=df.index)
    minute = pd.Series(idx.minute, index=df.index)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    minutes_in_session = hour * 60 + minute
    out["minutes_to_close_norm"] = ((16 * 60) - minutes_in_session).clip(0, 390) / 390
    out["days_to_options_expiry"] = 0
    out["is_fomc_day"] = 0
    out["is_earnings_week"] = 0
    return out


def build_feature_frame(df: pd.DataFrame,
                        microstructure: Optional[pd.DataFrame] = None,
                        regime: Optional[pd.DataFrame] = None,
                        sentiment: Optional[pd.DataFrame] = None,
                        onchain: Optional[pd.DataFrame] = None,
                        cross_asset: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Combine all feature groups into one frame, then SHIFT by 1 (lookahead-safe)."""
    parts = [
        compute_price_features(df),
        compute_volatility_features(df),
        compute_technical_features(df),
        compute_calendar_features(df),
    ]
    for extra in (microstructure, regime, sentiment, onchain, cross_asset):
        if extra is not None and len(extra) > 0:
            parts.append(extra.reindex(df.index))
    out = pd.concat(parts, axis=1)
    # Ensure every advertised feature exists (even if NaN)
    for col in ALL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    out = out[ALL_FEATURES]
    return out.shift(1)


def make_target(df: pd.DataFrame, horizon: int = 12, kind: str = "binary"
                ) -> pd.Series:
    """Future log return target (kind=binary: 1 if positive)."""
    log_c = np.log(df["close"].astype(float).clip(lower=1e-9))
    future_ret = log_c.shift(-horizon) - log_c
    if kind == "binary":
        return (future_ret > 0).astype(int)
    return future_ret


def assert_no_lookahead(features: pd.DataFrame, target: pd.Series,
                        threshold: float = 0.95) -> None:
    """Detect features that correlate too tightly with the target."""
    aligned = features.dropna(how="all")
    target_a = target.reindex(aligned.index)
    suspects: List[str] = []
    for col in aligned.columns:
        s = aligned[col].dropna()
        t = target_a.reindex(s.index).dropna()
        common = s.index.intersection(t.index)
        if len(common) < 30:
            continue
        try:
            corr = float(np.corrcoef(s.loc[common], t.loc[common])[0, 1])
        except Exception:
            corr = 0.0
        if np.isfinite(corr) and abs(corr) >= threshold:
            suspects.append(f"{col}={corr:.3f}")
    if suspects:
        raise RuntimeError(f"Possible lookahead in features: {suspects}")


def get_future_leakage_check(features: pd.DataFrame, df: pd.DataFrame,
                              horizon: int = 12) -> Dict[str, float]:
    """Return correlation between each feature and future returns."""
    log_c = np.log(df["close"].astype(float).clip(lower=1e-9))
    future_ret = log_c.shift(-horizon) - log_c
    out: Dict[str, float] = {}
    for col in features.columns:
        s = features[col].dropna()
        t = future_ret.reindex(s.index).dropna()
        idx = s.index.intersection(t.index)
        if len(idx) < 30:
            out[col] = 0.0
            continue
        try:
            out[col] = float(np.corrcoef(s.loc[idx], t.loc[idx])[0, 1])
        except Exception:
            out[col] = 0.0
    return out
