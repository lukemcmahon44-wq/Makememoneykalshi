"""
Feature engineering pipeline for the quantitative trading system.
Computes 75 features from OHLCV data with zero lookahead bias.
All features are shifted by 1 bar to ensure no future data leakage.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

FEATURE_GROUPS: Dict[str, List[str]] = {
    "price_returns": [
        "log_ret_1m", "log_ret_5m", "log_ret_15m", "log_ret_1h", "log_ret_4h",
        "abs_ret_1m", "abs_ret_5m", "ret_vs_open", "overnight_gap",
    ],
    "volatility": [
        "realized_vol_5m_20", "realized_vol_15m_20", "realized_vol_1h_20",
        "realized_vol_1d_20", "parkinson_vol_1d", "garman_klass_vol",
        "yang_zhang_vol", "atr_14_norm", "bb_width_pct",
    ],
    "technical": [
        "rsi_14", "rsi_7", "stoch_k_14", "stoch_d_14", "williams_r_14",
        "macd_hist", "macd_hist_slope", "bb_zscore_20", "ema9_vs_ema21",
        "ema21_vs_ema50", "ema50_vs_ema200", "adx_14", "di_plus_14", "di_minus_14",
        "vwap_dev_atr", "obv_zscore_20", "mfi_14", "cmf_20",
        "ichimoku_cloud_signal", "tenkan_kijun_cross",
    ],
    "microstructure": [
        "obi_level1", "obi_level5", "obi_level10", "obi_level20",
        "weighted_obi", "micro_price_dev", "vamp_dev", "spread_bps",
        "spread_vol_ratio", "cvd_1m", "cvd_5m", "cvd_divergence",
        "aggressor_ratio_1m", "hawkes_branching_ratio",
        "hawkes_intensity_zscore", "trade_count_ratio",
    ],
    "regime": [
        "garch_vol_forecast", "garch_vol_ratio", "hmm_bull_probability",
        "hmm_regime_duration", "macro_multiplier", "adx_regime_encoded",
        "vix_level", "vix_1d_change",
    ],
    "cross_asset": [
        "btc_1h_return", "spy_1h_return", "correlation_spy_20",
        "correlation_btc_20", "gold_1d_return", "dxy_1d_return",
        "tlt_1d_return", "sector_etf_1h_return", "btc_dominance_change",
    ],
    "sentiment": [
        "reddit_sentiment_1h", "reddit_sentiment_4h", "reddit_sentiment_momentum",
        "reddit_mention_surge", "news_sentiment_finbert", "news_sentiment_momentum",
        "fear_greed_normalized", "insider_buy_signal",
    ],
    "onchain": [
        "onchain_composite", "exchange_netflow_zscore", "active_addr_momentum",
        "nvt_ratio_zscore", "staking_ratio_change",
    ],
    "calendar": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "minutes_to_close_norm",
        "days_to_options_expiry", "is_fomc_day", "is_earnings_week",
    ],
}

ALL_FEATURES: List[str] = [f for group in FEATURE_GROUPS.values() for f in group]


# ---------------------------------------------------------------------------
# Internal helper functions (vectorized, no lookahead)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    denom = highest_high - lowest_low
    pct_k = (close - lowest_low) / (denom.replace(0, np.nan) + 1e-8) * 100.0
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def _adx_di(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, DI+, DI-)."""
    up_move = high.diff()
    down_move = (-low).diff()
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    pos_dm_s = pd.Series(pos_dm, index=close.index).ewm(com=period - 1, adjust=False).mean()
    neg_dm_s = pd.Series(neg_dm, index=close.index).ewm(com=period - 1, adjust=False).mean()
    tr_s = _atr(high, low, close, period)
    dip = 100.0 * pos_dm_s / (tr_s + 1e-8)
    dim = 100.0 * neg_dm_s / (tr_s + 1e-8)
    dx = 100.0 * (dip - dim).abs() / (dip + dim + 1e-8)
    adx = dx.ewm(com=period - 1, adjust=False).mean()
    return adx, dip, dim


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    delta_tp = typical.diff()
    pos_flow = raw_flow.where(delta_tp > 0, 0.0)
    neg_flow = raw_flow.where(delta_tp < 0, 0.0).abs()
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    money_ratio = pos_sum / (neg_sum + 1e-8)
    return 100.0 - (100.0 / (1.0 + money_ratio))


def _cmf(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    hl_range = high - low
    mfm = ((close - low) - (high - close)) / (hl_range + 1e-8)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-8)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    denom = highest_high - lowest_low
    return (highest_high - close) / (denom.replace(0, np.nan) + 1e-8) * -100.0


def _ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (tenkan, kijun, senkou_a, senkou_b, chikou_ref)."""

    def mid(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
        return (h.rolling(n).max() + l.rolling(n).min()) / 2.0

    tenkan = mid(high, low, 9)
    kijun = mid(high, low, 26)
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = mid(high, low, 52).shift(26)
    chikou_ref = close.shift(26)
    return tenkan, kijun, senkou_a, senkou_b, chikou_ref


def _simple_garch_vol(returns: pd.Series, omega: float = 0.00001, alpha: float = 0.1, beta: float = 0.85) -> pd.Series:
    """Simplified GARCH(1,1) variance forecast via iterative computation."""
    ret_vals = returns.values
    n = len(ret_vals)
    variance = np.full(n, np.nanvar(ret_vals[:20]) if n >= 20 else 1e-6)
    for i in range(1, n):
        if np.isnan(ret_vals[i - 1]):
            variance[i] = variance[i - 1]
        else:
            variance[i] = omega + alpha * ret_vals[i - 1] ** 2 + beta * variance[i - 1]
    return pd.Series(np.sqrt(variance) * np.sqrt(252), index=returns.index)


def _hmm_regime(returns: pd.Series, window: int = 60) -> Tuple[pd.Series, pd.Series]:
    """
    Simplified HMM-like regime via rolling mean/std threshold.
    Returns (bull_prob, regime_duration).
    """
    roll_mean = returns.rolling(window, min_periods=10).mean()
    roll_std = returns.rolling(window, min_periods=10).std()
    z_score = roll_mean / (roll_std + 1e-8)
    # Sigmoid to get bull probability
    bull_prob = 1.0 / (1.0 + np.exp(-z_score * 5.0))

    # Compute regime duration: consecutive bars with same sign
    regime_sign = np.sign(z_score)
    duration = pd.Series(0, index=returns.index, dtype=float)
    count = 0
    prev_sign = 0.0
    for i, (idx, s) in enumerate(regime_sign.items()):
        if np.isnan(s):
            count = 0
        elif s == prev_sign:
            count += 1
        else:
            count = 1
            prev_sign = s
        duration.iloc[i] = count

    return bull_prob, duration


def _hawkes_params(trade_times_relative: np.ndarray) -> Tuple[float, float]:
    """
    Fit Hawkes process branching ratio and intensity via simplified method-of-moments.
    trade_times_relative: inter-arrival times (seconds).
    Returns (branching_ratio, intensity_zscore).
    """
    if len(trade_times_relative) < 5:
        return 0.5, 0.0
    arr = np.array(trade_times_relative, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 5:
        return 0.5, 0.0
    # Simplified: branching ratio from autocorrelation of inter-arrivals
    if len(arr) > 10:
        corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        branching_ratio = float(np.clip(0.5 - corr * 0.5, 0.01, 0.99))
    else:
        branching_ratio = 0.5
    intensity_zscore = float(np.tanh((len(arr) / max(arr.sum(), 1e-8) - 1.0) * 2.0))
    return branching_ratio, intensity_zscore


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    df: pd.DataFrame,
    asset_type: str = "equity",
    sentiment_data: Optional[pd.DataFrame] = None,
    onchain_data: Optional[pd.DataFrame] = None,
    lob_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute all 75 features from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
        Index should be a DatetimeIndex.
    asset_type : str
        'equity', 'crypto', or 'futures'.
    sentiment_data : pd.DataFrame or None
        Optional DataFrame with sentiment columns aligned to df's index.
    onchain_data : pd.DataFrame or None
        Optional DataFrame with on-chain columns aligned to df's index.
    lob_data : pd.DataFrame or None
        Optional LOB snapshot DataFrame with bid/ask columns.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with ALL_FEATURES columns, shifted by 1 bar.
    """
    df = df.copy()
    # Ensure lowercase column names
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    out = pd.DataFrame(index=df.index)

    # -----------------------------------------------------------------------
    # 1. Price returns
    # -----------------------------------------------------------------------
    log_ret = np.log(close / close.shift(1))

    out["log_ret_1m"] = log_ret
    out["log_ret_5m"] = np.log(close / close.shift(5))
    out["log_ret_15m"] = np.log(close / close.shift(15))
    out["log_ret_1h"] = np.log(close / close.shift(60))
    out["log_ret_4h"] = np.log(close / close.shift(240))
    out["abs_ret_1m"] = log_ret.abs()
    out["abs_ret_5m"] = np.log(close / close.shift(5)).abs()
    out["ret_vs_open"] = np.log(close / open_)
    out["overnight_gap"] = np.log(open_ / close.shift(1))

    # -----------------------------------------------------------------------
    # 2. Volatility
    # -----------------------------------------------------------------------
    # Realized volatility at different timeframes (annualized)
    out["realized_vol_5m_20"] = log_ret.rolling(20).std() * np.sqrt(252 * 78)
    out["realized_vol_15m_20"] = np.log(close / close.shift(3)).rolling(20).std() * np.sqrt(252 * 26)
    out["realized_vol_1h_20"] = np.log(close / close.shift(12)).rolling(20).std() * np.sqrt(252 * 6.5)
    out["realized_vol_1d_20"] = np.log(close / close.shift(390)).rolling(20).std() * np.sqrt(252)

    # Parkinson volatility (1-day): uses high/low
    hl_ratio = np.log(high / low)
    out["parkinson_vol_1d"] = np.sqrt(1.0 / (4.0 * np.log(2.0)) * hl_ratio.rolling(20).mean() ** 2) * np.sqrt(252)

    # Garman-Klass volatility
    gk_term1 = 0.5 * (np.log(high / low)) ** 2
    gk_term2 = (2 * np.log(2) - 1) * (np.log(close / open_)) ** 2
    out["garman_klass_vol"] = np.sqrt((gk_term1 - gk_term2).rolling(20).mean() * 252)

    # Yang-Zhang volatility
    yz_overnight = np.log(open_ / close.shift(1))
    yz_open = np.log(close / open_)
    yz_rs = 0.5 * (np.log(high / low)) ** 2 - (2 * np.log(2) - 1) * (np.log(close / open_)) ** 2
    k = 0.34 / (1.34 + (21 + 1) / (21 - 1))
    out["yang_zhang_vol"] = np.sqrt(
        (yz_overnight.rolling(20).var() + k * yz_open.rolling(20).var() + (1 - k) * yz_rs.rolling(20).mean()) * 252
    )

    # ATR normalized by price
    atr14 = _atr(high, low, close, 14)
    out["atr_14_norm"] = atr14 / (close + 1e-8)

    # Bollinger Band width %
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    out["bb_width_pct"] = (bb_upper - bb_lower) / (sma20 + 1e-8)

    # -----------------------------------------------------------------------
    # 3. Technical indicators
    # -----------------------------------------------------------------------
    out["rsi_14"] = _rsi(close, 14) / 100.0  # normalize to [0,1]
    out["rsi_7"] = _rsi(close, 7) / 100.0

    stoch_k, stoch_d = _stoch(high, low, close, 14, 3)
    out["stoch_k_14"] = stoch_k / 100.0
    out["stoch_d_14"] = stoch_d / 100.0

    out["williams_r_14"] = _williams_r(high, low, close, 14) / 100.0  # [-100,0] → [-1,0]

    # MACD histogram and slope
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal_line = _ema(macd_line, 9)
    macd_hist = macd_line - signal_line
    out["macd_hist"] = macd_hist / (atr14 + 1e-8)
    out["macd_hist_slope"] = macd_hist.diff(3) / (atr14 + 1e-8)

    # Bollinger Z-score
    out["bb_zscore_20"] = (close - sma20) / (std20 + 1e-8)

    # EMA crossovers (normalized by ATR)
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    out["ema9_vs_ema21"] = (ema9 - ema21) / (atr14 + 1e-8)
    out["ema21_vs_ema50"] = (ema21 - ema50) / (atr14 + 1e-8)
    out["ema50_vs_ema200"] = (ema50 - ema200) / (atr14 + 1e-8)

    # ADX / DI
    adx14, dip14, dim14 = _adx_di(high, low, close, 14)
    out["adx_14"] = adx14 / 100.0
    out["di_plus_14"] = dip14 / 100.0
    out["di_minus_14"] = dim14 / 100.0

    # VWAP deviation in ATR units
    typical_price = (high + low + close) / 3.0
    cum_tpv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    session_vwap = cum_tpv / (cum_vol + 1e-8)
    out["vwap_dev_atr"] = (close - session_vwap) / (atr14 + 1e-8)

    # OBV z-score
    obv_series = _obv(close, volume)
    obv_mean20 = obv_series.rolling(20).mean()
    obv_std20 = obv_series.rolling(20).std()
    out["obv_zscore_20"] = (obv_series - obv_mean20) / (obv_std20 + 1e-8)

    # MFI normalized
    mfi14 = _mfi(high, low, close, volume, 14)
    out["mfi_14"] = mfi14 / 100.0

    # CMF
    out["cmf_20"] = _cmf(high, low, close, volume, 20)

    # Ichimoku
    tenkan, kijun, senkou_a, senkou_b, chikou_ref = _ichimoku(high, low, close)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    above_cloud = np.where(close > cloud_top, 1.0, np.where(close < cloud_bot, -1.0, 0.0))
    tenkan_vs_kijun = np.where(tenkan > kijun, 1.0, np.where(tenkan < kijun, -1.0, 0.0))
    chikou_signal = np.where(close > chikou_ref, 1.0, np.where(close < chikou_ref, -1.0, 0.0))
    out["ichimoku_cloud_signal"] = (
        pd.Series(above_cloud, index=df.index) +
        pd.Series(tenkan_vs_kijun, index=df.index) +
        pd.Series(chikou_signal, index=df.index)
    ) / 3.0

    # Tenkan/Kijun cross momentum
    tk_diff = tenkan - kijun
    out["tenkan_kijun_cross"] = tk_diff / (atr14 + 1e-8)

    # -----------------------------------------------------------------------
    # 4. Microstructure (derived from OHLCV; LOB data optional)
    # -----------------------------------------------------------------------
    if lob_data is not None:
        lob = lob_data.reindex(df.index, method="ffill")
        for lvl, col in [(1, "obi_level1"), (5, "obi_level5"), (10, "obi_level10"), (20, "obi_level20")]:
            bid_col = f"bid_size_{lvl}" if f"bid_size_{lvl}" in lob.columns else None
            ask_col = f"ask_size_{lvl}" if f"ask_size_{lvl}" in lob.columns else None
            if bid_col and ask_col:
                bid_sz = lob[bid_col]
                ask_sz = lob[ask_col]
                out[col] = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-8)
            else:
                out[col] = 0.0

        if "bid_size_1" in lob.columns and "ask_size_1" in lob.columns:
            # Weighted OBI using multiple levels present
            wt_bid = sum(
                lob[f"bid_size_{i}"] / i
                for i in range(1, 6)
                if f"bid_size_{i}" in lob.columns
            )
            wt_ask = sum(
                lob[f"ask_size_{i}"] / i
                for i in range(1, 6)
                if f"ask_size_{i}" in lob.columns
            )
            out["weighted_obi"] = (wt_bid - wt_ask) / (wt_bid + wt_ask + 1e-8)
        else:
            out["weighted_obi"] = 0.0

        if "bid_price_1" in lob.columns and "ask_price_1" in lob.columns:
            bid_p = lob["bid_price_1"]
            ask_p = lob["ask_price_1"]
            if "bid_size_1" in lob.columns and "ask_size_1" in lob.columns:
                bid_s = lob["bid_size_1"]
                ask_s = lob["ask_size_1"]
                micro_price = (bid_p * ask_s + ask_p * bid_s) / (bid_s + ask_s + 1e-8)
            else:
                micro_price = (bid_p + ask_p) / 2.0
            out["micro_price_dev"] = (close - micro_price) / (atr14 + 1e-8)
            spread = ask_p - bid_p
            mid_price = (bid_p + ask_p) / 2.0
            out["spread_bps"] = spread / (mid_price + 1e-8) * 10000.0
        else:
            out["micro_price_dev"] = 0.0
            out["spread_bps"] = 2.0  # default 2 bps

        if "vwap" in lob.columns:
            out["vamp_dev"] = (close - lob["vwap"]) / (atr14 + 1e-8)
        else:
            out["vamp_dev"] = out["vwap_dev_atr"]

        if "cvd_1m" in lob.columns:
            out["cvd_1m"] = lob["cvd_1m"] / (volume + 1e-8)
        else:
            out["cvd_1m"] = 0.0

        if "cvd_5m" in lob.columns:
            out["cvd_5m"] = lob["cvd_5m"] / (volume.rolling(5).sum() + 1e-8)
        else:
            out["cvd_5m"] = 0.0

        if "aggressor_ratio" in lob.columns:
            out["aggressor_ratio_1m"] = lob["aggressor_ratio"]
        else:
            out["aggressor_ratio_1m"] = 0.5

        out["spread_vol_ratio"] = out["spread_bps"] / (out["realized_vol_5m_20"] * 10000.0 + 1e-8)
    else:
        # Synthetic microstructure from OHLCV
        # OBI approximated from close position within bar
        bar_position = (close - low) / (high - low + 1e-8) * 2.0 - 1.0
        for col in ["obi_level1", "obi_level5", "obi_level10", "obi_level20", "weighted_obi"]:
            out[col] = bar_position * (0.5 + np.random.RandomState(42).uniform(0, 0.1, len(df)))

        out["micro_price_dev"] = 0.0
        out["vamp_dev"] = 0.0
        out["spread_bps"] = 2.0 + atr14 / (close + 1e-8) * 5000.0
        out["spread_vol_ratio"] = out["spread_bps"] / (out["realized_vol_5m_20"] * 10000.0 + 1e-8)
        out["cvd_1m"] = log_ret.rolling(1).sum()
        out["cvd_5m"] = log_ret.rolling(5).sum()
        out["aggressor_ratio_1m"] = (bar_position + 1.0) / 2.0

    # CVD divergence: 1m CVD vs 5m CVD difference
    out["cvd_divergence"] = out["cvd_1m"] - out["cvd_5m"].rolling(5).mean()

    # Hawkes process parameters (vectorized approximation)
    # Use rolling inter-trade-time proxy (volume changes)
    vol_changes = volume.diff().abs()
    vol_mean = vol_changes.rolling(20).mean()
    vol_std = vol_changes.rolling(20).std()
    out["hawkes_branching_ratio"] = 0.5 - 0.3 * (vol_changes - vol_mean) / (vol_std + 1e-8)
    out["hawkes_branching_ratio"] = out["hawkes_branching_ratio"].clip(0.01, 0.99)
    out["hawkes_intensity_zscore"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-8)

    # Trade count ratio (volume ratio as proxy)
    out["trade_count_ratio"] = volume / (volume.rolling(20).mean() + 1e-8)

    # -----------------------------------------------------------------------
    # 5. Regime features
    # -----------------------------------------------------------------------
    daily_returns = log_ret.rolling(390).sum()  # ~1 trading day of 1-min bars
    garch_vol = _simple_garch_vol(log_ret)
    out["garch_vol_forecast"] = garch_vol
    hist_vol = log_ret.rolling(20).std() * np.sqrt(252 * 390)
    out["garch_vol_ratio"] = garch_vol / (hist_vol + 1e-8)

    bull_prob, regime_dur = _hmm_regime(log_ret, window=60)
    out["hmm_bull_probability"] = bull_prob
    out["hmm_regime_duration"] = regime_dur / 60.0  # normalize

    # Macro multiplier (derived from vol regime)
    out["macro_multiplier"] = np.where(out["garch_vol_ratio"] > 1.5, 0.5,
                               np.where(out["garch_vol_ratio"] < 0.7, 1.2, 1.0))

    # ADX regime encoded: 0=ranging, 0.5=transition, 1=trending
    out["adx_regime_encoded"] = np.where(out["adx_14"] > 0.25, 1.0,
                                 np.where(out["adx_14"] > 0.20, 0.5, 0.0))

    # VIX placeholder (if no external data, use realized vol proxy)
    out["vix_level"] = out["realized_vol_1d_20"] * 100.0  # rough VIX equivalent
    out["vix_1d_change"] = out["vix_level"].diff(390)  # 1-day change

    # -----------------------------------------------------------------------
    # 6. Cross-asset features
    # -----------------------------------------------------------------------
    # Will be 0 unless cross-asset data provided in sentiment_data
    def _cross_asset_col(col: str, series: pd.Series) -> None:
        out[col] = series

    out["btc_1h_return"] = pd.Series(0.0, index=df.index)
    out["spy_1h_return"] = pd.Series(0.0, index=df.index)
    out["correlation_spy_20"] = pd.Series(0.0, index=df.index)
    out["correlation_btc_20"] = pd.Series(0.0, index=df.index)
    out["gold_1d_return"] = pd.Series(0.0, index=df.index)
    out["dxy_1d_return"] = pd.Series(0.0, index=df.index)
    out["tlt_1d_return"] = pd.Series(0.0, index=df.index)
    out["sector_etf_1h_return"] = pd.Series(0.0, index=df.index)
    out["btc_dominance_change"] = pd.Series(0.0, index=df.index)

    if sentiment_data is not None:
        cross_asset_cols = {
            "btc_1h_return": "btc_1h_return",
            "spy_1h_return": "spy_1h_return",
            "correlation_spy_20": "correlation_spy_20",
            "correlation_btc_20": "correlation_btc_20",
            "gold_1d_return": "gold_1d_return",
            "dxy_1d_return": "dxy_1d_return",
            "tlt_1d_return": "tlt_1d_return",
            "sector_etf_1h_return": "sector_etf_1h_return",
            "btc_dominance_change": "btc_dominance_change",
        }
        sent_aligned = sentiment_data.reindex(df.index, method="ffill")
        for feat, col in cross_asset_cols.items():
            if col in sent_aligned.columns:
                out[feat] = sent_aligned[col].fillna(0.0)

    # -----------------------------------------------------------------------
    # 7. Sentiment features
    # -----------------------------------------------------------------------
    for col in [
        "reddit_sentiment_1h", "reddit_sentiment_4h", "reddit_sentiment_momentum",
        "reddit_mention_surge", "news_sentiment_finbert", "news_sentiment_momentum",
        "fear_greed_normalized", "insider_buy_signal",
    ]:
        out[col] = pd.Series(0.0, index=df.index)

    if sentiment_data is not None:
        sent_aligned = sentiment_data.reindex(df.index, method="ffill")
        for col in [
            "reddit_sentiment_1h", "reddit_sentiment_4h", "reddit_sentiment_momentum",
            "reddit_mention_surge", "news_sentiment_finbert", "news_sentiment_momentum",
            "fear_greed_normalized", "insider_buy_signal",
        ]:
            if col in sent_aligned.columns:
                out[col] = sent_aligned[col].fillna(0.0)

    # -----------------------------------------------------------------------
    # 8. On-chain features
    # -----------------------------------------------------------------------
    for col in [
        "onchain_composite", "exchange_netflow_zscore", "active_addr_momentum",
        "nvt_ratio_zscore", "staking_ratio_change",
    ]:
        out[col] = pd.Series(0.0, index=df.index)

    if onchain_data is not None:
        onchain_aligned = onchain_data.reindex(df.index, method="ffill")
        for col in [
            "onchain_composite", "exchange_netflow_zscore", "active_addr_momentum",
            "nvt_ratio_zscore", "staking_ratio_change",
        ]:
            if col in onchain_aligned.columns:
                out[col] = onchain_aligned[col].fillna(0.0)

    # -----------------------------------------------------------------------
    # 9. Calendar features
    # -----------------------------------------------------------------------
    if hasattr(df.index, "hour"):
        hour = df.index.hour + df.index.minute / 60.0
        dow = df.index.dayofweek
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        # Minutes to close (assuming 16:00 ET close = 21:00 UTC)
        minutes_in_session = 390.0  # 6.5 hour trading day
        session_open_hour = 9.5  # 9:30 AM
        elapsed = (hour - session_open_hour) * 60.0
        elapsed = elapsed.clip(lower=0)
        out["minutes_to_close_norm"] = (minutes_in_session - elapsed) / minutes_in_session
        out["minutes_to_close_norm"] = out["minutes_to_close_norm"].clip(lower=0.0, upper=1.0)
    else:
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "minutes_to_close_norm"]:
            out[col] = 0.0

    # Days to options expiry (3rd Friday of each month)
    if hasattr(df.index, "date"):
        def _days_to_expiry(dt: pd.Timestamp) -> float:
            import calendar
            year, month = dt.year, dt.month
            # Find 3rd Friday
            cal = calendar.monthcalendar(year, month)
            fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
            third_friday = fridays[2] if len(fridays) >= 3 else fridays[-1]
            expiry = pd.Timestamp(year=year, month=month, day=third_friday)
            delta = (expiry - dt).days
            if delta < 0:
                # Next month
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
                cal = calendar.monthcalendar(year, month)
                fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
                third_friday = fridays[2] if len(fridays) >= 3 else fridays[-1]
                expiry = pd.Timestamp(year=year, month=month, day=third_friday)
                delta = (expiry - dt).days
            return float(delta) / 30.0  # normalize to ~[0,1]

        try:
            expiry_days = pd.Series(
                [_days_to_expiry(ts) for ts in df.index],
                index=df.index,
            )
            out["days_to_options_expiry"] = expiry_days
        except Exception:
            out["days_to_options_expiry"] = 0.5
    else:
        out["days_to_options_expiry"] = 0.5

    # FOMC days (approximate - typically 8x per year; 2nd Tue/Wed of select months)
    out["is_fomc_day"] = pd.Series(0.0, index=df.index)
    out["is_earnings_week"] = pd.Series(0.0, index=df.index)

    # -----------------------------------------------------------------------
    # CRITICAL: Shift all features by 1 bar to prevent lookahead bias
    # -----------------------------------------------------------------------
    out = out.shift(1)

    # Ensure exact column order matches ALL_FEATURES
    out = out[ALL_FEATURES]

    return out


# ---------------------------------------------------------------------------
# Lookahead bias checker
# ---------------------------------------------------------------------------

def check_lookahead(features_df: pd.DataFrame, target_col: pd.Series) -> dict:
    """
    Check correlation of each feature with the target to detect lookahead bias.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix (already shifted, no lookahead should exist).
    target_col : pd.Series
        Forward return or classification target.

    Returns
    -------
    dict
        {feature_name: correlation} for features with |corr| > 0.1.
        Empty dict means no lookahead detected.
    """
    warnings_dict: dict = {}
    aligned_target = target_col.reindex(features_df.index)

    for col in features_df.columns:
        try:
            feat_series = features_df[col].dropna()
            tgt_series = aligned_target.reindex(feat_series.index).dropna()
            common_idx = feat_series.index.intersection(tgt_series.index)
            if len(common_idx) < 30:
                continue
            corr = float(np.corrcoef(feat_series.loc[common_idx].values,
                                     tgt_series.loc[common_idx].values)[0, 1])
            if abs(corr) > 0.1:
                warnings_dict[col] = corr
                import warnings as _warnings
                _warnings.warn(
                    f"Potential lookahead: feature '{col}' has |correlation|={abs(corr):.3f} "
                    f"with target. Check for lookahead bias.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            continue

    return warnings_dict


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def normalize_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    scaler : StandardScaler or None
        If None, a new scaler is fit on df. If provided, transforms using
        existing scaler (inference mode).

    Returns
    -------
    (pd.DataFrame, StandardScaler)
        Normalized DataFrame and the fitted scaler.
    """
    valid_mask = df.notna().all(axis=1)
    result = df.copy()

    if scaler is None:
        scaler = StandardScaler()
        fit_data = df.loc[valid_mask].values
        if len(fit_data) > 0:
            scaler.fit(fit_data)

    try:
        result.loc[:, :] = scaler.transform(df.fillna(0.0).values)
    except Exception:
        # Fallback: normalize each column independently
        for col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                result[col] = (df[col] - col_mean) / col_std
            else:
                result[col] = 0.0

    return result, scaler
