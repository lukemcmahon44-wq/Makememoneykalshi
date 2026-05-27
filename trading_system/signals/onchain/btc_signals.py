"""
BTC on-chain composite signal from CoinMetrics data.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .._common import EPS, tanh_clip


def compute_btc_onchain_signal(metrics_df: pd.DataFrame
                                ) -> Tuple[float, Dict[str, float]]:
    """
    metrics_df: indexed by date with columns AdrActCnt, TxCnt,
                FlowInExNtv, FlowOutExNtv, SplyAct1yr, FeeTotNtv.
    Returns (composite_signal in [-1, +1], breakdown dict).
    """
    signals: Dict[str, float] = {}
    if metrics_df is None or len(metrics_df) < 35:
        return 0.0, signals

    df = metrics_df.copy().sort_index()

    # 1. Active address momentum
    if "AdrActCnt" in df.columns:
        addr_ma7 = float(df["AdrActCnt"].rolling(7).mean().iloc[-1])
        addr_ma30 = float(df["AdrActCnt"].rolling(30).mean().iloc[-1])
        if addr_ma30 > 0:
            signals["addr_momentum"] = float(tanh_clip(
                (addr_ma7 / addr_ma30 - 1) * 10, 1.0))

    # 2. Exchange net flow (outflow bullish)
    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        net = df["FlowInExNtv"] - df["FlowOutExNtv"]
        mu = float(net.rolling(30).mean().iloc[-1])
        sd = float(net.rolling(30).std().iloc[-1] + EPS)
        z = (float(net.iloc[-1]) - mu) / sd if sd > 0 else 0.0
        signals["exchange_flow"] = float(-tanh_clip(z, 2.0))

    # 3. NVT proxy
    if "TxCnt" in df.columns:
        nvt = float((df["TxCnt"].rolling(28).mean().iloc[-1]
                     / max(float(df["TxCnt"].rolling(90).mean().iloc[-1]), EPS)))
        signals["nvt_proxy"] = float(-tanh_clip((nvt - 1) * 3, 1.0))

    # 4. Miner activity
    if "FeeTotNtv" in df.columns:
        fma7 = float(df["FeeTotNtv"].rolling(7).mean().iloc[-1])
        fma30 = float(df["FeeTotNtv"].rolling(30).mean().iloc[-1])
        if fma30 > 0:
            signals["miner_activity"] = float(tanh_clip((fma7 / fma30 - 1) * 2, 1.0))

    # 5. HODL wave
    if "SplyAct1yr" in df.columns:
        ratio = float(df["SplyAct1yr"].iloc[-1]
                       / max(float(df["SplyAct1yr"].rolling(90).mean().iloc[-1]), EPS))
        signals["hodl_wave"] = float(tanh_clip((1 - ratio) * 5, 1.0))

    if not signals:
        return 0.0, signals
    composite = float(np.clip(np.mean(list(signals.values())), -1.0, 1.0))
    return composite, signals
