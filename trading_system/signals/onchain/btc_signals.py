"""
Bitcoin on-chain signals.

Expected metrics_df columns:
    active_addr   : int/float  – number of active addresses (daily)
    exchange_inflow  : float   – BTC flowing into exchanges
    exchange_outflow : float   – BTC flowing out of exchanges
    transaction_volume: float  – total on-chain transaction volume (BTC)
    market_cap    : float      – BTC market cap (USD)
    miner_fees    : float      – total miner fees (BTC)
    block_reward  : float      – total block reward (BTC) [optional, for fee ratio]
    supply_active : float      – supply active in last 1yr (BTC)
    total_supply  : float      – total circulating supply (BTC)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def compute_btc_onchain_signal(
    metrics_df: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite Bitcoin on-chain signal.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Daily on-chain metrics (see module docstring for columns).
        Must have at least 30 rows for meaningful signals.

    Returns
    -------
    (composite_signal, signals_dict)
        composite_signal : float in [-1, +1]
        signals_dict     : dict mapping signal name → float in [-1, +1]
    """
    signals: Dict[str, float] = {}
    df = metrics_df.copy()

    # ------------------------------------------------------------------
    # 1. Address momentum
    # ------------------------------------------------------------------
    if "active_addr" in df.columns and len(df) >= 30:
        addr = df["active_addr"].astype(float).replace(0, np.nan)
        addr_ma7 = addr.rolling(7).mean()
        addr_ma30 = addr.rolling(30).mean()
        ratio = (addr_ma7.iloc[-1] / (addr_ma30.iloc[-1] + 1e-8)) - 1.0
        signals["addr_momentum"] = float(np.tanh(ratio * 10.0))
    else:
        signals["addr_momentum"] = 0.0

    # ------------------------------------------------------------------
    # 2. Exchange flow (negative = bullish: coins leaving exchanges)
    # ------------------------------------------------------------------
    if "exchange_inflow" in df.columns and "exchange_outflow" in df.columns:
        net_flow = df["exchange_inflow"].astype(float) - df["exchange_outflow"].astype(float)
        net_flow_mean = net_flow.rolling(30).mean()
        net_flow_std = net_flow.rolling(30).std()
        last_flow = float(net_flow.iloc[-1])
        last_mean = float(net_flow_mean.iloc[-1])
        last_std = float(net_flow_std.iloc[-1])
        if last_std > 0 and not np.isnan(last_mean):
            net_flow_z = (last_flow - last_mean) / last_std
        else:
            net_flow_z = 0.0
        # Negative flow (coins leaving exchanges) → bullish
        signals["exchange_flow"] = float(-np.tanh(net_flow_z / 2.0))
    else:
        signals["exchange_flow"] = 0.0

    # ------------------------------------------------------------------
    # 3. NVT proxy (Network Value to Transactions ratio)
    # ------------------------------------------------------------------
    if "transaction_volume" in df.columns and "market_cap" in df.columns:
        tx_vol = df["transaction_volume"].astype(float).replace(0, np.nan)
        mktcap = df["market_cap"].astype(float).replace(0, np.nan)
        nvt = mktcap / (tx_vol + 1e-8)
        nvt_ma = nvt.rolling(30).mean()
        current_nvt = float(nvt.iloc[-1])
        avg_nvt = float(nvt_ma.iloc[-1])
        if not np.isnan(avg_nvt) and avg_nvt > 0:
            nvt_ratio = current_nvt / avg_nvt
        else:
            nvt_ratio = 1.0
        # High NVT relative to average = overvalued = bearish
        signals["nvt_proxy"] = float(-np.tanh((nvt_ratio - 1.0) * 3.0))
    else:
        signals["nvt_proxy"] = 0.0

    # ------------------------------------------------------------------
    # 4. Miner activity (fee pressure = network demand)
    # ------------------------------------------------------------------
    if "miner_fees" in df.columns:
        fees = df["miner_fees"].astype(float).replace(0, np.nan)
        fee_ma = fees.rolling(30).mean()
        current_fee = float(fees.iloc[-1])
        avg_fee = float(fee_ma.iloc[-1])
        if not np.isnan(avg_fee) and avg_fee > 0:
            fee_ratio = current_fee / avg_fee
        else:
            fee_ratio = 1.0
        # High fees = high demand = bullish
        signals["miner_activity"] = float(np.tanh((fee_ratio - 1.0) * 2.0))
    else:
        signals["miner_activity"] = 0.0

    # ------------------------------------------------------------------
    # 5. HODL wave (coins not moving = accumulation = bullish)
    # ------------------------------------------------------------------
    if "supply_active" in df.columns and "total_supply" in df.columns:
        supply_active = df["supply_active"].astype(float)
        total_supply = df["total_supply"].astype(float)
        supply_active_ratio = (supply_active / (total_supply + 1e-8)).iloc[-1]
        if not np.isnan(supply_active_ratio):
            # Low active ratio → more HODLing → bullish
            signals["hodl_wave"] = float(np.tanh((1.0 - float(supply_active_ratio)) * 5.0))
        else:
            signals["hodl_wave"] = 0.0
    else:
        signals["hodl_wave"] = 0.0

    # ------------------------------------------------------------------
    # Composite: equal-weighted average of available signals
    # ------------------------------------------------------------------
    weights = {
        "addr_momentum": 0.25,
        "exchange_flow": 0.25,
        "nvt_proxy": 0.20,
        "miner_activity": 0.15,
        "hodl_wave": 0.15,
    }

    weighted_sum = sum(weights[k] * signals[k] for k in weights)
    total_weight = sum(weights.values())
    composite = float(np.clip(weighted_sum / total_weight, -1.0, 1.0))

    return (composite, signals)
