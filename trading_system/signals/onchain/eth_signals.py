"""
Ethereum on-chain signals.

Expected metrics_df columns:
    active_addr      : int/float  – number of active addresses (daily)
    exchange_inflow  : float      – ETH flowing into exchanges
    exchange_outflow : float      – ETH flowing out of exchanges
    tx_count         : int/float  – daily transaction count
    gas_fees_gwei    : float      – average gas fee in Gwei
    base_fee_gwei    : float      – EIP-1559 base fee in Gwei [optional]
    total_supply     : float      – total ETH supply (may decrease due to burn)
    staked_eth       : float      – ETH staked in beacon chain [optional]
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def compute_eth_onchain_signal(
    metrics_df: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite Ethereum on-chain signal.

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
    # 1. Active address momentum (same formula as BTC)
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
    # 2. Exchange flow (coins leaving exchanges = bullish accumulation)
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
        # Positive net flow = inflow exceeds outflow = selling pressure = bearish
        signals["exchange_flow"] = float(-np.tanh(net_flow_z / 2.0))
    else:
        signals["exchange_flow"] = 0.0

    # ------------------------------------------------------------------
    # 3. Transaction count momentum
    # ------------------------------------------------------------------
    if "tx_count" in df.columns and len(df) >= 30:
        tx = df["tx_count"].astype(float).replace(0, np.nan)
        tx_ma7 = tx.rolling(7).mean()
        tx_ma30 = tx.rolling(30).mean()
        last_tx_ma7 = float(tx_ma7.iloc[-1])
        last_tx_ma30 = float(tx_ma30.iloc[-1])
        if not np.isnan(last_tx_ma7) and not np.isnan(last_tx_ma30) and last_tx_ma30 > 0:
            tx_ratio = last_tx_ma7 / last_tx_ma30
        else:
            tx_ratio = 1.0
        # Rising tx count = increased network usage = bullish
        signals["tx_count_signal"] = float(np.tanh((tx_ratio - 1.0) * 5.0))
    else:
        signals["tx_count_signal"] = 0.0

    # ------------------------------------------------------------------
    # 4. Gas fee momentum (high gas = high demand = bullish congestion)
    # ------------------------------------------------------------------
    fee_col = "gas_fees_gwei" if "gas_fees_gwei" in df.columns else (
        "base_fee_gwei" if "base_fee_gwei" in df.columns else None
    )
    if fee_col is not None and len(df) >= 30:
        fees = df[fee_col].astype(float).replace(0, np.nan)
        fee_ma = fees.rolling(30).mean()
        current_fee = float(fees.iloc[-1])
        avg_fee = float(fee_ma.iloc[-1])
        if not np.isnan(avg_fee) and avg_fee > 0:
            fee_ratio = current_fee / avg_fee
        else:
            fee_ratio = 1.0
        signals["fee_momentum"] = float(np.tanh((fee_ratio - 1.0) * 2.0))
    else:
        signals["fee_momentum"] = 0.0

    # ------------------------------------------------------------------
    # 5. Staking ratio momentum (more ETH staked = less circulating = bullish)
    # ------------------------------------------------------------------
    if "staked_eth" in df.columns and "total_supply" in df.columns and len(df) >= 30:
        staked = df["staked_eth"].astype(float)
        total = df["total_supply"].astype(float).replace(0, np.nan)
        stake_ratio = staked / total
        stake_ma7 = stake_ratio.rolling(7).mean()
        stake_ma30 = stake_ratio.rolling(30).mean()
        last_ma7 = float(stake_ma7.iloc[-1])
        last_ma30 = float(stake_ma30.iloc[-1])
        if not np.isnan(last_ma7) and not np.isnan(last_ma30) and last_ma30 > 0:
            stake_momentum = (last_ma7 / last_ma30) - 1.0
            signals["stake_momentum"] = float(np.tanh(stake_momentum * 10.0))
        else:
            signals["stake_momentum"] = 0.0
    else:
        signals["stake_momentum"] = 0.0

    # ------------------------------------------------------------------
    # Composite: weighted average of available signals
    # ------------------------------------------------------------------
    weights = {
        "addr_momentum": 0.25,
        "exchange_flow": 0.25,
        "tx_count_signal": 0.20,
        "fee_momentum": 0.15,
        "stake_momentum": 0.15,
    }

    weighted_sum = sum(weights[k] * signals[k] for k in weights)
    total_weight = sum(weights.values())
    composite = float(np.clip(weighted_sum / total_weight, -1.0, 1.0))

    return (composite, signals)
